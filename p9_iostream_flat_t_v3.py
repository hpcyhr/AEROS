#!/usr/bin/env python
"""
AEROS §5.4 — IO-streaming flat-T experiment, cross-family headline.

For each of the 16 archs (10 SNN + 6 extended stateful sequence networks),
sweep logical horizon T at fixed κ=8 under two modes:

  Mode 2 (retain-IO baseline): input + output retained on GPU; peak HBM
                                grows linearly with T at slope α_in × T.
  Mode 4 (IO-streaming):       input + output streamed (CPU-resident,
                                per-segment H2D / output sink); peak HBM
                                should be nearly invariant w.r.t. T --
                                only M_0 + (α_in + α_K + α_out) × κ.

The paper's cross-family headline: Mode 4 collapses peak HBM growth from
O(T) to O(κ) for ALL stateful sequence families tested, including SNN,
ConvLSTM, ConvGRU, LSTM, GRU, CausalTCN, and minimal SSM.

Per-arch reports:
  - mode2_peaks_GB[T]  — linear in T
  - mode4_peaks_GB[T]  — flat in T
  - mode2_slope_MB_per_T  — large (input-residency dominant)
  - mode4_slope_MB_per_T  — small (residency contract holds)
  - T_max_reduction_ratio = mode2_peak[T_max] / mode4_peak[T_max]

Cross-family aggregate:
  - median mode2 slope, median mode4 slope, median reduction ratio at T_max

Usage:
  python p9_iostream_flat_t.py \\
      --coeffs p9_1a_full16.json \\
      --Ts 128 256 512 1024 2048 4096 8192 16384 \\
      --kappa 8 \\
      --output p9_iostream_flat_t
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18, spiking_resnet34, spiking_resnet50)
    from spikingjelly.activation_based.model.sew_resnet import (
        sew_resnet18, sew_resnet50, sew_resnet101)
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg11_bn, spiking_vgg13_bn, spiking_vgg16_bn,
        spiking_vgg19_bn)
except Exception as e:
    print(f"[WARN] SJ unavailable: {e}")
    sys.exit(1)

try:
    sys.path.insert(0, ".")
    from aeros_dispatch import (load_unified_bundle, get_arch_config,
                                  build_any_net, reset_any_net)
    DISPATCH_OK = True
except Exception as e:
    print(f"[FATAL] aeros_dispatch required: {e}")
    sys.exit(1)


BYTES_PER_GB = 1024 ** 3


def reset_state(net, provenance="snn"):
    if provenance == "extended":
        reset_any_net(net, provenance)
        return
    try:
        functional.reset_net(net)
    except Exception:
        pass


def forward_segmented(net, x, mode_id, kappa, device, provenance="snn"):
    T = x.shape[0]
    if mode_id == 1 or kappa >= T:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state(net, provenance)
        return net(x)
    if mode_id == 2:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state(net, provenance)
        chunks = []
        i = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous()
            chunks.append(net(x_seg))
            del x_seg
            i += sz
        return torch.cat(chunks, dim=0)
    if mode_id == 4:
        assert x.device.type == "cpu", \
            f"mode 4 streaming requires CPU x; got {x.device}"
        reset_state(net, provenance)
        i = 0
        sink = None
        n = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            y_seg = net(x_seg)
            if sink is None:
                sink = y_seg.sum(dim=0)
                n = sz
            else:
                sink = sink + y_seg.sum(dim=0)
                n += sz
            del x_seg, y_seg
            i += sz
        return sink / n
    raise ValueError(f"unknown mode {mode_id}")


@torch.no_grad()
def measure_peak(net, x, mode_id, kappa, device, provenance="snn"):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        _ = forward_segmented(net, x, mode_id, kappa, device, provenance)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
        return peak, "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return -1.0, "oom"
    except Exception as e:
        torch.cuda.empty_cache()
        return -1.0, f"error:{type(e).__name__}"


def run_arch_T_sweep(net_name, cfg, T_list, kappa, device,
                     skip_t_above_ms=120000):
    """Per-arch sweep across T for both Mode 2 and Mode 4.

    cfg: {b, C, H, num_classes, provenance}
    Returns dict with per-T peaks (mode2 + mode4) and slope estimates.

    skip_t_above_ms: if previous T's mode-4 forward exceeded this wall-time,
    skip larger T (saves time on CausalTCN at T=16384).
    """
    out = {
        "net": net_name, "provenance": cfg["provenance"],
        "T_list": [], "mode2_peaks_GB": [], "mode4_peaks_GB": [],
        "mode2_wall_ms": [], "mode4_wall_ms": [],
        "errors": [],
    }
    last_mode4_ms = 0.0
    skip_remaining = False
    for T in T_list:
        if skip_remaining:
            out["errors"].append((T, "skipped (previous T too slow)"))
            continue

        # Build net for this (arch, T) — fresh state guaranteed
        net = build_any_net(net_name, cfg["provenance"],
                             num_classes=cfg["num_classes"],
                             H=cfg["H"], C=cfg["C"]).to(device)

        # Mode 2: retain-IO baseline (x on GPU)
        # NOTE: x_gpu allocation itself can OOM at very large T (e.g.
        # SNN suite at T=16384, b=32, H=128 needs 96 GB just for input).
        # That's actually a positive signal: it shows Mode 2 has hit the
        # physical residency wall well before Mode 4. We catch it
        # gracefully and continue to Mode 4.
        peak2 = -1.0
        ms2 = -1.0
        status2 = "ok"
        try:
            x_gpu = torch.randn(T, cfg["b"], cfg["C"], cfg["H"], cfg["H"],
                                 device=device)
            t0 = time.time()
            peak2, status2 = measure_peak(net, x_gpu, 2, kappa, device,
                                           cfg["provenance"])
            ms2 = (time.time() - t0) * 1000
            del x_gpu
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            status2 = "oom_alloc"
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   M2 input alloc OOM at T={T} (input alone exceeds HBM); "
                  f"M4 streaming may still succeed")

        if status2.startswith("error") or status2 == "oom" \
                or status2 == "oom_alloc" or peak2 < 0:
            # Record the M2 failure but DON'T skip to next T — we still
            # try M4 (input-streaming) since M4 is precisely the design
            # for cases where M2 fails.
            if status2 != "oom_alloc":
                # other error type: full skip
                print(f"   M2 status={status2} T={T}; skip larger T")
                out["errors"].append((T, f"mode2 {status2}"))
                del net
                torch.cuda.empty_cache()
                gc.collect()
                skip_remaining = True
                continue
            # status2 == oom_alloc: continue to try M4
            out["errors"].append((T, f"mode2 oom_alloc (input>{cfg['H']}^2 too large)"))

        # Mode 4: IO-streaming (x on CPU)
        # Even CPU alloc can fail at very large T for SNN suite at H=128
        # (T=16384 b=32 C=3 H=128 fp32 = 103 GB, exceeds host RAM).
        # That's a real-deployment finding: "even with input streaming,
        # truly large T requires disk-resident input." We catch it and
        # skip larger T cleanly.
        peak4 = -1.0
        ms4 = -1.0
        status4 = "ok"
        try:
            x_cpu = torch.randn(T, cfg["b"], cfg["C"], cfg["H"], cfg["H"],
                                 device="cpu")
            t0 = time.time()
            peak4, status4 = measure_peak(net, x_cpu, 4, kappa, device,
                                           cfg["provenance"])
            ms4 = (time.time() - t0) * 1000
            del x_cpu
        except (RuntimeError, MemoryError) as e:
            status4 = "oom_cpu_alloc"
            print(f"   M4 CPU input alloc failed at T={T}: {type(e).__name__} "
                  f"(input >host RAM); marking and skipping larger T")
            out["errors"].append((T, f"mode4 oom_cpu_alloc"))
            del net
            torch.cuda.empty_cache()
            gc.collect()
            skip_remaining = True
            continue
        last_mode4_ms = ms4

        if status4.startswith("error") or peak4 < 0:
            print(f"   M4 status={status4}")
            out["errors"].append((T, f"mode4 {status4}"))
            del net
            torch.cuda.empty_cache()
            gc.collect()
            continue

        out["T_list"].append(T)
        out["mode2_peaks_GB"].append(peak2)  # -1 if OOM'd
        out["mode4_peaks_GB"].append(peak4)
        out["mode2_wall_ms"].append(ms2)
        out["mode4_wall_ms"].append(ms4)

        if peak2 > 0:
            ratio = peak2 / peak4 if peak4 > 0 else float("inf")
            print(f"   T={T:>5}  M2 peak={peak2:.3f}GB ({ms2:.0f}ms)  "
                  f"M4 peak={peak4:.3f}GB ({ms4:.0f}ms)  ratio={ratio:.2f}x")
        else:
            # M2 OOM'd alloc — Mode 4 succeeds where Mode 2 cannot.
            # This is a key qualitative finding for the headline.
            print(f"   T={T:>5}  M2 OOM (input > HBM)         "
                  f"M4 peak={peak4:.3f}GB ({ms4:.0f}ms)  M4 succeeds where M2 fails")

        del net
        torch.cuda.empty_cache()
        gc.collect()

        if last_mode4_ms > skip_t_above_ms:
            print(f"   [time-skip] mode4 took {last_mode4_ms:.0f}ms > "
                  f"{skip_t_above_ms}ms; skipping larger T")
            skip_remaining = True

    # Compute slopes via linear regression over T
    if len(out["T_list"]) >= 2:
        Ts = np.array(out["T_list"], dtype=np.float64)
        m2 = np.array(out["mode2_peaks_GB"]) * 1024  # MB
        m4 = np.array(out["mode4_peaks_GB"]) * 1024  # MB
        # Filter to valid M2 entries only (peak2 > 0); M4 should be all valid
        m2_valid_mask = m2 > 0
        m4_valid_mask = m4 > 0

        # M2 slope from valid M2 cells
        if m2_valid_mask.sum() >= 2:
            A2 = np.vstack([np.ones(m2_valid_mask.sum()),
                             Ts[m2_valid_mask]]).T
            b2 = np.linalg.lstsq(A2, m2[m2_valid_mask], rcond=None)[0][1]
            out["mode2_slope_MB_per_T"] = float(b2)
        else:
            out["mode2_slope_MB_per_T"] = float("nan")

        # M4 slope from all M4 cells
        if m4_valid_mask.sum() >= 2:
            A4 = np.vstack([np.ones(m4_valid_mask.sum()),
                             Ts[m4_valid_mask]]).T
            b4 = np.linalg.lstsq(A4, m4[m4_valid_mask], rcond=None)[0][1]
            out["mode4_slope_MB_per_T"] = float(b4)
        else:
            out["mode4_slope_MB_per_T"] = float("nan")

        out["T_max_tested"] = int(max(out["T_list"]))

        # Reduction ratio at the largest T where BOTH M2 and M4 succeeded
        m2_arr = np.array(out["mode2_peaks_GB"])
        m4_arr = np.array(out["mode4_peaks_GB"])
        both_valid = (m2_arr > 0) & (m4_arr > 0)
        if both_valid.any():
            valid_indices = np.where(both_valid)[0]
            last_valid = valid_indices[-1]
            out["T_max_both_valid"] = int(out["T_list"][last_valid])
            out["T_max_reduction_ratio"] = float(
                m2_arr[last_valid] / m4_arr[last_valid])
        else:
            out["T_max_both_valid"] = 0
            out["T_max_reduction_ratio"] = float("nan")

        # Also report Mode 2 OOM-onset T (qualitative headline:
        # "Mode 2 fails to allocate input at T=X, but Mode 4 succeeds at T_max")
        m2_oom_Ts = [out["T_list"][i] for i, v in enumerate(m2_arr) if v < 0]
        out["mode2_first_oom_T"] = int(min(m2_oom_Ts)) if m2_oom_Ts else None
        out["mode4_max_succeeded_T"] = int(max(
            [out["T_list"][i] for i, v in enumerate(m4_arr) if v > 0]
            or [0]))
    else:
        out["mode2_slope_MB_per_T"] = float("nan")
        out["mode4_slope_MB_per_T"] = float("nan")
        out["T_max_tested"] = 0
        out["T_max_reduction_ratio"] = float("nan")
        out["T_max_both_valid"] = 0
        out["mode2_first_oom_T"] = None
        out["mode4_max_succeeded_T"] = 0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True,
                        help="path to unified coefficient bundle")
    parser.add_argument("--output", default="p9_iostream_flat_t")
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--nets", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_above_ms", type=float, default=120000,
                        help="skip larger T if mode4 forward > this ms "
                             "(default 120000 = 2 min)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)

    bundle = load_unified_bundle(args.coeffs)
    archs = list(bundle["coeffs"].keys())
    if args.nets:
        archs = [a for a in archs if a in args.nets]

    print("=" * 78)
    print("AEROS §5.4 — IO-streaming flat-T cross-family experiment")
    print("=" * 78)
    print(f"  archs ({len(archs)}): {archs}")
    print(f"  T_list: {args.Ts}")
    print(f"  kappa: {args.kappa}")
    print()

    per_arch = {}
    for net_name in archs:
        cfg = get_arch_config(bundle, net_name)
        print(f"\n=== {net_name} (prov={cfg['provenance']}, "
              f"b={cfg['b']}, C={cfg['C']}, H={cfg['H']}) ===")
        result = run_arch_T_sweep(net_name, cfg, args.Ts, args.kappa,
                                   device, args.skip_above_ms)
        per_arch[net_name] = result

        if not np.isnan(result.get("mode2_slope_MB_per_T", float("nan"))):
            print(f"   summary: M2 slope={result['mode2_slope_MB_per_T']:.3f}"
                  f" MB/T  M4 slope={result['mode4_slope_MB_per_T']:.3f} MB/T"
                  f"  ratio at T={result['T_max_tested']}: "
                  f"{result['T_max_reduction_ratio']:.2f}x")

    # Cross-family aggregate
    valid = [r for r in per_arch.values()
             if not np.isnan(r.get("mode2_slope_MB_per_T", float("nan")))]
    aggregate = {}
    if valid:
        m2_slopes = [r["mode2_slope_MB_per_T"] for r in valid]
        m4_slopes = [r["mode4_slope_MB_per_T"] for r in valid]
        ratios = [r["T_max_reduction_ratio"] for r in valid]
        aggregate = {
            "n_archs_valid": len(valid),
            "median_mode2_slope_MB_per_T": float(np.median(m2_slopes)),
            "median_mode4_slope_MB_per_T": float(np.median(m4_slopes)),
            "median_T_max_reduction_ratio": float(np.median(ratios)),
            "max_T_max_reduction_ratio": float(np.max(ratios)),
            "min_T_max_reduction_ratio": float(np.min(ratios)),
            "max_T_tested": max(r["T_max_tested"] for r in valid),
        }

    print()
    print("=" * 78)
    print("Cross-family summary")
    print("=" * 78)
    if valid:
        print(f"  Median M2 slope: {aggregate['median_mode2_slope_MB_per_T']:.3f} MB/T")
        print(f"  Median M4 slope: {aggregate['median_mode4_slope_MB_per_T']:.3f} MB/T")
        print(f"  Median M2/M4 reduction at max T: "
              f"{aggregate['median_T_max_reduction_ratio']:.2f}x")
        print(f"  M2/M4 ratio range: "
              f"{aggregate['min_T_max_reduction_ratio']:.2f}x — "
              f"{aggregate['max_T_max_reduction_ratio']:.2f}x")
        print(f"  Max T tested: {aggregate['max_T_tested']}")
        print()
        print(f"{'Arch':<14s} {'M2 slope':>12s} {'M4 slope':>12s} "
              f"{'Ratio':>10s} {'T_max':>8s}")
        for net_name, r in per_arch.items():
            if "mode2_slope_MB_per_T" in r and not np.isnan(r["mode2_slope_MB_per_T"]):
                print(f"  {net_name:<12s} "
                      f"{r['mode2_slope_MB_per_T']:>10.3f} MB "
                      f"{r['mode4_slope_MB_per_T']:>10.3f} MB "
                      f"{r['T_max_reduction_ratio']:>8.2f}x "
                      f"{r['T_max_tested']:>8}")

    output_data = {
        "experiment": "IO-streaming flat-T cross-family (§5.4 headline)",
        "config": vars(args),
        "per_arch": per_arch,
        "aggregate": aggregate,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()