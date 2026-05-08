#!/usr/bin/env python
"""
AEROS Phase 2 Exp 1A — Coefficient profiling.

For each (net, T, mode), fits the live-set memory coefficients
(M_0, alpha_K, alpha_in, alpha_out) by running a small kappa sweep
and least-squares fitting the policy-aware live-set model:

  M(T, kappa, pi) = A(T, pi) + B(pi) * kappa

For four modes:
  Mode 1: full-horizon -> M(T, T, retIO) = M_0 + (alpha_in + alpha_out)*T + alpha_K * T
                                         = (M_0) + (alpha_in + alpha_out + alpha_K) * T
  Mode 2: seg retIO    -> M(T, kappa, retIO) = M_0 + (alpha_in + alpha_out)*T + alpha_K * kappa
  Mode 3: input-stream -> M(T, kappa, in-stream) = M_0 + alpha_out*T + (alpha_in + alpha_K) * kappa
  Mode 4: IO-stream    -> M(T, kappa, io-stream) = M_0 + (alpha_in + alpha_K + alpha_out) * kappa

Strategy: Run mode 2 (segmented retIO) at multiple kappa values for fixed T
to fit alpha_K (slope) and (M_0 + (alpha_in + alpha_out) * T) (offset).
Then (alpha_in + alpha_out)*T is computed from input/output bytes and shape.

For mode 4 we directly run at multiple kappa to fit
(alpha_in + alpha_K + alpha_out) (slope) and M_0 (offset).

Usage:
    python p9_1a_coeff_profile.py --output p9_1a_coeffs
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18, spiking_resnet34, spiking_resnet50)
    from spikingjelly.activation_based.model.sew_resnet import (
        sew_resnet18, sew_resnet50, sew_resnet101)
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg11_bn, spiking_vgg13_bn, spiking_vgg16_bn,
        spiking_vgg19_bn)
    SJ_OK = True
except Exception as e:
    print(f"[WARN] SJ unavailable: {e}")
    SJ_OK = False


# ============================================================================
# Network builders (subset — full 17 set takes long; here we cover the 10
# core SJ standard SNNs; AlexNet/ZFNet/MobileNet/Transformers can be added
# from CATFuse imports if available)
# ============================================================================

def _bn(name, fn):
    """Helper: build, eval, set step_mode='m'."""
    def _build():
        net = fn()
        net.eval()
        functional.set_step_mode(net, "m")
        return net
    _build.__name__ = name
    return _build


def build_nets() -> Dict[str, callable]:
    out = {}
    if not SJ_OK:
        return out
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=10,
    )
    out["SR-18"]   = _bn("SR-18",  lambda: spiking_resnet18(**common))
    out["SR-34"]   = _bn("SR-34",  lambda: spiking_resnet34(**common))
    out["SR-50"]   = _bn("SR-50",  lambda: spiking_resnet50(**common))
    out["SEW-18"]  = _bn("SEW-18", lambda: sew_resnet18(cnf="ADD", **common))
    out["SEW-50"]  = _bn("SEW-50", lambda: sew_resnet50(cnf="ADD", **common))
    out["SEW-101"] = _bn("SEW-101",lambda: sew_resnet101(cnf="ADD", **common))
    out["VGG-11-BN"] = _bn("VGG-11-BN", lambda: spiking_vgg11_bn(**common))
    out["VGG-13-BN"] = _bn("VGG-13-BN", lambda: spiking_vgg13_bn(**common))
    out["VGG-16-BN"] = _bn("VGG-16-BN", lambda: spiking_vgg16_bn(**common))
    out["VGG-19-BN"] = _bn("VGG-19-BN", lambda: spiking_vgg19_bn(**common))
    return out


def reset_state(net):
    try:
        functional.reset_net(net)
    except Exception:
        pass


# ============================================================================
# Mode runners (same as Exp 4, single-iter peak measurement per Doris 7 P0-3)
# ============================================================================

@torch.no_grad()
def run_mode(net, T, kappa, b, C, H, mode_id, device, num_classes=10):
    """Run one (T, kappa, mode) cell. Returns (peak_bytes or -1 if OOM, wall_ms)."""
    try:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        reset_state(net)

        t0 = time.time()
        if mode_id == 1:
            # Mode 1: full horizon, kappa = T
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            y = net(x)
            torch.cuda.synchronize(device)
            del x, y
        elif mode_id == 2:
            # Mode 2: segmented retained-IO
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            chunks = []
            i = 0
            while i < T:
                sz = min(kappa, T - i)
                chunks.append(net(x[i:i + sz]))
                i += sz
            y = torch.cat(chunks, dim=0)
            torch.cuda.synchronize(device)
            del x, y, chunks
        elif mode_id == 3:
            # Mode 3: input-streaming (full output retained)
            chunks = []
            g = torch.Generator(device=device).manual_seed(42)
            i = 0
            while i < T:
                sz = min(kappa, T - i)
                x_seg = torch.randn(sz, b, C, H, H, generator=g,
                                    device=device, dtype=torch.float32)
                chunks.append(net(x_seg))
                del x_seg
                i += sz
            y = torch.cat(chunks, dim=0)
            torch.cuda.synchronize(device)
            del y, chunks
        elif mode_id == 4:
            # Mode 4: IO-streaming (sink)
            g = torch.Generator(device=device).manual_seed(42)
            running_sum = torch.zeros(b, num_classes, device=device, dtype=torch.float32)
            n = 0; i = 0
            while i < T:
                sz = min(kappa, T - i)
                x_seg = torch.randn(sz, b, C, H, H, generator=g,
                                    device=device, dtype=torch.float32)
                y_seg = net(x_seg)
                running_sum += y_seg.sum(dim=0)
                n += sz
                del x_seg, y_seg
                i += sz
            torch.cuda.synchronize(device)
            del running_sum
        wall_ms = (time.time() - t0) * 1000
        peak = torch.cuda.max_memory_allocated(device)
        return peak, wall_ms
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return -1, -1.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache(); gc.collect()
            return -1, -1.0
        raise


# ============================================================================
# Coefficient fitting
# ============================================================================

def fit_coefficients(measurements: Dict, T: int, b: int, C: int, H: int,
                     num_classes: int = 10) -> Dict:
    """Given mode-2 measurements at multiple kappa, fit (M_0, alpha_K, alpha_in, alpha_out).

    Mode 2: M(T, kappa) = (M_0 + (alpha_in + alpha_out)*T) + alpha_K * kappa
    Linear in kappa: y = a + b*kappa
      a = M_0 + (alpha_in + alpha_out)*T
      b = alpha_K
    alpha_in is computed analytically from input shape (b*C*H*H*4 bytes/step).
    alpha_out is computed analytically from (b*num_classes*4 bytes/step).
    Then M_0 = a - (alpha_in + alpha_out)*T.
    """
    BYTES_PER_GB = 1024 ** 3
    alpha_in_bytes = b * C * H * H * 4    # bytes per timestep for input
    alpha_out_bytes = b * num_classes * 4  # bytes per timestep for output

    mode2 = measurements.get(2, {})
    if len(mode2) < 2:
        return {"error": "insufficient mode-2 data"}

    kappas = sorted(mode2.keys())
    peaks = np.array([mode2[k] for k in kappas], dtype=np.float64)
    valid = peaks > 0
    if valid.sum() < 2:
        return {"error": "insufficient valid mode-2 data after OOM filtering"}

    kappas_arr = np.array(kappas, dtype=np.float64)
    A_mat = np.vstack([np.ones_like(kappas_arr[valid]),
                       kappas_arr[valid]]).T
    coef, *_ = np.linalg.lstsq(A_mat, peaks[valid], rcond=None)
    a_offset, alpha_K = coef[0], coef[1]
    M_0 = a_offset - (alpha_in_bytes + alpha_out_bytes) * T

    return {
        "M_0_bytes":     float(M_0),
        "alpha_K_bytes": float(alpha_K),
        "alpha_in_bytes":  float(alpha_in_bytes),
        "alpha_out_bytes": float(alpha_out_bytes),
        "fit_kappas":   kappas,
        "fit_peaks":    [int(p) for p in peaks],
        "fit_residuals_GB": [
            float((peaks[i] - (a_offset + alpha_K * kappas_arr[i])) / BYTES_PER_GB)
            for i in range(len(kappas)) if valid[i]
        ],
    }


def predict_peak(coeffs: Dict, T: int, kappa: int, mode_id: int) -> float:
    """Closed-form peak prediction in bytes using fitted coefficients."""
    if "error" in coeffs:
        return -1.0
    M_0 = coeffs["M_0_bytes"]
    aK = coeffs["alpha_K_bytes"]
    a_in = coeffs["alpha_in_bytes"]
    a_out = coeffs["alpha_out_bytes"]
    if mode_id == 1:
        return M_0 + (a_in + a_out) * T + aK * T
    elif mode_id == 2:
        return M_0 + (a_in + a_out) * T + aK * kappa
    elif mode_id == 3:
        return M_0 + a_out * T + (a_in + aK) * kappa
    elif mode_id == 4:
        return M_0 + (a_in + aK + a_out) * kappa
    return -1.0


# ============================================================================
# Main profiling loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T_sweep", default="128,1024,4096")
    parser.add_argument("--kappa_sweep", default="1,2,4,8,16,32",
                        help="kappa values for fitting (Mode 2)")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--nets", default="all")
    parser.add_argument("--output", default="p9_1a_coeffs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    builders = build_nets()
    if args.nets.lower() != "all":
        names = [n.strip() for n in args.nets.split(",")]
        builders = {k: v for k, v in builders.items() if k in names}

    Ts = [int(t) for t in args.T_sweep.split(",")]
    kappas = [int(k) for k in args.kappa_sweep.split(",")]

    print(f"=== AEROS Phase 2 Exp 1A — Coefficient Profiling ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  T sweep: {Ts}  kappa sweep (mode 2): {kappas}")
    print(f"  b={args.b}  C=3  H={args.H}")
    print(f"  Nets: {list(builders.keys())}")

    all_coeffs = {}
    all_raw = {}

    for net_name, builder in builders.items():
        print(f"\n{'='*72}\n=== {net_name} ===\n{'='*72}")
        try:
            net = builder().to(device)
        except Exception as e:
            print(f"  Failed to build: {e}")
            continue

        net_coeffs = {}
        net_raw = {}

        for T in Ts:
            print(f"\n  --- T={T} ---")
            kappa_to_use = [k for k in kappas if k <= T]
            mode2_measurements = {}
            for k in kappa_to_use:
                peak, wall = run_mode(net, T, k, args.b, 3, args.H,
                                      mode_id=2, device=device)
                mode2_measurements[k] = peak
                if peak > 0:
                    print(f"    kappa={k:4d}  peak={peak/1024**3:7.3f} GB  wall={wall:8.1f} ms")
                else:
                    print(f"    kappa={k:4d}  OOM")

            coeffs = fit_coefficients({2: mode2_measurements}, T, args.b, 3,
                                       args.H, num_classes=10)
            net_coeffs[T] = coeffs
            net_raw[T] = {2: mode2_measurements}

            if "error" not in coeffs:
                BYTES_PER_GB = 1024 ** 3
                print(f"    fit: M_0={coeffs['M_0_bytes']/BYTES_PER_GB:.3f}GB  "
                      f"alpha_K={coeffs['alpha_K_bytes']/BYTES_PER_GB*1000:.1f}MB/kappa  "
                      f"alpha_in={coeffs['alpha_in_bytes']/1024**2:.2f}MB/step  "
                      f"alpha_out={coeffs['alpha_out_bytes']:.0f}B/step")
                # Sanity: predict each mode-2 cell vs measured
                for k in kappa_to_use:
                    pred = predict_peak(coeffs, T, k, mode_id=2)
                    meas = mode2_measurements.get(k, -1)
                    if meas > 0:
                        err_gb = (meas - pred) / BYTES_PER_GB
                        print(f"      verify kappa={k}: pred={pred/BYTES_PER_GB:.3f}GB  "
                              f"meas={meas/BYTES_PER_GB:.3f}GB  err={err_gb:+.4f}GB")
            else:
                print(f"    fit error: {coeffs['error']}")

        all_coeffs[net_name] = net_coeffs
        all_raw[net_name] = net_raw
        del net
        torch.cuda.empty_cache(); gc.collect()

    # Save
    output = {
        "config": {
            "T_sweep": Ts, "kappa_sweep": kappas,
            "b": args.b, "C": 3, "H": args.H, "num_classes": 10,
        },
        "coeffs": all_coeffs,
        "raw": all_raw,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved JSON: {args.output}.json")

    # Summary table
    print(f"\n{'='*72}")
    print(f"=== Summary: alpha_K (GB/kappa) by (Net, T) ===")
    print(f"{'='*72}")
    BYTES_PER_GB = 1024 ** 3
    nets = sorted(all_coeffs.keys())
    print(f"  {'Net':<14s}" + "".join(f"  T={t:<6d}" for t in Ts))
    for n in nets:
        row = f"  {n:<14s}"
        for T in Ts:
            c = all_coeffs[n].get(T, {})
            if "error" in c:
                row += f"  {'ERR':<8s}"
            else:
                row += f"  {c['alpha_K_bytes']/BYTES_PER_GB:<8.4f}"
        print(row)


if __name__ == "__main__":
    main()