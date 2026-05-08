#!/usr/bin/env python
"""
AEROS Phase 2 Exp 1A — Coefficient profiling, EXTENDED suite.

Profiles the 6 non-SNN stateful sequence architectures
(ConvLSTM-2L, ConvGRU-2L, LSTM-4L, GRU-4L, CausalTCN-8L, MinimalSSM-2L)
defined in aeros_models_extended.py. Output JSON has the same schema as
p9_1a_full.json so the two bundles can be merged into a unified 16-arch
coefficient set.

Methodology mirrors p9_1a_coeff_profile.py:
  Mode 2 sweep over kappa for fixed T -> fit (M_0, alpha_K)
  alpha_in computed analytically from input shape
  alpha_out computed analytically from output shape

Critical config note: extended-suite archs default to H=32 (not H=128 as
the SNN suite uses), because LSTM/GRU/CausalTCN/SSM flatten spatial dims
and a 128x128 spatial input would inflate the feature dim to ~50K which
makes nn.LSTM weights unrealistically large. We use H=32 to keep param
counts comparable and the fit clean. The fit coefficients are
shape-specific; the disjoint-fold protocol still works because Ts and
kappas vary while (b, C, H) is fixed.

Usage:
  python p9_1a_extended.py --output p9_1a_extended

Then merge:
  python p9_1a_extended.py --merge p9_1a_full.json p9_1a_extended.json \\
      --merged_output p9_1a_full16.json
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


BYTES_PER_GB = 1024 ** 3


def setup_determinism(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.manual_seed(seed)


# ============================================================================
# Mode runners — adapted from p9_1a_coeff_profile.py for extended suite
# ============================================================================

@torch.no_grad()
def run_mode(net, T, kappa, b, C, H, mode_id, device, num_classes=10,
             reset_fn=None):
    """Run one (T, kappa, mode) cell. Returns (peak_bytes or -1 on OOM, wall_ms)."""
    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        if reset_fn is not None:
            reset_fn(net)

        t0 = time.time()
        if mode_id == 1:
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            y = net(x)
            torch.cuda.synchronize(device)
            del x, y
        elif mode_id == 2:
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
            g = torch.Generator(device=device).manual_seed(42)
            running_sum = torch.zeros(b, num_classes, device=device,
                                      dtype=torch.float32)
            n = 0
            i = 0
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
        torch.cuda.empty_cache()
        gc.collect()
        return -1, -1.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return -1, -1.0
        raise


# ============================================================================
# Coefficient fitting — mirror of p9_1a_coeff_profile.py
# ============================================================================

def fit_coefficients(measurements, T, b, C, H, num_classes=10):
    alpha_in_bytes = b * C * H * H * 4
    alpha_out_bytes = b * num_classes * 4

    mode2 = measurements.get(2, {})
    if len(mode2) < 2:
        return {"error": "insufficient mode-2 data"}

    kappas = sorted(mode2.keys())
    peaks = np.array([mode2[k] for k in kappas], dtype=np.float64)
    valid = peaks > 0
    if valid.sum() < 2:
        return {"error": "insufficient valid mode-2 data after OOM filter"}

    kappas_arr = np.array(kappas, dtype=np.float64)
    A_mat = np.vstack([np.ones_like(kappas_arr[valid]),
                       kappas_arr[valid]]).T
    coef, *_ = np.linalg.lstsq(A_mat, peaks[valid], rcond=None)
    a_offset, alpha_K = coef[0], coef[1]
    M_0 = a_offset - (alpha_in_bytes + alpha_out_bytes) * T

    # Compute fit residuals (in GB; positive = under-prediction, neg = over)
    pred = a_offset + alpha_K * kappas_arr[valid]
    residuals = (peaks[valid] - pred) / BYTES_PER_GB

    return {
        "M_0_bytes":     float(M_0),
        "alpha_K_bytes": float(alpha_K),
        "alpha_in_bytes":  float(alpha_in_bytes),
        "alpha_out_bytes": float(alpha_out_bytes),
        "fit_kappas":   kappas,
        "fit_peaks":    [int(p) for p in peaks],
        "fit_residuals_GB": [float(r) for r in residuals],
    }


# ============================================================================
# Per-arch profiling
# ============================================================================

def profile_arch(name, build_fn, reset_fn, T_sweep, kappa_sweep,
                 b, C, H, device, num_classes=10):
    """Profile one arch across T_sweep × kappa_sweep × {mode 2}.

    Returns (per-T coeffs dict, raw measurements dict).
    """
    print(f"\n=== Profiling {name} (b={b}, C={C}, H={H}) ===")
    coeffs_per_T = {}
    raw_per_T = {}

    for T in T_sweep:
        print(f"  T={T}:")
        net = build_fn().to(device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"    params: {n_params:,}")

        # Mode 2 kappa sweep
        mode2_meas = {}
        for kappa in kappa_sweep:
            if kappa > T:
                continue
            peak, ms = run_mode(net, T, kappa, b, C, H, mode_id=2,
                                 device=device, num_classes=num_classes,
                                 reset_fn=reset_fn)
            if peak < 0:
                print(f"    kappa={kappa:>4}  OOM")
            else:
                print(f"    kappa={kappa:>4}  peak={peak/BYTES_PER_GB:.3f}GB"
                      f"  wall={ms:.1f}ms")
                mode2_meas[kappa] = peak

        measurements = {2: mode2_meas}
        c = fit_coefficients(measurements, T, b, C, H, num_classes)
        coeffs_per_T[str(T)] = c
        raw_per_T[str(T)] = {"mode2_kappa_to_peak_bytes": {
            str(k): int(v) for k, v in mode2_meas.items()
        }}

        del net
        torch.cuda.empty_cache()
        gc.collect()

    return coeffs_per_T, raw_per_T


# ============================================================================
# Main: profile all 6 extended archs
# ============================================================================

def main_profile(args):
    sys.path.insert(0, ".")
    from aeros_models_extended import (build_extended_net,
                                        reset_state_extended,
                                        EXTENDED_NETS)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    setup_determinism(args.seed)

    print("=" * 78)
    print("AEROS Exp 1A EXTENDED — coefficient profiling for 6 new archs")
    print("=" * 78)
    print(f"  archs: {EXTENDED_NETS}")
    print(f"  T_sweep: {args.Ts}")
    print(f"  kappa_sweep: {args.kappas}")
    print(f"  b={args.b}, C={args.C}, H={args.H}, num_classes={args.num_classes}")
    print()

    out_coeffs = {}
    out_raw = {}

    for name in EXTENDED_NETS:
        if args.nets and name not in args.nets:
            continue
        # Curry build_fn with config
        def build_fn(_name=name):
            return build_extended_net(_name, num_classes=args.num_classes,
                                       H=args.H, in_ch=args.C)
        coeffs_T, raw_T = profile_arch(name, build_fn, reset_state_extended,
                                        args.Ts, args.kappas,
                                        args.b, args.C, args.H, device,
                                        num_classes=args.num_classes)
        out_coeffs[name] = coeffs_T
        out_raw[name] = raw_T

    output = {
        "config": {
            "T_sweep": args.Ts,
            "kappa_sweep": args.kappas,
            "b": args.b, "C": args.C, "H": args.H,
            "num_classes": args.num_classes,
            "suite": "extended (non-SNN stateful sequence)",
        },
        "coeffs": out_coeffs,
        "raw": out_raw,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}.json")


# ============================================================================
# Merge mode: combine SNN bundle + extended bundle into 16-arch unified
# ============================================================================

def main_merge(args):
    print(f"=== Merging coefficient bundles ===")
    print(f"  bundle 1: {args.merge[0]}")
    print(f"  bundle 2: {args.merge[1]}")

    with open(args.merge[0]) as f:
        b1 = json.load(f)
    with open(args.merge[1]) as f:
        b2 = json.load(f)

    c1 = b1.get("coeffs", b1)
    c2 = b2.get("coeffs", b2)

    # Conflict check: any overlapping arch names?
    overlap = set(c1.keys()) & set(c2.keys())
    if overlap:
        print(f"  [WARN] overlapping arch names: {overlap}")
        print(f"  using bundle 2 values for overlap")

    merged_coeffs = {**c1, **c2}
    merged_raw = {**b1.get("raw", {}), **b2.get("raw", {})}

    # Configs differ between bundles (H=128 for SNN, H=32 for extended).
    # Record both as separate sub-dicts so downstream tools can introspect.
    merged = {
        "config_snn":      b1.get("config", {}),
        "config_extended": b2.get("config", {}),
        "coeffs": merged_coeffs,
        "raw": merged_raw,
        "arch_provenance": {
            **{name: "snn" for name in c1},
            **{name: "extended" for name in c2},
        },
    }

    with open(args.merged_output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  merged: {len(merged_coeffs)} archs total")
    print(f"  saved: {args.merged_output}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Default mode: profile
    parser.add_argument("--output", default="p9_1a_extended")
    parser.add_argument("--Ts", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--kappas", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--H", type=int, default=32,
                        help="spatial size for extended-suite archs (32 default)")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--nets", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Merge mode
    parser.add_argument("--merge", type=str, nargs=2, default=None,
                        help="two bundle JSON paths to merge")
    parser.add_argument("--merged_output", type=str,
                        default="p9_1a_full16.json")
    args = parser.parse_args()

    if args.merge is not None:
        main_merge(args)
    else:
        main_profile(args)


if __name__ == "__main__":
    main()