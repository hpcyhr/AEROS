#!/usr/bin/env python
"""
AEROS Phase 2 — Ablation 2: no-residual-envelope solver.

Goal: prove that the calibrated residual envelope Delta_resid is the
operative contributor to compliance, not just decoration. We rerun the
exact same compliance protocol as Sec 5.6 (Exp 1B) but with the envelope
disabled, and compare violation/OOM rates.

Two configurations:
  CALIBRATED (current v14 §5.6 setup):
      Delta_resid = q_{0.95}(max(0, residuals))
      epsilon_safety = 0.02
      eff_budget = M_budget - Delta_resid - 0.02 * M_budget

  NO-ENVELOPE (ablation):
      Delta_resid = 0
      epsilon_safety = 0.0
      eff_budget = M_budget   (raw predicted-peak comparison)

Same (net, T, M_budget, mode) cell grid; same fitted coefficients; only
the solver epsilon changes. The cell grid + coefficients are loaded from
p9_1a profiling JSON (already produced; no re-fit needed). Forward runs
and peak-HBM measurements happen here.

Expected:
  - Calibrated:   compliance ≈ 97.4%, OOM = 0%, violation ≈ 2.6% bounded
  - No-envelope:  compliance drops; some cells OOM; violation rate up
  - Conclusion:   the envelope is the principal contributor to compliance,
                  validating the calibrated solver design.

Usage:
  python p9_6b_no_envelope_solver.py \\
      --coeffs p9_1a_full.json \\
      --output p9_6b_no_envelope_ablation
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Dict, List, Tuple

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


BYTES_PER_GB = 1024 ** 3


# ============================================================================
# Net builders (mirrors p9_1b_compliance.py)
# ============================================================================

def build_net(name, num_classes=10):
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=num_classes,
    )
    table = {
        "SR-18":   lambda: spiking_resnet18(**common),
        "SR-34":   lambda: spiking_resnet34(**common),
        "SR-50":   lambda: spiking_resnet50(**common),
        "SEW-18":  lambda: sew_resnet18(cnf="ADD", **common),
        "SEW-50":  lambda: sew_resnet50(cnf="ADD", **common),
        "SEW-101": lambda: sew_resnet101(cnf="ADD", **common),
        "VGG-11-BN": lambda: spiking_vgg11_bn(**common),
        "VGG-13-BN": lambda: spiking_vgg13_bn(**common),
        "VGG-16-BN": lambda: spiking_vgg16_bn(**common),
        "VGG-19-BN": lambda: spiking_vgg19_bn(**common),
    }
    if name not in table:
        raise ValueError(f"unknown net: {name}")
    net = table[name]()
    net.eval()
    functional.set_step_mode(net, "m")
    return net


def reset_state(net):
    try:
        functional.reset_net(net)
    except Exception:
        pass


# ============================================================================
# Coeff helpers (memory model A + B*kappa form per (T, mode))
# ============================================================================

def AB_for_mode(coeffs_arch, T, mode_id):
    """Return (A_bytes, B_bytes) for given (T, mode_id).

    Reads from p9_1a per-(arch, T) coefficient bundle. Uses the unified
    live-set decomposition: peak = A(T, pi) + B(pi) * kappa.

    coeffs_arch is the dict for a single arch. It has per-T sub-dicts keyed
    by str(T), each with M_0_bytes / alpha_K_bytes / alpha_in_bytes /
    alpha_out_bytes / fit_residuals_GB.
    """
    T_key = str(T)
    if T_key not in coeffs_arch:
        raise KeyError(f"T={T} not in coeffs_arch keys {list(coeffs_arch.keys())}")
    c = coeffs_arch[T_key]
    M0 = c["M_0_bytes"]
    a_K = c["alpha_K_bytes"]
    a_in = c["alpha_in_bytes"]
    a_out = c["alpha_out_bytes"]

    # Per-mode policy (pi_in, pi_out)
    if mode_id == 1:
        # Full-residency: input and output both retained over T
        A = M0 + a_in * T + a_out * T
        B = 0.0  # kappa = T doesn't matter; degenerate
    elif mode_id == 2:
        # retained-IO: input + output retained, segment-internal sized by kappa
        A = M0 + a_in * T + a_out * T
        B = a_K
    elif mode_id == 3:
        # input-stream: input only kappa-resident, output retained
        A = M0 + a_out * T
        B = a_in + a_K
    elif mode_id == 4:
        # IO-stream + sink: both input and output kappa-resident
        A = M0
        B = a_in + a_K + a_out
    else:
        raise ValueError(f"unknown mode_id {mode_id}")
    return A, B


def compute_residual_envelope(coeffs_arch, T, delta=0.05):
    """One-sided underprediction quantile in bytes for (arch, T) fit."""
    T_key = str(T)
    if T_key not in coeffs_arch:
        return 0.0
    c = coeffs_arch[T_key]
    if "fit_residuals_GB" not in c:
        return 0.0
    resids_under = [max(0.0, r) * BYTES_PER_GB
                    for r in c["fit_residuals_GB"]]
    if not resids_under:
        return 0.0
    return float(np.quantile(resids_under, 1.0 - delta))


def kappa_solver(coeffs_arch, T, mode_id, M_budget_bytes, Delta_resid_bytes,
                 epsilon_safety):
    """Memory-budget solver. Returns (kappa_star, status, predicted_peak)."""
    A, B = AB_for_mode(coeffs_arch, T, mode_id)
    eff_budget = (M_budget_bytes - Delta_resid_bytes
                  - epsilon_safety * M_budget_bytes)

    if mode_id == 1:
        pred = A + B * T  # = A since B=0 for mode 1
        if pred <= eff_budget:
            return T, "feasible", pred
        else:
            return -1, "INFEASIBLE", pred

    if B <= 1e3:  # bytes; degenerate slope
        if A <= eff_budget:
            return T, "feasible_degenerate", A
        else:
            return -1, "INFEASIBLE", A

    kappa_raw = (eff_budget - A) / B
    if kappa_raw < 1:
        return -1, "INFEASIBLE", A + B
    kappa_star = min(int(np.floor(kappa_raw)), T)
    pred = A + B * kappa_star
    return kappa_star, "feasible", pred


# ============================================================================
# Forward + peak measurement (lean version of p9_1b runner)
# ============================================================================

def forward_segmented(net, x, mode_id, kappa, device):
    """Run forward in given mode with given kappa.

    Input shape conventions:
      Mode 1, 2: x is on GPU (retained-input semantics)
      Mode 3, 4: x is on CPU (input-streaming semantics) — segments are
                 H2D'd one at a time. This matches the policy
                 pi_in = stream of Sec 3.4.
    """
    T = x.shape[0]
    if mode_id == 1 or kappa == T:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state(net)
        return net(x)

    if mode_id == 2:
        # retained-IO: x is fully on GPU
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state(net)
        chunks = []
        i = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous()
            y_seg = net(x_seg)
            chunks.append(y_seg)
            del x_seg
            i += sz
        return torch.cat(chunks, dim=0)

    if mode_id in (3, 4):
        # input-streaming: x must NOT be GPU-resident on entry. We H2D one
        # segment at a time so peak HBM input residency is alpha_in * kappa,
        # not alpha_in * T.
        assert x.device.type == "cpu", (
            f"mode {mode_id} input-streaming requires CPU x; got device={x.device}")
        reset_state(net)
        i = 0
        if mode_id == 3:
            chunks = []
        else:
            sink = None
            n = 0
        while i < T:
            sz = min(kappa, T - i)
            # H2D one segment
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            y_seg = net(x_seg)
            if mode_id == 4:
                # sink: accumulate per-step; only keep running sum on GPU
                if sink is None:
                    sink = y_seg.sum(dim=0)
                    n = sz
                else:
                    sink = sink + y_seg.sum(dim=0)
                    n += sz
            else:
                chunks.append(y_seg)
            del x_seg, y_seg
            i += sz
        if mode_id == 4:
            return sink / n
        return torch.cat(chunks, dim=0)

    raise ValueError(f"unknown mode_id {mode_id}")


@torch.no_grad()
def measure_peak_hbm(net, x, mode_id, kappa, device):
    """Single forward; returns (peak_GB, success_or_oom_str)."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        _ = forward_segmented(net, x, mode_id, kappa, device)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
        return peak, "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return -1.0, "oom"
    except Exception as e:
        torch.cuda.empty_cache()
        return -1.0, f"error:{type(e).__name__}"


# ============================================================================
# Per-cell evaluation under one solver configuration
# ============================================================================

def eval_cell(net_name, T, mode_id, M_budget_GB, coeffs_bundle,
              Delta_resid_bytes, epsilon_safety, device, b=32, C=3, H=128):
    """One (net, T, mode, M_budget) cell under one solver config.

    Returns dict with: kappa_star, status, predicted_GB, measured_GB,
    compliant (bool), violation_GB (signed), oom (bool).
    """
    coeffs_arch = coeffs_bundle[net_name]
    M_budget_bytes = M_budget_GB * BYTES_PER_GB

    kappa_star, status, pred_bytes = kappa_solver(
        coeffs_arch, T, mode_id, M_budget_bytes,
        Delta_resid_bytes, epsilon_safety)

    out = {
        "net": net_name, "T": T, "mode": mode_id,
        "M_budget_GB": M_budget_GB,
        "kappa_star": kappa_star, "status": status,
        "predicted_GB": pred_bytes / BYTES_PER_GB,
        "measured_GB": -1.0, "compliant": None,
        "violation_GB": 0.0, "oom": False,
    }
    if status == "INFEASIBLE":
        return out  # solver refuses; not counted as violation, but not feasible

    # Build net + run forward
    net = build_net(net_name).to(device)
    # Allocate x on CPU for streaming modes (3, 4), GPU for retained (1, 2).
    # This honors the policy pi_in defined in Sec 3.4.
    if mode_id in (3, 4):
        x = torch.randn(T, b, C, H, H, device="cpu",
                        pin_memory=True)
    else:
        x = torch.randn(T, b, C, H, H, device=device)
    measured_GB, run_status = measure_peak_hbm(net, x, mode_id,
                                                kappa_star, device)
    out["measured_GB"] = measured_GB
    if run_status == "oom":
        out["oom"] = True
        out["compliant"] = False
        out["violation_GB"] = 999.0  # marker
    elif run_status.startswith("error"):
        out["compliant"] = False
        out["violation_GB"] = 999.0
        out["error"] = run_status
    else:
        out["compliant"] = (measured_GB <= M_budget_GB)
        out["violation_GB"] = measured_GB - M_budget_GB
    # cleanup
    del net, x
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ============================================================================
# Aggregate metrics
# ============================================================================

def summarize(results, label):
    """Compute compliance / OOM / violation rates over all cells."""
    feasible = [r for r in results if r["status"] != "INFEASIBLE"]
    n_total = len(results)
    n_feasible = len(feasible)
    if n_feasible == 0:
        return {
            "config": label,
            "n_total_cells": n_total, "n_feasible": 0,
            "n_infeasible": n_total,
            "compliance_pct": 0.0,
            "oom_pct": 0.0, "violation_pct": 0.0,
            "avg_violation_GB": 0.0, "max_violation_GB": 0.0,
        }
    n_compliant = sum(1 for r in feasible if r["compliant"] is True)
    n_oom = sum(1 for r in feasible if r["oom"])
    n_violation = sum(1 for r in feasible
                      if (r["compliant"] is False) and (not r["oom"]))
    violations = [r["violation_GB"] for r in feasible
                  if (r["compliant"] is False) and (not r["oom"])]
    return {
        "config": label,
        "n_total_cells": n_total,
        "n_feasible": n_feasible,
        "n_infeasible": n_total - n_feasible,
        "n_compliant": n_compliant,
        "n_oom": n_oom,
        "n_bounded_violations": n_violation,
        "compliance_pct": 100.0 * n_compliant / n_feasible,
        "oom_pct": 100.0 * n_oom / n_feasible,
        "violation_pct": 100.0 * n_violation / n_feasible,
        "avg_violation_GB": float(np.mean(violations)) if violations else 0.0,
        "max_violation_GB": float(np.max(violations)) if violations else 0.0,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True,
                        help="path to p9_1a fitted coefficients JSON")
    parser.add_argument("--output", default="p9_6b_no_envelope_ablation")
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[128, 1024, 4096])
    parser.add_argument("--Ms", type=float, nargs="+",
                        default=[4.0, 8.0, 16.0],
                        help="memory budgets in GB; defaults selected to "
                             "exercise compliance pressure on b=32 H=128 fits")
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--nets", type=str, nargs="+", default=None,
                        help="optional restrict to subset; default = all")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="must match p9_1a fit setup (default 32)")
    parser.add_argument("--H", type=int, default=128,
                        help="must match p9_1a fit setup (default 128)")
    parser.add_argument("--C", type=int, default=3,
                        help="must match p9_1a fit setup (default 3)")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    # Determinism (so the same kappa_star deterministically reaches same peak)
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)

    # Load coefficients
    with open(args.coeffs) as f:
        data = json.load(f)
    if "coeffs" in data:
        coeffs_bundle = data["coeffs"]
        print(f"=== AEROS Ablation 2: no-residual-envelope solver ===")
        print(f"  coeffs: {args.coeffs}  (loaded {len(coeffs_bundle)} archs)")
        if "config" in data:
            print(f"  fit config: T_sweep={data['config'].get('T_sweep')} "
                  f"kappa_sweep={data['config'].get('kappa_sweep')} "
                  f"b={data['config'].get('b')} C={data['config'].get('C')} "
                  f"H={data['config'].get('H')}")
    else:
        coeffs_bundle = data
        print(f"=== AEROS Ablation 2: no-residual-envelope solver ===")
        print(f"  coeffs: {args.coeffs}  (legacy schema, "
              f"{len(coeffs_bundle)} archs)")
    if args.nets:
        coeffs_bundle = {k: v for k, v in coeffs_bundle.items()
                         if k in args.nets}
        print(f"  restricted to: {list(coeffs_bundle.keys())}")
    print(f"  Ts: {args.Ts}, M_budgets: {args.Ms} GB, modes: {args.modes}")
    print(f"  delta={args.delta}, epsilon_safety={args.epsilon_safety}")
    print(f"  Total cells: "
          f"{len(coeffs_bundle)} * {len(args.Ts)} * {len(args.Ms)} * "
          f"{len(args.modes)} = "
          f"{len(coeffs_bundle) * len(args.Ts) * len(args.Ms) * len(args.modes)}")

    # ---- Run both configs back-to-back on the same cell grid ----
    results_calibrated = []
    results_no_envelope = []

    cells = []
    for net_name in coeffs_bundle:
        for T in args.Ts:
            for M_GB in args.Ms:
                for mode_id in args.modes:
                    cells.append((net_name, T, M_GB, mode_id))

    for idx, (net_name, T, M_GB, mode_id) in enumerate(cells, 1):
        coeffs_arch = coeffs_bundle[net_name]
        Delta_resid_bytes = compute_residual_envelope(coeffs_arch, T, args.delta)

        print(f"\n[{idx}/{len(cells)}] {net_name} T={T} mode={mode_id} "
              f"M={M_GB}GB Δ={Delta_resid_bytes/BYTES_PER_GB:.3f}GB")

        # CALIBRATED
        r_cal = eval_cell(net_name, T, mode_id, M_GB, coeffs_bundle,
                          Delta_resid_bytes, args.epsilon_safety,
                          device, b=args.batch_size, C=args.C, H=args.H)
        results_calibrated.append(r_cal)
        print(f"   CAL : kappa*={r_cal['kappa_star']:>4} "
              f"status={r_cal['status']:<14s} "
              f"pred={r_cal['predicted_GB']:.2f}GB "
              f"meas={r_cal['measured_GB']:.2f}GB "
              f"comp={r_cal['compliant']} oom={r_cal['oom']}")

        # NO-ENVELOPE (Δ=0, ε_safety=0)
        r_ne = eval_cell(net_name, T, mode_id, M_GB, coeffs_bundle,
                         0.0, 0.0, device, b=args.batch_size,
                         C=args.C, H=args.H)
        results_no_envelope.append(r_ne)
        print(f"   NE  : kappa*={r_ne['kappa_star']:>4} "
              f"status={r_ne['status']:<14s} "
              f"pred={r_ne['predicted_GB']:.2f}GB "
              f"meas={r_ne['measured_GB']:.2f}GB "
              f"comp={r_ne['compliant']} oom={r_ne['oom']}")

    # ---- Summarize ----
    sum_cal = summarize(results_calibrated, "calibrated")
    sum_ne = summarize(results_no_envelope, "no-envelope")

    print("\n" + "=" * 78)
    print("Summary — no-envelope solver ablation")
    print("=" * 78)
    print(f"{'Metric':<28s} {'Calibrated':>14s} {'No-Envelope':>14s}")
    print("-" * 60)
    for key in ("n_total_cells", "n_feasible", "n_infeasible",
                "n_compliant", "n_oom", "n_bounded_violations",
                "compliance_pct", "oom_pct", "violation_pct",
                "avg_violation_GB", "max_violation_GB"):
        v_cal = sum_cal.get(key, "-")
        v_ne = sum_ne.get(key, "-")
        if isinstance(v_cal, float):
            print(f"{key:<28s} {v_cal:>14.2f} {v_ne:>14.2f}")
        else:
            print(f"{key:<28s} {v_cal:>14} {v_ne:>14}")

    out = {
        "ablation": "no-residual-envelope solver (Ablation 2)",
        "config": vars(args),
        "summary": {
            "calibrated": sum_cal,
            "no_envelope": sum_ne,
        },
        "results": {
            "calibrated": results_calibrated,
            "no_envelope": results_no_envelope,
        },
    }
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()