#!/usr/bin/env python
"""
AEROS Phase 2 — Experiment E1: Disjoint-fold solver calibration.

Goal: address the methodological softness in v15's compliance claim. Doris's
review noted that the v14/v15 97.4% compliance figure was reported on the
same (arch, T, kappa) cells used for coefficient fitting; the calibrated
residual envelope quantile was therefore in-sample. This experiment splits
(T, M_budget) cells into three disjoint sets:

  FIT (60%):           used to derive coefficients (already done in p9_1a;
                       this script does NOT re-fit, only consumes)
  CALIBRATION (20%):   on these cells, we run forward at AEROS-chosen kappa,
                       measure (predicted - measured) residuals, and compute
                       the 95% quantile envelope Delta_resid_held
  EVALUATION (20%):    on these cells, we run the solver USING
                       Delta_resid_held (NOT the in-sample fit residuals),
                       then forward, then measure compliance / OOM /
                       violation. This is the held-out compliance figure.

Three numbers are reported side-by-side:
  - in-sample compliance (using fit residuals; matches v14/v15 §5.6)
  - held-out compliance with calibrated envelope (the new defensible figure)
  - delta between them (how much in-sample optimism inflated the reported
    figure)

Cell split is deterministic (modular arithmetic on (T, M_budget) tuple
index) so re-runs match.

Usage:
  python p9_6d_disjoint_calibration.py \\
      --coeffs p9_1a_full.json \\
      --output p9_6d_disjoint_calibration
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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

# Try unified dispatch (16-arch suite). Falls back to SJ-only when not present.
try:
    sys.path.insert(0, ".")
    from aeros_dispatch import (load_unified_bundle, get_arch_config,
                                  build_any_net, reset_any_net)
    DISPATCH_OK = True
except Exception as e:
    print(f"[INFO] aeros_dispatch unavailable: {e}; SNN-only mode")
    DISPATCH_OK = False


BYTES_PER_GB = 1024 ** 3


# ============================================================================
# Net builders + reset (same as Ablation 2/3)
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


def reset_state(net, provenance="snn"):
    if provenance == "extended" and DISPATCH_OK:
        reset_any_net(net, provenance)
        return
    try:
        functional.reset_net(net)
    except Exception:
        pass


# ============================================================================
# Memory model: peak = A + B*kappa per (arch, T, mode) — same as Ablation 2
# ============================================================================

def AB_for_mode(coeffs_arch, T, mode_id):
    T_key = str(T)
    if T_key not in coeffs_arch:
        raise KeyError(f"T={T} not in coeffs_arch keys")
    c = coeffs_arch[T_key]
    M0 = c["M_0_bytes"]
    a_K = c["alpha_K_bytes"]
    a_in = c["alpha_in_bytes"]
    a_out = c["alpha_out_bytes"]

    if mode_id == 1:
        A = M0 + a_in * T + a_out * T
        B = 0.0
    elif mode_id == 2:
        A = M0 + a_in * T + a_out * T
        B = a_K
    elif mode_id == 3:
        A = M0 + a_out * T
        B = a_in + a_K
    elif mode_id == 4:
        A = M0
        B = a_in + a_K + a_out
    else:
        raise ValueError(f"unknown mode {mode_id}")
    return A, B


def in_sample_envelope(coeffs_arch, T, delta=0.05):
    """Original v15 envelope: quantile over fit residuals. In-sample."""
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


def kappa_solver(coeffs_arch, T, mode_id, M_budget_bytes,
                 Delta_resid_bytes, epsilon_safety):
    A, B = AB_for_mode(coeffs_arch, T, mode_id)
    eff_budget = (M_budget_bytes - Delta_resid_bytes
                  - epsilon_safety * M_budget_bytes)
    if mode_id == 1:
        pred = A + B * T
        return (T, "feasible", pred) if pred <= eff_budget \
            else (-1, "INFEASIBLE", pred)
    if B <= 1e3:
        return (T, "feasible_degenerate", A) if A <= eff_budget \
            else (-1, "INFEASIBLE", A)
    kappa_raw = (eff_budget - A) / B
    if kappa_raw < 1:
        return -1, "INFEASIBLE", A + B
    kappa_star = min(int(np.floor(kappa_raw)), T)
    pred = A + B * kappa_star
    return kappa_star, "feasible", pred


# ============================================================================
# Forward + peak measurement (same as Ablation 2/3)
# ============================================================================

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
            y_seg = net(x_seg)
            chunks.append(y_seg)
            del x_seg
            i += sz
        return torch.cat(chunks, dim=0)

    if mode_id in (3, 4):
        assert x.device.type == "cpu"
        reset_state(net, provenance)
        i = 0
        chunks = []
        sink = None
        n = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            y_seg = net(x_seg)
            if mode_id == 4:
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
def measure_peak_hbm(net, x, mode_id, kappa, device, provenance="snn"):
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


def run_cell_with_envelope(net_name, T, mode_id, M_budget_GB,
                            coeffs_bundle, Delta_resid_bytes, epsilon_safety,
                            device, b=32, C=3, H=128,
                            provenance="snn", num_classes=10):
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
        "residual_GB": 0.0,
        "provenance": provenance,
    }
    if status == "INFEASIBLE":
        return out
    if DISPATCH_OK:
        net = build_any_net(net_name, provenance, num_classes=num_classes,
                             H=H, C=C).to(device)
    else:
        net = build_net(net_name).to(device)
    if mode_id in (3, 4):
        x = torch.randn(T, b, C, H, H, device="cpu", pin_memory=False)
    else:
        x = torch.randn(T, b, C, H, H, device=device)
    measured_GB, run_status = measure_peak_hbm(
        net, x, mode_id, kappa_star, device, provenance=provenance)
    out["measured_GB"] = measured_GB
    if run_status == "oom":
        out["oom"] = True
        out["compliant"] = False
        out["violation_GB"] = 999.0
    elif run_status.startswith("error"):
        out["compliant"] = False
        out["violation_GB"] = 999.0
        out["error"] = run_status
    else:
        out["compliant"] = (measured_GB <= M_budget_GB)
        out["violation_GB"] = max(0.0, measured_GB - M_budget_GB)
        out["residual_GB"] = measured_GB - out["predicted_GB"]
    del net, x
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ============================================================================
# Fit / calibration / evaluation cell split
# ============================================================================

def split_cells(cells, seed=42):
    """Deterministic 60/20/20 split on (T, M_budget) cells.

    Same (T, M_budget) goes to same fold across all archs/modes — this is
    the disjoint property we want for the calibration claim.
    """
    # Get unique (T, M_budget) tuples
    tm_keys = sorted({(c[1], c[2]) for c in cells})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tm_keys))
    n = len(tm_keys)
    n_fit = int(round(n * 0.60))
    n_cal = max(1, int(round(n * 0.20)))
    n_eval = n - n_fit - n_cal
    if n_eval < 1:
        # rebalance if too few
        n_eval = 1
        n_cal = max(1, n - n_fit - n_eval)
    fit_keys = {tm_keys[i] for i in perm[:n_fit]}
    cal_keys = {tm_keys[i] for i in perm[n_fit:n_fit+n_cal]}
    eval_keys = {tm_keys[i] for i in perm[n_fit+n_cal:]}
    print(f"  fit (T,M_GB)         : {sorted(fit_keys)}")
    print(f"  calibration (T,M_GB) : {sorted(cal_keys)}")
    print(f"  evaluation (T,M_GB)  : {sorted(eval_keys)}")
    fit_cells = [c for c in cells if (c[1], c[2]) in fit_keys]
    cal_cells = [c for c in cells if (c[1], c[2]) in cal_keys]
    eval_cells = [c for c in cells if (c[1], c[2]) in eval_keys]
    return fit_cells, cal_cells, eval_cells


def held_out_envelope_per_arch(cal_results, delta=0.05):
    """Per-arch residual envelope from CALIBRATION cell measured residuals.

    For each arch, compute the (1-delta) quantile of max(0, measured-pred)
    over its calibration cells. This is the disjoint-cell counterpart of
    the in-sample fit-residual envelope: same arch, different cells.

    Returns dict: {arch_name: Delta_GB}
    """
    by_arch = {}
    for r in cal_results:
        if r["status"] == "INFEASIBLE" or r["measured_GB"] < 0:
            continue
        arch = r["net"]
        by_arch.setdefault(arch, []).append(
            max(0.0, r["residual_GB"]) * BYTES_PER_GB)
    out = {}
    for arch, resids in by_arch.items():
        if not resids:
            out[arch] = 0.0
        else:
            out[arch] = float(np.quantile(resids, 1.0 - delta))
    return out


def held_out_envelope(cal_results, delta=0.05):
    """Pooled held-out envelope (one number across all archs). Kept for
    reference but DEPRECATED in favor of held_out_envelope_per_arch — the
    pooled version inflates VGG-19's underestimation by SR-18/SEW-50's
    smaller residuals."""
    resids_under_bytes = []
    for r in cal_results:
        if r["status"] == "INFEASIBLE" or r["measured_GB"] < 0:
            continue
        resids_under_bytes.append(max(0.0, r["residual_GB"]) * BYTES_PER_GB)
    if not resids_under_bytes:
        return 0.0
    return float(np.quantile(resids_under_bytes, 1.0 - delta))


def summarize(results, label):
    feasible = [r for r in results if r["status"] != "INFEASIBLE"]
    n_total = len(results)
    n_feasible = len(feasible)
    if n_feasible == 0:
        return {"config": label, "n_total": n_total, "n_feasible": 0,
                "compliance_pct": 0.0, "oom_pct": 0.0, "violation_pct": 0.0}
    n_compliant = sum(1 for r in feasible if r["compliant"] is True)
    n_oom = sum(1 for r in feasible if r["oom"])
    n_violation = sum(1 for r in feasible
                      if (r["compliant"] is False) and (not r["oom"]))
    violations = [r["violation_GB"] for r in feasible
                  if (r["compliant"] is False) and (not r["oom"])]
    return {
        "config": label, "n_total": n_total, "n_feasible": n_feasible,
        "n_infeasible": n_total - n_feasible,
        "n_compliant": n_compliant, "n_oom": n_oom,
        "n_bounded_violations": n_violation,
        "compliance_pct": 100.0 * n_compliant / n_feasible,
        "oom_pct": 100.0 * n_oom / n_feasible,
        "violation_pct": 100.0 * n_violation / n_feasible,
        "avg_violation_GB": float(np.mean(violations)) if violations else 0.0,
        "max_violation_GB": float(np.max(violations)) if violations else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True)
    parser.add_argument("--output", default="p9_6d_disjoint_calibration")
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[128, 1024, 4096])
    parser.add_argument("--Ms", type=float, nargs="+",
                        default=[4.0, 8.0, 16.0, 24.0, 32.0])
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--nets", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
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

    if DISPATCH_OK:
        bundle = load_unified_bundle(args.coeffs)
    else:
        with open(args.coeffs) as f:
            data = json.load(f)
        coeffs_only = data.get("coeffs", data)
        bundle = {
            "config_snn": data.get("config", {}),
            "coeffs": coeffs_only,
            "raw": data.get("raw", {}),
            "arch_provenance": {n: "snn" for n in coeffs_only},
        }
    coeffs_bundle = bundle["coeffs"]
    if args.nets:
        coeffs_bundle = {k: v for k, v in coeffs_bundle.items()
                         if k in args.nets}

    def cfg_for(net_name):
        if DISPATCH_OK:
            return get_arch_config(bundle, net_name)
        return {"b": args.batch_size, "C": args.C, "H": args.H,
                "num_classes": 10, "provenance": "snn"}

    # Build cell grid
    all_cells = []
    for net_name in coeffs_bundle:
        for T in args.Ts:
            for M_GB in args.Ms:
                for mode_id in args.modes:
                    all_cells.append((net_name, T, M_GB, mode_id))

    print("=" * 78)
    print("AEROS Experiment E1: Disjoint-fold solver calibration")
    print("=" * 78)
    print(f"  archs ({len(coeffs_bundle)}): {list(coeffs_bundle.keys())}")
    snn_count = sum(1 for n in coeffs_bundle
                    if bundle["arch_provenance"].get(n) == "snn")
    ext_count = sum(1 for n in coeffs_bundle
                    if bundle["arch_provenance"].get(n) == "extended")
    print(f"  provenance: SNN={snn_count}, extended={ext_count}")
    print(f"  Ts={args.Ts}  Ms={args.Ms}GB  modes={args.modes}")
    print(f"  total cells: {len(all_cells)}")
    print()
    print("Disjoint split on (T, M_budget):")
    fit_cells, cal_cells, eval_cells = split_cells(all_cells, args.seed)
    print(f"  fit cells (used by p9_1a): {len(fit_cells)}")
    print(f"  calibration cells (held-out): {len(cal_cells)}")
    print(f"  evaluation cells (held-out): {len(eval_cells)}")
    print()

    # ---- Step 1: in-sample compliance (v15 baseline replication) ----
    print("=" * 78)
    print("Step 1: in-sample compliance (v15 baseline; uses fit residuals)")
    print("=" * 78)
    in_sample_results = []
    for idx, (net_name, T, M_GB, mode_id) in enumerate(eval_cells, 1):
        coeffs_arch = coeffs_bundle[net_name]
        Delta_in = in_sample_envelope(coeffs_arch, T, args.delta)
        cfg = cfg_for(net_name)
        print(f"\n[{idx}/{len(eval_cells)}] IN-SAMPLE  {net_name} "
              f"({cfg['provenance']}) T={T} mode={mode_id} M={M_GB}GB  "
              f"Δ_in={Delta_in/BYTES_PER_GB:.3f}GB")
        r = run_cell_with_envelope(
            net_name, T, mode_id, M_GB, coeffs_bundle,
            Delta_in, args.epsilon_safety, device,
            b=cfg["b"], C=cfg["C"], H=cfg["H"],
            provenance=cfg["provenance"], num_classes=cfg["num_classes"])
        in_sample_results.append(r)
        print(f"   κ*={r['kappa_star']:>4} {r['status']:<14s} "
              f"pred={r['predicted_GB']:.2f}GB meas={r['measured_GB']:.2f}GB "
              f"comp={r['compliant']} oom={r['oom']}")

    # ---- Step 2: forward all calibration cells with in-sample envelope ----
    print()
    print("=" * 78)
    print("Step 2: build held-out envelope from calibration cells")
    print("=" * 78)
    cal_results = []
    for idx, (net_name, T, M_GB, mode_id) in enumerate(cal_cells, 1):
        coeffs_arch = coeffs_bundle[net_name]
        Delta_in = in_sample_envelope(coeffs_arch, T, args.delta)
        cfg = cfg_for(net_name)
        print(f"\n[{idx}/{len(cal_cells)}] CALIBRATE {net_name} "
              f"({cfg['provenance']}) T={T} mode={mode_id} M={M_GB}GB")
        r = run_cell_with_envelope(
            net_name, T, mode_id, M_GB, coeffs_bundle,
            Delta_in, args.epsilon_safety, device,
            b=cfg["b"], C=cfg["C"], H=cfg["H"],
            provenance=cfg["provenance"], num_classes=cfg["num_classes"])
        cal_results.append(r)
        print(f"   κ*={r['kappa_star']:>4} {r['status']:<14s} "
              f"pred={r['predicted_GB']:.2f}GB meas={r['measured_GB']:.2f}GB "
              f"residual={r['residual_GB']:+.3f}GB")

    # Compute both pooled (deprecated) and per-arch held-out envelopes
    Delta_held_pooled = held_out_envelope(cal_results, args.delta)
    Delta_held_per_arch = held_out_envelope_per_arch(cal_results, args.delta)
    print()
    print(f"  Pooled held-out envelope (deprecated, for reference): "
          f"{Delta_held_pooled/BYTES_PER_GB:.4f} GB")
    print(f"  Per-arch held-out envelopes (used for Step 3):")
    for arch, d in sorted(Delta_held_per_arch.items()):
        print(f"    {arch:<14s}  Δ_held = {d/BYTES_PER_GB:.4f} GB")

    # ---- Step 3: rerun evaluation cells using held-out envelope ----
    print()
    print("=" * 78)
    print("Step 3: held-out compliance (use per-arch Δ_held in solver)")
    print("=" * 78)
    held_out_results = []
    for idx, (net_name, T, M_GB, mode_id) in enumerate(eval_cells, 1):
        Delta_held_arch = Delta_held_per_arch.get(net_name, Delta_held_pooled)
        cfg = cfg_for(net_name)
        print(f"\n[{idx}/{len(eval_cells)}] HELD-OUT  {net_name} "
              f"({cfg['provenance']}) T={T} mode={mode_id} M={M_GB}GB  "
              f"Δ_held_arch={Delta_held_arch/BYTES_PER_GB:.3f}GB")
        r = run_cell_with_envelope(
            net_name, T, mode_id, M_GB, coeffs_bundle,
            Delta_held_arch, args.epsilon_safety, device,
            b=cfg["b"], C=cfg["C"], H=cfg["H"],
            provenance=cfg["provenance"], num_classes=cfg["num_classes"])
        held_out_results.append(r)
        print(f"   κ*={r['kappa_star']:>4} {r['status']:<14s} "
              f"pred={r['predicted_GB']:.2f}GB meas={r['measured_GB']:.2f}GB "
              f"comp={r['compliant']} oom={r['oom']}")

    # ---- Summary ----
    sum_in = summarize(in_sample_results, "in-sample (v15 baseline)")
    sum_held = summarize(held_out_results, "held-out (disjoint calibration)")

    print()
    print("=" * 78)
    print("Summary — disjoint-fold solver calibration")
    print("=" * 78)
    print(f"{'Metric':<28s} {'In-Sample':>16s} {'Held-Out':>16s} {'Δ':>10s}")
    print("-" * 75)
    for key, label in [
            ("n_total", "n_eval_cells"),
            ("n_feasible", "n_feasible"),
            ("n_compliant", "n_compliant"),
            ("n_oom", "n_oom"),
            ("n_bounded_violations", "n_violation"),
            ("compliance_pct", "compliance_pct"),
            ("oom_pct", "oom_pct"),
            ("violation_pct", "violation_pct"),
            ("avg_violation_GB", "avg_violation_GB"),
            ("max_violation_GB", "max_violation_GB"),
    ]:
        v_in = sum_in.get(key, "-")
        v_h = sum_held.get(key, "-")
        if isinstance(v_in, float) and isinstance(v_h, float):
            d = v_h - v_in
            print(f"{label:<28s} {v_in:>16.2f} {v_h:>16.2f} {d:>+10.2f}")
        else:
            print(f"{label:<28s} {v_in:>16} {v_h:>16}")

    out = {
        "experiment": "Disjoint-fold solver calibration (E1)",
        "config": vars(args),
        "split": {
            "fit_cells_count": len(fit_cells),
            "calibration_cells_count": len(cal_cells),
            "evaluation_cells_count": len(eval_cells),
        },
        "Delta_held_pooled_GB": Delta_held_pooled / BYTES_PER_GB,
        "Delta_held_per_arch_GB": {
            arch: d / BYTES_PER_GB
            for arch, d in Delta_held_per_arch.items()
        },
        "summary": {"in_sample": sum_in, "held_out": sum_held},
        "results": {
            "in_sample": in_sample_results,
            "calibration": cal_results,
            "held_out": held_out_results,
        },
    }
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()