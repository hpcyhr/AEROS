#!/usr/bin/env python
"""
AEROS Experiment E2 — Cross-architecture held-out solver calibration.

E1 (already done) shows the calibrated bundle generalizes to unseen
(T, M_budget) cells for the same architecture-hardware pair. E2 asks
the stronger question: does a residual envelope built from K
training-fold architectures generalize to a held-out test-fold
architecture from a DIFFERENT family?

Protocol:
  Given 16 unified-suite archs, repeat for n_splits random folds:
    1. Randomly partition 16 archs into 13 fit + 3 held-out (test-fold).
    2. Build POOLED held-out envelope: from the fit-fold archs, gather
       ALL fit residuals; take Δ_held = (1-δ)-quantile (δ=0.05).
    3. Run solver with Δ_held on each held-out arch's eval cells; report
       compliance, OOM rate, violations.
    4. Compare to E1 same-arch held-out compliance for the same arch.

For paper:
  - If cross-arch held-out compliance is comparable to E1 (~95-98%),
    the calibration generalizes across families. Strong claim.
  - If significantly degraded (e.g. <90%), the calibration is
    architecture-specific. Honest finding, paper limitation.

Output:
  /data/yhr/AEROS/p9_6e_cross_arch_holdout.json

Usage:
  python p9_6e_cross_arch_holdout.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
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

from aeros_dispatch import (load_unified_bundle, get_arch_config,
                              build_any_net, reset_any_net)


BYTES_PER_GB = 1024 ** 3


# ============================================================================
# Solver and forward (mirrors p9_6d)
# ============================================================================

def reset_state(net, provenance="snn"):
    if provenance == "extended":
        reset_any_net(net, provenance)
        return
    try:
        functional.reset_net(net)
    except Exception:
        pass


def kappa_solver(coeffs, T, mode_id, M_budget_bytes,
                  Delta_resid_bytes, epsilon_safety):
    """Mirrors p9_6d kappa_solver. Returns (kappa*, status, pred_bytes)."""
    M_eff = (1.0 - epsilon_safety) * M_budget_bytes - Delta_resid_bytes
    if M_eff <= 0:
        return 1, "INFEASIBLE", 0

    def predict(kappa):
        m = coeffs[f"mode_{mode_id}"]
        return m["A"] + m["B"] * kappa

    pred_at_T = predict(T)
    if pred_at_T <= M_eff:
        return T, "feasible_degenerate", pred_at_T

    pred_at_1 = predict(1)
    if pred_at_1 > M_eff:
        return 1, "INFEASIBLE", pred_at_1

    lo, hi = 1, T
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if predict(mid) <= M_eff:
            lo = mid
        else:
            hi = mid - 1
    return lo, "feasible", predict(lo)


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
            chunks.append(net(x[i:i+sz].contiguous()))
            i += sz
        return torch.cat(chunks, dim=0)
    if mode_id in (3, 4):
        assert x.device.type == "cpu"
        reset_state(net, provenance)
        i = 0
        sink, n = None, 0
        chunks = []
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            y_seg = net(x_seg)
            if mode_id == 4:
                if sink is None:
                    sink = y_seg.sum(dim=0); n = sz
                else:
                    sink = sink + y_seg.sum(dim=0); n += sz
            else:
                chunks.append(y_seg)
            i += sz
        if mode_id == 4:
            return sink / n
        return torch.cat(chunks, dim=0)
    raise ValueError(mode_id)


@torch.no_grad()
def measure_peak(net, x, mode_id, kappa, device, provenance="snn"):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        _ = forward_segmented(net, x, mode_id, kappa, device, provenance)
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / BYTES_PER_GB, "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return -1.0, "oom"
    except Exception as e:
        torch.cuda.empty_cache()
        return -1.0, f"error:{type(e).__name__}"


def run_cell(net_name, T, mode_id, M_GB, coeffs_bundle, Delta_resid,
              eps_safety, device, b, C, H, provenance, num_classes=10):
    coeffs = coeffs_bundle[net_name]
    M_bytes = M_GB * BYTES_PER_GB
    kappa, status, pred = kappa_solver(coeffs, T, mode_id, M_bytes,
                                        Delta_resid, eps_safety)
    out = {
        "net": net_name, "T": T, "mode": mode_id, "M_budget_GB": M_GB,
        "kappa_star": kappa, "status": status,
        "predicted_GB": pred / BYTES_PER_GB, "measured_GB": -1.0,
        "compliant": None, "oom": False, "provenance": provenance,
    }
    if status == "INFEASIBLE":
        return out

    net = build_any_net(net_name, provenance, num_classes=num_classes,
                         H=H, C=C).to(device)
    if mode_id in (3, 4):
        x = torch.randn(T, b, C, H, H, device="cpu", pin_memory=False)
    else:
        x = torch.randn(T, b, C, H, H, device=device)
    measured, run_status = measure_peak(net, x, mode_id, kappa, device,
                                          provenance)
    out["measured_GB"] = measured
    if run_status == "oom":
        out["oom"] = True
        out["compliant"] = False
    elif run_status.startswith("error"):
        out["compliant"] = False
        out["error"] = run_status
    else:
        out["compliant"] = (measured <= M_GB)
    del net, x
    torch.cuda.empty_cache(); gc.collect()
    return out


# ============================================================================
# Pooled envelope from fit-fold archs' raw residuals
# ============================================================================

def pooled_held_out_envelope(coeffs_bundle, fit_archs, delta=0.05):
    """Aggregate fit residuals across all fit-fold archs to a single
    pooled (1-delta)-quantile bytes envelope.

    The fit residuals for an arch are: across all (T, kappa, mode) profiling
    cells, the signed prediction error in bytes (measured - predicted).
    We take only positive residuals (under-prediction) for envelope purposes.
    """
    all_residuals = []
    for arch in fit_archs:
        coeffs = coeffs_bundle[arch]
        # raw_residuals stored in coeffs["raw"]["residuals_bytes"] (per cell)
        raw = coeffs.get("raw", {})
        # If structured per-mode:
        for mode_key in ("mode_2", "mode_3", "mode_4"):
            if mode_key in raw and "residuals_bytes" in raw[mode_key]:
                all_residuals.extend(raw[mode_key]["residuals_bytes"])
        # If a flat list:
        if "residuals_bytes" in raw and isinstance(raw["residuals_bytes"], list):
            all_residuals.extend(raw["residuals_bytes"])

    if not all_residuals:
        # Fallback: aggregate per-mode A_resid+B_resid quantiles from coeffs
        # (each arch's coeffs has per-mode "Delta_underpred_bytes_q95")
        for arch in fit_archs:
            coeffs = coeffs_bundle[arch]
            for mode_key in ("mode_2", "mode_3", "mode_4"):
                m = coeffs.get(mode_key, {})
                if "Delta_underpred_bytes_q95" in m:
                    all_residuals.append(m["Delta_underpred_bytes_q95"])

    if not all_residuals:
        return 0.0

    pos = [r for r in all_residuals if r > 0]
    if not pos:
        return 0.0
    return float(np.quantile(pos, 1.0 - delta))


def per_arch_envelope_from_fit(coeffs_bundle, arch, delta=0.05):
    """Same-arch envelope (E1-style) for comparison."""
    coeffs = coeffs_bundle[arch]
    residuals = []
    raw = coeffs.get("raw", {})
    for mode_key in ("mode_2", "mode_3", "mode_4"):
        if mode_key in raw and "residuals_bytes" in raw[mode_key]:
            residuals.extend(raw[mode_key]["residuals_bytes"])
    if not residuals:
        for mode_key in ("mode_2", "mode_3", "mode_4"):
            m = coeffs.get(mode_key, {})
            if "Delta_underpred_bytes_q95" in m:
                residuals.append(m["Delta_underpred_bytes_q95"])
    pos = [r for r in residuals if r > 0]
    if not pos:
        return 0.0
    return float(np.quantile(pos, 1.0 - delta))


def summarize(results, label):
    n_total = len(results)
    n_feasible = sum(1 for r in results if r["status"] != "INFEASIBLE")
    n_compliant = sum(1 for r in results if r.get("compliant") is True)
    n_oom = sum(1 for r in results if r.get("oom"))
    n_violation = sum(1 for r in results
                      if r.get("compliant") is False and not r.get("oom")
                      and r["status"] != "INFEASIBLE")
    return {
        "label": label, "n_total": n_total,
        "n_feasible": n_feasible, "n_compliant": n_compliant,
        "n_oom": n_oom, "n_violation": n_violation,
        "compliance_pct": 100.0 * n_compliant / n_feasible if n_feasible else 0,
        "oom_pct": 100.0 * n_oom / n_total if n_total else 0,
        "violation_pct": 100.0 * n_violation / n_feasible if n_feasible else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True)
    parser.add_argument("--Ts", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--Ms", type=float, nargs="+",
                        default=[4.0, 8.0, 16.0, 24.0])
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--output", default="p9_6e_cross_arch_holdout")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_holdout", type=int, default=3,
                        help="archs in each held-out fold")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    bundle = load_unified_bundle(args.coeffs)
    coeffs_bundle = bundle["coeffs"]
    archs = list(coeffs_bundle.keys())

    print("=" * 78)
    print("AEROS Experiment E2 — cross-architecture held-out calibration")
    print("=" * 78)
    print(f"  archs ({len(archs)}): {archs}")
    print(f"  Ts={args.Ts}  Ms={args.Ms}  modes={args.modes}")
    print(f"  n_splits={args.n_splits}, n_holdout={args.n_holdout} per split")

    rng = random.Random(args.seed)
    splits = []
    for i in range(args.n_splits):
        archs_shuffled = list(archs)
        rng.shuffle(archs_shuffled)
        held_out = archs_shuffled[:args.n_holdout]
        fit = archs_shuffled[args.n_holdout:]
        splits.append({"split_id": i, "fit_archs": fit, "held_archs": held_out})
        print(f"\nSplit {i}: held-out = {held_out}")

    all_split_results = []
    for split in splits:
        print("\n" + "=" * 78)
        print(f"Split {split['split_id']}: held-out archs = {split['held_archs']}")
        print("=" * 78)

        Delta_pooled = pooled_held_out_envelope(coeffs_bundle,
                                                  split["fit_archs"],
                                                  args.delta)
        print(f"\n  Pooled cross-arch held-out envelope: "
              f"{Delta_pooled / BYTES_PER_GB:.4f} GB "
              f"(from {len(split['fit_archs'])} fit archs)")

        # Eval cells: only for held-out archs
        eval_results_pooled = []
        eval_results_per_arch = []  # E1-style for comparison

        for arch in split["held_archs"]:
            cfg = get_arch_config(bundle, arch)
            print(f"\n  --- held-out arch: {arch} (prov={cfg['provenance']}) ---")
            for T in args.Ts:
                for M in args.Ms:
                    for mode in args.modes:
                        # POOLED cross-arch envelope
                        rp = run_cell(arch, T, mode, M, coeffs_bundle,
                                       Delta_pooled, args.epsilon_safety,
                                       device, cfg["b"], cfg["C"], cfg["H"],
                                       cfg["provenance"], cfg["num_classes"])
                        eval_results_pooled.append(rp)
                        # PER-ARCH envelope (E1-style baseline)
                        Delta_arch = per_arch_envelope_from_fit(coeffs_bundle,
                                                                  arch,
                                                                  args.delta)
                        ra = run_cell(arch, T, mode, M, coeffs_bundle,
                                       Delta_arch, args.epsilon_safety,
                                       device, cfg["b"], cfg["C"], cfg["H"],
                                       cfg["provenance"], cfg["num_classes"])
                        eval_results_per_arch.append(ra)
                        print(f"    {arch} T={T} mode={mode} M={M}GB  "
                              f"POOLED kappa*={rp['kappa_star']} "
                              f"comp={rp['compliant']} oom={rp['oom']}  | "
                              f"PER-ARCH kappa*={ra['kappa_star']} "
                              f"comp={ra['compliant']} oom={ra['oom']}")

        sum_pooled = summarize(eval_results_pooled, "pooled cross-arch")
        sum_per_arch = summarize(eval_results_per_arch, "per-arch (E1-style)")

        print(f"\n  Split {split['split_id']} summary:")
        print(f"    POOLED   compliance={sum_pooled['compliance_pct']:.2f}%  "
              f"oom={sum_pooled['oom_pct']:.2f}%  "
              f"viol={sum_pooled['violation_pct']:.2f}%")
        print(f"    PER-ARCH compliance={sum_per_arch['compliance_pct']:.2f}%  "
              f"oom={sum_per_arch['oom_pct']:.2f}%  "
              f"viol={sum_per_arch['violation_pct']:.2f}%")

        all_split_results.append({
            "split_id": split["split_id"],
            "fit_archs": split["fit_archs"],
            "held_archs": split["held_archs"],
            "Delta_pooled_GB": Delta_pooled / BYTES_PER_GB,
            "results_pooled": eval_results_pooled,
            "results_per_arch": eval_results_per_arch,
            "summary_pooled": sum_pooled,
            "summary_per_arch": sum_per_arch,
        })

    # Cross-split aggregate
    print("\n" + "=" * 78)
    print("Cross-split aggregate (n_splits={})".format(args.n_splits))
    print("=" * 78)
    pooled_compliance = [s["summary_pooled"]["compliance_pct"]
                          for s in all_split_results]
    per_arch_compliance = [s["summary_per_arch"]["compliance_pct"]
                            for s in all_split_results]
    print(f"\n  Pooled cross-arch held-out compliance:")
    print(f"    mean={np.mean(pooled_compliance):.2f}%  "
          f"std={np.std(pooled_compliance):.2f}  "
          f"per-split={[f'{x:.2f}%' for x in pooled_compliance]}")
    print(f"  Per-arch (E1-style same-arch) compliance:")
    print(f"    mean={np.mean(per_arch_compliance):.2f}%  "
          f"std={np.std(per_arch_compliance):.2f}  "
          f"per-split={[f'{x:.2f}%' for x in per_arch_compliance]}")

    output = {
        "config": vars(args),
        "splits": all_split_results,
        "aggregate": {
            "n_splits": args.n_splits,
            "pooled_compliance_per_split": pooled_compliance,
            "per_arch_compliance_per_split": per_arch_compliance,
            "pooled_compliance_mean": float(np.mean(pooled_compliance)),
            "pooled_compliance_std": float(np.std(pooled_compliance)),
            "per_arch_compliance_mean": float(np.mean(per_arch_compliance)),
            "per_arch_compliance_std": float(np.std(per_arch_compliance)),
        },
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()