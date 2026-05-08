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
    1. Randomly partition into 13 fit + 3 held-out.
    2. Build POOLED held-out envelope by aggregating fit_residuals_GB
       across all fit-fold archs and Ts; take (1-δ)-quantile of
       positive residuals.
    3. Run solver with Δ_pooled on each held-out arch's eval cells;
       report compliance, OOM rate, violations.
    4. Compare to per-arch (E1-style) envelope as baseline.

Output:
  /data/yhr/AEROS/p9_6e_cross_arch_holdout.json

Usage:
  python p9_6e_cross_arch_holdout.py --coeffs p9_1a_full16.json \
      --Ts 128 1024 --Ms 4 8 16 24 --modes 2 4 \
      --n_splits 3 --n_holdout 3 \
      --output p9_6e_cross_arch_holdout
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
# Solver — use the SAME schema as p9_6d_disjoint_calibration_v3.py
# ============================================================================

def reset_state(net, provenance="snn"):
    if provenance == "extended":
        reset_any_net(net, provenance)
        return
    try:
        functional.reset_net(net)
    except Exception:
        pass


def AB_for_mode(coeffs_arch, T, mode_id):
    """Reduce per-mode peak HBM model to A + B * kappa."""
    T_key = str(T)
    if T_key not in coeffs_arch:
        raise KeyError(f"T={T} not in coeffs_arch keys ({list(coeffs_arch.keys())})")
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
# Envelope construction from coefficient bundle's fit_residuals_GB
# ============================================================================

def per_arch_envelope_from_fit(coeffs_bundle, arch, delta=0.05):
    """Same-arch envelope (E1-style baseline): aggregate fit_residuals_GB
    across all T-keyed fits for this arch, take (1-delta)-quantile of
    positive residuals."""
    coeffs = coeffs_bundle[arch]
    residuals_bytes = []
    for T_key, c in coeffs.items():
        if "fit_residuals_GB" in c:
            for r in c["fit_residuals_GB"]:
                if r > 0:
                    residuals_bytes.append(r * BYTES_PER_GB)
    if not residuals_bytes:
        return 0.0
    return float(np.quantile(residuals_bytes, 1.0 - delta))


def pooled_cross_arch_envelope(coeffs_bundle, fit_archs, delta=0.05):
    """Pooled cross-arch envelope: aggregate ALL fit-fold archs' residuals,
    take (1-delta)-quantile."""
    all_residuals_bytes = []
    for arch in fit_archs:
        coeffs = coeffs_bundle[arch]
        for T_key, c in coeffs.items():
            if "fit_residuals_GB" in c:
                for r in c["fit_residuals_GB"]:
                    if r > 0:
                        all_residuals_bytes.append(r * BYTES_PER_GB)
    if not all_residuals_bytes:
        return 0.0
    return float(np.quantile(all_residuals_bytes, 1.0 - delta))


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

        Delta_pooled = pooled_cross_arch_envelope(coeffs_bundle,
                                                    split["fit_archs"],
                                                    args.delta)
        print(f"\n  Pooled cross-arch held-out envelope: "
              f"{Delta_pooled / BYTES_PER_GB:.4f} GB "
              f"(from {len(split['fit_archs'])} fit archs)")

        eval_results_pooled = []
        eval_results_per_arch = []

        for arch in split["held_archs"]:
            cfg = get_arch_config(bundle, arch)
            print(f"\n  --- held-out arch: {arch} (prov={cfg['provenance']}) ---")
            for T in args.Ts:
                for M in args.Ms:
                    for mode in args.modes:
                        rp = run_cell(arch, T, mode, M, coeffs_bundle,
                                       Delta_pooled, args.epsilon_safety,
                                       device, cfg["b"], cfg["C"], cfg["H"],
                                       cfg["provenance"], cfg["num_classes"])
                        eval_results_pooled.append(rp)
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

    print("\n" + "=" * 78)
    print(f"Cross-split aggregate (n_splits={args.n_splits})")
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