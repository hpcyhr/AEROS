#!/usr/bin/env python
"""
AEROS Phase 2 — Ablation 3: manual fixed-kappa baseline.

Goal: prove that the calibrated contract solver is Pareto-optimal in
(safety, efficiency) space. Manual kappa-selection heuristics either OOM
(unsafe) or over-provision (wasteful). The calibrated AEROS kappa* is the
only choice that consistently respects budget while utilizing it.

Strategies compared:
  S1. NO-SEGMENT  (kappa=T, full Mode 1):
      The "user does nothing" baseline. Should OOM at large T or large net.

  S2. NAIVE HALVE (kappa=T/2):
      Simplest non-trivial heuristic. Mid-ground; some cells safe, some
      OOM at small budgets.

  S3. CONSERVATIVE FIXED (kappa=8):
      A "small-enough-everywhere" hand-pick. Should not OOM, but leaves
      large unused budget on small nets / loose budgets (waste).

  S4. AEROS CALIBRATED (kappa* read from Ablation 2 JSON):
      The contract solver's choice. Expected to be Pareto-optimal:
      compliance close to S3, efficiency close to S2.

Same cell grid as Ablation 2 (10 archs * 2 Ts * 2 Ms * 2 modes = 80 cells).

Measurement per (cell, strategy):
  - kappa_chosen (clipped to T)
  - measured_peak_GB (or oom)
  - compliant (peak <= budget)
  - unused_budget_GB (budget - peak, signed; negative = violation)

Aggregate per strategy:
  - n_oom, n_compliant, n_violation
  - oom_pct, compliance_pct
  - avg_unused_GB (across safe cells; lower = more efficient)
  - p95_unused_GB

Usage:
  python p9_6c_manual_fixed_kappa.py \\
      --coeffs p9_1a_full.json \\
      --ablation2_json p9_6b_no_envelope_medium.json \\
      --output p9_6c_manual_kappa
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


BYTES_PER_GB = 1024 ** 3


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


def forward_segmented(net, x, mode_id, kappa, device):
    """Mode-aware segmented forward. Same semantics as Ablation 2."""
    T = x.shape[0]
    if mode_id == 1 or kappa >= T:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state(net)
        return net(x)

    if mode_id == 2:
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
        assert x.device.type == "cpu", \
            f"mode {mode_id} streaming requires CPU x; got {x.device}"
        reset_state(net)
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
def measure_peak_hbm(net, x, mode_id, kappa, device):
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
# Strategies for picking kappa
# ============================================================================

def strategy_no_segment(T):
    """S1: no segmentation (full Mode 1; kappa = T)."""
    return T


def strategy_naive_halve(T):
    """S2: naive halving (kappa = T/2, min 1)."""
    return max(1, T // 2)


def strategy_conservative(T, fixed=8):
    """S3: conservative fixed kappa=8 (clipped to T if T < 8)."""
    return min(fixed, T)


# ============================================================================
# Per-cell evaluation under each strategy
# ============================================================================

def eval_strategy(net_name, T, mode_id, M_budget_GB, kappa_chosen, device,
                  b=32, C=3, H=128):
    """Run forward at kappa_chosen, return measured peak + compliance."""
    out = {
        "net": net_name, "T": T, "mode": mode_id,
        "M_budget_GB": M_budget_GB,
        "kappa": kappa_chosen,
        "measured_GB": -1.0, "compliant": None,
        "unused_GB": 0.0, "oom": False,
    }
    if kappa_chosen is None or kappa_chosen < 1:
        out["compliant"] = False
        out["error"] = "no_kappa"
        return out

    net = build_net(net_name).to(device)
    if mode_id in (3, 4):
        x = torch.randn(T, b, C, H, H, device="cpu", pin_memory=False)
    else:
        x = torch.randn(T, b, C, H, H, device=device)

    measured_GB, run_status = measure_peak_hbm(net, x, mode_id,
                                                kappa_chosen, device)
    out["measured_GB"] = measured_GB
    if run_status == "oom":
        out["oom"] = True
        out["compliant"] = False
        out["unused_GB"] = -999.0  # marker: violation by OOM
    elif run_status.startswith("error"):
        out["compliant"] = False
        out["unused_GB"] = -999.0
        out["error"] = run_status
    else:
        out["compliant"] = (measured_GB <= M_budget_GB)
        out["unused_GB"] = M_budget_GB - measured_GB

    del net, x
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ============================================================================
# Aggregate
# ============================================================================

def summarize(results, label):
    n_total = len(results)
    n_safe = sum(1 for r in results
                 if r["compliant"] is True and not r["oom"])
    n_oom = sum(1 for r in results if r["oom"])
    n_violation = sum(1 for r in results
                      if r["compliant"] is False and not r["oom"])
    safe_unused = [r["unused_GB"] for r in results
                   if r["compliant"] is True and not r["oom"]]
    return {
        "strategy": label,
        "n_total": n_total,
        "n_compliant": n_safe,
        "n_oom": n_oom,
        "n_violation": n_violation,
        "compliance_pct": 100.0 * n_safe / n_total if n_total else 0.0,
        "oom_pct": 100.0 * n_oom / n_total if n_total else 0.0,
        "violation_pct": 100.0 * n_violation / n_total if n_total else 0.0,
        "avg_unused_GB_safe": (float(np.mean(safe_unused))
                                if safe_unused else 0.0),
        "p95_unused_GB_safe": (float(np.percentile(safe_unused, 95))
                                if safe_unused else 0.0),
        "max_unused_GB_safe": (float(np.max(safe_unused))
                                if safe_unused else 0.0),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True,
                        help="path to p9_1a fitted coefficients JSON "
                             "(used to enumerate the cell grid only)")
    parser.add_argument("--ablation2_json", required=True,
                        help="path to p9_6b_no_envelope output JSON "
                             "(provides AEROS calibrated kappa* per cell)")
    parser.add_argument("--output", default="p9_6c_manual_kappa")
    parser.add_argument("--Ts", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--Ms", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--nets", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--conservative_kappa", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    # Determinism (so peak measurement is consistent)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)

    # Load coeffs (only for arch enumeration; coeff values not needed here)
    with open(args.coeffs) as f:
        data = json.load(f)
    coeffs_bundle = data.get("coeffs", data)
    if args.nets:
        coeffs_bundle = {k: v for k, v in coeffs_bundle.items()
                         if k in args.nets}

    # Load Ablation 2 JSON to get AEROS calibrated kappa* per cell
    with open(args.ablation2_json) as f:
        abl2 = json.load(f)
    cal_results = abl2["results"]["calibrated"]
    # Index by (net, T, mode, M_budget) -> kappa*
    cal_kappa = {}
    for r in cal_results:
        key = (r["net"], r["T"], r["mode"], r["M_budget_GB"])
        cal_kappa[key] = (r["kappa_star"], r["status"], r["measured_GB"],
                          r.get("compliant"), r.get("oom"))

    print("=" * 78)
    print("AEROS Ablation 3: manual fixed-kappa baseline")
    print("=" * 78)
    print(f"  archs: {list(coeffs_bundle.keys())}")
    print(f"  Ts={args.Ts}  Ms={args.Ms}GB  modes={args.modes}")
    print(f"  conservative kappa = {args.conservative_kappa}")
    print(f"  AEROS kappa* read from: {args.ablation2_json}")
    print()

    # Build cell grid
    cells = []
    for net_name in coeffs_bundle:
        for T in args.Ts:
            for M_GB in args.Ms:
                for mode_id in args.modes:
                    cells.append((net_name, T, M_GB, mode_id))

    # Per-strategy results
    by_strategy = {
        "S1_no_segment": [],
        "S2_naive_halve": [],
        "S3_conservative": [],
        "S4_AEROS": [],
    }

    for idx, (net_name, T, M_GB, mode_id) in enumerate(cells, 1):
        # Strategy choices
        k1 = strategy_no_segment(T)
        k2 = strategy_naive_halve(T)
        k3 = strategy_conservative(T, args.conservative_kappa)
        # S4: AEROS kappa* from Ablation 2 calibrated config
        key = (net_name, T, mode_id, M_GB)
        if key in cal_kappa:
            k4_star, k4_status, k4_meas, k4_comp, k4_oom = cal_kappa[key]
            if k4_status == "INFEASIBLE":
                k4 = None  # solver refused; mark as INFEASIBLE
            else:
                k4 = k4_star
        else:
            k4 = None

        print(f"\n[{idx}/{len(cells)}] {net_name} T={T} mode={mode_id} "
              f"M={M_GB}GB  | S1={k1} S2={k2} S3={k3} S4={k4}")

        # S1: no segment (kappa=T, mode 1 effectively, but keep mode for fairness)
        # NOTE: at kappa=T the segmented loop degenerates to single forward;
        # for mode 4 we still apply the sink for fair comparison.
        r1 = eval_strategy(net_name, T, mode_id, M_GB, k1, device,
                           b=args.batch_size, C=args.C, H=args.H)
        by_strategy["S1_no_segment"].append(r1)
        print(f"   S1 NO-SEG    : meas={r1['measured_GB']:.2f}GB "
              f"comp={r1['compliant']} oom={r1['oom']} "
              f"unused={r1['unused_GB']:.2f}GB")

        # S2: naive halve
        r2 = eval_strategy(net_name, T, mode_id, M_GB, k2, device,
                           b=args.batch_size, C=args.C, H=args.H)
        by_strategy["S2_naive_halve"].append(r2)
        print(f"   S2 HALVE     : meas={r2['measured_GB']:.2f}GB "
              f"comp={r2['compliant']} oom={r2['oom']} "
              f"unused={r2['unused_GB']:.2f}GB")

        # S3: conservative fixed
        r3 = eval_strategy(net_name, T, mode_id, M_GB, k3, device,
                           b=args.batch_size, C=args.C, H=args.H)
        by_strategy["S3_conservative"].append(r3)
        print(f"   S3 CONSV-{args.conservative_kappa:>2d}  : meas={r3['measured_GB']:.2f}GB "
              f"comp={r3['compliant']} oom={r3['oom']} "
              f"unused={r3['unused_GB']:.2f}GB")

        # S4: AEROS — read from Ablation 2 (avoid re-measure for efficiency)
        if k4 is None:
            r4 = {"net": net_name, "T": T, "mode": mode_id,
                  "M_budget_GB": M_GB, "kappa": None,
                  "measured_GB": -1.0, "compliant": False,
                  "unused_GB": -999.0, "oom": False, "infeasible": True}
        else:
            r4 = {"net": net_name, "T": T, "mode": mode_id,
                  "M_budget_GB": M_GB, "kappa": k4,
                  "measured_GB": k4_meas, "compliant": k4_comp,
                  "unused_GB": M_GB - k4_meas if k4_meas > 0 else -999.0,
                  "oom": k4_oom or False}
        by_strategy["S4_AEROS"].append(r4)
        print(f"   S4 AEROS κ*  : meas={r4['measured_GB']:.2f}GB "
              f"comp={r4['compliant']} oom={r4['oom']} "
              f"unused={r4['unused_GB']:.2f}GB")

    # Summarize
    sums = {k: summarize(v, k) for k, v in by_strategy.items()}

    print("\n" + "=" * 95)
    print("Summary — manual kappa strategies")
    print("=" * 95)
    hdr = f"{'Strategy':<22s}  {'compliance%':>11s}  {'oom%':>7s}  {'viol%':>7s}  {'avg_unused':>11s}  {'p95_unused':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for label in ("S1_no_segment", "S2_naive_halve",
                  "S3_conservative", "S4_AEROS"):
        s = sums[label]
        print(f"  {label:<20s}  {s['compliance_pct']:>10.2f}%  "
              f"{s['oom_pct']:>6.2f}%  {s['violation_pct']:>6.2f}%  "
              f"{s['avg_unused_GB_safe']:>10.3f}GB  "
              f"{s['p95_unused_GB_safe']:>10.3f}GB")

    out = {
        "ablation": "manual fixed-kappa baseline (Ablation 3)",
        "config": vars(args),
        "summary": sums,
        "results_by_strategy": by_strategy,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()