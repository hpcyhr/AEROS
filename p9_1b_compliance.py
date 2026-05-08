#!/usr/bin/env python
"""
AEROS Phase 2 Exp 1B — Contract compliance multi-budget.

For each (net, T, mode, M_budget):
  1. Load fitted coefficients from p9_1a JSON.
  2. Compute residual envelope Delta_resid = q_{1-delta}(max(0, meas - pred))
     across the cells fit for this (net, T) (one-sided underprediction).
  3. Run memory-budget solver:
        kappa_A = floor((M_budget * (1 - epsilon) - A(T, pi)) / B(pi))
     where epsilon comes from Delta_resid / M_budget + safety margin.
  4. If kappa_A >= 1, run forward at kappa = kappa_A in mode_id, measure peak.
  5. Record: feasible/infeasible, predicted, measured, violation flag.

Compliance metric: fraction of feasible cells where measured <= M_budget.
Target: >= (1 - delta) = 95% under nominal contract.

Usage:
  python p9_1b_compliance.py --coeffs p9_1a_full.json --output p9_1b_results
"""

from __future__ import annotations

import argparse
import gc
import json
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


# ============================================================================
# Net builders + runners (copied from p9_1a)
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
    try: functional.reset_net(net)
    except Exception: pass


@torch.no_grad()
def run_mode(net, T, kappa, b, C, H, mode_id, device, num_classes=10):
    try:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        reset_state(net)
        t0 = time.time()
        if mode_id == 1:
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            y = net(x)
            torch.cuda.synchronize(device)
            del x, y
        elif mode_id == 2:
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            chunks = []; i = 0
            while i < T:
                sz = min(kappa, T - i)
                chunks.append(net(x[i:i+sz])); i += sz
            y = torch.cat(chunks, dim=0)
            torch.cuda.synchronize(device)
            del x, y, chunks
        elif mode_id == 3:
            chunks = []
            g = torch.Generator(device=device).manual_seed(42)
            i = 0
            while i < T:
                sz = min(kappa, T - i)
                xs = torch.randn(sz, b, C, H, H, generator=g, device=device,
                                 dtype=torch.float32)
                chunks.append(net(xs)); del xs; i += sz
            y = torch.cat(chunks, dim=0)
            torch.cuda.synchronize(device)
            del y, chunks
        elif mode_id == 4:
            g = torch.Generator(device=device).manual_seed(42)
            running_sum = torch.zeros(b, num_classes, device=device, dtype=torch.float32)
            n = 0; i = 0
            while i < T:
                sz = min(kappa, T - i)
                xs = torch.randn(sz, b, C, H, H, generator=g, device=device,
                                 dtype=torch.float32)
                ys = net(xs)
                running_sum += ys.sum(dim=0); n += sz
                del xs, ys; i += sz
            torch.cuda.synchronize(device)
            del running_sum
        wall_ms = (time.time() - t0) * 1000
        return torch.cuda.max_memory_allocated(device), wall_ms
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return -1, -1.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache(); gc.collect()
            return -1, -1.0
        raise


# ============================================================================
# Memory model (matches p9_1a fits)
# ============================================================================

def AB_for_mode(coeffs: Dict, T: int, mode_id: int):
    """Return (A(T,pi), B(pi)) for the given mode using fitted coefficients."""
    M_0 = coeffs["M_0_bytes"]
    aK = coeffs["alpha_K_bytes"]
    a_in = coeffs["alpha_in_bytes"]
    a_out = coeffs["alpha_out_bytes"]
    if mode_id == 1:
        A = M_0 + (a_in + a_out) * T
        B = aK
    elif mode_id == 2:
        A = M_0 + (a_in + a_out) * T
        B = aK
    elif mode_id == 3:
        A = M_0 + a_out * T
        B = a_in + aK
    elif mode_id == 4:
        A = M_0
        B = a_in + aK + a_out
    else:
        raise ValueError(f"unknown mode {mode_id}")
    return A, B


def predict_peak(coeffs, T, kappa, mode_id):
    A, B = AB_for_mode(coeffs, T, mode_id)
    if mode_id == 1:
        return A + B * T
    return A + B * kappa


# ============================================================================
# Calibrated residual envelope from fit residuals
# ============================================================================

def compute_residual_envelope(coeffs: Dict, delta: float = 0.05) -> float:
    """One-sided underprediction quantile in bytes from p9_1a fit.

    fit_residuals_GB has been stored as (meas - pred) in GB; positive means
    underprediction (meas > pred), negative means overprediction.
    The envelope is the (1-delta) quantile of max(0, residual).
    """
    BYTES_PER_GB = 1024 ** 3
    if "fit_residuals_GB" not in coeffs:
        return 0.0
    resids_under = [max(0.0, r) * BYTES_PER_GB
                    for r in coeffs["fit_residuals_GB"]]
    if not resids_under:
        return 0.0
    return float(np.quantile(resids_under, 1.0 - delta))


# ============================================================================
# Solver
# ============================================================================

def kappa_solver(coeffs, T, mode_id, M_budget_bytes, delta=0.05,
                 epsilon_safety=0.02):
    """Memory-budget solver. Returns (kappa_star, status, predicted_peak_bytes,
    epsilon_used)."""
    A, B = AB_for_mode(coeffs, T, mode_id)
    Delta_resid = compute_residual_envelope(coeffs, delta=delta)

    eff_budget = M_budget_bytes - Delta_resid - epsilon_safety * M_budget_bytes

    # Mode 1: kappa is fixed = T; just check feasibility
    if mode_id == 1:
        pred = A + B * T
        if pred <= eff_budget:
            return T, "feasible", pred, epsilon_safety
        else:
            return -1, "INFEASIBLE", pred, epsilon_safety

    # Modes 2/3/4: solve for kappa
    if B <= 1e3:                            # bytes; degenerate slope
        if A <= eff_budget:
            return T, "feasible_degenerate", A, epsilon_safety
        else:
            return -1, "INFEASIBLE", A, epsilon_safety

    kappa_raw = (eff_budget - A) / B
    if kappa_raw < 1:
        return -1, "INFEASIBLE", A + B, epsilon_safety
    kappa_star = min(int(np.floor(kappa_raw)), T)
    pred = A + B * kappa_star
    return kappa_star, "feasible", pred, epsilon_safety


# ============================================================================
# Compliance sweep
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True,
                        help="JSON from p9_1a")
    parser.add_argument("--budgets_GB", default="1,2,4,8,16,32")
    parser.add_argument("--T_sweep", default="128,1024,4096")
    parser.add_argument("--modes", default="2,3,4")
    parser.add_argument("--nets", default="all")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    parser.add_argument("--output", default="p9_1b_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    with open(args.coeffs) as f:
        coeffs_json = json.load(f)
    config = coeffs_json["config"]
    all_coeffs = coeffs_json["coeffs"]
    BYTES_PER_GB = 1024 ** 3

    nets = list(all_coeffs.keys())
    if args.nets.lower() != "all":
        names = [n.strip() for n in args.nets.split(",")]
        nets = [n for n in nets if n in names]
    Ts = [int(t) for t in args.T_sweep.split(",")]
    modes = [int(m) for m in args.modes.split(",")]
    budgets_bytes = [float(b) * BYTES_PER_GB for b in args.budgets_GB.split(",")]

    b_size = config["b"]; H = config["H"]; C = config["C"]; nc = config["num_classes"]
    print(f"=== AEROS Phase 2 Exp 1B — Contract compliance ===")
    print(f"  Coeffs from: {args.coeffs}")
    print(f"  Nets: {len(nets)}  T: {Ts}  modes: {modes}  budgets: {args.budgets_GB} GB")
    print(f"  delta={args.delta}  epsilon_safety={args.epsilon_safety}")
    print(f"  b={b_size} C={C} H={H}")

    cells = []
    for net_name in nets:
        print(f"\n{'='*72}\n=== {net_name} ===\n{'='*72}")
        try:
            net = build_net(net_name).to(device)
        except Exception as e:
            print(f"  Failed to build: {e}")
            continue

        for T in Ts:
            T_str = str(T)
            if T_str not in all_coeffs[net_name]:
                continue
            coeffs = all_coeffs[net_name][T_str]
            if "error" in coeffs:
                print(f"  [skip] T={T}: fit error")
                continue
            envelope_GB = compute_residual_envelope(coeffs, args.delta) / BYTES_PER_GB
            print(f"\n  T={T}  envelope(q_{{1-{args.delta:.2f}}})={envelope_GB:.4f} GB")

            for mode_id in modes:
                for M_budget in budgets_bytes:
                    M_budget_GB = M_budget / BYTES_PER_GB
                    kappa, status, pred, eps = kappa_solver(
                        coeffs, T, mode_id, M_budget,
                        delta=args.delta,
                        epsilon_safety=args.epsilon_safety)
                    record = {
                        "net": net_name, "T": T, "mode": mode_id,
                        "budget_GB": M_budget_GB,
                        "kappa": kappa, "solver_status": status,
                        "predicted_GB": pred / BYTES_PER_GB if pred > 0 else -1.0,
                        "envelope_GB": envelope_GB,
                        "epsilon_safety": eps,
                    }
                    if status.startswith("feasible"):
                        peak, wall = run_mode(net, T, kappa, b_size, C, H,
                                              mode_id, device, nc)
                        record["measured_GB"] = peak / BYTES_PER_GB if peak > 0 else -1.0
                        record["wall_ms"] = wall
                        if peak < 0:
                            record["status"] = "OOM"
                        elif peak > M_budget:
                            record["status"] = "VIOLATION"
                        else:
                            record["status"] = "compliant"
                        marker = ("✓" if record["status"] == "compliant"
                                  else "✗" if record["status"] == "VIOLATION"
                                  else "!")
                        print(f"    M={M_budget_GB:5.1f}GB  mode={mode_id}  "
                              f"kappa={kappa:5d}  pred={record['predicted_GB']:6.3f}GB  "
                              f"meas={record['measured_GB']:6.3f}GB  "
                              f"{record['status']:10s} {marker}")
                    else:
                        record["measured_GB"] = -1.0
                        record["wall_ms"] = -1.0
                        record["status"] = "INFEASIBLE"
                        print(f"    M={M_budget_GB:5.1f}GB  mode={mode_id}  "
                              f"INFEASIBLE (predicted {record['predicted_GB']:.3f}GB)")
                    cells.append(record)
        del net
        torch.cuda.empty_cache(); gc.collect()

    with open(args.output + ".json", "w") as f:
        json.dump({"config": config, "cells": cells,
                   "params": {"delta": args.delta,
                              "epsilon_safety": args.epsilon_safety,
                              "budgets_GB": args.budgets_GB}}, f, indent=2)
    print(f"\nSaved JSON: {args.output}.json")

    # Summary
    print(f"\n{'='*72}")
    print(f"=== Compliance summary ===")
    print(f"{'='*72}")
    feasible_cells = [c for c in cells if c["status"] != "INFEASIBLE"]
    n_total = len(cells)
    n_inf = sum(1 for c in cells if c["status"] == "INFEASIBLE")
    n_compliant = sum(1 for c in cells if c["status"] == "compliant")
    n_violation = sum(1 for c in cells if c["status"] == "VIOLATION")
    n_oom = sum(1 for c in cells if c["status"] == "OOM")
    print(f"  Total cells: {n_total}")
    print(f"  INFEASIBLE: {n_inf}  (solver refused)")
    print(f"  Feasible runs: {len(feasible_cells)}")
    print(f"    compliant:  {n_compliant}  ({n_compliant/max(1,len(feasible_cells))*100:.1f}%)")
    print(f"    VIOLATION:  {n_violation}  ({n_violation/max(1,len(feasible_cells))*100:.1f}%)")
    print(f"    OOM:        {n_oom}")
    if n_violation > 0:
        print(f"\n  Violations:")
        for c in cells:
            if c["status"] == "VIOLATION":
                excess = (c["measured_GB"] - c["budget_GB"])
                print(f"    {c['net']:12s} T={c['T']:5d} mode={c['mode']} "
                      f"M={c['budget_GB']:5.1f}GB  "
                      f"meas={c['measured_GB']:.3f}  excess=+{excess:.3f}GB")


if __name__ == "__main__":
    main()