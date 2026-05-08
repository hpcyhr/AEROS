#!/usr/bin/env python
"""
AEROS Phase 2 Exp 2 — Regret Pareto frontier.

For each (net, T, mode, M_budget) cell:
  1. Exhaustive kappa sweep: measure (peak_HBM, wall_time) for each
     kappa in a geometric grid.
  2. AEROS solver picks kappa_AEROS using the calibrated formula.
  3. kappa_opt = argmin wall_time s.t. peak_HBM <= M_budget (over the sweep).
  4. regret = wall_AEROS / wall_opt.

Scope (subset of Exp 1 cells):
  4 nets x 2 T x 2 modes x 3 budgets = 48 contract cells
  ~8 kappa values per cell (geometric)
  ~384 forward measurements total

Outputs:
  - p9_2_regret.json    : per-cell {kappa_AEROS, kappa_opt, regret, peak, wall}
  - tab_regret.tex      : LaTeX summary table
  - p9_2_pareto.json    : per-(net, T, mode) full Pareto sweep for plotting

Usage:
  python p9_2_regret.py --coeffs p9_1a_full.json --output p9_2_results
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import torch

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18)
    from spikingjelly.activation_based.model.sew_resnet import sew_resnet50
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg11_bn, spiking_vgg19_bn)
    SJ_OK = True
except Exception as e:
    print(f"[WARN] SJ unavailable: {e}")
    SJ_OK = False


# ============================================================================
# Net builders (subset to keep regret experiment tractable)
# ============================================================================

def build_net(name, num_classes=10):
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=num_classes,
    )
    table = {
        "SR-18":     lambda: spiking_resnet18(**common),
        "SEW-50":    lambda: sew_resnet50(cnf="ADD", **common),
        "VGG-11-BN": lambda: spiking_vgg11_bn(**common),
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


# ============================================================================
# Mode runners (per-iter peak measurement; copied from p9_1b)
# ============================================================================

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
# Solver (matches p9_1b)
# ============================================================================

def AB_for_mode(coeffs, T, mode_id):
    M_0 = coeffs["M_0_bytes"]
    aK = coeffs["alpha_K_bytes"]
    a_in = coeffs["alpha_in_bytes"]
    a_out = coeffs["alpha_out_bytes"]
    if mode_id == 1:
        A = M_0 + (a_in + a_out) * T; B = aK
    elif mode_id == 2:
        A = M_0 + (a_in + a_out) * T; B = aK
    elif mode_id == 3:
        A = M_0 + a_out * T; B = a_in + aK
    elif mode_id == 4:
        A = M_0; B = a_in + aK + a_out
    else:
        raise ValueError(f"unknown mode {mode_id}")
    return A, B


def compute_residual_envelope(coeffs, delta=0.05):
    BYTES_PER_GB = 1024 ** 3
    if "fit_residuals_GB" not in coeffs:
        return 0.0
    rs = [max(0.0, r) * BYTES_PER_GB for r in coeffs["fit_residuals_GB"]]
    if not rs: return 0.0
    return float(np.quantile(rs, 1 - delta))


def kappa_solver(coeffs, T, mode_id, M_budget_bytes,
                 delta=0.05, epsilon_safety=0.02):
    A, B = AB_for_mode(coeffs, T, mode_id)
    Delta_resid = compute_residual_envelope(coeffs, delta)
    eff_budget = M_budget_bytes - Delta_resid - epsilon_safety * M_budget_bytes
    if mode_id == 1:
        pred = A + B * T
        return (T if pred <= eff_budget else -1), pred
    if B <= 1e3:
        return (T if A <= eff_budget else -1), A
    kappa_raw = (eff_budget - A) / B
    if kappa_raw < 1: return -1, A + B
    return min(int(np.floor(kappa_raw)), T), A + B * min(int(np.floor(kappa_raw)), T)


# ============================================================================
# Kappa sweep grid
# ============================================================================

def kappa_grid(T):
    """Geometric grid of kappa values up to T, capped at T."""
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return sorted(set(k for k in base if k <= T) | {T})


# ============================================================================
# Sweep
# ============================================================================

@dataclass
class CellResult:
    net: str
    T: int
    mode: int
    budget_GB: float
    kappa_AEROS: int
    peak_AEROS_GB: float
    wall_AEROS_ms: float
    kappa_opt: int
    peak_opt_GB: float
    wall_opt_ms: float
    regret: float
    n_feasible: int
    pareto_kappas: List[int] = field(default_factory=list)
    pareto_peaks_GB: List[float] = field(default_factory=list)
    pareto_walls_ms: List[float] = field(default_factory=list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True)
    parser.add_argument("--nets", default="SR-18,SEW-50,VGG-11-BN,VGG-19-BN")
    parser.add_argument("--T_sweep", default="128,1024")
    parser.add_argument("--modes", default="3,4")
    parser.add_argument("--budgets_GB", default="4,8,16")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    parser.add_argument("--output", default="p9_2_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    with open(args.coeffs) as f:
        coeffs_data = json.load(f)
    all_coeffs = coeffs_data["coeffs"]

    nets = [n.strip() for n in args.nets.split(",")]
    Ts = [int(t) for t in args.T_sweep.split(",")]
    modes = [int(m) for m in args.modes.split(",")]
    BYTES_PER_GB = 1024 ** 3
    budgets_bytes = [float(b) * BYTES_PER_GB for b in args.budgets_GB.split(",")]

    print(f"=== AEROS Phase 2 Exp 2 — Regret Pareto frontier ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Nets: {nets}  T: {Ts}  modes: {modes}  budgets: {args.budgets_GB} GB")
    print(f"  b={args.b}  C=3  H={args.H}")

    cells = []
    pareto_data = {}                 # (net, T, mode) -> list of (kappa, peak_GB, wall_ms)

    for net_name in nets:
        if net_name not in all_coeffs:
            print(f"  [skip] {net_name}: no coefficients")
            continue
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
            for mode_id in modes:
                kappas = kappa_grid(T)
                print(f"\n  --- T={T}, mode={mode_id}, kappa sweep: {kappas} ---")

                # Sweep all kappas; cache (peak, wall) per kappa
                pareto = {}
                for k in kappas:
                    peak, wall = run_mode(net, T, k, args.b, 3, args.H,
                                          mode_id, device)
                    pareto[k] = (peak, wall)
                    if peak > 0:
                        print(f"    kappa={k:5d}  peak={peak/BYTES_PER_GB:7.3f}GB  wall={wall:8.1f}ms")
                    else:
                        print(f"    kappa={k:5d}  OOM")

                # Save pareto data for plotting
                key = f"{net_name}::{T}::{mode_id}"
                pareto_data[key] = [
                    {"kappa": k, "peak_bytes": p, "wall_ms": w}
                    for k, (p, w) in pareto.items()
                ]

                # For each budget, compute kappa_AEROS, kappa_opt, regret
                for M_budget in budgets_bytes:
                    M_GB = M_budget / BYTES_PER_GB
                    k_aeros, _ = kappa_solver(coeffs, T, mode_id, M_budget,
                                               args.delta, args.epsilon_safety)
                    if k_aeros < 1:
                        print(f"    M={M_GB:5.1f}GB  AEROS=INFEASIBLE")
                        continue
                    feasible = [(k, p, w) for k, (p, w) in pareto.items()
                                if 0 < p <= M_budget]
                    if not feasible:
                        print(f"    M={M_GB:5.1f}GB  no feasible kappa in sweep")
                        continue
                    k_opt, p_opt, w_opt = min(feasible, key=lambda x: x[2])

                    # Get wall_AEROS at k_aeros: use closest measured kappa
                    if k_aeros in pareto and pareto[k_aeros][0] > 0:
                        p_aeros, w_aeros = pareto[k_aeros]
                    else:
                        # k_aeros may not be in the sweep grid; pick closest <=
                        valid = [k for k in kappas if k <= k_aeros and pareto[k][0] > 0]
                        if not valid:
                            continue
                        k_aeros_closest = max(valid)
                        p_aeros, w_aeros = pareto[k_aeros_closest]
                        k_aeros = k_aeros_closest

                    regret = w_aeros / w_opt

                    cell = CellResult(
                        net=net_name, T=T, mode=mode_id, budget_GB=M_GB,
                        kappa_AEROS=k_aeros,
                        peak_AEROS_GB=p_aeros / BYTES_PER_GB,
                        wall_AEROS_ms=w_aeros,
                        kappa_opt=k_opt,
                        peak_opt_GB=p_opt / BYTES_PER_GB,
                        wall_opt_ms=w_opt,
                        regret=regret,
                        n_feasible=len(feasible),
                        pareto_kappas=[k for k, _, _ in feasible],
                        pareto_peaks_GB=[p / BYTES_PER_GB for _, p, _ in feasible],
                        pareto_walls_ms=[w for _, _, w in feasible],
                    )
                    cells.append(cell)
                    print(f"    M={M_GB:5.1f}GB  AEROS k={k_aeros:5d} (peak={p_aeros/BYTES_PER_GB:.3f}GB  wall={w_aeros:.1f}ms)  "
                          f"OPT k={k_opt:5d} (peak={p_opt/BYTES_PER_GB:.3f}GB  wall={w_opt:.1f}ms)  "
                          f"regret={regret:.3f}")
        del net
        torch.cuda.empty_cache(); gc.collect()

    # Save
    output = {
        "config": {
            "nets": nets, "Ts": Ts, "modes": modes,
            "budgets_GB": args.budgets_GB,
            "b": args.b, "C": 3, "H": args.H,
            "delta": args.delta, "epsilon_safety": args.epsilon_safety,
        },
        "cells": [asdict(c) for c in cells],
        "pareto": pareto_data,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {args.output}.json")

    # Summary
    print(f"\n{'='*72}")
    print(f"=== Regret summary ({len(cells)} contract cells) ===")
    print(f"{'='*72}")
    if cells:
        regrets = [c.regret for c in cells]
        print(f"  Mean regret:    {np.mean(regrets):.4f}")
        print(f"  Median regret:  {np.median(regrets):.4f}")
        print(f"  Min regret:     {np.min(regrets):.4f}")
        print(f"  Max regret:     {np.max(regrets):.4f}")
        print(f"  P95 regret:     {np.quantile(regrets, 0.95):.4f}")
        print(f"  Cells with regret < 1.10: "
              f"{sum(1 for r in regrets if r < 1.10)}/{len(regrets)} "
              f"({sum(1 for r in regrets if r < 1.10)/len(regrets)*100:.1f}%)")
        print(f"\n  Cells with highest regret:")
        for c in sorted(cells, key=lambda c: -c.regret)[:5]:
            print(f"    {c.net:12s} T={c.T:5d} mode={c.mode} M={c.budget_GB:.0f}GB  "
                  f"kappa_AEROS={c.kappa_AEROS:5d}  kappa_opt={c.kappa_opt:5d}  "
                  f"regret={c.regret:.3f}")


if __name__ == "__main__":
    main()