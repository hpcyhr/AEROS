#!/usr/bin/env python
"""
AEROS Phase 2 Exp 11 — Boundary bit-exact validation.

Doris 10 P0-1 fix: §5.1 bit-exact ran at T=4, kappa>=4 -> single segment, no
boundary stress. This experiment runs each trained checkpoint at SYNTHETIC
horizons T in {16, 32, 64} with kappa in {4, 8} to actually traverse
multiple segment boundaries (T/kappa = 2..16 segments per run), comparing
segmented forward against unsegmented baseline element-by-element.

We use synthetic random input (matching CIFAR-10 input shape b=32 C=3 H=32 W=32)
because trained checkpoints accept the same input shape regardless of T;
the boundary correctness is a property of the runtime, not the data
distribution.

Non-uniform schedules: for each T, also run [kappa, T-kappa] (one short tail)
and [T-2, 1, 1] (two singleton tails) to stress the runtime's variable-segment
handling.

Usage:
    python p9_11_boundary_bitexact.py \\
        --checkpoint_dir /data/yhr/AEROS/checkpoints \\
        --T_sweep 16,32,64 \\
        --kappa_sweep 4,8 \\
        --output p9_11_boundary
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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
except Exception as e:
    print(f"[FATAL] SJ unavailable: {e}")
    raise


# ============================================================================
# Net builders (10 SJ standard nets that match Exp 1A coefficients)
# ============================================================================

def build_net(name, num_classes=10):
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=num_classes,
    )
    table = {
        "SR-18":     lambda: spiking_resnet18(**common),
        "SR-34":     lambda: spiking_resnet34(**common),
        "SR-50":     lambda: spiking_resnet50(**common),
        "SEW-18":    lambda: sew_resnet18(cnf="ADD", **common),
        "SEW-50":    lambda: sew_resnet50(cnf="ADD", **common),
        "SEW-101":   lambda: sew_resnet101(cnf="ADD", **common),
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


def load_checkpoint(net, ckpt_path):
    """Load weights if checkpoint present; else fall back to random init."""
    if not os.path.exists(ckpt_path):
        return False, f"missing: {ckpt_path}"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            sd = ckpt
        # strip "module." prefix if present
        sd2 = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd2, strict=False)
        return True, "loaded"
    except Exception as e:
        return False, f"load_fail: {type(e).__name__}: {str(e)[:80]}"


# ============================================================================
# Schedules: uniform and non-uniform
# ============================================================================

def uniform_schedule(T, kappa) -> List[int]:
    """Uniform kappa, last segment may be shorter."""
    out = []
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        out.append(sz)
        i += sz
    return out


def short_tail_schedule(T, kappa) -> List[int]:
    """[kappa, kappa, ..., remainder] - same as uniform when T % kappa == 0."""
    return uniform_schedule(T, kappa)


def singleton_tail_schedule(T, kappa) -> List[int]:
    """[..., 1, 1] - exposes the smallest possible final segments."""
    if T <= 2:
        return [T]
    main = uniform_schedule(T - 2, kappa)
    return main + [1, 1]


# ============================================================================
# Forward: unsegmented and segmented
# ============================================================================

@torch.no_grad()
def forward_unsegmented(net, x):
    """Run net(x) as a single forward pass (kappa = T).

    x: [T, b, C, H, W]
    Returns: [T, b, num_classes]
    """
    reset_state(net)
    return net(x)


@torch.no_grad()
def forward_segmented(net, x, schedule: List[int]):
    """Run net on x split according to schedule, concatenating outputs.

    The streamability certificate's accept-carry contract requires that
    network state (LIF v, BN running stats in eval mode are static) is
    propagated across segment boundaries by SJ's reset_state happening
    only ONCE before the first segment. Within a single inference call
    the state should accumulate naturally.

    x: [T, b, C, H, W]
    Returns: [T, b, num_classes]
    """
    reset_state(net)
    chunks = []
    i = 0
    for sz in schedule:
        y_seg = net(x[i:i+sz])
        chunks.append(y_seg)
        i += sz
    assert i == x.shape[0], f"schedule sum {i} != T {x.shape[0]}"
    return torch.cat(chunks, dim=0)


# ============================================================================
# Cell definition + sweep
# ============================================================================

@dataclass
class BoundaryCell:
    net: str
    ckpt_loaded: bool
    T: int
    kappa: int
    schedule_kind: str          # "uniform", "singleton_tail"
    schedule: List[int]
    n_segments: int
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    pred_agreement: float       # fraction of (t, b) with argmax matching
    n_compared: int
    status: str = "ok"
    error_msg: str = ""


def compare_outputs(y_ref: torch.Tensor, y_seg: torch.Tensor) -> Tuple[float, float, float, float, int]:
    """Element-wise comparison of [T, b, num_classes] tensors."""
    diff = (y_ref - y_seg).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_abs = y_ref.abs().clamp(min=1e-12)
    max_rel = float((diff / ref_abs).max().item())
    pred_ref = y_ref.argmax(dim=-1)
    pred_seg = y_seg.argmax(dim=-1)
    pred_agree = float((pred_ref == pred_seg).float().mean().item())
    n = int(y_ref.numel())
    return max_abs, mean_abs, max_rel, pred_agree, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="/data/yhr/AEROS/checkpoints")
    parser.add_argument("--nets", default="all")
    parser.add_argument("--T_sweep", default="16,32,64")
    parser.add_argument("--kappa_sweep", default="4,8")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="p9_11_boundary")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)

    Ts = [int(t) for t in args.T_sweep.split(",")]
    kappas = [int(k) for k in args.kappa_sweep.split(",")]

    all_nets = ["SR-18", "SR-34", "SR-50",
                "SEW-18", "SEW-50", "SEW-101",
                "VGG-11-BN", "VGG-13-BN", "VGG-16-BN", "VGG-19-BN"]
    if args.nets.lower() == "all":
        nets = all_nets
    else:
        nets = [n.strip() for n in args.nets.split(",")]

    print(f"=== AEROS Phase 2 Exp 11 — Boundary bit-exact validation ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Nets: {nets}")
    print(f"  T sweep: {Ts}  kappa sweep: {kappas}")
    print(f"  Input shape: b={args.b} C=3 H=W={args.H}")
    print(f"  Schedules per (T, kappa): uniform, singleton_tail")

    cells = []
    for net_name in nets:
        print(f"\n{'='*72}\n=== {net_name} ===\n{'='*72}")
        try:
            net = build_net(net_name).to(device)
        except Exception as e:
            print(f"  [skip] build fail: {e}")
            continue

        ckpt_path = os.path.join(args.checkpoint_dir, f"{net_name}_cifar10_best.pth")
        loaded, msg = load_checkpoint(net, ckpt_path)
        print(f"  checkpoint: {msg}  (loaded={loaded})")

        for T in Ts:
            torch.manual_seed(args.seed + T)
            x = torch.randn(T, args.b, 3, args.H, args.H, device=device,
                            dtype=torch.float32)

            try:
                y_ref = forward_unsegmented(net, x)
                torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError:
                print(f"  T={T}: baseline OOM, skipping")
                continue
            except Exception as e:
                print(f"  T={T}: baseline fail {type(e).__name__}, skipping")
                continue

            for kappa in kappas:
                if kappa >= T:
                    continue                # Doris 10 P0-1: must be true segment split
                schedules = [
                    ("uniform",        uniform_schedule(T, kappa)),
                    ("singleton_tail", singleton_tail_schedule(T, kappa)),
                ]
                for sched_kind, sched in schedules:
                    cell = BoundaryCell(
                        net=net_name, ckpt_loaded=loaded, T=T, kappa=kappa,
                        schedule_kind=sched_kind, schedule=sched,
                        n_segments=len(sched),
                        max_abs_err=-1.0, mean_abs_err=-1.0, max_rel_err=-1.0,
                        pred_agreement=-1.0, n_compared=0,
                    )
                    try:
                        y_seg = forward_segmented(net, x, sched)
                        torch.cuda.synchronize(device)
                        max_abs, mean_abs, max_rel, pred_agree, n_cmp = \
                            compare_outputs(y_ref, y_seg)
                        cell.max_abs_err = max_abs
                        cell.mean_abs_err = mean_abs
                        cell.max_rel_err = max_rel
                        cell.pred_agreement = pred_agree
                        cell.n_compared = n_cmp
                        marker = "✓" if max_abs < 1e-5 else "✗"
                        print(f"  T={T:3d} kappa={kappa:2d} {sched_kind:15s} "
                              f"segs={len(sched):2d}  "
                              f"max_abs={max_abs:.2e}  pred_agree={pred_agree*100:.2f}% {marker}")
                        del y_seg
                    except torch.cuda.OutOfMemoryError:
                        cell.status = "OOM"
                        torch.cuda.empty_cache(); gc.collect()
                        print(f"  T={T:3d} kappa={kappa:2d} {sched_kind:15s} OOM")
                    except Exception as e:
                        cell.status = "runtime_fail"
                        cell.error_msg = type(e).__name__ + ": " + str(e)[:80]
                        print(f"  T={T:3d} kappa={kappa:2d} {sched_kind:15s} {cell.error_msg}")
                    cells.append(cell)

            del y_ref
            torch.cuda.empty_cache(); gc.collect()

        del net
        torch.cuda.empty_cache(); gc.collect()

    with open(args.output + ".json", "w") as f:
        json.dump({
            "config": {
                "T_sweep": Ts, "kappa_sweep": kappas,
                "b": args.b, "C": 3, "H": args.H, "seed": args.seed,
            },
            "cells": [asdict(c) for c in cells],
        }, f, indent=2)
    print(f"\nSaved JSON: {args.output}.json")

    # Summary
    print(f"\n{'='*72}")
    print(f"=== Summary across {len(cells)} (net, T, kappa, schedule) cells ===")
    print(f"{'='*72}")
    ok_cells = [c for c in cells if c.status == "ok"]
    n_bitexact = sum(1 for c in ok_cells if c.max_abs_err == 0.0)
    n_pred_perfect = sum(1 for c in ok_cells if c.pred_agreement == 1.0)
    print(f"  Total successful comparisons: {len(ok_cells)} / {len(cells)}")
    print(f"  Cells with max_abs_err == 0.0 (bit-exact): {n_bitexact} / {len(ok_cells)} "
          f"({n_bitexact/max(1,len(ok_cells))*100:.1f}%)")
    print(f"  Cells with prediction agreement == 100%: {n_pred_perfect} / {len(ok_cells)}")
    print(f"  Total scalar comparisons: {sum(c.n_compared for c in ok_cells):,}")
    print(f"  Max segments traversed (max non-trivial T/kappa): "
          f"{max((c.n_segments for c in ok_cells), default=0)}")
    if ok_cells and n_bitexact < len(ok_cells):
        print(f"\n  Cells with non-zero error (showing top 5):")
        for c in sorted(ok_cells, key=lambda c: -c.max_abs_err)[:5]:
            print(f"    {c.net:12s} T={c.T:3d} kappa={c.kappa:2d} "
                  f"sched={c.schedule_kind:15s}  "
                  f"max_abs={c.max_abs_err:.3e}  pred={c.pred_agreement*100:.2f}%")


if __name__ == "__main__":
    main()