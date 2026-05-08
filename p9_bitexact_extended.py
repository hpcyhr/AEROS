#!/usr/bin/env python
"""
AEROS §5.1 extension — bit-exact wrapper invariance + boundary stress for
6 trained extended-suite archs.

Mirrors the 17-arch SNN protocol (already in §5.1):
  Stage 1 — wrapper invariance: load trained ckpt, T=4 with kappa>=4 (single
            segment), compare to baseline on the full test set the arch was
            trained on. Should be max_abs_err = 0 across all comparisons.
  Stage 2 — boundary stress: random-input synthetic horizons T={16, 32, 64}
            with kappa={4, 8} so every forward traverses multiple segment
            boundaries. Verifies carry-state propagation is bit-exact.
            Uses 2 schedules: uniform (kappa, kappa, ...) and singleton-tail
            (kappa, kappa, ..., 1, 1) exposing smallest final segments.

Output:
  /data/yhr/AEROS/p9_bitexact_extended.json

Usage:
  python p9_bitexact_extended.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from aeros_models_extended import (build_extended_net, reset_state_extended,
                                    EXTENDED_NETS)
from train_extended import (get_smnist_loaders, get_moving_mnist_loaders,
                             CHECKPOINT_DIR)


def task_for(arch_name):
    """Same dispatch as train_extended.py."""
    if arch_name in ("ConvLSTM-2L", "ConvGRU-2L"):
        return "moving_mnist"
    return "smnist"


def get_test_loader(arch_name, batch_size=64):
    if task_for(arch_name) == "moving_mnist":
        _, test_loader = get_moving_mnist_loaders(batch_size=batch_size,
                                                    T=16, H=32)
    else:
        _, test_loader = get_smnist_loaders(batch_size=batch_size)
    return test_loader


def forward_segmented_uniform(net, x, kappa, mode_id, device,
                                provenance="extended"):
    """Mode-aware segmented forward, uniform schedule. Mirrors p9_6 protocol."""
    T = x.shape[0]
    if mode_id == 1 or kappa >= T:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state_extended(net)
        return net(x)
    if mode_id == 2:
        if x.device != device:
            x = x.to(device, non_blocking=True)
        reset_state_extended(net)
        chunks = []
        i = 0
        while i < T:
            sz = min(kappa, T - i)
            chunks.append(net(x[i:i+sz].contiguous()))
            i += sz
        return torch.cat(chunks, dim=0)
    if mode_id == 3:  # input-stream, output-retain
        assert x.device.type == "cpu"
        reset_state_extended(net)
        chunks = []
        i = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            chunks.append(net(x_seg))
            i += sz
        return torch.cat(chunks, dim=0)
    if mode_id == 4:  # input-stream, output-sink
        assert x.device.type == "cpu"
        reset_state_extended(net)
        sink, n = None, 0
        i = 0
        while i < T:
            sz = min(kappa, T - i)
            x_seg = x[i:i+sz].contiguous().to(device, non_blocking=False)
            y_seg = net(x_seg)
            if sink is None:
                sink = y_seg.sum(dim=0)
                n = sz
            else:
                sink = sink + y_seg.sum(dim=0)
                n += sz
            i += sz
        return sink / n
    raise ValueError(f"unknown mode {mode_id}")


def forward_singleton_tail(net, x, kappa, device):
    """Schedule: [kappa, kappa, ..., 1, 1] -- last 2 segments are singletons."""
    T = x.shape[0]
    if x.device != device:
        x = x.to(device, non_blocking=True)
    reset_state_extended(net)
    if T <= 2:
        return net(x)
    chunks = []
    # Determine number of full-kappa segments such that final 2 are singletons
    # Schedule: kappa-segments fill [0, T-2), then [T-2, T-1], [T-1, T]
    body_T = T - 2
    n_full = body_T // kappa
    rem = body_T - n_full * kappa
    i = 0
    for _ in range(n_full):
        chunks.append(net(x[i:i+kappa].contiguous()))
        i += kappa
    if rem > 0:
        chunks.append(net(x[i:i+rem].contiguous()))
        i += rem
    # Two singleton segments
    chunks.append(net(x[i:i+1].contiguous()))
    chunks.append(net(x[i+1:i+2].contiguous()))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def stage1_wrapper_invariance(arch_name, ckpt_path, device, batch_size=64):
    """For trained arch, compare 3 mode wrappers to baseline on full test set
    at T_train (16 for Moving MNIST, 28 for sMNIST). All kappa>=T should
    produce single segment, so this isolates wrapper invariance only."""
    print(f"\n[Stage 1] {arch_name}: wrapper invariance")
    print(f"  ckpt: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    net = build_extended_net(arch_name, num_classes=10, H=32, in_ch=3).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"  loaded checkpoint, original test_acc: {ckpt['test_acc']*100:.2f}%")

    test_loader = get_test_loader(arch_name, batch_size=batch_size)

    # T from data
    sample_x, _ = next(iter(test_loader))
    T = sample_x.shape[0]
    # For wrapper invariance, use kappa = T (forces single segment per mode)
    kappa = T
    print(f"  T={T}, kappa={kappa} (single segment per call)")

    n_total, n_correct_baseline = 0, 0
    max_err_per_mode = {2: 0.0, 3: 0.0, 4: 0.0}
    pred_agree_per_mode = {2: 0, 3: 0, 4: 0}
    n_compared = {2: 0, 3: 0, 4: 0}

    for xb, yb in test_loader:
        xb_gpu = xb.to(device, non_blocking=True)
        yb_gpu = yb.to(device, non_blocking=True)

        # Baseline (Mode 1, kappa=T)
        reset_state_extended(net)
        y_base = net(xb_gpu)            # [T, B, num_classes]
        pred_base = y_base[-1].argmax(dim=-1)
        n_correct_baseline += (pred_base == yb_gpu).sum().item()
        n_total += yb.size(0)

        # Mode 2: retain-IO segmented (with kappa=T -> equivalent to baseline)
        y2 = forward_segmented_uniform(net, xb_gpu, kappa, 2, device)
        err2 = (y2 - y_base).abs().max().item()
        max_err_per_mode[2] = max(max_err_per_mode[2], err2)
        pred_agree_per_mode[2] += (y2[-1].argmax(dim=-1) == pred_base).sum().item()
        n_compared[2] += yb.size(0)

        # Mode 3: input-stream
        xb_cpu = xb.cpu()
        y3 = forward_segmented_uniform(net, xb_cpu, kappa, 3, device)
        err3 = (y3 - y_base).abs().max().item()
        max_err_per_mode[3] = max(max_err_per_mode[3], err3)
        pred_agree_per_mode[3] += (y3[-1].argmax(dim=-1) == pred_base).sum().item()
        n_compared[3] += yb.size(0)

        # Mode 4: input-stream + output-sink (compare scalar logit per sample)
        # Mode 4 collapses time -> we compare the sink logits to baseline.mean(dim=0)
        y4 = forward_segmented_uniform(net, xb_cpu, kappa, 4, device)
        y_base_sink = y_base.mean(dim=0)
        err4 = (y4 - y_base_sink).abs().max().item()
        max_err_per_mode[4] = max(max_err_per_mode[4], err4)
        pred_agree_per_mode[4] += (y4.argmax(dim=-1) == y_base_sink.argmax(dim=-1)).sum().item()
        n_compared[4] += yb.size(0)

    baseline_acc = n_correct_baseline / n_total
    print(f"  baseline test_acc: {baseline_acc*100:.2f}%, n_total={n_total}")
    print(f"  Mode 2: max_err={max_err_per_mode[2]:.2e}, "
          f"pred_agree={pred_agree_per_mode[2]}/{n_compared[2]}")
    print(f"  Mode 3: max_err={max_err_per_mode[3]:.2e}, "
          f"pred_agree={pred_agree_per_mode[3]}/{n_compared[3]}")
    print(f"  Mode 4: max_err={max_err_per_mode[4]:.2e}, "
          f"pred_agree={pred_agree_per_mode[4]}/{n_compared[4]}")

    return {
        "arch": arch_name,
        "task": task_for(arch_name),
        "T": T, "kappa": kappa,
        "baseline_test_acc": baseline_acc,
        "n_total": n_total,
        "max_err_mode2": max_err_per_mode[2],
        "max_err_mode3": max_err_per_mode[3],
        "max_err_mode4": max_err_per_mode[4],
        "pred_agree_mode2": pred_agree_per_mode[2],
        "pred_agree_mode3": pred_agree_per_mode[3],
        "pred_agree_mode4": pred_agree_per_mode[4],
        "n_compared_mode2": n_compared[2],
        "n_compared_mode3": n_compared[3],
        "n_compared_mode4": n_compared[4],
    }


@torch.no_grad()
def stage2_boundary_stress(arch_name, device, T_list=None, kappa_list=None,
                            b=32, C=3, H=32, n_seeds=3):
    """Random-init network, synthetic random input, T={16,32,64} kappa={4,8}.
    Each forward traverses multiple segment boundaries; verify carry-state
    propagation is bit-exact across both uniform and singleton-tail schedules.
    """
    if T_list is None:
        T_list = [16, 32, 64]
    if kappa_list is None:
        kappa_list = [4, 8]

    print(f"\n[Stage 2] {arch_name}: boundary stress (random init)")

    cells = []
    for T in T_list:
        for kappa in kappa_list:
            if kappa >= T:
                continue
            cells.append((T, kappa))

    out_cells = []
    for T, kappa in cells:
        for sched in ("uniform", "singleton_tail"):
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                net = build_extended_net(arch_name, num_classes=10,
                                          H=H, in_ch=C).to(device)
                net.eval()
                # Same seed for input
                x = torch.randn(T, b, C, H, H, device=device)

                # Baseline (full forward)
                reset_state_extended(net)
                y_base = net(x)

                # Segmented forward
                if sched == "uniform":
                    y_seg = forward_segmented_uniform(net, x, kappa, 2, device)
                else:
                    y_seg = forward_singleton_tail(net, x, kappa, device)

                err = (y_seg - y_base).abs().max().item()
                pred_match = (y_seg[-1].argmax(dim=-1) ==
                              y_base[-1].argmax(dim=-1)).sum().item()

                # Number of boundary crossings
                if sched == "uniform":
                    n_boundaries = (T + kappa - 1) // kappa - 1
                else:
                    body = T - 2
                    n_full = body // kappa
                    rem = body - n_full * kappa
                    # +1 if rem > 0 plus the 2 singleton segments
                    n_segments = n_full + (1 if rem > 0 else 0) + 2
                    n_boundaries = n_segments - 1

                cell = {
                    "arch": arch_name, "T": T, "kappa": kappa,
                    "schedule": sched, "seed": seed,
                    "n_boundaries": n_boundaries,
                    "max_abs_err": err,
                    "pred_match": pred_match, "b": b,
                    "n_compared_scalars": int(np.prod(y_base.shape)),
                }
                out_cells.append(cell)

                print(f"  T={T:>2} kappa={kappa} sched={sched:<14s} "
                      f"seed={seed}  max_err={err:.2e}  "
                      f"pred_match={pred_match}/{b}  "
                      f"boundaries={n_boundaries}")

                del net, x, y_base, y_seg
                torch.cuda.empty_cache()
                gc.collect()

    return out_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", type=str, nargs="+", default=None,
                        help="archs to test (default: all 6 extended)")
    parser.add_argument("--ckpt_dir", type=str,
                        default=str(CHECKPOINT_DIR))
    parser.add_argument("--output", type=str,
                        default="p9_bitexact_extended")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="skip wrapper invariance (e.g. if no ckpt yet)")
    parser.add_argument("--skip_stage2", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archs = args.archs or list(EXTENDED_NETS)

    # Deterministic everywhere
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("=" * 78)
    print("AEROS §5.1 ext — bit-exact for 6 trained extended-suite archs")
    print("=" * 78)
    print(f"  archs: {archs}")
    print(f"  ckpt dir: {args.ckpt_dir}")
    print(f"  device: {device}")

    all_stage1, all_stage2 = [], []
    ckpt_dir = Path(args.ckpt_dir)

    for arch in archs:
        ckpt = ckpt_dir / f"{arch}_best.pth"
        if not args.skip_stage1:
            if not ckpt.exists():
                print(f"\n[SKIP] {arch}: no ckpt at {ckpt}")
            else:
                try:
                    s1 = stage1_wrapper_invariance(arch, ckpt, device)
                    all_stage1.append(s1)
                except Exception as e:
                    print(f"[ERR Stage 1 {arch}] {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
        if not args.skip_stage2:
            try:
                s2 = stage2_boundary_stress(arch, device)
                all_stage2.extend(s2)
            except Exception as e:
                print(f"[ERR Stage 2 {arch}] {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

    # Aggregate
    print("\n" + "=" * 78)
    print("Aggregate")
    print("=" * 78)

    if all_stage1:
        max_err_2 = max(s["max_err_mode2"] for s in all_stage1)
        max_err_3 = max(s["max_err_mode3"] for s in all_stage1)
        max_err_4 = max(s["max_err_mode4"] for s in all_stage1)
        n_total = sum(s["n_total"] for s in all_stage1)
        n_compared_total = sum(s["n_compared_mode2"] +
                                s["n_compared_mode3"] +
                                s["n_compared_mode4"] for s in all_stage1)
        print("\nStage 1 (wrapper invariance):")
        print(f"  archs tested: {len(all_stage1)}")
        print(f"  total samples: {n_total}")
        print(f"  total comparisons: {n_compared_total}")
        print(f"  max_err  M2={max_err_2:.2e}  M3={max_err_3:.2e}  "
              f"M4={max_err_4:.2e}")

    if all_stage2:
        max_err_s2 = max(c["max_abs_err"] for c in all_stage2)
        n_cells = len(all_stage2)
        n_scalars = sum(c["n_compared_scalars"] for c in all_stage2)
        cells_zero_err = sum(1 for c in all_stage2 if c["max_abs_err"] == 0.0)
        print("\nStage 2 (boundary stress, random init):")
        print(f"  cells: {n_cells}")
        print(f"  scalars compared: {n_scalars:,}")
        print(f"  cells with max_err=0: {cells_zero_err}/{n_cells}")
        print(f"  max_err: {max_err_s2:.2e}")

    output = {
        "stage1_wrapper_invariance": all_stage1,
        "stage2_boundary_stress": all_stage2,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()