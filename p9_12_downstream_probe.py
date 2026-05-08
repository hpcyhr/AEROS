#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — downstream probe (post-LIF localization).

LIF carry probe at b=4 and b=16 both showed all 7 LIF outputs bit-exact
between M1 (kappa=T=16) and M2 (kappa=4). The 6.25e-03 max_err observed in
inference at b=16, kappa=4 must therefore be downstream of the last LIF
(conv_fc.26).

DVSGestureNet ending (SJ standard):
  conv_fc[20]: Flatten
  conv_fc[21]: Dropout
  conv_fc[22]: Linear
  conv_fc[23]: LIF       <-- last hooked in carry probe but second-to-last LIF
  conv_fc[24]: Dropout
  conv_fc[25]: Linear
  conv_fc[26]: LIF       <-- final LIF
  conv_fc[27]: VotingLayer

This probe hooks:
  - every Flatten / Linear / Dropout / VotingLayer
  - the net's final output [T, B, 11]
and reports M1 vs M2 max_abs_err per layer per timestep.

Usage:
    python p9_12_downstream_probe.py \\
        --ckpt /data/yhr/AEROS/checkpoints_dvs/dvs128_gesture_best.pth \\
        --data_root /data/yhr/AEROS/DVS128Gesture \\
        --T 16 --kappa 4 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def setup_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def reset_state(net):
    from spikingjelly.activation_based import functional
    try:
        functional.reset_net(net)
    except Exception:
        pass


class AllLayerCapture:
    """Hook every named module's forward output (filtering trivial wrappers)."""

    SKIP_TYPES = (nn.Sequential, nn.ModuleList)

    def __init__(self, net, only_after: str = None):
        self.handles = []
        self.records: Dict[str, List[torch.Tensor]] = {}
        self.layer_names: List[str] = []
        seen_after = (only_after is None)
        for name, mod in net.named_modules():
            if isinstance(mod, self.SKIP_TYPES) or name == "":
                continue
            # If only_after specified, skip until we have passed that layer
            if not seen_after:
                if name == only_after:
                    seen_after = True
                continue
            self.layer_names.append(name)
            self.records[name] = []
            handle = mod.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self.records[name].append(out.detach().cpu())
        return hook

    def clear(self):
        for k in self.records:
            self.records[k] = []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_mode_1(net, x, capture):
    capture.clear()
    reset_state(net)
    out = net(x)
    return ({name: torch.cat(parts, dim=0) if len(parts) > 0 else None
             for name, parts in capture.records.items()},
            out.detach().cpu())


@torch.no_grad()
def run_mode_2(net, x, kappa, capture):
    capture.clear()
    reset_state(net)
    T = x.shape[0]
    i = 0
    final_chunks = []
    while i < T:
        sz = min(kappa, T - i)
        out_seg = net(x[i:i+sz])
        final_chunks.append(out_seg.detach().cpu())
        i += sz
    return ({name: torch.cat(parts, dim=0) if len(parts) > 0 else None
             for name, parts in capture.records.items()},
            torch.cat(final_chunks, dim=0))


def fmt_err(v):
    if v == 0.0:
        return "      0"
    exp = int(f"{v:e}".split("e")[1])
    mantissa = v / (10 ** exp)
    return f"{mantissa:.2f}e{exp:+d}"


def compare(m1_layers, m2_layers, m1_out, m2_out, layer_names, T, kappa):
    print(f"\n{'='*100}")
    print(f"=== Per-layer per-timestep max_abs_err: M1 (kappa=T={T}) vs M2 (kappa={kappa}) ===")
    print(f"{'='*100}")
    print(f"  segments: {(T+kappa-1)//kappa}, kappa={kappa}, T={T}")
    print(f"  boundaries at t = {[i for i in range(kappa, T, kappa)]}")
    print()

    first_div_layer = None
    first_div_t = None

    for name in layer_names:
        s1 = m1_layers.get(name)
        s2 = m2_layers.get(name)
        if s1 is None or s2 is None:
            continue
        if s1.shape != s2.shape:
            print(f"  {name:<35s}  SHAPE MISMATCH: M1={tuple(s1.shape)} M2={tuple(s2.shape)}")
            continue
        # If first dim is T, do per-t. Otherwise just one value.
        if s1.dim() >= 1 and s1.shape[0] == T:
            diffs = (s1 - s2).abs()
            per_t = diffs.flatten(1).max(dim=1).values
            cells = [fmt_err(per_t[t].item()) for t in range(T)]
            row = " ".join(f"{c:>9s}" for c in cells)
            row_max = per_t.max().item()
            print(f"  {name:<35s} {tuple(s1.shape)!s:<25s} max={fmt_err(row_max):<10s} | {row}")
            for t in range(T):
                if per_t[t].item() > 0 and first_div_layer is None:
                    first_div_layer = name
                    first_div_t = t
        else:
            diff = (s1 - s2).abs().max().item()
            print(f"  {name:<35s} {tuple(s1.shape)!s:<25s} max={fmt_err(diff)}")

    print()
    print(f"=== Final net output (returned by net(x)) ===")
    if m1_out.shape == m2_out.shape:
        out_diff = (m1_out - m2_out).abs()
        out_max = out_diff.max().item()
        print(f"  shape: {tuple(m1_out.shape)}")
        print(f"  max_abs_err: {fmt_err(out_max)}  ({out_max:.6e})")
        if m1_out.dim() >= 1 and m1_out.shape[0] == T:
            per_t = out_diff.flatten(1).max(dim=1).values
            print(f"  per-t: " + " ".join(fmt_err(per_t[t].item()) for t in range(T)))
    else:
        print(f"  SHAPE MISMATCH: M1={tuple(m1_out.shape)} M2={tuple(m2_out.shape)}")

    print()
    if first_div_layer is None:
        print(f"  [VERDICT] All hooked intermediate outputs bit-exact.")
        print(f"            Divergence (if any) appears only in net's final output.")
    else:
        print(f"  [VERDICT] First non-LIF divergence at:")
        print(f"    layer = {first_div_layer}")
        print(f"    timestep t = {first_div_t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--kappa", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--only_after", default=None,
                        help="Only hook layers after this named module "
                             "(e.g. 'conv_fc.18' to focus on tail).")
    args = parser.parse_args()

    setup_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.parametric_lif_net import (
        DVSGestureNet)
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    print(f"=== AEROS Phase 2 Exp 12 — Downstream probe ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  T={args.T}  kappa={args.kappa}  b={args.batch_size}")
    print(f"  only_after: {args.only_after}")

    test_set = DVS128Gesture(
        root=args.data_root, train=False, data_type="frame",
        frames_number=args.T, split_by="number")
    loader = DataLoader(test_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)

    net = DVSGestureNet(
        channels=args.channels,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    ).to(device)
    functional.set_step_mode(net, "m")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval()

    # Print the net structure tail to understand layer indices
    print(f"\n  === Tail of conv_fc Sequential ===")
    if hasattr(net, "conv_fc"):
        for i, mod in enumerate(net.conv_fc):
            cls = type(mod).__name__
            print(f"    [{i:2d}] {cls}")

    capture = AllLayerCapture(net, only_after=args.only_after)
    print(f"\n  Hooked {len(capture.layer_names)} layer(s):")
    for n in capture.layer_names:
        print(f"    - {n}")

    x_batch = None
    for x, y in loader:
        x_batch = x.to(device).float().transpose(0, 1)
        break
    print(f"\n  Probe batch shape: {tuple(x_batch.shape)}")

    print(f"\n  Running Mode 1 (single {args.T}-step call) ...")
    m1_layers, m1_out = run_mode_1(net, x_batch, capture)

    print(f"  Running Mode 2 (kappa={args.kappa}) ...")
    m2_layers, m2_out = run_mode_2(net, x_batch, args.kappa, capture)

    compare(m1_layers, m2_layers, m1_out, m2_out, capture.layer_names,
            args.T, args.kappa)
    capture.remove()


if __name__ == "__main__":
    main()