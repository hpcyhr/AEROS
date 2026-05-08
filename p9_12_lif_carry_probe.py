#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — LIF carry-state probe.

Diagnoses where the 6.25e-03 max_abs_err between M1 (kappa=T) and M2 (kappa<T)
on trained DVS128 Gesture enters. Hooks every LIFNode in DVSGestureNet,
captures the spike output [T, B, C, H, W] per layer per timestep for both
M1 and M2, and reports per-(layer, timestep) max_abs_err.

Hypothesis: LIF membrane potential v is implicitly reset at the entry of each
segment call when step_mode='m', so segment 2 starts from v=0 instead of from
the v at end of segment 1. Test: per-LIF, per-t spike diff. Expected pattern
if hypothesis holds:

  segment 1 (t=0..3): all LIFs identical (max_err=0)
  segment 2 (t=4..7): first LIF diverges at t=4, divergence propagates downstream
  segment 3, 4: divergence accumulates

If hypothesis is wrong, divergence will appear inside segment 1 already, or be
randomly scattered across timesteps.

Usage:
    python p9_12_lif_carry_probe.py \\
        --ckpt /data/yhr/AEROS/checkpoints_dvs/dvs128_gesture_best.pth \\
        --data_root /data/yhr/AEROS/DVS128Gesture \\
        --T 16 --kappa 4
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def setup_determinism():
    """Same flags as p9_12_dvs_inference_det.py."""
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


class LIFSpikeCapture:
    """Hooks every LIFNode and captures its multi-step output [T, B, C, H, W]."""

    def __init__(self, net):
        from spikingjelly.activation_based.neuron import LIFNode
        self.handles = []
        self.records: Dict[str, List[torch.Tensor]] = {}
        self.lif_names: List[str] = []
        for name, mod in net.named_modules():
            if isinstance(mod, LIFNode):
                self.lif_names.append(name)
                self.records[name] = []
                handle = mod.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(mod, inp, out):
            # out shape under step_mode='m' is [T, B, C, H, W] for conv-LIFs
            self.records[name].append(out.detach().cpu())
        return hook

    def clear(self):
        for k in self.records:
            self.records[k] = []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_mode_1(net, x, capture: LIFSpikeCapture):
    """Single 16-step call. capture.records[lif_name] = [tensor of shape (16,B,C,H,W)]."""
    capture.clear()
    reset_state(net)
    _ = net(x)  # x: [T, B, 2, 128, 128]
    # Each LIF was called exactly once with all 16 steps
    return {name: torch.cat(parts, dim=0) for name, parts in capture.records.items()}


@torch.no_grad()
def run_mode_2(net, x, kappa, capture: LIFSpikeCapture):
    """Multiple kappa-step calls. capture should accumulate across segments."""
    capture.clear()
    reset_state(net)
    T = x.shape[0]
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        _ = net(x[i:i+sz])
        i += sz
    # Each LIF was called T/kappa times. Concat along dim=0 gives full [T,...].
    return {name: torch.cat(parts, dim=0) for name, parts in capture.records.items()}


def compare(m1_spikes, m2_spikes, lif_names, T, kappa):
    """Print per-(layer, t) max_abs_err table."""
    print(f"\n{'='*78}")
    print(f"=== Per-LIF per-timestep max_abs_err: M1 (kappa=T={T}) vs M2 (kappa={kappa}) ===")
    print(f"{'='*78}")

    # Header: timesteps
    n_seg = (T + kappa - 1) // kappa
    print(f"  segments: {n_seg}, kappa={kappa}, T={T}")
    print(f"  segment boundaries at t = {[i for i in range(kappa, T, kappa)]}")
    print()

    hdr = f"{'LIF layer':<35s} | " + " ".join(f"t={t:02d}" for t in range(T))
    print(hdr)
    print("-" * len(hdr))

    first_div_layer = None
    first_div_t = None

    for name in lif_names:
        s1 = m1_spikes[name]  # [T, B, C, H, W]
        s2 = m2_spikes[name]
        if s1.shape != s2.shape:
            print(f"  {name:<35s}  shape mismatch: M1={s1.shape} M2={s2.shape}")
            continue
        # Per-t max_abs_err
        diffs = (s1 - s2).abs()
        per_t = diffs.flatten(1).max(dim=1).values  # [T]
        cells = []
        for t in range(T):
            v = per_t[t].item()
            if v == 0:
                cells.append(" 0.0")
            else:
                # Format as "X.Xe-N" compact
                exp = int(f"{v:e}".split("e")[1])
                mantissa = v / (10 ** exp)
                cells.append(f"{mantissa:.1f}e{exp:+d}")
                if first_div_layer is None:
                    first_div_layer = name
                    first_div_t = t
        print(f"  {name:<35s} | " + " ".join(f"{c:>6s}" for c in cells))

    print()
    if first_div_layer is None:
        print("  [VERDICT] No divergence found. Carry is bit-exact.")
    else:
        print(f"  [VERDICT] First divergence at:")
        print(f"    layer = {first_div_layer}")
        print(f"    timestep t = {first_div_t}")
        print(f"    boundary index (t // kappa) = {first_div_t // kappa}")
        if first_div_t == 0:
            print(f"    --> Divergence at t=0: bug is INSIDE forward, not at boundary.")
        elif first_div_t < kappa:
            print(f"    --> Divergence inside segment 1 (before any boundary): "
                  f"forward path differs even within first call.")
        elif first_div_t == kappa:
            print(f"    --> Divergence appears EXACTLY at first segment boundary "
                  f"(t={kappa}): hypothesis confirmed -- LIF carry across calls failed.")
        else:
            print(f"    --> Divergence appears mid-segment {first_div_t // kappa + 1}: "
                  f"unexpected pattern, may be downstream propagation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--kappa", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)  # smaller for probe
    parser.add_argument("--channels", type=int, default=128)
    args = parser.parse_args()

    setup_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.parametric_lif_net import (
        DVSGestureNet)
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    print(f"=== AEROS Phase 2 Exp 12 — LIF carry probe ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  T={args.T}  kappa={args.kappa}  b={args.batch_size}")

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

    capture = LIFSpikeCapture(net)
    print(f"  Hooked {len(capture.lif_names)} LIFNode(s):")
    for n in capture.lif_names:
        print(f"    - {n}")

    # First batch only -- enough for diagnosis
    x_batch = None
    for x, y in loader:
        x_batch = x.to(device).float().transpose(0, 1)  # [T, B, 2, 128, 128]
        break
    print(f"  Probe batch shape: {tuple(x_batch.shape)}")

    print(f"\n  Running Mode 1 (single {args.T}-step call) ...")
    m1_spikes = run_mode_1(net, x_batch, capture)

    print(f"  Running Mode 2 (kappa={args.kappa}, "
          f"{args.T // args.kappa} calls) ...")
    m2_spikes = run_mode_2(net, x_batch, args.kappa, capture)

    compare(m1_spikes, m2_spikes, capture.lif_names, args.T, args.kappa)
    capture.remove()


if __name__ == "__main__":
    main()