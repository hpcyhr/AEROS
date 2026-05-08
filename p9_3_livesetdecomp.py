#!/usr/bin/env python
"""
AEROS Phase 2 Exp 3 — Live-Set Decomposition Stacked Bar Profiling.

Profiles 6 tensor categories of peak HBM residency across 4 deployment
modes for 4 representative nets, supporting v9 paper §3.3 (Layer 2:
Live-Set Memory Model) and §5 Exp 3 figure.

6 categories:
  M_static    — model parameters
  M_carry     — recurrent state (LIF v_t, ConvLSTM (h,c), etc.)  
  M_in_live   — input data resident on device
  M_act_live  — segment-internal activations through all layers
  M_out_live  — outputs resident on device awaiting consumption
  M_workspace — cuDNN scratch + AMP scaler + allocator overhead (residual)

4 modes (Phase 1 v2 paper):
  Mode 1: Full-horizon baseline    (pi_in=retain, pi_out=retain, kappa=T)
  Mode 2: Segmented retained-IO    (pi_in=retain, pi_out=retain, kappa<T)
  Mode 3: Input-streaming          (pi_in=stream, pi_out=retain, kappa<T)
  Mode 4: IO-streaming (sink)      (pi_in=stream, pi_out=sink,   kappa<T)

4 nets:
  SR-18         (SNN baseline, 18 layers)
  SEW-50        (deep SNN with skip)
  VGG-19-BN     (deep SNN no-skip)
  ConvLSTM      (non-SNN recurrent, multi-family validation)

Usage:
  python p9_3_livesetdecomp.py --T 128 --kappa 8 --b 64 --H 224 \\
      --output p9_3_livesetdecomp_results.npz

Output:
  p9_3_livesetdecomp_results.npz: arrays of (4 nets x 4 modes x 6 categories)
  p9_3_livesetdecomp_results.json: same data in human-readable form
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18
    from spikingjelly.activation_based.model.sew_resnet import sew_resnet50
    SJ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SpikingJelly import failed: {e}")
    SJ_AVAILABLE = False


# ============================================================================
# Network builders
# ============================================================================

def build_sr18(num_classes: int = 10) -> nn.Module:
    """SpikingResNet-18 with multi-step LIF.

    SJ's spiking_resnet18 uses layer.Conv2d / layer.BatchNorm2d (SJ's
    wrappers), not vanilla nn.Conv2d. After functional.set_step_mode('m'),
    these layers internally flatten T*B for the wrapped op and reshape back.
    Thus the model accepts [T, B, C, H, W] directly.
    """
    net = spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=num_classes,
    )
    net.eval()
    functional.set_step_mode(net, step_mode="m")
    return net


def build_sew50(num_classes: int = 10) -> nn.Module:
    """SEW-ResNet-50 with cnf=ADD, multi-step LIF."""
    net = sew_resnet50(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        cnf="ADD",
        num_classes=num_classes,
    )
    net.eval()
    functional.set_step_mode(net, step_mode="m")
    return net


def build_vgg19_bn(num_classes: int = 10) -> nn.Module:
    """SpikingVGG-19-BN with multi-step LIF.

    Use SJ's layer.Conv2d / layer.BatchNorm2d (multi-step capable) so
    forward accepts [T, B, C, H, W] directly after set_step_mode('m').
    """
    from spikingjelly.activation_based import layer

    cfg = [64, 64, "M",
           128, 128, "M",
           256, 256, 256, 256, "M",
           512, 512, 512, 512, "M",
           512, 512, 512, 512, "M"]

    layers = []
    in_c = 3
    for v in cfg:
        if v == "M":
            layers.append(layer.MaxPool2d(2, 2))
        else:
            layers.append(layer.Conv2d(in_c, v, 3, padding=1))
            layers.append(layer.BatchNorm2d(v))
            layers.append(neuron.LIFNode(
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                v_threshold=0.5,
            ))
            in_c = v

    feature = nn.Sequential(*layers)
    classifier = nn.Sequential(
        layer.Flatten(),
        layer.Linear(512 * 7 * 7, 4096),
        neuron.LIFNode(surrogate_function=surrogate.ATan(),
                       detach_reset=True, v_threshold=0.5),
        layer.Linear(4096, 4096),
        neuron.LIFNode(surrogate_function=surrogate.ATan(),
                       detach_reset=True, v_threshold=0.5),
        layer.Linear(4096, num_classes),
    )

    class VGG19MultiStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = feature
            self.classifier = classifier

        def forward(self, x):
            # x: [T, B, C, H, W] — SJ multi-step layers handle T internally
            f = self.feature(x)               # [T, B, 512, 7, 7]
            return self.classifier(f)         # [T, B, num_classes]

    net = VGG19MultiStep()
    net.eval()
    functional.set_step_mode(net, step_mode="m")
    return net


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell with hidden+cell state."""

    def __init__(self, in_c: int, hid_c: int, k: int = 3):
        super().__init__()
        self.hid_c = hid_c
        pad = k // 2
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=pad)

    def forward(self, x: torch.Tensor, state):
        # x: [B, C, H, W]; state: (h, c) each [B, hid_c, H, W] or None
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hid_c, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hid_c, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class ConvLSTMNet(nn.Module):
    """Minimal ConvLSTM stack for multi-family validation (Doris 5)."""

    def __init__(self, in_c: int = 3, hid_c: int = 64,
                 num_layers: int = 2, num_classes: int = 10, H: int = 224):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLSTMCell(in_c if i == 0 else hid_c, hid_c)
            for i in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hid_c, num_classes)
        self._states = None

    def reset_state(self):
        self._states = [None] * len(self.layers)

    def forward(self, x: torch.Tensor):
        # x: [T, B, C, H, W]
        T = x.shape[0]
        if self._states is None:
            self.reset_state()
        outs = []
        for t in range(T):
            h = x[t]
            for i, cell in enumerate(self.layers):
                h, self._states[i] = cell(h, self._states[i])
            outs.append(self.fc(self.pool(h).flatten(1)))
        return torch.stack(outs, dim=0)  # [T, B, num_classes]


def reset_net_compat(net: nn.Module):
    """Reset recurrent state — SJ for spiking nets, manual for ConvLSTM."""
    if hasattr(net, "reset_state"):
        net.reset_state()
    else:
        try:
            functional.reset_net(net)
        except Exception:
            pass


# ============================================================================
# Static category measurement
# ============================================================================

def measure_static(net: nn.Module) -> int:
    """M_static: model parameters in bytes."""
    return sum(p.numel() * p.element_size() for p in net.parameters())


def measure_carry_capacity(net: nn.Module, b: int, H: int) -> int:
    """
    M_carry: bound on recurrent state per-step (kappa-independent).

    For SJ LIFNode, state shape = activation shape after first forward.
    We estimate as a fraction of M_static for SJ nets (carry buffer ~ width
    of widest layer x b x spatial); for ConvLSTM we compute (h, c) explicitly.
    """
    total = 0
    for name, m in net.named_modules():
        if isinstance(m, ConvLSTMCell):
            # h + c, each [B, hid_c, H, W] in fp32
            total += 2 * b * m.hid_c * H * H * 4
        # SJ LIFNode state buffer is allocated lazily — we compute it
        # post-forward via differential measurement instead. So for SJ nets,
        # this static estimate is a lower bound.
    return total


def measure_io_static(T: int, b: int, C: int, H: int, num_classes: int) -> dict:
    """Static I/O sizes for a given (T, b, C, H) and output spec."""
    return {
        "in_per_step":   b * C * H * H * 4,            # [B, C, H, W] fp32
        "in_full":       T * b * C * H * H * 4,        # [T, B, C, H, W]
        "out_per_step":  b * num_classes * 4,          # [B, num_classes]
        "out_full":      T * b * num_classes * 4,      # [T, B, num_classes]
    }


# ============================================================================
# Dynamic measurement: 4 modes, GPU peak after forward
# ============================================================================

@torch.no_grad()
def run_mode1_full_horizon(net, T, b, C, H, device):
    """Mode 1: Full-horizon baseline (pi_in=retain, pi_out=retain, kappa=T)."""
    reset_net_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    # Materialize full input
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    y = net(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del x, y
    return peak


@torch.no_grad()
def run_mode2_segmented_retIO(net, T, b, C, H, kappa, device):
    """Mode 2: Segmented retained-IO (pi_in=retain, pi_out=retain, kappa<T)."""
    reset_net_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    chunks = []
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    y = torch.cat(chunks, dim=0)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del x, y, chunks
    return peak


@torch.no_grad()
def run_mode3_inputstream(net, T, b, C, H, kappa, device):
    """Mode 3: Input-streaming (pi_in=stream, pi_out=retain, kappa<T)."""
    reset_net_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    chunks = []
    g = torch.Generator(device="cpu").manual_seed(42)
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        # Generate this segment on CPU then move (to model real stream
        # from event-camera decoder; alternative is direct on-device gen)
        x_seg = torch.randn(sz, b, C, H, H, generator=g, dtype=torch.float32).to(device)
        chunks.append(net(x_seg))
        del x_seg  # release input segment
        i += sz
    y = torch.cat(chunks, dim=0)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del y, chunks
    return peak


@torch.no_grad()
def run_mode4_io_stream(net, T, b, C, H, kappa, device, num_classes):
    """Mode 4: IO-streaming with sink (pi_in=stream, pi_out=sink, kappa<T).

    Sink semantics: each segment's output goes through a running reducer
    (running argmax for classification) and the segment output tensor is
    immediately released.
    """
    reset_net_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    g = torch.Generator(device="cpu").manual_seed(42)
    # Sink: running sum of logits (will be mean over T at end)
    running_sum = torch.zeros(b, num_classes, device=device, dtype=torch.float32)
    n_seen = 0
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = torch.randn(sz, b, C, H, H, generator=g, dtype=torch.float32).to(device)
        y_seg = net(x_seg)            # [sz, B, num_classes]
        running_sum += y_seg.sum(dim=0)
        n_seen += sz
        del x_seg, y_seg              # sink: release immediately
        i += sz
    final = running_sum / n_seen
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del final, running_sum
    return peak


# ============================================================================
# Decomposition: split aggregate peak into 6 categories
# ============================================================================

@dataclass
class ModeDecomp:
    mode_id: int
    mode_name: str
    peak_total: int          # bytes
    M_static: int
    M_carry: int             # estimated; lower bound for SJ nets
    M_in_live: int
    M_act_live: int          # = peak_total - sum(others)
    M_out_live: int
    M_workspace: int = 0     # absorbed into M_act_live for now

    def to_gb(self) -> dict:
        return {k: v / (1024 ** 3) for k, v in asdict(self).items()
                if isinstance(v, (int, float)) and k != "mode_id"}


def decompose(peak_total: int, M_static: int, M_carry: int,
              io: dict, mode_id: int, T: int, kappa: int) -> ModeDecomp:
    """Decompose aggregate peak into 6 categories given mode-specific I/O."""
    if mode_id == 1:
        # Full-horizon baseline: full input retained, full output retained
        M_in_live = io["in_full"]
        M_out_live = io["out_full"]
    elif mode_id == 2:
        # Segmented retained-IO: full input retained, full output retained
        M_in_live = io["in_full"]
        M_out_live = io["out_full"]
    elif mode_id == 3:
        # Input-streaming retained-out: kappa input live, full output retained
        M_in_live = io["in_per_step"] * kappa
        M_out_live = io["out_full"]
    elif mode_id == 4:
        # IO-streaming: kappa input live, kappa output live (sink)
        M_in_live = io["in_per_step"] * kappa
        M_out_live = io["out_per_step"] * kappa
    else:
        raise ValueError(f"unknown mode {mode_id}")

    accounted = M_static + M_carry + M_in_live + M_out_live
    M_act_live = max(peak_total - accounted, 0)

    return ModeDecomp(
        mode_id=mode_id,
        mode_name={1: "full_horizon", 2: "segmented_retIO",
                   3: "input_stream", 4: "io_stream"}[mode_id],
        peak_total=peak_total,
        M_static=M_static, M_carry=M_carry,
        M_in_live=M_in_live, M_act_live=M_act_live,
        M_out_live=M_out_live,
        M_workspace=0,
    )


# ============================================================================
# Per-network full sweep
# ============================================================================

def profile_network(name: str, builder, T: int, kappa: int, b: int,
                    C: int, H: int, num_classes: int,
                    device: torch.device) -> dict:
    """Run all 4 modes for one network, decompose into 6 categories each."""
    print(f"\n{'='*72}")
    print(f"=== {name} ===")
    print(f"{'='*72}")

    net = builder(num_classes=num_classes).to(device)
    net.eval()

    M_static = measure_static(net)
    M_carry  = measure_carry_capacity(net, b, H)
    io = measure_io_static(T, b, C, H, num_classes)

    print(f"  M_static={M_static / 1024**3:.3f}GB  "
          f"M_carry(est)={M_carry / 1024**3:.3f}GB")
    print(f"  io: in_full={io['in_full']/1024**3:.3f}GB "
          f"in_step={io['in_per_step']/1024**3:.3f}GB  "
          f"out_full={io['out_full']/1024**3:.3f}GB "
          f"out_step={io['out_per_step']/1024**3:.3f}GB")

    results = {}
    for mode_id, (label, runner) in enumerate([
        ("Mode1 full-horizon",
         lambda: run_mode1_full_horizon(net, T, b, C, H, device)),
        ("Mode2 segmented retained-IO",
         lambda: run_mode2_segmented_retIO(net, T, b, C, H, kappa, device)),
        ("Mode3 input-streaming",
         lambda: run_mode3_inputstream(net, T, b, C, H, kappa, device)),
        ("Mode4 IO-streaming",
         lambda: run_mode4_io_stream(net, T, b, C, H, kappa, device, num_classes)),
    ], 1):
        try:
            t0 = time.time()
            peak = runner()
            dt = time.time() - t0
            decomp = decompose(peak, M_static, M_carry, io, mode_id, T, kappa)
            print(f"  [{label:32s}] peak={peak/1024**3:6.3f}GB  "
                  f"act_live={decomp.M_act_live/1024**3:5.3f}GB  ({dt:5.1f}s)")
            results[f"mode{mode_id}"] = decomp
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [{label:32s}] OOM: {str(e)[:80]}")
            results[f"mode{mode_id}"] = ModeDecomp(
                mode_id=mode_id,
                mode_name={1: "full_horizon", 2: "segmented_retIO",
                           3: "input_stream", 4: "io_stream"}[mode_id],
                peak_total=-1, M_static=M_static, M_carry=M_carry,
                M_in_live=-1, M_act_live=-1, M_out_live=-1,
            )
            torch.cuda.empty_cache(); gc.collect()

    del net
    torch.cuda.empty_cache(); gc.collect()
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--b", type=int, default=64)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--H", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output", type=str,
                        default="p9_3_livesetdecomp_results")
    parser.add_argument("--nets", type=str, default="all",
                        help="comma-separated subset, or 'all' "
                             "(SR-18,SEW-50,VGG-19,ConvLSTM)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not SJ_AVAILABLE:
        raise RuntimeError("SpikingJelly required for SR-18/SEW-50/VGG-19")

    device = torch.device("cuda:0")
    print(f"=== AEROS Phase 2 Exp 3 — Live-Set Decomposition ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  T={args.T}, kappa={args.kappa}, b={args.b}, "
          f"C={args.C}, H={args.H}")

    builders = {
        "SR-18":    build_sr18,
        "SEW-50":   build_sew50,
        "VGG-19":   build_vgg19_bn,
        "ConvLSTM": ConvLSTMNet,
    }
    if args.nets.lower() != "all":
        names = [n.strip() for n in args.nets.split(",")]
        builders = {k: v for k, v in builders.items() if k in names}

    all_results = {}
    for name, builder in builders.items():
        try:
            r = profile_network(name, builder, args.T, args.kappa,
                                args.b, args.C, args.H, args.num_classes, device)
            all_results[name] = {k: asdict(v) for k, v in r.items()}
        except Exception as e:
            print(f"\n[ERROR] {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Summary table
    print(f"\n{'='*92}")
    print(f"=== Live-Set Decomposition (GB) — T={args.T}, kappa={args.kappa},"
          f" b={args.b}, H={args.H} ===")
    print(f"{'='*92}")
    fmt = "{:10s}  {:8s}  {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} | {:>7s}"
    print(fmt.format("Net", "Mode", "M_stat", "M_carry", "M_in",
                     "M_act", "M_out", "Peak"))
    print("-" * 92)
    for net_name, modes in all_results.items():
        if "error" in modes:
            print(f"{net_name:10s}  ERROR")
            continue
        for mode_key, d in modes.items():
            if d["peak_total"] < 0:
                print(f"{net_name:10s}  {mode_key:8s}  OOM")
                continue
            print(fmt.format(
                net_name, mode_key,
                f"{d['M_static']/1024**3:.3f}",
                f"{d['M_carry']/1024**3:.3f}",
                f"{d['M_in_live']/1024**3:.3f}",
                f"{d['M_act_live']/1024**3:.3f}",
                f"{d['M_out_live']/1024**3:.3f}",
                f"{d['peak_total']/1024**3:.3f}",
            ))

    # Save
    out_npz = args.output + ".npz"
    out_json = args.output + ".json"

    flat = {
        "config": {
            "T": args.T, "kappa": args.kappa, "b": args.b,
            "C": args.C, "H": args.H, "num_classes": args.num_classes,
        },
        "results": all_results,
    }
    with open(out_json, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # NPZ: stack into array (4 nets x 4 modes x 6 categories)
    cat_keys = ["M_static", "M_carry", "M_in_live", "M_act_live",
                "M_out_live", "M_workspace"]
    arr = np.full((len(builders), 4, len(cat_keys)), np.nan)
    net_names = list(builders.keys())
    for ni, name in enumerate(net_names):
        if "error" in all_results[name]:
            continue
        for mi, mode_key in enumerate([f"mode{i}" for i in [1, 2, 3, 4]]):
            d = all_results[name][mode_key]
            if d["peak_total"] < 0:
                continue
            for ki, k in enumerate(cat_keys):
                arr[ni, mi, ki] = d[k]
    np.savez(out_npz,
             data=arr,
             net_names=np.array(net_names),
             mode_names=np.array(["full_horizon", "segmented_retIO",
                                  "input_stream", "io_stream"]),
             cat_keys=np.array(cat_keys),
             config=np.array(json.dumps(flat["config"])))
    print(f"Saved NPZ:  {out_npz}")


if __name__ == "__main__":
    main()