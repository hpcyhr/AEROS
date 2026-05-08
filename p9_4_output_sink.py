#!/usr/bin/env python
"""
AEROS Phase 2 Exp 4 — Output streaming sink + T sweep.

Validates the headline claim of v9 paper §1 abstract:
    Mode 4 (IO-streaming) peak HBM is T-independent, only kappa-dependent.

Sweeps T over 9 orders of magnitude {128, 1024, 4096, 16384, 65536} for
4 representative nets and 4 deployment modes. The expected pattern:
  Mode 1 (full-horizon): peak grows as O(T), OOM beyond moderate T.
  Mode 2 (segmented retIO): peak grows as O(T) due to input/output retention.
  Mode 3 (input-stream): peak grows as O(T) due to output retention only.
  Mode 4 (IO-stream): peak is constant in T (eq M_stream(kappa) in paper).

Setup: T sweep, kappa=8, b=16, C=3, H=64. We use H=64 (smaller than Exp3)
because at T=65536 even a single-step input tensor must be allocated;
we want H modest enough that 32 GB V100 can host all kappa=8 segments
even at the tail of the sweep.

Usage:
    python p9_4_output_sink.py --output p9_4_results
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18)
    from spikingjelly.activation_based.model.sew_resnet import sew_resnet18
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg11_bn)
    SJ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SpikingJelly unavailable: {e}")
    SJ_AVAILABLE = False


# ============================================================================
# Network builders
# ============================================================================

def build_sr18(num_classes=10):
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True, num_classes=num_classes)
    net.eval()
    functional.set_step_mode(net, "m")
    return net


def build_sew18(num_classes=10):
    net = sew_resnet18(spiking_neuron=neuron.LIFNode,
                       surrogate_function=surrogate.ATan(),
                       detach_reset=True, cnf="ADD", num_classes=num_classes)
    net.eval()
    functional.set_step_mode(net, "m")
    return net


def build_vgg11(num_classes=10):
    net = spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
                          surrogate_function=surrogate.ATan(),
                          detach_reset=True, num_classes=num_classes)
    net.eval()
    functional.set_step_mode(net, "m")
    return net


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_c=3, hid_c=32, k=3):
        super().__init__()
        self.hid_c = hid_c
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=k // 2)

    def forward(self, x, state):
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


class _ConvLSTMNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cell1 = _ConvLSTMCell(3, 32)
        self.cell2 = _ConvLSTMCell(32, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
        self._states = None

    def reset_state(self):
        self._states = [None, None]

    def forward(self, x):
        T = x.shape[0]
        if self._states is None:
            self.reset_state()
        outs = []
        for t in range(T):
            h = x[t]
            h, self._states[0] = self.cell1(h, self._states[0])
            h, self._states[1] = self.cell2(h, self._states[1])
            outs.append(self.fc(self.pool(h).flatten(1)))
        return torch.stack(outs, dim=0)


def build_convlstm(num_classes=10):
    net = _ConvLSTMNet(num_classes=num_classes)
    net.eval()
    return net


def reset_state_compat(net):
    if hasattr(net, "reset_state"):
        net.reset_state()
    else:
        try:
            functional.reset_net(net)
        except Exception:
            pass


# ============================================================================
# Modes (all torch.no_grad to match deployment-inference setting)
# ============================================================================

@torch.no_grad()
def run_mode1_full_horizon(net, T, b, C, H, device):
    """Full-horizon: materialize [T,B,C,H,W], run net once."""
    reset_state_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    y = net(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del x, y
    return peak


@torch.no_grad()
def run_mode2_segmented_retIO(net, T, b, C, H, kappa, device):
    """Segmented retained-IO: full input retained, full output retained."""
    reset_state_compat(net)
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
    """Input-streaming: kappa input live, full output retained."""
    reset_state_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    chunks = []
    g = torch.Generator(device=device).manual_seed(42)
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = torch.randn(sz, b, C, H, H, generator=g, device=device,
                            dtype=torch.float32)
        chunks.append(net(x_seg))
        del x_seg
        i += sz
    y = torch.cat(chunks, dim=0)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del y, chunks
    return peak


@torch.no_grad()
def run_mode4_io_stream(net, T, b, C, H, kappa, device, num_classes):
    """IO-streaming: kappa input live, output sink (running sum/argmax)."""
    reset_state_compat(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    g = torch.Generator(device=device).manual_seed(42)
    running_sum = torch.zeros(b, num_classes, device=device, dtype=torch.float32)
    n_seen = 0
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = torch.randn(sz, b, C, H, H, generator=g, device=device,
                            dtype=torch.float32)
        y_seg = net(x_seg)
        running_sum += y_seg.sum(dim=0)
        n_seen += sz
        del x_seg, y_seg                     # immediate release
        i += sz
    final = running_sum / n_seen
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del final, running_sum
    return peak


# ============================================================================
# Sweep
# ============================================================================

@dataclass
class Cell:
    net: str
    mode: int
    T: int
    kappa: int
    peak_bytes: int = -1            # -1 means OOM
    wall_ms: float = -1.0
    error: str = ""


def run_one_cell(net, name, mode_id, T, kappa, b, C, H, device, num_classes):
    """Run one (net, mode, T) configuration. Returns Cell."""
    cell = Cell(net=name, mode=mode_id, T=T, kappa=kappa)
    try:
        t0 = time.time()
        if mode_id == 1:
            peak = run_mode1_full_horizon(net, T, b, C, H, device)
        elif mode_id == 2:
            peak = run_mode2_segmented_retIO(net, T, b, C, H, kappa, device)
        elif mode_id == 3:
            peak = run_mode3_inputstream(net, T, b, C, H, kappa, device)
        elif mode_id == 4:
            peak = run_mode4_io_stream(net, T, b, C, H, kappa, device, num_classes)
        else:
            raise ValueError(f"unknown mode {mode_id}")
        wall = (time.time() - t0) * 1000
        cell.peak_bytes = peak
        cell.wall_ms = wall
    except torch.cuda.OutOfMemoryError as e:
        cell.error = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            cell.error = "OOM"
        else:
            cell.error = type(e).__name__ + ": " + msg[:80]
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.error = type(e).__name__ + ": " + str(e)[:80]
        torch.cuda.empty_cache(); gc.collect()
    return cell


def sweep(args, device):
    builders = {
        "SR-18":    build_sr18,
        "SEW-18":   build_sew18,
        "VGG-11":   build_vgg11,
        "ConvLSTM": build_convlstm,
    }
    if args.nets.lower() != "all":
        names = [n.strip() for n in args.nets.split(",")]
        builders = {k: v for k, v in builders.items() if k in names}

    Ts = [int(t) for t in args.T_sweep.split(",")]
    kappa = args.kappa

    print(f"=== AEROS Phase 2 Exp 4 — Output streaming + T sweep ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  T sweep: {Ts}  kappa={kappa}  b={args.b}  C=3  H={args.H}")
    print(f"  Nets: {list(builders.keys())}")
    print(f"  Modes: 1 (full-horizon), 2 (seg retIO), 3 (input-stream), 4 (IO-stream)")
    print()

    all_cells = []

    for name, builder in builders.items():
        print(f"\n{'='*72}\n=== {name} ===\n{'='*72}")
        # build net once, reuse across T (state is reset per call)
        try:
            net = builder().to(device)
        except Exception as e:
            print(f"  Failed to build {name}: {e}")
            continue

        for T in Ts:
            for mode_id in [1, 2, 3, 4]:
                cell = run_one_cell(
                    net, name, mode_id, T, kappa,
                    args.b, 3, args.H, device, num_classes=10,
                )
                all_cells.append(cell)
                if cell.error:
                    print(f"  T={T:6d} mode={mode_id} : {cell.error}")
                else:
                    print(f"  T={T:6d} mode={mode_id} : peak={cell.peak_bytes/1024**3:6.3f}GB  "
                          f"wall={cell.wall_ms:7.1f}ms")
        del net
        torch.cuda.empty_cache(); gc.collect()

    return all_cells


def summarize(cells, output_path):
    """Print summary table + save JSON/NPZ."""
    print(f"\n{'='*92}")
    print(f"=== Summary: peak HBM (GB) by (Net, Mode, T)  [OOM = '-'] ===")
    print(f"{'='*92}")

    nets = sorted(set(c.net for c in cells))
    Ts = sorted(set(c.T for c in cells))
    modes = [1, 2, 3, 4]

    for net in nets:
        print(f"\n--- {net} ---")
        # Header
        hdr = f"{'Mode':<10s}" + "".join(f"  T={t:<8d}" for t in Ts)
        print(hdr)
        print("-" * len(hdr))
        for mode_id in modes:
            row = f"Mode {mode_id:<5d}"
            for T in Ts:
                cell = next((c for c in cells
                             if c.net == net and c.mode == mode_id and c.T == T),
                            None)
                if cell is None:
                    row += f"  {'?':<10s}"
                elif cell.error == "OOM":
                    row += f"  {'OOM':<10s}"
                elif cell.error:
                    row += f"  {'ERR':<10s}"
                else:
                    row += f"  {cell.peak_bytes/1024**3:<10.3f}"
            print(row)

    # Save JSON
    rows = [{
        "net": c.net, "mode": c.mode, "T": c.T, "kappa": c.kappa,
        "peak_bytes": c.peak_bytes, "wall_ms": c.wall_ms, "error": c.error,
    } for c in cells]
    with open(output_path + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved JSON: {output_path}.json")

    # Save NPZ: 3D array (n_nets, 4 modes, n_T) of peak bytes (NaN for OOM)
    arr = np.full((len(nets), len(modes), len(Ts)), np.nan)
    for c in cells:
        ni = nets.index(c.net)
        mi = modes.index(c.mode)
        ti = Ts.index(c.T)
        if not c.error and c.peak_bytes >= 0:
            arr[ni, mi, ti] = c.peak_bytes

    np.savez(output_path + ".npz",
             data=arr,
             net_names=np.array(nets),
             mode_ids=np.array(modes),
             T_values=np.array(Ts),
             config=np.array(json.dumps({
                 "kappa": cells[0].kappa if cells else 8,
                 "b": None, "C": 3, "H": None,
             })))
    print(f"Saved NPZ:  {output_path}.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T_sweep", type=str, default="128,1024,4096,16384,65536")
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--nets", type=str, default="all",
                        help="comma-separated subset, or 'all'")
    parser.add_argument("--output", type=str, default="p9_4_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")
    cells = sweep(args, device)
    summarize(cells, args.output)


if __name__ == "__main__":
    main()