#!/usr/bin/env python
"""
AEROS Phase 2 Exp 10 — Full-horizon graph scaling stress test.

Compares 5 systems for long-horizon temporal model deployment:

  S1: PyTorch eager
        Standard execution, no graph optimization.
  S2: TorchScript trace
        Captures forward into a JIT-traced graph. SJ stateful LIF
        is expected to fail trace; this is itself a useful finding.
  S3: Full-horizon CUDA Graph capture
        torch.cuda.graph context captures all kernels for [T,B,...]
        forward. Capture overhead (warmup + capture) and replay.
  S4: AEROS segmented (no graph)
        Equivalent to Mode 4 in Exp 4. Per-segment forward, IO sink.
  S5: AEROS + segment-CUDA-Graph
        Capture one kappa-segment graph; replay across T/kappa segments.
        Demonstrates the section 3.4 chunk-internal backend story.

Metrics per (system, net, T):
  - capture_time_ms     : time to compile/trace/capture (S2/S3/S5)
  - capture_peak_GB     : peak HBM during capture/compile
  - first_inference_ms  : wall time of first inference call
  - steady_inference_ms : wall time of subsequent inferences (5-iter median)
  - peak_memory_GB      : peak HBM during inference
  - status              : ok / OOM / compile_fail / capture_fail / runtime_fail

The killer claim is feasibility, NOT speedup: all full-horizon graph
systems exhibit capture-time / compile-time / memory growth with T,
and fail beyond moderate T; AEROS systems are kappa-bounded.

Usage:
    python p9_10_fullhorizon_stress.py --output p9_10_results
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
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18)
    SJ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SJ unavailable: {e}")
    SJ_AVAILABLE = False


# ============================================================================
# Networks
# ============================================================================

def build_sr18(num_classes=10):
    """SpikingResNet-18 multi-step."""
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
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
# Result record
# ============================================================================

@dataclass
class Cell:
    system: str
    net: str
    T: int
    kappa: int
    status: str = "ok"          # ok / OOM / compile_fail / capture_fail / runtime_fail
    error_msg: str = ""
    capture_time_ms: float = -1.0
    capture_peak_GB: float = -1.0
    first_inference_ms: float = -1.0
    steady_inference_ms: float = -1.0
    peak_memory_GB: float = -1.0
    n_repeats: int = 5


# ============================================================================
# System runners
# ============================================================================

def reset_gpu(device):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def run_s1_pytorch_eager(net, T, b, C, H, device, n_repeats=5) -> Cell:
    cell = Cell(system="S1_eager", net="?", T=T, kappa=T)
    try:
        # Single-run peak measurement (Doris 7 P0-3 fix: reset between iters)
        x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
        reset_state_compat(net)
        reset_gpu(device)
        t0 = time.time()
        y = net(x)
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        times = []
        for _ in range(n_repeats):
            reset_state_compat(net)
            t0 = time.time()
            y = net(x)
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.n_repeats = n_repeats
        del x, y
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "runtime_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


@torch.no_grad()
def run_s2_torchscript(net, T, b, C, H, device, n_repeats=5) -> Cell:
    """JIT-trace the network. SJ stateful LIF is expected to fail."""
    cell = Cell(system="S2_torchscript", net="?", T=T, kappa=T)
    try:
        reset_gpu(device)
        x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
        reset_state_compat(net)
        t0 = time.time()
        # Try jit.trace — SJ stateful neurons will likely error
        traced = torch.jit.trace(net, x, check_trace=False)
        torch.cuda.synchronize(device)
        cell.capture_time_ms = (time.time() - t0) * 1000
        cell.capture_peak_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        reset_gpu(device)
        reset_state_compat(net)
        t0 = time.time()
        y = traced(x)
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000

        times = []
        for _ in range(n_repeats):
            reset_state_compat(net)
            t0 = time.time()
            y = traced(x)
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3
        cell.n_repeats = n_repeats
        del x, y, traced
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            cell.status = "OOM"
        else:
            cell.status = "compile_fail"
            cell.error_msg = msg[:160]
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "compile_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


@torch.no_grad()
def run_s3_full_horizon_cuda_graph(net, T, b, C, H, device, n_repeats=5) -> Cell:
    """Capture full-horizon [T,B,C,H,W] forward into one CUDA Graph; replay."""
    cell = Cell(system="S3_full_horizon_cuda_graph", net="?", T=T, kappa=T)
    try:
        reset_gpu(device)
        x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)

        # Warmup — CUDA Graph capture requires non-default stream warmup
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                reset_state_compat(net)
                _ = net(x)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)
        reset_gpu(device)

        # Capture
        reset_state_compat(net)
        g = torch.cuda.CUDAGraph()
        t0 = time.time()
        with torch.cuda.graph(g):
            y_static = net(x)
        torch.cuda.synchronize(device)
        cell.capture_time_ms = (time.time() - t0) * 1000
        cell.capture_peak_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        # First replay
        t0 = time.time()
        g.replay()
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000

        # Steady replay
        times = []
        for _ in range(n_repeats):
            t0 = time.time()
            g.replay()
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3
        cell.n_repeats = n_repeats
        del x, y_static, g
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            cell.status = "OOM"
        else:
            cell.status = "capture_fail"
            cell.error_msg = msg[:160]
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "capture_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


@torch.no_grad()
def run_s4a_aeros_retainedinput_seg(net, T, b, C, H, kappa, device, n_repeats=5) -> Cell:
    """S4a: AEROS retained-input segment with output sink.
    pi_in=retain, pi_out=sink, kappa<T.
    Predicted memory: M_0 + alpha_in * T + alpha_K * kappa.

    Note: for Doris 7 P0-3, this is the variant with input materialized as
    a single [T,b,C,H,W] tensor at start, with segments indexed off it,
    and outputs sunk per-segment.
    """
    cell = Cell(system="S4a_aeros_retainedinput_seg", net="?", T=T, kappa=kappa)
    try:
        def one_run():
            reset_state_compat(net)
            x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
            running_sum = torch.zeros(b, 10, device=device, dtype=torch.float32)
            n = 0; i = 0
            while i < T:
                sz = min(kappa, T - i)
                y_seg = net(x[i:i+sz])
                running_sum += y_seg.sum(dim=0)
                n += sz
                del y_seg
                i += sz
            torch.cuda.synchronize(device)
            out = running_sum / n
            del x, running_sum
            return out

        # Per-iteration peak measurement (not accumulated; Doris 7 P0-3 fix)
        reset_gpu(device)
        t0 = time.time()
        _ = one_run()
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        times = []
        for _ in range(n_repeats):
            reset_gpu(device)                           # reset between repeats
            t0 = time.time()
            _ = one_run()
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.n_repeats = n_repeats
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "runtime_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


@torch.no_grad()
def run_s4b_aeros_iostream(net, T, b, C, H, kappa, device, n_repeats=5) -> Cell:
    """S4b: AEROS IO-streaming (Mode 4 in paper).
    pi_in=stream, pi_out=sink, kappa<T.
    Predicted memory: M_0 + (alpha_in + alpha_K + alpha_out) * kappa  (T-independent)
    """
    cell = Cell(system="S4b_aeros_iostream", net="?", T=T, kappa=kappa)
    try:
        def one_run():
            reset_state_compat(net)
            g_gen = torch.Generator(device=device).manual_seed(42)
            running_sum = torch.zeros(b, 10, device=device, dtype=torch.float32)
            n = 0; i = 0
            while i < T:
                sz = min(kappa, T - i)
                x_seg = torch.randn(sz, b, C, H, H, generator=g_gen,
                                    device=device, dtype=torch.float32)
                y_seg = net(x_seg)
                running_sum += y_seg.sum(dim=0)
                n += sz
                del x_seg, y_seg
                i += sz
            torch.cuda.synchronize(device)
            out = running_sum / n
            del running_sum
            return out

        reset_gpu(device)
        t0 = time.time()
        _ = one_run()
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        times = []
        for _ in range(n_repeats):
            reset_gpu(device)
            t0 = time.time()
            _ = one_run()
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.n_repeats = n_repeats
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "runtime_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


@torch.no_grad()
def run_s5_aeros_segment_cuda_graph(net, T, b, C, H, kappa, device, n_repeats=5) -> Cell:
    """AEROS + capture one kappa-segment graph; replay across segments.

    The capture unit is bounded by kappa, so the residency advantage of
    AEROS is preserved. CUDA Graph reduces per-segment kernel-launch
    overhead. This is paper section 3.4's "optional segment CUDA Graph backend".

    Note on state: SJ LIF state is updated in-place and is part of the
    captured kernel sequence. Replaying the same graph N times equals
    running N segments only if the state is reset between captures and
    the captured state-update is run forward each replay. We do an
    initial capture + replay and warn that this is exploratory; full
    state-flow CUDA Graph integration is implementation work beyond
    scope.
    """
    cell = Cell(system="S5_aeros_segment_cuda_graph", net="?", T=T, kappa=kappa)
    if T % kappa != 0:
        # For simplicity require divisibility; non-uniform schedule is paper §3.4
        cell.status = "skipped_non_divisible"
        cell.error_msg = f"T={T} not divisible by kappa={kappa}"
        return cell
    try:
        reset_gpu(device)
        n_segs = T // kappa
        x_seg_static = torch.randn(kappa, b, C, H, H, device=device, dtype=torch.float32)

        # Warmup
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                reset_state_compat(net)
                _ = net(x_seg_static)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)
        reset_gpu(device)

        # Capture single kappa-segment graph
        reset_state_compat(net)
        g = torch.cuda.CUDAGraph()
        t0 = time.time()
        with torch.cuda.graph(g):
            y_seg_static = net(x_seg_static)
        torch.cuda.synchronize(device)
        cell.capture_time_ms = (time.time() - t0) * 1000
        cell.capture_peak_GB = torch.cuda.max_memory_allocated(device) / 1024**3

        # First inference: replay graph n_segs times, accumulating outputs
        running_sum = torch.zeros(b, 10, device=device, dtype=torch.float32)
        t0 = time.time()
        for _ in range(n_segs):
            x_seg_static.copy_(torch.randn_like(x_seg_static))
            g.replay()
            running_sum += y_seg_static.sum(dim=0)
        torch.cuda.synchronize(device)
        cell.first_inference_ms = (time.time() - t0) * 1000

        # Steady
        times = []
        for _ in range(n_repeats):
            running_sum.zero_()
            t0 = time.time()
            for _ in range(n_segs):
                x_seg_static.copy_(torch.randn_like(x_seg_static))
                g.replay()
                running_sum += y_seg_static.sum(dim=0)
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        cell.steady_inference_ms = float(np.median(times))
        cell.peak_memory_GB = torch.cuda.max_memory_allocated(device) / 1024**3
        cell.n_repeats = n_repeats
        del g, x_seg_static, y_seg_static, running_sum
    except torch.cuda.OutOfMemoryError:
        cell.status = "OOM"
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            cell.status = "OOM"
        else:
            cell.status = "capture_fail"
            cell.error_msg = msg[:160]
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        cell.status = "capture_fail"
        cell.error_msg = type(e).__name__ + ": " + str(e)[:120]
        torch.cuda.empty_cache(); gc.collect()
    return cell


# ============================================================================
# Main sweep
# ============================================================================

def sweep(args, device):
    builders = {"SR-18": build_sr18, "ConvLSTM": build_convlstm}
    if args.nets.lower() != "all":
        names = [n.strip() for n in args.nets.split(",")]
        builders = {k: v for k, v in builders.items() if k in names}

    Ts = [int(t) for t in args.T_sweep.split(",")]
    kappa = args.kappa

    print(f"=== AEROS Phase 2 Exp 10 — Full-horizon graph scaling stress ===")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  T sweep: {Ts}  kappa={kappa}  b={args.b}  H={args.H}")
    print(f"  Nets: {list(builders.keys())}")
    print()

    runners = {
        "S1_eager":                  lambda net, T: run_s1_pytorch_eager(
            net, T, args.b, 3, args.H, device, args.n_repeats),
        "S2_torchscript":            lambda net, T: run_s2_torchscript(
            net, T, args.b, 3, args.H, device, args.n_repeats),
        "S3_fullhorizon_cudagraph":  lambda net, T: run_s3_full_horizon_cuda_graph(
            net, T, args.b, 3, args.H, device, args.n_repeats),
        "S4a_aeros_retainedinput_seg": lambda net, T: run_s4a_aeros_retainedinput_seg(
            net, T, args.b, 3, args.H, kappa, device, args.n_repeats),
        "S4b_aeros_iostream":        lambda net, T: run_s4b_aeros_iostream(
            net, T, args.b, 3, args.H, kappa, device, args.n_repeats),
        "S5_aeros_segment_cudagraph": lambda net, T: run_s5_aeros_segment_cuda_graph(
            net, T, args.b, 3, args.H, kappa, device, args.n_repeats),
    }

    all_cells = []
    for name, builder in builders.items():
        print(f"\n{'='*72}\n=== {name} ===\n{'='*72}")
        try:
            net = builder().to(device)
        except Exception as e:
            print(f"  Failed to build {name}: {e}")
            continue

        for T in Ts:
            for sys_name, runner in runners.items():
                cell = runner(net, T)
                cell.net = name
                cell.system = sys_name
                cell.T = T
                if cell.system in ("S4a_aeros_retainedinput_seg",
                                   "S4b_aeros_iostream",
                                   "S5_aeros_segment_cudagraph"):
                    cell.kappa = kappa
                else:
                    cell.kappa = T

                all_cells.append(cell)
                if cell.status == "ok":
                    cap = (f"cap={cell.capture_time_ms:.1f}ms"
                           if cell.capture_time_ms >= 0 else "cap=-")
                    print(f"  {sys_name:30s} T={T:5d} : OK     "
                          f"{cap:20s} steady={cell.steady_inference_ms:8.1f}ms  "
                          f"peak={cell.peak_memory_GB:6.3f}GB")
                else:
                    print(f"  {sys_name:30s} T={T:5d} : {cell.status:14s} "
                          f"{cell.error_msg[:80]}")
        del net
        torch.cuda.empty_cache(); gc.collect()
    return all_cells


def summarize(cells, output_path):
    # JSON
    with open(output_path + ".json", "w") as f:
        json.dump([asdict(c) for c in cells], f, indent=2)
    print(f"\nSaved JSON: {output_path}.json")

    # Summary table
    nets = sorted(set(c.net for c in cells))
    Ts = sorted(set(c.T for c in cells))
    systems = ["S1_eager", "S2_torchscript", "S3_fullhorizon_cudagraph",
               "S4a_aeros_retainedinput_seg", "S4b_aeros_iostream",
               "S5_aeros_segment_cudagraph"]

    print(f"\n{'='*92}")
    print(f"=== Summary: status by (Net, System, T) ===")
    print(f"{'='*92}")
    for net in nets:
        print(f"\n--- {net} ---")
        hdr = f"{'System':30s}" + "".join(f"  T={t:<6d}" for t in Ts)
        print(hdr)
        print("-" * len(hdr))
        for sys_n in systems:
            row = f"{sys_n:30s}"
            for T in Ts:
                cell = next((c for c in cells
                             if c.net == net and c.system == sys_n and c.T == T),
                            None)
                if cell is None:
                    row += f"  {'?':<8s}"
                elif cell.status == "ok":
                    row += f"  {cell.peak_memory_GB:<8.2f}"
                elif cell.status == "OOM":
                    row += f"  {'OOM':<8s}"
                elif cell.status == "compile_fail":
                    row += f"  {'CF':<8s}"
                elif cell.status == "capture_fail":
                    row += f"  {'CapF':<8s}"
                else:
                    row += f"  {cell.status[:6]:<8s}"
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T_sweep", type=str, default="16,32,64,128,256,512,1024,2048")
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--nets", type=str, default="all")
    parser.add_argument("--output", type=str, default="p9_10_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")
    cells = sweep(args, device)
    summarize(cells, args.output)


if __name__ == "__main__":
    main()