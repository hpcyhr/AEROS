#!/usr/bin/env python
"""
AEROS Phase 2 Exp 10 v3 — Full-horizon graph scaling stress, subprocess-isolated.

Doris 8 P0-1 root cause: in v2, all 6 systems (S1, S2, S3, S4a, S4b, S5)
ran in sequence on the SAME process / SAME `net` object. Even with
torch.cuda.empty_cache() between cells, the cached allocator pool keeps
freed buffers around. Earlier full-horizon S1/S2/S3 allocations of
`x = torch.randn(T, b, C, H, H)` made the pool grow large; later S4b
peak measurements via reset_peak_memory_stats only count NEW allocations
relative to the reset, but max_memory_allocated reads the high-water mark
which can still reflect the pool's existing reservation in some workloads
(in particular when CUDAGraphs / TorchScript hold onto static tensors).

Fix: each (net, system, T) cell runs in its own Python subprocess. Each
subprocess builds the net, runs ONE cell, writes JSON to stdout, exits.
The OS reclaims the entire CUDA context. No state leaks between cells.

Usage:
    python p9_10_fullhorizon_stress_v3.py --nets all \\
        --T_sweep 16,32,64,128,256,512,1024,2048 \\
        --kappa 8 --b 16 --H 64 --output p9_10_v3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


# ============================================================================
# Worker: runs ONE cell and prints JSON to stdout
# ============================================================================

WORKER_SCRIPT = r'''
import gc
import json
import sys
import time

import torch

@torch.no_grad()
def reset_state(net):
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(net)
    except Exception:
        pass


def build_net(name, num_classes=10):
    from spikingjelly.activation_based import functional, neuron, surrogate
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=num_classes,
    )
    if name == "SR-18":
        from spikingjelly.activation_based.model.spiking_resnet import (
            spiking_resnet18)
        net = spiking_resnet18(**common)
    elif name == "ConvLSTM":
        import torch.nn as tnn
        class ConvLSTMCell(tnn.Module):
            def __init__(self, in_ch=3, hid=128, ks=3, pad=1):
                super().__init__()
                self.conv = tnn.Conv2d(in_ch + hid, 4*hid, ks, padding=pad)
                self.hid = hid
            def forward(self, x_t, h, c):
                z = self.conv(torch.cat([x_t, h], dim=1))
                i, f, g, o = z.chunk(4, dim=1)
                i = torch.sigmoid(i); f = torch.sigmoid(f)
                g = torch.tanh(g);    o = torch.sigmoid(o)
                c = f*c + i*g
                h = o * torch.tanh(c)
                return h, c
        class ConvLSTMNet(tnn.Module):
            def __init__(self, in_ch=3, hid=128, n_classes=10):
                super().__init__()
                self.cell = ConvLSTMCell(in_ch, hid)
                self.head = tnn.Linear(hid, n_classes)
                self.hid = hid
                self.h = None; self.c = None
            def forward(self, x):
                T, B, C, H, W = x.shape
                if self.h is None or self.h.shape[0] != B:
                    self.h = torch.zeros(B, self.hid, H, W, device=x.device)
                    self.c = torch.zeros(B, self.hid, H, W, device=x.device)
                outs = []
                for t in range(T):
                    self.h, self.c = self.cell(x[t], self.h, self.c)
                    outs.append(self.head(self.h.mean(dim=[2,3])))
                return torch.stack(outs, dim=0)
            def reset(self):
                self.h = None; self.c = None
        net = ConvLSTMNet()
    else:
        raise ValueError("unknown net: " + name)
    net.eval()
    if name != "ConvLSTM":
        functional.set_step_mode(net, "m")
    return net


@torch.no_grad()
def run_s1_eager(net, T, b, C, H, device, n_repeats=3):
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    reset_state(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); y = net(x); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated(device)
    times = []
    for _ in range(n_repeats):
        reset_state(net)
        t0 = time.time(); y = net(x); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    return {"capture_time_ms": -1, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


@torch.no_grad()
def run_s2_torchscript(net, T, b, C, H, device, n_repeats=3):
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    reset_state(net)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        scripted = torch.jit.trace(net, x, check_trace=False)
    except Exception as e:
        return {"error": "trace_fail: " + type(e).__name__}
    torch.cuda.synchronize(device)
    cap_ms = (time.time() - t0) * 1000
    reset_state(net)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); y = scripted(x); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated(device)
    times = []
    for _ in range(n_repeats):
        reset_state(net)
        t0 = time.time(); y = scripted(x); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    return {"capture_time_ms": cap_ms, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


@torch.no_grad()
def run_s3_fullhorizon_cudagraph(net, T, b, C, H, device, n_repeats=3):
    x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(3):
            reset_state(net)
            _ = net(x)
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    reset_state(net)
    g = torch.cuda.CUDAGraph()
    t0 = time.time()
    with torch.cuda.graph(g):
        y_static = net(x)
    torch.cuda.synchronize(device)
    cap_ms = (time.time() - t0) * 1000
    t0 = time.time(); g.replay(); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    times = []
    for _ in range(n_repeats):
        t0 = time.time(); g.replay(); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated(device)
    return {"capture_time_ms": cap_ms, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


@torch.no_grad()
def run_s4a_retainedinput(net, T, b, C, H, kappa, device, n_repeats=3):
    def one_run():
        reset_state(net)
        x = torch.randn(T, b, C, H, H, device=device, dtype=torch.float32)
        running_sum = torch.zeros(b, 10, device=device, dtype=torch.float32)
        n = 0; i = 0
        while i < T:
            sz = min(kappa, T - i)
            y_seg = net(x[i:i+sz])
            running_sum += y_seg.sum(dim=0)
            n += sz; del y_seg; i += sz
        torch.cuda.synchronize(device)
        out = running_sum / n
        del x, running_sum
        return out
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated(device)
    times = []
    for _ in range(n_repeats):
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    return {"capture_time_ms": -1, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


@torch.no_grad()
def run_s4b_iostream(net, T, b, C, H, kappa, device, n_repeats=3):
    def one_run():
        reset_state(net)
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
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated(device)
    times = []
    for _ in range(n_repeats):
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    return {"capture_time_ms": -1, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


@torch.no_grad()
def run_s5_segment_cudagraph(net, T, b, C, H, kappa, device, n_repeats=3):
    g_gen = torch.Generator(device=device).manual_seed(42)
    x_static = torch.randn(kappa, b, C, H, H, generator=g_gen,
                           device=device, dtype=torch.float32)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(3):
            reset_state(net)
            _ = net(x_static)
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    reset_state(net)
    g = torch.cuda.CUDAGraph()
    t0 = time.time()
    with torch.cuda.graph(g):
        y_seg_static = net(x_static)
    torch.cuda.synchronize(device)
    cap_ms = (time.time() - t0) * 1000

    def one_run():
        reset_state(net)
        running_sum = torch.zeros(b, 10, device=device, dtype=torch.float32)
        n = 0; i = 0
        while i < T:
            sz = min(kappa, T - i)
            if sz == kappa:
                x_static.copy_(torch.randn(kappa, b, C, H, H,
                                           generator=g_gen, device=device,
                                           dtype=torch.float32))
                g.replay()
                running_sum += y_seg_static.sum(dim=0)
            else:
                x_partial = torch.randn(sz, b, C, H, H, generator=g_gen,
                                        device=device, dtype=torch.float32)
                y_partial = net(x_partial)
                running_sum += y_partial.sum(dim=0)
                del x_partial, y_partial
            n += sz; i += sz
        torch.cuda.synchronize(device)
        return running_sum / n

    t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
    first_ms = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated(device)
    times = []
    for _ in range(n_repeats):
        t0 = time.time(); _ = one_run(); torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000)
    return {"capture_time_ms": cap_ms, "first_inference_ms": first_ms,
            "steady_inference_ms": float(sorted(times)[len(times)//2]),
            "peak_memory_GB": peak / 1024**3, "n_repeats": n_repeats}


def main():
    payload = json.loads(sys.argv[1])
    net_name = payload["net"]
    system = payload["system"]
    T = int(payload["T"])
    kappa = int(payload["kappa"])
    b = int(payload["b"])
    H = int(payload["H"])
    n_repeats = int(payload.get("n_repeats", 3))

    if not torch.cuda.is_available():
        print(json.dumps({"error": "no CUDA"})); return
    device = torch.device("cuda:0")

    try:
        net = build_net(net_name).to(device)
    except Exception as e:
        print(json.dumps({"error": "build_fail: " + type(e).__name__ + ": " + str(e)[:80]}))
        return

    runners = {
        "S1_eager":                  lambda: run_s1_eager(net, T, b, 3, H, device, n_repeats),
        "S2_torchscript":            lambda: run_s2_torchscript(net, T, b, 3, H, device, n_repeats),
        "S3_fullhorizon_cudagraph":  lambda: run_s3_fullhorizon_cudagraph(net, T, b, 3, H, device, n_repeats),
        "S4a_aeros_retainedinput_seg": lambda: run_s4a_retainedinput(net, T, b, 3, H, kappa, device, n_repeats),
        "S4b_aeros_iostream":        lambda: run_s4b_iostream(net, T, b, 3, H, kappa, device, n_repeats),
        "S5_aeros_segment_cudagraph": lambda: run_s5_segment_cudagraph(net, T, b, 3, H, kappa, device, n_repeats),
    }

    if system not in runners:
        print(json.dumps({"error": "unknown_system: " + system})); return

    try:
        result = runners[system]()
        print(json.dumps(result))
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"error": "OOM"}))
    except Exception as e:
        print(json.dumps({"error": "runtime_fail: " + type(e).__name__ + ": " + str(e)[:80]}))


if __name__ == "__main__":
    main()
'''


# ============================================================================
# Driver: launches subprocess per cell
# ============================================================================

@dataclass
class Cell:
    net: str
    system: str
    T: int
    kappa: int
    status: str = "ok"
    error_msg: str = ""
    capture_time_ms: float = -1.0
    first_inference_ms: float = -1.0
    steady_inference_ms: float = -1.0
    peak_memory_GB: float = -1.0
    n_repeats: int = 0


def run_cell_subprocess(net, system, T, kappa, b, H, n_repeats,
                         worker_path, timeout=900):
    payload = json.dumps({
        "net": net, "system": system, "T": T, "kappa": kappa,
        "b": b, "H": H, "n_repeats": n_repeats,
    })
    cmd = [sys.executable, worker_path, payload]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if r.returncode != 0:
        return {"error": "subprocess_exit_" + str(r.returncode),
                "stderr": (r.stderr[-200:] if r.stderr else "")}
    out_lines = [l for l in r.stdout.splitlines() if l.strip()]
    if not out_lines:
        return {"error": "no_output", "stderr": r.stderr[-200:]}
    try:
        return json.loads(out_lines[-1])
    except json.JSONDecodeError:
        return {"error": "bad_json", "raw": out_lines[-1][:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T_sweep", default="16,32,64,128,256,512,1024,2048")
    parser.add_argument("--nets", default="all")
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--output", default="p9_10_v3")
    args = parser.parse_args()

    Ts = [int(t) for t in args.T_sweep.split(",")]
    if args.nets.lower() == "all":
        nets = ["SR-18", "ConvLSTM"]
    else:
        nets = [n.strip() for n in args.nets.split(",")]

    systems = ["S1_eager", "S2_torchscript", "S3_fullhorizon_cudagraph",
               "S4a_aeros_retainedinput_seg", "S4b_aeros_iostream",
               "S5_aeros_segment_cudagraph"]

    worker_path = "/tmp/p9_10_v3_worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    print(f"=== AEROS Phase 2 Exp 10 v3 — Subprocess-isolated stress ===")
    print(f"  Nets: {nets}")
    print(f"  T sweep: {Ts}  kappa={args.kappa}  b={args.b}  H={args.H}")
    print(f"  Each cell runs in its own Python subprocess (clean CUDA context).")

    all_cells = []
    for name in nets:
        print(f"\n{'='*72}\n=== {name} ===\n{'='*72}")
        for T in Ts:
            for sys_name in systems:
                kappa = args.kappa if sys_name in (
                    "S4a_aeros_retainedinput_seg",
                    "S4b_aeros_iostream",
                    "S5_aeros_segment_cudagraph") else T
                cell = Cell(net=name, system=sys_name, T=T, kappa=kappa)

                t0 = time.time()
                result = run_cell_subprocess(
                    name, sys_name, T, kappa, args.b, args.H, args.n_repeats,
                    worker_path, timeout=900)
                elapsed = time.time() - t0

                if "error" in result:
                    cell.status = result["error"][:32]
                    cell.error_msg = result.get("stderr", "")[:120]
                    print(f"  {sys_name:30s} T={T:5d} : {cell.status:20s} "
                          f"({elapsed:.1f}s elapsed)")
                else:
                    cell.capture_time_ms = float(result.get("capture_time_ms", -1))
                    cell.first_inference_ms = float(result.get("first_inference_ms", -1))
                    cell.steady_inference_ms = float(result.get("steady_inference_ms", -1))
                    cell.peak_memory_GB = float(result.get("peak_memory_GB", -1))
                    cell.n_repeats = int(result.get("n_repeats", 0))
                    cap_str = (f"cap={cell.capture_time_ms:.1f}ms"
                               if cell.capture_time_ms >= 0 else "cap=-")
                    print(f"  {sys_name:30s} T={T:5d} : OK     "
                          f"{cap_str:20s} steady={cell.steady_inference_ms:8.1f}ms  "
                          f"peak={cell.peak_memory_GB:6.3f}GB  ({elapsed:.1f}s)")

                all_cells.append(asdict(cell))

    with open(args.output + ".json", "w") as f:
        json.dump(all_cells, f, indent=2)
    print(f"\nSaved JSON: {args.output}.json")

    print(f"\n{'='*92}")
    print(f"=== Summary: peak HBM (GB) by (Net, System, T) ===")
    print(f"{'='*92}")
    for name in nets:
        print(f"\n--- {name} ---")
        header = f"{'System':<32s}" + "".join(f"  T={t:<6d}" for t in Ts)
        print(header)
        print("-" * len(header))
        for sys_name in systems:
            row = f"{sys_name:<32s}"
            for T in Ts:
                cell = next((c for c in all_cells
                             if c["net"] == name and c["system"] == sys_name
                             and c["T"] == T), None)
                if cell is None:
                    row += f"  {'?':<8s}"
                elif cell["status"] == "ok":
                    row += f"  {cell['peak_memory_GB']:<8.3f}"
                else:
                    short = "OOM" if "OOM" in cell["status"] else cell["status"][:6]
                    row += f"  {short:<8s}"
            print(row)


if __name__ == "__main__":
    main()