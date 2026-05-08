#!/usr/bin/env python
"""
AEROS — comparison against SpikingJelly torch and cupy backends.

For chosen SNN architectures, sweep T and measure peak HBM + per-iteration
latency under THREE backends:

  1. SJ torch backend (default, eager LIF over time)
  2. SJ cupy backend (fused multi-step LIF kernel via CuPy)
  3. AEROS Mode 4 (κ-segment carry-stream forward; uses default torch
     LIF node, but executes in segments of size κ with input streamed
     from CPU)

The comparison is intentionally honest:
  - At small T (≤256-1024): SJ cupy should win on latency (fused kernel
    + no segment overhead). AEROS M4 may be 1.1-2× slower, but peak
    HBM is comparable.
  - At large T (≥2048): SJ cupy and SJ torch should OOM (full T-step
    activation tensor exceeds 32 GB). AEROS M4 should sustain.

Output: JSON + console table.

Usage:
  python p9_compare_sj_backends.py \\
      --archs SR-18 SEW-18 VGG-19-BN \\
      --Ts 128 256 512 1024 2048 4096 \\
      --kappa 8 \\
      --b 32 --H 128 \\
      --output p9_compare_sj_backends
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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
    print(f"[FATAL] SJ unavailable: {e}")
    sys.exit(1)


BYTES_PER_GB = 1024 ** 3

ARCH_BUILDERS = {
    "SR-18":     lambda nc: spiking_resnet18(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "SR-34":     lambda nc: spiking_resnet34(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "SR-50":     lambda nc: spiking_resnet50(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "SEW-18":    lambda nc: sew_resnet18(spiking_neuron=neuron.LIFNode,
                                          num_classes=nc, cnf="ADD"),
    "SEW-50":    lambda nc: sew_resnet50(spiking_neuron=neuron.LIFNode,
                                          num_classes=nc, cnf="ADD"),
    "SEW-101":   lambda nc: sew_resnet101(spiking_neuron=neuron.LIFNode,
                                           num_classes=nc, cnf="ADD"),
    "VGG-11-BN": lambda nc: spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "VGG-13-BN": lambda nc: spiking_vgg13_bn(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "VGG-16-BN": lambda nc: spiking_vgg16_bn(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
    "VGG-19-BN": lambda nc: spiking_vgg19_bn(spiking_neuron=neuron.LIFNode,
                                              num_classes=nc),
}


def build_net(arch_name, num_classes=10):
    return ARCH_BUILDERS[arch_name](num_classes)


def reset_state(net):
    try:
        functional.reset_net(net)
    except Exception:
        pass


def set_step_mode_m(net):
    """Configure net for multi-step (T-major) forward."""
    try:
        functional.set_step_mode(net, "m")
    except Exception as e:
        print(f"[WARN] set_step_mode('m') failed: {e}")


def set_backend(net, backend_name):
    """Switch all LIFNodes (and similar neurons) to the requested backend.

    backend_name: 'torch' or 'cupy'
    """
    if backend_name == "torch":
        try:
            functional.set_backend(net, "torch")
        except Exception as e:
            # Fallback: per-node setattr
            for m in net.modules():
                if hasattr(m, "backend"):
                    try:
                        m.backend = "torch"
                    except Exception:
                        pass
        return
    if backend_name == "cupy":
        try:
            functional.set_backend(net, "cupy")
        except Exception as e:
            print(f"[WARN] functional.set_backend('cupy') failed: {e}")
            for m in net.modules():
                if hasattr(m, "backend"):
                    try:
                        m.backend = "cupy"
                    except Exception:
                        pass
        return
    raise ValueError(f"unknown backend {backend_name}")


# ============================================================================
# Forward implementations
# ============================================================================

@torch.no_grad()
def forward_sj_full(net, x, device):
    """SJ multi-step forward: x is [T, B, C, H, W], net runs in step_mode='m'.

    Returns y of shape [T, B, num_classes].
    """
    if x.device != device:
        x = x.to(device, non_blocking=True)
    reset_state(net)
    return net(x)


@torch.no_grad()
def forward_aeros_mode4(net, x, kappa, device):
    """AEROS Mode 4: input on CPU, segmented forward, output sink (mean).

    Returns y of shape [B, num_classes].
    """
    assert x.device.type == "cpu", \
        f"AEROS Mode 4 requires CPU x, got {x.device}"
    T = x.shape[0]
    reset_state(net)
    sink = None
    n = 0
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
        del x_seg, y_seg
        i += sz
    return sink / n


# ============================================================================
# Measurement: peak HBM + latency p50/p95
# ============================================================================

@torch.no_grad()
def measure(forward_fn, net, x_or_x_cpu, device, n_warmup=2, n_iters=5,
             label=""):
    """Run forward_fn several times. Return {peak_GB, p50_ms, p95_ms, mean_ms}.

    forward_fn signature: (net, x, device) -> output.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup (not measured)
    for _ in range(n_warmup):
        try:
            _ = forward_fn(net, x_or_x_cpu, device)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                    "mean_ms": -1.0, "status": "oom"}
        except Exception as e:
            return {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                    "mean_ms": -1.0, "status": f"error:{type(e).__name__}",
                    "err_str": str(e)[:200]}

    # Measured iters
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        for _ in range(n_iters):
            torch.cuda.synchronize(device)
            t0 = time.time()
            _ = forward_fn(net, x_or_x_cpu, device)
            torch.cuda.synchronize(device)
            times.append((time.time() - t0) * 1000)
        peak = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
        return {
            "peak_GB": peak,
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(np.mean(times)),
            "n_iters": n_iters,
            "status": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                "mean_ms": -1.0, "status": "oom"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                "mean_ms": -1.0, "status": f"error:{type(e).__name__}",
                "err_str": str(e)[:200]}


def run_one_cell(arch_name, T, kappa, b, C, H, num_classes,
                  device, n_warmup=2, n_iters=5):
    """Run all 3 backends on one (arch, T) cell.

    Returns dict mapping backend -> measurement dict.
    """
    out = {"arch": arch_name, "T": T, "kappa": kappa, "b": b, "C": C, "H": H}

    # ---- 1. SJ torch backend (full forward) ----
    print(f"  [SJ-torch] arch={arch_name} T={T}", end="", flush=True)
    net = build_net(arch_name, num_classes).to(device)
    set_step_mode_m(net)
    set_backend(net, "torch")
    net.eval()
    try:
        x = torch.randn(T, b, C, H, H, device=device)
        m_sjt = measure(forward_sj_full, net, x, device, n_warmup, n_iters)
        del x
    except torch.cuda.OutOfMemoryError:
        m_sjt = {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                 "mean_ms": -1.0, "status": "oom_alloc_input"}
    out["sj_torch"] = m_sjt
    print(f"  -> peak={m_sjt['peak_GB']:.2f}GB p50={m_sjt['p50_ms']:.0f}ms"
          f" status={m_sjt['status']}")
    del net
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 2. SJ cupy backend (full forward) ----
    print(f"  [SJ-cupy ] arch={arch_name} T={T}", end="", flush=True)
    net = build_net(arch_name, num_classes).to(device)
    set_step_mode_m(net)
    set_backend(net, "cupy")
    net.eval()
    try:
        x = torch.randn(T, b, C, H, H, device=device)
        m_sjc = measure(forward_sj_full, net, x, device, n_warmup, n_iters)
        del x
    except torch.cuda.OutOfMemoryError:
        m_sjc = {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                 "mean_ms": -1.0, "status": "oom_alloc_input"}
    out["sj_cupy"] = m_sjc
    print(f"  -> peak={m_sjc['peak_GB']:.2f}GB p50={m_sjc['p50_ms']:.0f}ms"
          f" status={m_sjc['status']}")
    del net
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 3. AEROS Mode 4 (segmented, input-streaming) ----
    print(f"  [AEROS-M4] arch={arch_name} T={T} kappa={kappa}",
          end="", flush=True)
    net = build_net(arch_name, num_classes).to(device)
    set_step_mode_m(net)
    set_backend(net, "torch")
    net.eval()
    try:
        x_cpu = torch.randn(T, b, C, H, H, device="cpu")
        m_aeros = measure(
            lambda n, xx, d: forward_aeros_mode4(n, xx, kappa, d),
            net, x_cpu, device, n_warmup, n_iters)
        del x_cpu
    except (RuntimeError, MemoryError) as e:
        m_aeros = {"peak_GB": -1.0, "p50_ms": -1.0, "p95_ms": -1.0,
                   "mean_ms": -1.0, "status": "oom_cpu_alloc",
                   "err_str": str(e)[:200]}
    out["aeros_m4"] = m_aeros
    print(f"  -> peak={m_aeros['peak_GB']:.2f}GB"
          f" p50={m_aeros['p50_ms']:.0f}ms status={m_aeros['status']}")
    del net
    torch.cuda.empty_cache()
    gc.collect()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", type=str, nargs="+",
                        default=["SR-18", "SEW-18", "VGG-19-BN"])
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_iters", type=int, default=5)
    parser.add_argument("--output", default="p9_compare_sj_backends")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)

    # Sanity: probe cupy availability by loading a small net
    print("=" * 78)
    print("Probing SJ cupy backend availability...")
    try:
        probe = build_net("SR-18", args.num_classes).to(device)
        set_step_mode_m(probe)
        set_backend(probe, "cupy")
        probe.eval()
        x_probe = torch.randn(2, 1, 3, 32, 32, device=device)
        _ = probe(x_probe)
        torch.cuda.synchronize(device)
        print("  cupy backend is available and works")
        del probe, x_probe
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [WARN] cupy backend probe failed: {type(e).__name__}: {e}")
        print(f"  Continuing; sj_cupy cells may all show 'error' status.")

    print()
    print("=" * 78)
    print("AEROS vs SpikingJelly torch / cupy backends")
    print("=" * 78)
    print(f"  archs: {args.archs}")
    print(f"  Ts: {args.Ts}")
    print(f"  kappa (AEROS M4): {args.kappa}")
    print(f"  b={args.b}, C={args.C}, H={args.H}")
    print(f"  iters: {args.n_iters} (warmup {args.n_warmup})")
    print()

    cells = []
    for arch in args.archs:
        for T in args.Ts:
            print(f"\n=== {arch} T={T} ===")
            cell = run_one_cell(arch, T, args.kappa, args.b, args.C, args.H,
                                  args.num_classes, device,
                                  args.n_warmup, args.n_iters)
            cells.append(cell)

    # ---- Summary table ----
    print()
    print("=" * 110)
    print(f"{'Arch':<12} {'T':>6} | {'SJ-torch peak':>14} {'SJ-torch p50':>13} | "
          f"{'SJ-cupy peak':>14} {'SJ-cupy p50':>13} | "
          f"{'AEROS peak':>11} {'AEROS p50':>10}")
    print("-" * 110)
    for c in cells:
        s = c["sj_torch"]
        cu = c["sj_cupy"]
        a = c["aeros_m4"]

        def fmt(d, key, suffix=""):
            if d.get("status") == "ok":
                return f"{d[key]:.2f}{suffix}"
            elif d.get("status", "").startswith("oom"):
                return "OOM"
            else:
                return "ERR"

        print(f"{c['arch']:<12} {c['T']:>6} | "
              f"{fmt(s, 'peak_GB', 'GB'):>14} {fmt(s, 'p50_ms', 'ms'):>13} | "
              f"{fmt(cu, 'peak_GB', 'GB'):>14} {fmt(cu, 'p50_ms', 'ms'):>13} | "
              f"{fmt(a, 'peak_GB', 'GB'):>11} {fmt(a, 'p50_ms', 'ms'):>10}")

    output = {
        "config": vars(args),
        "cells": cells,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()