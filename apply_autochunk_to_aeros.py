#!/usr/bin/env python
"""
AEROS — Head-to-head against AutoChunk on stateful temporal models.

Applies ColossalAI's AutoChunk to three AEROS architectures spanning
the SNN family (SR-18, VGG-19-BN) and the extended family (ConvLSTM-2L)
to determine, empirically, what AutoChunk does when it encounters a
stateful operator along the temporal dimension.

The hypothesis (from reading AutoChunk paper §3.3, Equation 4 and
Rule 3 of chunk-flow legality): AutoChunk's chunk-flow algorithm
breaks at the first stateful op it cannot prove output-independent
across chunks.

Empirical finding pathways:
  (a) symbolic_trace explodes (RecursionError / trace failure on
      explicit Python time loops) — AutoChunk cannot even start.
  (b) trace succeeds but unrolls T-axis into O(T) static graph,
      AutoChunkCodeGen returns 0 chunks (no legal chunk found).
  (c) AutoChunk produces a chunk plan that *excludes* the stateful
      op (chunks only inter-stateful sub-regions).

Each is a paper-grade datapoint demonstrating AutoChunk's structural
limit vs AEROS's complementary scope.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Raise recursion limit BEFORE any heavy imports to allow FX trace of
# explicit-time-loop models like ConvLSTM-2L
sys.setrecursionlimit(100000)

import torch
import torch.fx
import torch.nn as nn

# ----------------------------------------------------------------------
# ColossalAI / AutoChunk imports
# ----------------------------------------------------------------------
try:
    import colossalai
    from colossalai.autochunk.autochunk_codegen import (
        AUTOCHUNK_AVAILABLE,
        AutoChunkCodeGen,
    )
    from colossalai.fx.passes.meta_info_prop import MetaInfoProp
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import symbolic_trace
except Exception as e:
    print(f"[FATAL] ColossalAI / AutoChunk unavailable: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# SpikingJelly (for SNN builders)
# ----------------------------------------------------------------------
try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18,
    )
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg19_bn,
    )
    SJ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SpikingJelly unavailable: {e}")
    SJ_AVAILABLE = False


# ======================================================================
# Model builders
# ======================================================================

class SNN_SR18_Wrapper(nn.Module):
    """Wrapper that exposes SpikingJelly's SR-18 with [T, B, C, H, W] input.

    AutoChunk will see `forward(x)` and try to chunk along each dim of x.
    We want it to consider the T (dim 0) axis as a chunk candidate.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = spiking_resnet18(
            spiking_neuron=neuron.LIFNode,
            num_classes=num_classes,
            surrogate_function=surrogate.ATan(),
        )
        try:
            functional.set_step_mode(self.net, "m")
        except Exception:
            pass

    def forward(self, x):
        # x: [T, B, C, H, W]
        return self.net(x)


class SNN_VGG19_Wrapper(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = spiking_vgg19_bn(
            spiking_neuron=neuron.LIFNode,
            num_classes=num_classes,
            surrogate_function=surrogate.ATan(),
        )
        try:
            functional.set_step_mode(self.net, "m")
        except Exception:
            pass

    def forward(self, x):
        return self.net(x)


class ConvLSTMCellLite(nn.Module):
    """Minimal ConvLSTM cell (no SJ dependency)."""
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch,
                              kernel_size, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        # x: [B, C, H, W]; h, c: [B, hid, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM2L_Wrapper(nn.Module):
    """2-layer ConvLSTM with explicit T loop, [T, B, C, H, W] input."""
    def __init__(self, in_ch=3, hid_ch=16, num_classes=10, H=32):
        super().__init__()
        self.cell1 = ConvLSTMCellLite(in_ch, hid_ch)
        self.cell2 = ConvLSTMCellLite(hid_ch, hid_ch)
        self.head = nn.Linear(hid_ch * H * H, num_classes)
        self.hid_ch = hid_ch

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        h1 = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        c1 = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        h2 = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        c2 = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        for t in range(T):
            h1, c1 = self.cell1(x[t], h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
        out = self.head(h2.flatten(1))
        return out


# ======================================================================
# AutoChunk runner
# ======================================================================

def try_autochunk(model, sample_input, max_memory_mb,
                  arch_name="?", device="cuda:0"):
    """Apply AutoChunk to (model, sample_input). Returns dict with status,
    chunk plan, and diagnostic info."""

    result = {
        "arch": arch_name,
        "input_shape": list(sample_input.shape),
        "max_memory_mb": max_memory_mb,
        "status": "unknown",
        "n_chunks": 0,
        "chunk_infos": [],
        "exception": None,
        "exception_type": None,
        "exception_traceback": None,
    }

    print(f"\n{'=' * 68}")
    print(f"  AutoChunk on {arch_name}")
    print(f"  input shape: {list(sample_input.shape)}")
    print(f"  memory budget: {max_memory_mb} MB")
    print(f"{'=' * 68}")

    try:
        # Step 1: Symbolic trace with meta args
        print(f"  [1/3] symbolic_trace + MetaInfoProp...", flush=True)
        meta_args = {"x": sample_input.to(torch.device("meta"))}
        try:
            meta_graph = symbolic_trace(model, meta_args=meta_args)
        except RecursionError as e:
            result["status"] = "trace_exploded"
            result["exception"] = "RecursionError during symbolic_trace"
            result["exception_type"] = "RecursionError"
            result["recursion_limit"] = sys.getrecursionlimit()
            print(f"       [TRACE EXPLODED] symbolic_trace hit RecursionError"
                  f" (recursion_limit={sys.getrecursionlimit()})")
            print(f"       [VERDICT] AutoChunk cannot even trace this model"
                  f" — explicit Python time loop unrolls past Python's stack")
            return result
        n_nodes = len(list(meta_graph.graph.nodes))
        print(f"       symbolic_trace OK, {n_nodes} graph nodes")
        result["n_graph_nodes"] = n_nodes

        # Step 2: MetaInfoProp
        try:
            interp = MetaInfoProp(meta_graph)
            meta_tensor = MetaTensor(sample_input, fake_device=device)
            interp.propagate(meta_tensor)
            print(f"       MetaInfoProp OK")
        except Exception as e:
            result["status"] = "metainfo_failed"
            result["exception"] = str(e)[:500]
            result["exception_type"] = type(e).__name__
            result["exception_traceback"] = traceback.format_exc()[:2000]
            print(f"       [METAINFO FAILED] {type(e).__name__}: {str(e)[:200]}")
            print(f"       [VERDICT] AutoChunk fails before chunk planning"
                  f" — graph structure incompatible with meta-tensor analysis")
            return result

        # Step 3: AutoChunkCodeGen — chunk-flow legality + plan generation
        print(f"  [2/3] AutoChunkCodeGen (chunk-flow legality check)...",
              flush=True)
        try:
            codegen = AutoChunkCodeGen(
                meta_graph,
                max_memory=max_memory_mb,
                print_mem=False,
                print_progress=False,
                eval_mem=False,
            )
            chunks = codegen.chunk_infos
        except Exception as e:
            result["status"] = "autochunk_failed"
            result["exception"] = str(e)[:500]
            result["exception_type"] = type(e).__name__
            result["exception_traceback"] = traceback.format_exc()[:2000]
            print(f"       [AUTOCHUNK FAILED] {type(e).__name__}: {str(e)[:200]}")
            print(f"       [VERDICT] AutoChunkCodeGen crashed during"
                  f" chunk-flow analysis or plan generation")
            return result

        n_chunks = len(chunks) if chunks else 0
        print(f"       AutoChunkCodeGen produced {n_chunks} chunk plan(s)")
        result["n_chunks"] = n_chunks

        # Extract chunk info for reporting
        for i, c in enumerate(chunks):
            info = {}
            for key in ("region", "chunk_size", "chunk_dim", "n_dim"):
                if key in c:
                    val = c[key]
                    info[key] = (val if isinstance(val, (int, float, str, list, tuple))
                                 else str(val))
            result["chunk_infos"].append(info)
            print(f"       chunk[{i}]: {info}")

        if n_chunks == 0:
            result["status"] = "no_chunk_found"
            print(f"       [VERDICT] No legal chunk found — model REJECTED"
                  f" by chunk-flow legality check")
        else:
            result["status"] = "ok"
            print(f"       [VERDICT] {n_chunks} chunk plan(s) generated"
                  f" — partially or fully accepted")

    except Exception as e:
        # Catch-all for anything we didn't specifically classify above
        result["status"] = "unclassified_exception"
        result["exception"] = str(e)[:500]
        result["exception_type"] = type(e).__name__
        result["exception_traceback"] = traceback.format_exc()[:2000]
        print(f"       [UNCLASSIFIED EXCEPTION] {type(e).__name__}: {e}")

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=128,
                        help="number of timesteps")
    parser.add_argument("--B", type=int, default=4,
                        help="batch size (kept small to avoid OOM during trace)")
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--H_snn", type=int, default=128,
                        help="spatial size for SNN models")
    parser.add_argument("--H_extended", type=int, default=32,
                        help="spatial size for ConvLSTM")
    parser.add_argument("--max_memory_mb", type=int, default=8192,
                        help="memory budget passed to AutoChunk (MB)")
    parser.add_argument("--archs", type=str, nargs="+",
                        default=["SR-18", "VGG-19-BN", "ConvLSTM-2L"])
    parser.add_argument("--output", default="autochunk_aeros_headtohead.json")
    args = parser.parse_args()

    # Reduce log noise
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 78)
    print("AEROS — Head-to-head against AutoChunk")
    print("=" * 78)
    print(f"  archs:       {args.archs}")
    print(f"  T={args.T}, B={args.B}, C={args.C}")
    print(f"  H (SNN/ext): {args.H_snn}/{args.H_extended}")
    print(f"  max_memory:  {args.max_memory_mb} MB")
    print(f"  AUTOCHUNK_AVAILABLE: {AUTOCHUNK_AVAILABLE}")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda: {torch.cuda.is_available()}")
    if not AUTOCHUNK_AVAILABLE:
        print("[FATAL] AUTOCHUNK_AVAILABLE is False — bailing out")
        sys.exit(1)
    if not SJ_AVAILABLE:
        print("[WARN] SpikingJelly unavailable; SNN archs will be skipped")

    results = []

    for arch in args.archs:
        try:
            if arch == "SR-18":
                if not SJ_AVAILABLE:
                    print(f"[skip] {arch}: SJ unavailable")
                    continue
                model = SNN_SR18_Wrapper(num_classes=10)
                sample = torch.randn(args.T, args.B, args.C,
                                      args.H_snn, args.H_snn)
            elif arch == "VGG-19-BN":
                if not SJ_AVAILABLE:
                    print(f"[skip] {arch}: SJ unavailable")
                    continue
                model = SNN_VGG19_Wrapper(num_classes=10)
                sample = torch.randn(args.T, args.B, args.C,
                                      args.H_snn, args.H_snn)
            elif arch == "ConvLSTM-2L":
                model = ConvLSTM2L_Wrapper(in_ch=args.C, hid_ch=16,
                                            num_classes=10,
                                            H=args.H_extended)
                sample = torch.randn(args.T, args.B, args.C,
                                      args.H_extended, args.H_extended)
            else:
                print(f"[skip] unknown arch: {arch}")
                continue

            model.eval()
            r = try_autochunk(model, sample, args.max_memory_mb,
                               arch_name=arch)
            results.append(r)
        except Exception as e:
            results.append({
                "arch": arch,
                "status": "outer_exception",
                "exception": str(e)[:500],
                "exception_type": type(e).__name__,
                "exception_traceback": traceback.format_exc()[:2000],
            })
            print(f"\n[OUTER EXCEPTION on {arch}] {type(e).__name__}: {e}")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 78)
    print("AEROS-vs-AutoChunk head-to-head summary")
    print("=" * 78)
    print(f"{'Arch':<14} {'Status':<22} {'NodesGraph':<11} {'Chunks':<8} {'Verdict'}")
    print("-" * 100)
    for r in results:
        arch = r.get("arch", "?")
        status = r.get("status", "?")
        n_chunks = r.get("n_chunks", 0)
        n_nodes = r.get("n_graph_nodes", -1)
        if status == "ok" and n_chunks > 0:
            verdict = f"ACCEPTED ({n_chunks} chunk plans)"
        elif status == "no_chunk_found":
            verdict = "REJECTED at chunk-flow legality"
        elif status == "trace_exploded":
            verdict = "TRACE EXPLODED (RecursionError)"
        elif status == "metainfo_failed":
            etype = r.get("exception_type", "?")
            verdict = f"META-INFO FAILED ({etype})"
        elif status == "autochunk_failed":
            etype = r.get("exception_type", "?")
            verdict = f"AUTOCHUNK CRASHED ({etype})"
        elif status == "unclassified_exception":
            etype = r.get("exception_type", "?")
            verdict = f"UNCLASSIFIED ({etype})"
        else:
            verdict = status
        n_nodes_str = str(n_nodes) if n_nodes >= 0 else "--"
        print(f"{arch:<14} {status:<22} {n_nodes_str:<11} {n_chunks:<8} {verdict}")

    # Save
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results},
                  f, indent=2, default=str)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()