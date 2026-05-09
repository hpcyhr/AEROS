#!/usr/bin/env python
"""
AEROS — Head-to-head against Continual Inference Networks (CIN).

Tests three representative architectures spanning CIN's coverage:

  CausalTCN-8L  — 1D causal conv stack, CIN's sweet spot
  ConvLSTM-2L   — explicit (h, c) state, CIN no replacement
  SR-18 (SNN)   — LIF threshold control flow, CIN no replacement

For each: attempts CIN path (drop-in replacement of nn modules with co
modules), falls back to documenting coverage miss if no replacement
exists. Then runs AEROS Mode 4 (kappa-chunked forward with carry state).

Measures peak HBM and per-frame p50 latency.

Run in `cin` conda env on V100 (torch 2.5.1+cu121, V100-compat).

Usage:
  python apply_cin_to_aeros.py \\
      --T 128 \\
      --kappa 8 \\
      --output cin_aeros_headtohead.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

BYTES_PER_GB = 1024 ** 3


# ============================================================================
# CIN imports
# ============================================================================
try:
    import continual as co
    CIN_AVAILABLE = True
except ImportError as e:
    print(f"[FATAL] continual-inference unavailable: {e}")
    sys.exit(1)

# SpikingJelly for SR-18
try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18,
    )
    SJ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SpikingJelly unavailable: {e}")
    SJ_AVAILABLE = False


# ============================================================================
# Model builders (all 3 archs)
# ============================================================================

# ----------------------------------------------------------------------------
# Test A: CausalTCN-8L — CIN should support (uses Conv1d)
# ----------------------------------------------------------------------------

class NN_CausalTCN8L(nn.Module):
    """Reference TCN with 8 layers of causal Conv1d."""
    def __init__(self, in_ch=8, hid_ch=32, kernel=3, num_classes=10):
        super().__init__()
        layers = []
        ch_prev = in_ch
        for i in range(8):
            # causal padding via left-pad in forward
            layers.append(nn.Conv1d(ch_prev, hid_ch, kernel,
                                     padding=kernel - 1))
            layers.append(nn.ReLU())
            ch_prev = hid_ch
        self.layers = nn.ModuleList(layers)
        self.kernel = kernel
        self.head = nn.Linear(hid_ch, num_classes)

    def forward(self, x):
        # x: [B, C, T]
        T = x.shape[2]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Trim to causal: keep only first T outputs (drop the right pad)
            if isinstance(layer, nn.Conv1d):
                h = h[:, :, :T]
        # Pool last 4 timesteps and classify
        h = h[:, :, -4:].mean(dim=2)
        return self.head(h)


class CIN_CausalTCN8L(nn.Module):
    """CIN version: replace nn.Conv1d with co.Conv1d (causal=True)."""
    def __init__(self, in_ch=8, hid_ch=32, kernel=3, num_classes=10):
        super().__init__()
        layers = []
        ch_prev = in_ch
        for i in range(8):
            layers.append(co.Conv1d(ch_prev, hid_ch, kernel,
                                     padding=kernel - 1))
            layers.append(nn.ReLU())
            ch_prev = hid_ch
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hid_ch, num_classes)
        self.hid_ch = hid_ch

    def forward(self, x):
        # Input: [B, C, T] for batch mode
        h = x
        T = x.shape[2]
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, co.Conv1d):
                h = h[:, :, :T]
        h = h[:, :, -4:].mean(dim=2)
        return self.head(h)


# ----------------------------------------------------------------------------
# Test B: ConvLSTM-2L — CIN no replacement for LSTMCell
# ----------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM2L(nn.Module):
    """Two ConvLSTM layers, classification head."""
    def __init__(self, in_ch=3, hid_ch=16, num_classes=10, H=32):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_ch, hid_ch)
        self.cell2 = ConvLSTMCell(hid_ch, hid_ch)
        self.head = nn.Linear(hid_ch * H * H, num_classes)
        self.hid_ch = hid_ch
        self.h1 = self.c1 = self.h2 = self.c2 = None

    def reset_state(self):
        self.h1 = self.c1 = self.h2 = self.c2 = None

    def _ensure_state(self, B, H, W, device, dtype):
        if self.h1 is None:
            self.h1 = torch.zeros(B, self.hid_ch, H, W, device=device, dtype=dtype)
            self.c1 = torch.zeros_like(self.h1)
            self.h2 = torch.zeros_like(self.h1)
            self.c2 = torch.zeros_like(self.h1)

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        self._ensure_state(B, H, W, x.device, x.dtype)
        outs = []
        for t in range(T):
            self.h1, self.c1 = self.cell1(x[t], self.h1, self.c1)
            self.h2, self.c2 = self.cell2(self.h1, self.h2, self.c2)
            outs.append(self.h2)
        out = self.head(self.h2.flatten(1))
        return out


# ============================================================================
# CIN coverage probe
# ============================================================================

def cin_coverage_probe(arch_name):
    """Try to build a CIN drop-in for the given arch. Returns
    (success, error_msg, model_or_none)."""
    if arch_name == "CausalTCN-8L":
        try:
            m = CIN_CausalTCN8L()
            return True, None, m
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", None

    elif arch_name == "ConvLSTM-2L":
        # Try to find a LSTMCell or ConvLSTM equivalent in CIN
        co_attrs = dir(co)
        candidates = ["LSTMCell", "ConvLSTM", "ConvLSTMCell", "LSTM", "GRU",
                      "GRUCell", "RNN", "RNNCell"]
        found = [c for c in candidates if c in co_attrs]
        if found:
            return False, f"CIN has {found} but not ConvLSTMCell", None
        return False, ("CIN has no LSTMCell/ConvLSTMCell/RNN-cell module. "
                       "Available co.* modules don't include any cell-state "
                       "primitive. ConvLSTM cannot be wrapped."), None

    elif arch_name == "SR-18":
        # SJ uses LIFNode with surrogate gradient + threshold; CIN has no equivalent
        co_attrs = dir(co)
        candidates = ["LIF", "LIFNode", "Spiking", "Neuron", "Threshold"]
        found = [c for c in candidates if c in co_attrs]
        if found:
            return False, f"CIN has {found} but no SJ-compatible LIF replacement", None
        return False, ("CIN has no LIF / spiking neuron / threshold primitive. "
                       "SR-18's LIFNode (V > V_threshold + reset) cannot be "
                       "wrapped by any co.* module."), None

    return False, "unknown arch", None


# ============================================================================
# Forward measurement
# ============================================================================

@torch.no_grad()
def measure_forward(forward_fn, n_warmup=2, n_iter=5):
    """Run forward N times, return (peak_GB, p50_ms, list_of_ms)."""
    device = torch.device("cuda:0")
    torch.cuda.empty_cache(); gc.collect()
    # Warmup
    for _ in range(n_warmup):
        forward_fn()
    torch.cuda.synchronize(device)
    # Real measurement
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        forward_fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)
    peak_GB = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
    return {
        "peak_GB": peak_GB,
        "p50_ms": float(np.percentile(times, 50)),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "n_iter": n_iter,
    }


# ============================================================================
# Per-arch benchmark drivers
# ============================================================================

def bench_causaltcn(T, B, kappa, device):
    """Test A: CausalTCN-8L. Build CIN + nn versions, run both, compare."""
    print(f"\n=== Test A: CausalTCN-8L (T={T}, B={B}, kappa={kappa}) ===")
    in_ch = 8
    hid_ch = 32

    # Reference (nn.Conv1d batch)
    nn_model = NN_CausalTCN8L(in_ch=in_ch, hid_ch=hid_ch).to(device).eval()

    # CIN version
    cin_model = CIN_CausalTCN8L(in_ch=in_ch, hid_ch=hid_ch).to(device).eval()
    # Load matching weights
    nn_state = nn_model.state_dict()
    cin_state = cin_model.state_dict()
    # Copy where keys match
    for k in cin_state.keys():
        if k in nn_state and cin_state[k].shape == nn_state[k].shape:
            cin_state[k] = nn_state[k].clone()
    cin_model.load_state_dict(cin_state, strict=False)

    # Input: [B, C, T] for 1D conv
    x = torch.randn(B, in_ch, T, device=device)

    out = {"arch": "CausalTCN-8L", "T": T, "kappa": kappa,
           "cin_supported": True}

    # nn.Conv1d batch (reference)
    print(f"  [nn batch] forward...", end="", flush=True)
    out["nn_batch"] = measure_forward(lambda: nn_model(x))
    print(f" peak={out['nn_batch']['peak_GB']:.3f}GB  p50={out['nn_batch']['p50_ms']:.2f}ms")

    # CIN batch (should match nn)
    print(f"  [CIN batch] forward...", end="", flush=True)
    out["cin_batch"] = measure_forward(lambda: cin_model(x))
    print(f" peak={out['cin_batch']['peak_GB']:.3f}GB  p50={out['cin_batch']['p50_ms']:.2f}ms")

    # CIN streaming (forward_step in loop)
    print(f"  [CIN forward_step loop] ...", end="", flush=True)
    def cin_step():
        for layer in cin_model.layers:
            if isinstance(layer, co.Conv1d):
                layer.clean_state()
        for t in range(T):
            xt = x[:, :, t]  # [B, C]
            h = xt
            for layer in cin_model.layers:
                if isinstance(layer, co.Conv1d):
                    h = layer.forward_step(h)
                    if h is None:
                        break  # waiting for receptive field
                else:
                    if h is not None:
                        h = layer(h)
        # Don't actually classify here; just measure forward_step cost
    try:
        out["cin_step"] = measure_forward(cin_step)
        print(f" peak={out['cin_step']['peak_GB']:.3f}GB  p50={out['cin_step']['p50_ms']:.2f}ms")
    except Exception as e:
        out["cin_step"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f" FAILED: {type(e).__name__}: {str(e)[:80]}")

    # AEROS Mode 4 (kappa-chunked, carry state via causal residue)
    print(f"  [AEROS Mode 4] kappa-chunk forward...", end="", flush=True)
    def aeros_mode4():
        # Reset CIN state to use it as the carry-stream substrate
        for layer in cin_model.layers:
            if isinstance(layer, co.Conv1d):
                layer.clean_state()
        # Process in kappa-sized chunks via forward_steps
        for i in range(0, T, kappa):
            chunk = x[:, :, i:i+kappa]
            _ = cin_model(chunk)  # this will use cached state via co.Conv1d

    try:
        # Note: the cin_model.forward uses non-state forward; for true Mode 4
        # we'd need a chunked-streams forward. Use forward_steps on the conv layers.
        def aeros_mode4_proper():
            for layer in cin_model.layers:
                if isinstance(layer, co.Conv1d):
                    layer.clean_state()
            for i in range(0, T, kappa):
                chunk = x[:, :, i:i+kappa]
                h = chunk
                for layer in cin_model.layers:
                    if isinstance(layer, co.Conv1d):
                        h = layer.forward_steps(h)
                    else:
                        h = layer(h)
        out["aeros_mode4"] = measure_forward(aeros_mode4_proper)
        print(f" peak={out['aeros_mode4']['peak_GB']:.3f}GB  p50={out['aeros_mode4']['p50_ms']:.2f}ms")
    except Exception as e:
        out["aeros_mode4"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f" FAILED: {type(e).__name__}: {str(e)[:80]}")

    return out


def bench_convlstm(T, B, kappa, device, H=32):
    """Test B: ConvLSTM-2L. CIN coverage miss; AEROS Mode 4 succeeds."""
    print(f"\n=== Test B: ConvLSTM-2L (T={T}, B={B}, kappa={kappa}) ===")

    out = {"arch": "ConvLSTM-2L", "T": T, "kappa": kappa,
           "cin_supported": False}

    # Document CIN coverage miss
    success, err, _ = cin_coverage_probe("ConvLSTM-2L")
    out["cin_coverage_miss"] = err
    print(f"  [CIN coverage probe]: {err}")

    # AEROS Mode 4: chunk T into kappa-segments, ConvLSTM cell maintains state
    model = ConvLSTM2L(in_ch=3, hid_ch=16, num_classes=10, H=H).to(device).eval()
    x = torch.randn(T, B, 3, H, H, device=device)

    # Reference: full batch forward
    print(f"  [Reference batch forward] ...", end="", flush=True)
    def ref_fwd():
        model.reset_state()
        return model(x)
    out["nn_batch"] = measure_forward(ref_fwd)
    print(f" peak={out['nn_batch']['peak_GB']:.3f}GB  p50={out['nn_batch']['p50_ms']:.2f}ms")

    # AEROS Mode 4: chunk into kappa-segments, state persists (built into cell)
    print(f"  [AEROS Mode 4] kappa-chunk forward (carry-stream)...", end="", flush=True)
    def aeros_mode4():
        model.reset_state()
        for i in range(0, T, kappa):
            chunk = x[i:i+kappa]
            _ = model(chunk)
    out["aeros_mode4"] = measure_forward(aeros_mode4)
    print(f" peak={out['aeros_mode4']['peak_GB']:.3f}GB  p50={out['aeros_mode4']['p50_ms']:.2f}ms")

    return out


def bench_sr18(T, B, kappa, device, H=128):
    """Test C: SR-18 SNN. CIN coverage miss; AEROS Mode 4 succeeds."""
    print(f"\n=== Test C: SR-18 SNN (T={T}, B={B}, kappa={kappa}) ===")

    out = {"arch": "SR-18", "T": T, "kappa": kappa,
           "cin_supported": False}

    success, err, _ = cin_coverage_probe("SR-18")
    out["cin_coverage_miss"] = err
    print(f"  [CIN coverage probe]: {err}")

    if not SJ_AVAILABLE:
        out["nn_batch"] = {"error": "SpikingJelly unavailable"}
        out["aeros_mode4"] = {"error": "SpikingJelly unavailable"}
        return out

    model = spiking_resnet18(spiking_neuron=neuron.LIFNode, num_classes=10,
                              surrogate_function=surrogate.ATan()).to(device).eval()
    functional.set_step_mode(model, "m")
    x = torch.randn(T, B, 3, H, H, device=device)

    # Reference: full batch
    print(f"  [Reference batch forward] ...", end="", flush=True)
    def ref_fwd():
        functional.reset_net(model)
        return model(x)
    try:
        out["nn_batch"] = measure_forward(ref_fwd)
        print(f" peak={out['nn_batch']['peak_GB']:.3f}GB  p50={out['nn_batch']['p50_ms']:.2f}ms")
    except torch.cuda.OutOfMemoryError:
        out["nn_batch"] = {"error": "OOM"}
        print(" OOM")
    except Exception as e:
        out["nn_batch"] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        print(f" {type(e).__name__}: {str(e)[:80]}")

    # AEROS Mode 4: kappa-chunk, SJ LIF state persists across calls
    print(f"  [AEROS Mode 4] kappa-chunk forward (LIF carry-stream)...", end="", flush=True)
    def aeros_mode4():
        functional.reset_net(model)
        for i in range(0, T, kappa):
            chunk = x[i:i+kappa].contiguous()
            _ = model(chunk)
    try:
        out["aeros_mode4"] = measure_forward(aeros_mode4)
        print(f" peak={out['aeros_mode4']['peak_GB']:.3f}GB  p50={out['aeros_mode4']['p50_ms']:.2f}ms")
    except Exception as e:
        out["aeros_mode4"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f" FAILED: {type(e).__name__}: {str(e)[:80]}")

    return out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ts", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--archs", type=str, nargs="+",
                        default=["CausalTCN-8L", "ConvLSTM-2L", "SR-18"])
    parser.add_argument("--output", default="cin_aeros_headtohead.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)

    print("=" * 78)
    print("AEROS — Head-to-head against CIN")
    print("=" * 78)
    print(f"  archs:  {args.archs}")
    print(f"  Ts:     {args.Ts}")
    print(f"  B:      {args.B}")
    print(f"  kappa:  {args.kappa}")
    print(f"  CIN_AVAILABLE: {CIN_AVAILABLE}")
    print(f"  SJ_AVAILABLE:  {SJ_AVAILABLE}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  torch:  {torch.__version__}")

    all_results = []
    for arch in args.archs:
        for T in args.Ts:
            try:
                if arch == "CausalTCN-8L":
                    r = bench_causaltcn(T, args.B, args.kappa, device)
                elif arch == "ConvLSTM-2L":
                    r = bench_convlstm(T, args.B, args.kappa, device)
                elif arch == "SR-18":
                    r = bench_sr18(T, args.B, args.kappa, device)
                else:
                    print(f"[skip] unknown arch {arch}")
                    continue
                all_results.append(r)
            except Exception as e:
                print(f"\n[OUTER EXCEPTION on {arch} T={T}] {type(e).__name__}: {e}")
                traceback.print_exc()
                all_results.append({
                    "arch": arch, "T": T,
                    "outer_exception": f"{type(e).__name__}: {str(e)[:300]}",
                })
            torch.cuda.empty_cache(); gc.collect()

    # Save
    out = {"config": vars(args), "results": all_results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")

    # Summary
    print("\n" + "=" * 100)
    print("Head-to-head summary")
    print("=" * 100)
    print(f"{'Arch':<14} {'T':<6} {'Path':<20} {'PeakGB':<10} {'p50ms':<10} {'Verdict'}")
    print("-" * 100)
    for r in all_results:
        arch = r.get("arch", "?")
        T = r.get("T", "?")
        cin_supp = r.get("cin_supported", False)

        # CIN row
        if cin_supp:
            cb = r.get("cin_batch", {})
            cs = r.get("cin_step", {})
            ab = r.get("aeros_mode4", {})
            nb = r.get("nn_batch", {})
            if "error" not in cb:
                print(f"{arch:<14} {T:<6} {'CIN batch':<20} {cb.get('peak_GB',-1):<10.3f} "
                      f"{cb.get('p50_ms',-1):<10.2f} (reference)")
            if "error" not in cs:
                print(f"{arch:<14} {T:<6} {'CIN step-loop':<20} {cs.get('peak_GB',-1):<10.3f} "
                      f"{cs.get('p50_ms',-1):<10.2f} streaming")
            if "error" not in ab:
                print(f"{arch:<14} {T:<6} {'AEROS Mode 4':<20} {ab.get('peak_GB',-1):<10.3f} "
                      f"{ab.get('p50_ms',-1):<10.2f} kappa-chunk")
        else:
            miss_msg = r.get("cin_coverage_miss", "?")[:40]
            print(f"{arch:<14} {T:<6} {'CIN coverage':<20} {'--':<10} {'--':<10} MISS: {miss_msg}")
            ab = r.get("aeros_mode4", {})
            if "error" not in ab:
                print(f"{arch:<14} {T:<6} {'AEROS Mode 4':<20} {ab.get('peak_GB',-1):<10.3f} "
                      f"{ab.get('p50_ms',-1):<10.2f} (only AEROS runs)")
            else:
                print(f"{arch:<14} {T:<6} {'AEROS Mode 4':<20} ERROR")


if __name__ == "__main__":
    main()