#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — DVS128 Gesture inference under AEROS modes.

Loads trained DVSGestureNet checkpoint and runs inference under four modes
on the full DVS128 Gesture test set (288 samples, T=16, b=B):

    Mode 1: full-horizon baseline (kappa=T, single segment)
    Mode 2: retained-IO segmented (kappa<T, retain x and y)
    Mode 3: input-stream (kappa<T, stream x, retain y)
    Mode 4: IO-stream + sink (kappa<T, stream x, sink y)

For each mode we measure:
    - top-1 accuracy on test set (must be identical to baseline)
    - peak HBM (torch.cuda.max_memory_allocated)
    - per-sample wall-clock latency (median of repeats)
    - bit-exact comparison against Mode 1 (max_abs_err)

This is the Suite B real-event-camera anchor for Doris 9/10 P0-1.

Usage:
    python p9_12_dvs_inference.py \\
        --ckpt /data/yhr/AEROS/checkpoints/dvs128_gesture_best.pth \\
        --data_root /data/yhr/AEROS/DVS128Gesture \\
        --output p9_12_dvs_inference --kappa 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ModeResult:
    mode: str
    kappa: int
    top1_acc: float
    n_samples: int
    peak_memory_GB: float
    median_latency_ms: float
    p95_latency_ms: float
    max_abs_err_vs_baseline: float = -1.0
    pred_agreement_vs_baseline: float = -1.0
    n_compared: int = 0


def reset_state(net):
    from spikingjelly.activation_based import functional
    try: functional.reset_net(net)
    except Exception: pass


@torch.no_grad()
def forward_mode_1(net, x):
    """Mode 1: kappa=T, single forward."""
    reset_state(net)
    return net(x)  # [T, B, 11]


@torch.no_grad()
def forward_mode_2(net, x, kappa):
    """Mode 2: retain input x AND output, segmented forward.
    
    Both x and full y matrix are live in HBM. Splits computation into
    kappa-wide segments to bound segment-internal activation.
    """
    reset_state(net)
    chunks = []
    T = x.shape[0]
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        chunks.append(net(x[i:i+sz]))
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def forward_mode_3(net, x, kappa):
    """Mode 3: input-stream + retain output. Input arrives segment-at-a-time
    (we simulate this by slicing x inside the loop); output retained."""
    reset_state(net)
    chunks = []
    T = x.shape[0]
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = x[i:i+sz].contiguous()  # would be H2D in real streaming
        y_seg = net(x_seg)
        chunks.append(y_seg)
        del x_seg
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def forward_mode_4(net, x, kappa):
    """Mode 4: input-stream + output-sink. The deployment-realistic mode.
    
    For classification we sink by accumulating mean(y) across segments,
    returning a single [B, 11] logit vector. To verify equivalence with
    Mode 1, we do this by averaging Mode 1's [T, B, 11] over T.
    """
    reset_state(net)
    T = x.shape[0]
    sink_sum = None
    n = 0
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = x[i:i+sz].contiguous()
        y_seg = net(x_seg)  # [sz, B, 11]
        if sink_sum is None:
            sink_sum = y_seg.sum(dim=0)
        else:
            sink_sum = sink_sum + y_seg.sum(dim=0)
        n += sz
        del x_seg, y_seg
        i += sz
    return sink_sum / n  # [B, 11], same as Mode 1 mean(dim=0)


def measure_mode(name, fn, loader, device, n_warmup=2, n_repeats=3):
    """Run a mode across the full test set, measure peak HBM + latency."""
    print(f"\n  [{name}] running...")
    correct, total = 0, 0
    latencies = []
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup with first batch
    for x_w, y_w in loader:
        x_w = x_w.to(device).float().transpose(0, 1)
        for _ in range(n_warmup):
            _ = fn(x_w)
            torch.cuda.synchronize(device)
        break

    # Evaluation pass
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    all_preds = []
    all_targets = []
    all_logits = []  # for bit-exact comparison

    for x, y in loader:
        x = x.to(device).float().transpose(0, 1)  # [T, B, 2, 128, 128]
        y_int = y.to(device)

        # Time several repeats per batch
        ts = []
        for r in range(n_repeats):
            torch.cuda.synchronize(device)
            t0 = time.time()
            out = fn(x)
            torch.cuda.synchronize(device)
            ts.append((time.time() - t0) * 1000)
        latencies.append(sorted(ts)[len(ts)//2])  # median

        # If output is [T, B, 11] (Mode 1/2/3), reduce by mean to get [B, 11]
        if out.ndim == 3:
            logits = out.mean(dim=0)  # [B, 11]
        else:
            logits = out  # already [B, 11] for Mode 4
        pred = logits.argmax(1)
        correct += (pred == y_int).sum().item()
        total += y_int.shape[0]
        all_preds.append(pred.cpu())
        all_targets.append(y_int.cpu())
        all_logits.append(logits.cpu())

    peak_GB = torch.cuda.max_memory_allocated(device) / 1024**3
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat)//2]
    p95_lat = sorted_lat[min(len(sorted_lat)-1, int(0.95*len(sorted_lat)))]
    acc = correct / total if total > 0 else 0.0

    return {
        "top1_acc": acc, "n_samples": total,
        "peak_memory_GB": peak_GB,
        "median_latency_ms": median_lat, "p95_latency_ms": p95_lat,
        "preds": torch.cat(all_preds, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "logits": torch.cat(all_logits, dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Path to dvs128_gesture_best.pth")
    parser.add_argument("--data_root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--kappa", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--output", default="p9_12_dvs_inference")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.parametric_lif_net import (
        DVSGestureNet)
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    print(f"=== AEROS Phase 2 Exp 12 — DVS128 Gesture Inference (4 modes) ===")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  T={args.T}  kappa={args.kappa}  b={args.batch_size}")
    print(f"  Device: {torch.cuda.get_device_name(device)}")

    # ----- data -----
    test_set = DVS128Gesture(
        root=args.data_root, train=False, data_type="frame",
        frames_number=args.T, split_by="number")
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"  Test samples: {len(test_set)}  batches: {len(test_loader)}")

    # ----- model -----
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
    print(f"  Checkpoint loaded. Reported test_acc: "
          f"{ckpt.get('test_acc', '?')}")

    # ----- run modes -----
    results = {}

    results["Mode_1_baseline"] = measure_mode(
        "Mode 1 baseline (kappa=T)", lambda x: forward_mode_1(net, x),
        test_loader, device, n_repeats=args.n_repeats)

    results[f"Mode_2_retainedIO_kappa{args.kappa}"] = measure_mode(
        f"Mode 2 retained-IO (kappa={args.kappa})",
        lambda x: forward_mode_2(net, x, args.kappa),
        test_loader, device, n_repeats=args.n_repeats)

    results[f"Mode_3_inputstream_kappa{args.kappa}"] = measure_mode(
        f"Mode 3 input-stream (kappa={args.kappa})",
        lambda x: forward_mode_3(net, x, args.kappa),
        test_loader, device, n_repeats=args.n_repeats)

    results[f"Mode_4_IOstream_kappa{args.kappa}"] = measure_mode(
        f"Mode 4 IO-stream + sink (kappa={args.kappa})",
        lambda x: forward_mode_4(net, x, args.kappa),
        test_loader, device, n_repeats=args.n_repeats)

    # ----- bit-exact comparisons -----
    print(f"\n=== Bit-exact comparison vs Mode 1 baseline ===")
    base = results["Mode_1_baseline"]
    base_logits = base["logits"]
    base_preds = base["preds"]
    bitexact_log = {}
    for name, r in results.items():
        if name == "Mode_1_baseline":
            r["max_abs_err_vs_baseline"] = 0.0
            r["pred_agreement_vs_baseline"] = 1.0
            continue
        # For Mode 4, comparison is direct (mean-reduced logits)
        diff = (base_logits - r["logits"]).abs()
        r["max_abs_err_vs_baseline"] = float(diff.max().item())
        r["pred_agreement_vs_baseline"] = float(
            (base_preds == r["preds"]).float().mean().item())
        bitexact_log[name] = {
            "max_abs_err": r["max_abs_err_vs_baseline"],
            "pred_agreement": r["pred_agreement_vs_baseline"],
        }
        marker = "✓" if r["max_abs_err_vs_baseline"] < 1e-5 else "✗"
        print(f"  {name:35s}  max_abs_err={r['max_abs_err_vs_baseline']:.3e}  "
              f"pred_agreement={r['pred_agreement_vs_baseline']*100:.2f}% {marker}")

    # ----- summary table -----
    print(f"\n{'='*80}")
    print(f"=== Summary: DVS128 Gesture under AEROS modes (V100, T={args.T}, "
          f"b={args.batch_size}, kappa={args.kappa}) ===")
    print(f"{'='*80}")
    hdr = f"{'Mode':<35s}  {'Acc%':>6s}  {'Peak GB':>8s}  {'Med ms':>8s}  {'P95 ms':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        print(f"  {name:<35s}  {r['top1_acc']*100:>5.2f}%  "
              f"{r['peak_memory_GB']:>7.3f}  "
              f"{r['median_latency_ms']:>7.1f}  "
              f"{r['p95_latency_ms']:>7.1f}")

    # ----- save JSON -----
    save_data = {
        "config": {
            "ckpt": args.ckpt, "T": args.T, "kappa": args.kappa,
            "batch_size": args.batch_size, "channels": args.channels,
            "n_test": len(test_set),
        },
        "results": {k: {kk: vv for kk, vv in v.items()
                        if kk not in ("preds", "targets", "logits")}
                    for k, v in results.items()},
    }
    with open(args.output + ".json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved JSON: {args.output}.json")


if __name__ == "__main__":
    main()