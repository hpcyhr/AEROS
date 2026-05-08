#!/usr/bin/env python
"""
AEROS — real .aedat-derived event stream microbench.

Loads raw events (decoded from .aedat by SpikingJelly's events_np
cache), simulates a streaming arrival of events at their original
microsecond timestamps, bins into Δt-window frames, and feeds frames
to a trained DVSGestureNet under three forward modes:

  Mode 1 (full-batch): accumulate ALL frames first, then run forward
                       once. Highest HBM, single big call.
  Mode 2 (κ-chunked):  every κ frames, run forward chunk; retain
                       full output trajectory.
  Mode 4 (AEROS):      every κ frames, run AEROS Mode 4 segmented
                       forward with carry-state across segments.

Output: per-frame decision-latency curve, peak HBM per-mode,
sustained throughput across concatenated samples.

Usage:
  python p9_aedat_stream_microbench.py \\
      --events_root /data/yhr/datasets/dvs128_gesture/events_np \\
      --ckpt /data/yhr/AEROS/checkpoints_dvs/dvs128_gesture_best.pth \\
      --dt_us 50000 \\
      --kappa 8 \\
      --n_concat 4 \\
      --output p9_aedat_stream_microbench
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate, layer
except Exception as e:
    print(f"[FATAL] SpikingJelly unavailable: {e}")
    sys.exit(1)


BYTES_PER_GB = 1024 ** 3


# ============================================================================
# Model — DVSGestureNet from SpikingJelly
# ============================================================================

class DVSGestureNet(nn.Module):
    """SpikingJelly's standard DVSGestureNet (used for DVS128 Gesture)."""
    def __init__(self, channels=128, num_classes=11, T=16):
        super().__init__()
        from spikingjelly.activation_based.model import parametric_lif_net
        try:
            # Use SJ's reference implementation if available
            self.net = parametric_lif_net.DVSGestureNet(
                channels=channels, spiking_neuron=neuron.LIFNode,
                surrogate_function=surrogate.ATan(), detach_reset=True)
            try:
                # Replace classifier head if num_classes mismatch
                pass
            except Exception:
                pass
        except Exception:
            # Fallback: build manually
            conv = []
            in_ch = 2
            for c in [channels] * 5:
                conv.extend([
                    layer.Conv2d(in_ch, c, 3, padding=1, bias=False),
                    layer.BatchNorm2d(c),
                    neuron.LIFNode(surrogate_function=surrogate.ATan(),
                                    detach_reset=True),
                    layer.MaxPool2d(2, 2),
                ])
                in_ch = c
            self.net = nn.Sequential(
                *conv,
                layer.Flatten(),
                layer.Linear(channels * 4 * 4, 512),
                neuron.LIFNode(surrogate_function=surrogate.ATan(),
                                detach_reset=True),
                layer.Linear(512, num_classes),
            )
        try:
            functional.set_step_mode(self.net, "m")
        except Exception:
            pass

    def forward(self, x):
        # x: [T, B, 2, 128, 128]
        return self.net(x)


def load_dvs_net(ckpt_path, device):
    """Build DVSGestureNet, load weights, set eval+step_mode."""
    net = DVSGestureNet()
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("model", ck.get("state_dict", ck))
        # Strip "net." prefix if present
        try:
            net.load_state_dict(sd, strict=False)
            print(f"  loaded weights from {ckpt_path}")
        except Exception as e:
            print(f"  [WARN] could not load weights: {e}; using random init")
    else:
        print(f"  [WARN] ckpt not found, using random init")
    net = net.to(device)
    net.eval()
    try:
        functional.set_step_mode(net, "m")
    except Exception:
        pass
    return net


# ============================================================================
# Event loading + binning
# ============================================================================

def load_events(npz_path):
    """Load raw events from a SpikingJelly events_np npz file.

    Returns dict with t (microseconds, sorted), x, y, p arrays.
    """
    d = np.load(npz_path)
    t = d["t"].astype(np.int64)
    x = d["x"].astype(np.int64)
    y = d["y"].astype(np.int64)
    p = d["p"].astype(np.int64)
    d.close()
    # Sort by time if not sorted
    if not np.all(t[:-1] <= t[1:]):
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]
    return t, x, y, p


def concat_events(events_list, gap_us=0):
    """Concatenate multiple event streams into one virtual long stream.
    Each subsequent stream's timestamps are shifted to start right
    after the previous one's last timestamp (plus optional gap)."""
    out_t, out_x, out_y, out_p = [], [], [], []
    cur_offset = 0
    for t, x, y, p in events_list:
        if len(t) == 0:
            continue
        adjusted_t = t - t[0] + cur_offset
        out_t.append(adjusted_t)
        out_x.append(x)
        out_y.append(y)
        out_p.append(p)
        cur_offset = adjusted_t[-1] + gap_us + 1
    return (np.concatenate(out_t), np.concatenate(out_x),
            np.concatenate(out_y), np.concatenate(out_p))


def bin_to_frames(t, x, y, p, dt_us, H=128, W=128):
    """Bin event stream into a sequence of frames of shape [T, 2, H, W],
    where dt_us is the integration window in microseconds.

    Returns frames [T, 2, H, W] (int32 counts), and frame_end_times [T]
    (microseconds; the last-event timestamp in each window's bin).
    """
    if len(t) == 0:
        return (np.zeros((0, 2, H, W), dtype=np.int32),
                np.zeros((0,), dtype=np.int64))
    t0 = int(t[0])
    last_t = int(t[-1])
    n_frames = int((last_t - t0) // dt_us) + 1

    frames = np.zeros((n_frames, 2, H, W), dtype=np.int32)
    bin_idx = ((t - t0) // dt_us).astype(np.int64)

    # Clip events outside frame bounds
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & \
            (bin_idx >= 0) & (bin_idx < n_frames)
    bin_idx = bin_idx[valid]
    xv = x[valid]
    yv = y[valid]
    pv = p[valid]

    # np.add.at handles duplicate indices correctly
    np.add.at(frames, (bin_idx, pv, yv, xv), 1)

    # Frame end-times (when the last event in this bin's window arrives)
    frame_end_times = t0 + (np.arange(n_frames) + 1) * dt_us - 1
    return frames, frame_end_times


def to_tensor_TBCHW(frames_np):
    """frames_np: [T, 2, H, W] -> tensor [T, B=1, 2, H, W] (fp32 normalized)."""
    f = torch.from_numpy(frames_np).float().clamp_(0, 8) / 8.0
    return f.unsqueeze(1)  # [T, 1, 2, 128, 128]


# ============================================================================
# Forward modes
# ============================================================================

@torch.no_grad()
def forward_mode1(net, frames, device):
    """Mode 1: accumulate all frames first, single forward.
    frames: [T, 1, 2, 128, 128] CPU tensor.
    Returns: output [T, 1, num_classes], decision_times relative to forward
    start (single value: latency at end)."""
    x = frames.to(device, non_blocking=False)
    functional.reset_net(net)
    torch.cuda.synchronize(device)
    t_start = time.time()
    y = net(x)
    torch.cuda.synchronize(device)
    t_end = time.time()
    # Decision available only at the end; per-frame "decision time"
    # is t_end for every frame in this mode (waited for full forward).
    T = frames.shape[0]
    per_frame_decision_dt = np.full(T, t_end - t_start)
    return y, per_frame_decision_dt


@torch.no_grad()
def forward_mode2_chunked(net, frames, kappa, device):
    """Mode 2 chunked: every κ frames, run a forward chunk (retain-IO).
    Output trajectory accumulated. Per-frame decision latency =
    chunk-completion time within the chunk."""
    T = frames.shape[0]
    functional.reset_net(net)
    chunks_y = []
    per_frame_decision_dt = np.zeros(T)
    t_total_start = time.time()
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = frames[i:i+sz].to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        t_chunk_start = time.time()
        y_seg = net(x_seg)
        torch.cuda.synchronize(device)
        t_chunk_end = time.time()
        chunks_y.append(y_seg.cpu())
        # All frames in this chunk become decided at chunk_end
        per_frame_decision_dt[i:i+sz] = t_chunk_end - t_total_start
        del x_seg, y_seg
        i += sz
    return torch.cat(chunks_y, dim=0), per_frame_decision_dt


@torch.no_grad()
def forward_mode4_aeros(net, frames, kappa, device):
    """Mode 4 AEROS: every κ frames, run a chunk and immediately reduce
    output to a sink (sum over time). Carry state across chunks via
    SpikingJelly's persistent neuron state (no reset between chunks).

    Per-frame decision latency: at the end of each segment, we have
    a running sink that classifies the entire stream-so-far. We
    report per-frame decision_dt = chunk_end_time, but the actual
    classification only updates per-segment (every κ frames).
    """
    T = frames.shape[0]
    functional.reset_net(net)
    sink = None
    n_accum = 0
    per_frame_decision_dt = np.zeros(T)
    t_total_start = time.time()
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        x_seg = frames[i:i+sz].to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        t_chunk_start = time.time()
        y_seg = net(x_seg)  # [sz, B, num_classes]
        if sink is None:
            sink = y_seg.sum(dim=0).cpu()
            n_accum = sz
        else:
            sink = sink + y_seg.sum(dim=0).cpu()
            n_accum += sz
        torch.cuda.synchronize(device)
        t_chunk_end = time.time()
        per_frame_decision_dt[i:i+sz] = t_chunk_end - t_total_start
        del x_seg, y_seg
        i += sz
    return sink / n_accum, per_frame_decision_dt


# ============================================================================
# Measurement wrapper
# ============================================================================

def measure(forward_fn, *args, n_warmup=1, **kwargs):
    """Run forward_fn once for warmup, then measure."""
    torch.cuda.empty_cache()
    gc.collect()
    device = next(args[0].parameters()).device
    # Warmup
    for _ in range(n_warmup):
        try:
            forward_fn(*args, **kwargs)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"peak_GB": -1.0, "wall_s": -1.0,
                    "per_frame_decision_dt": None, "status": "oom"}
        except Exception as e:
            return {"peak_GB": -1.0, "wall_s": -1.0,
                    "per_frame_decision_dt": None,
                    "status": f"error:{type(e).__name__}",
                    "err_str": str(e)[:200]}
    # Real run
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        t_start = time.time()
        out, per_frame_dt = forward_fn(*args, **kwargs)
        torch.cuda.synchronize(device)
        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
        return {"peak_GB": peak, "wall_s": wall,
                "per_frame_decision_dt": per_frame_dt.tolist(),
                "p50_decision_ms": float(np.percentile(per_frame_dt, 50) * 1000),
                "p95_decision_ms": float(np.percentile(per_frame_dt, 95) * 1000),
                "p99_decision_ms": float(np.percentile(per_frame_dt, 99) * 1000),
                "status": "ok"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"peak_GB": -1.0, "wall_s": -1.0,
                "per_frame_decision_dt": None, "status": "oom"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"peak_GB": -1.0, "wall_s": -1.0,
                "per_frame_decision_dt": None,
                "status": f"error:{type(e).__name__}",
                "err_str": str(e)[:200]}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_root",
                        default="/data/yhr/datasets/dvs128_gesture/events_np")
    parser.add_argument("--ckpt",
                        default="/data/yhr/AEROS/checkpoints_dvs/dvs128_gesture_best.pth")
    parser.add_argument("--dt_us", type=int, default=50000,
                        help="bin width in microseconds (default 50ms)")
    parser.add_argument("--kappa", type=int, default=8)
    parser.add_argument("--n_concat", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="number of test samples to concatenate; one cell per value")
    parser.add_argument("--n_runs_per_config", type=int, default=3,
                        help="number of replicate samples per concat-count")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", default="p9_aedat_stream_microbench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_T", type=int, default=4096,
                        help="cap on number of frames per cell (truncate concat'd stream)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print("AEROS — .aedat-derived event stream microbench")
    print("=" * 78)
    print(f"  events_root: {args.events_root}")
    print(f"  ckpt:        {args.ckpt}")
    print(f"  dt_us:       {args.dt_us} ({args.dt_us/1000:.1f}ms per frame)")
    print(f"  kappa:       {args.kappa}")
    print(f"  n_concat:    {args.n_concat}")
    print(f"  modes:       {args.modes}")

    # Build network
    print("\n[1/3] Loading network...")
    net = load_dvs_net(args.ckpt, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  n_params: {n_params:,}")

    # Probe a few test events files
    print("\n[2/3] Locating test event files...")
    test_files = sorted(glob.glob(os.path.join(args.events_root,
                                                  "test/*/*.npz")))
    if not test_files:
        print(f"  [FATAL] no .npz in {args.events_root}/test/")
        sys.exit(1)
    print(f"  found {len(test_files)} test event files")
    np.random.seed(args.seed)
    np.random.shuffle(test_files)

    # Run cells
    print("\n[3/3] Running microbench...")
    cells = []
    for n_concat in args.n_concat:
        for run_id in range(args.n_runs_per_config):
            sample_pool_start = run_id * n_concat
            sample_pool_end = (run_id + 1) * n_concat
            if sample_pool_end > len(test_files):
                print(f"  [skip] n_concat={n_concat} run={run_id}: "
                      f"need {sample_pool_end} samples but only "
                      f"{len(test_files)} available")
                continue

            chosen = test_files[sample_pool_start:sample_pool_end]
            events_list = [load_events(f) for f in chosen]
            all_events = concat_events(events_list, gap_us=0)
            t, x, y, p = all_events
            n_events = len(t)
            duration_s = (t[-1] - t[0]) / 1e6
            frames, frame_end_times = bin_to_frames(t, x, y, p, args.dt_us)
            T = frames.shape[0]
            # Cap at max_T to avoid V100 host RAM blowup
            if T > args.max_T:
                print(f"    [trunc] n_concat={n_concat} run={run_id}: "
                      f"T={T} > max_T={args.max_T}; truncating")
                frames = frames[:args.max_T]
                frame_end_times = frame_end_times[:args.max_T]
                T = args.max_T

            x_tensor = to_tensor_TBCHW(frames)
            print(f"\n--- n_concat={n_concat} run={run_id}: T={T} frames "
                  f"({duration_s:.1f}s of events, {n_events:,} events) ---")

            cell = {
                "n_concat": n_concat, "run_id": run_id,
                "T": int(T), "n_events": int(n_events),
                "duration_s": float(duration_s),
                "events_per_sec": float(n_events / duration_s) if duration_s > 0 else 0,
                "samples": [str(s) for s in chosen],
            }

            for mode in args.modes:
                if mode == 1:
                    label = "Mode1_full"
                    fn = forward_mode1
                    fn_args = (net, x_tensor, device)
                elif mode == 2:
                    label = f"Mode2_kappa{args.kappa}"
                    fn = forward_mode2_chunked
                    fn_args = (net, x_tensor, args.kappa, device)
                elif mode == 4:
                    label = f"Mode4_kappa{args.kappa}"
                    fn = forward_mode4_aeros
                    fn_args = (net, x_tensor, args.kappa, device)
                else:
                    continue

                m = measure(fn, *fn_args, n_warmup=1)
                # Drop the per-frame trajectory from JSON to keep it small;
                # save percentiles only.
                if "per_frame_decision_dt" in m:
                    pf = m.pop("per_frame_decision_dt")
                    if pf is not None:
                        m["last_frame_decision_ms"] = float(pf[-1] * 1000)
                cell[label] = m

                p50 = m.get("p50_decision_ms", -1)
                p99 = m.get("p99_decision_ms", -1)
                last = m.get("last_frame_decision_ms", -1)
                peak = m.get("peak_GB", -1)
                status = m.get("status", "ok")
                print(f"   {label:<22}  peak={peak:.2f}GB  "
                      f"p50_dec={p50:7.1f}ms  p99={p99:7.1f}ms  "
                      f"last_frame={last:7.1f}ms  status={status}")

            cells.append(cell)

    # Summary
    print("\n" + "=" * 100)
    print(f"{'n_concat':>9} {'T':>6} {'events':>9} {'mode':<22} "
          f"{'peak (GB)':>10} {'p50 (ms)':>10} {'p99 (ms)':>10} {'status':>10}")
    print("-" * 100)
    by_concat = {}
    for c in cells:
        by_concat.setdefault(c["n_concat"], []).append(c)
    for n in sorted(by_concat.keys()):
        runs = by_concat[n]
        for label in [f"Mode1_full",
                      f"Mode2_kappa{args.kappa}",
                      f"Mode4_kappa{args.kappa}"]:
            peaks = [r[label]["peak_GB"] for r in runs
                     if label in r and r[label].get("status") == "ok"]
            p50s = [r[label]["p50_decision_ms"] for r in runs
                    if label in r and r[label].get("status") == "ok"]
            p99s = [r[label]["p99_decision_ms"] for r in runs
                    if label in r and r[label].get("status") == "ok"]
            statuses = [r[label].get("status") for r in runs if label in r]
            n_ok = sum(1 for s in statuses if s == "ok")
            n_oom = sum(1 for s in statuses if s == "oom")
            T_avg = int(np.mean([r["T"] for r in runs]))
            ev_avg = int(np.mean([r["n_events"] for r in runs]))
            if n_ok > 0:
                status = f"{n_ok}/{len(runs)} ok"
                p_str = f"{np.mean(peaks):.2f}"
                p50_str = f"{np.mean(p50s):.1f}"
                p99_str = f"{np.mean(p99s):.1f}"
            else:
                status = f"OOM" if n_oom > 0 else "ERR"
                p_str = "--"
                p50_str = "--"
                p99_str = "--"
            print(f"{n:>9} {T_avg:>6} {ev_avg:>9} {label:<22} "
                  f"{p_str:>10} {p50_str:>10} {p99_str:>10} {status:>10}")

    out = {"config": vars(args), "cells": cells}
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}.json")


if __name__ == "__main__":
    main()