#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — DVS128 Gesture reconnaissance.

Quick check before writing the full Suite B inference tool:
1. Does SpikingJelly's DVS128 Gesture data class import?
2. Is the dataset already downloaded somewhere on the V100?
3. Are there pre-trained checkpoints for DVS128 Gesture on V100?
4. What input shape does it expect?
5. What's the SJ-recommended baseline architecture for DVS128 Gesture?

Output: a single console report with PASS / FAIL flags so we decide
whether to (a) run inference on existing checkpoint, (b) train from
scratch, or (c) fall back to .aedat microbenchmark.

Usage:
    python p9_12_dvs_recon.py [--data_root /data/yhr/DVS128Gesture]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check(cond, label, detail=""):
    mark = "[PASS]" if cond else "[FAIL]"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--ckpt_dir", default="/data/yhr/AEROS/checkpoints")
    parser.add_argument("--alt_data_roots", nargs="*", default=[
        "/data/yhr/DVS128Gesture",
        "/data/yhr/data/DVS128Gesture",
        "/data/yhr/AEROS/data/DVS128Gesture",
        "/data/DVS128Gesture",
        os.path.expanduser("~/data/DVS128Gesture"),
    ])
    args = parser.parse_args()

    section("1. SpikingJelly DVS128 Gesture import")
    try:
        from spikingjelly.activation_based.model.parametric_lif_net import (
            DVSGestureNet)
        check(True, "DVSGestureNet (parametric_lif_net) imports")
        sj_pln = True
    except Exception as e:
        check(False, "DVSGestureNet import", f"{type(e).__name__}: {str(e)[:80]}")
        sj_pln = False

    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        check(True, "DVS128Gesture dataset class imports")
        sj_data = True
    except Exception as e:
        check(False, "DVS128Gesture dataset import",
              f"{type(e).__name__}: {str(e)[:80]}")
        sj_data = False

    try:
        from spikingjelly.activation_based import functional, neuron, surrogate
        check(True, "SJ activation_based imports")
    except Exception as e:
        check(False, "SJ activation_based import",
              f"{type(e).__name__}: {str(e)[:80]}")

    section("2. Locate downloaded DVS128 Gesture dataset")
    candidates = [args.data_root] + args.alt_data_roots
    found_root = None
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        # Check for SJ's expected layout: events_np / frames_number_T
        has_events = os.path.isdir(os.path.join(cand, "events_np"))
        has_extracted = os.path.isdir(os.path.join(cand, "extract"))
        has_download = os.path.isdir(os.path.join(cand, "download"))
        has_frames = any(
            d.startswith("frames_number_") and os.path.isdir(os.path.join(cand, d))
            for d in os.listdir(cand)
        )
        if has_events or has_extracted or has_download or has_frames:
            found_root = cand
            print(f"  [PASS] Found DVS128Gesture data root: {cand}")
            print(f"         events_np dir present: {has_events}")
            print(f"         extract dir present: {has_extracted}")
            print(f"         download dir present: {has_download}")
            print(f"         frames_number_T dir present: {has_frames}")
            if has_frames:
                frames_dirs = [d for d in os.listdir(cand)
                               if d.startswith("frames_number_")]
                print(f"         frames variants: {frames_dirs}")
            break
    if found_root is None:
        print(f"  [FAIL] No DVS128Gesture dataset found in any of:")
        for cand in candidates:
            print(f"           {cand}")
        print(f"         You will need to download to {args.data_root}")
        print(f"         (SJ will auto-download from IBM mirror to "
              f"{args.data_root}/download/)")

    section("3. Check for pre-trained DVS128 Gesture checkpoints")
    if not os.path.isdir(args.ckpt_dir):
        print(f"  [FAIL] Checkpoint directory does not exist: {args.ckpt_dir}")
    else:
        try:
            files = sorted(os.listdir(args.ckpt_dir))
            dvs_files = [f for f in files
                         if "dvs" in f.lower() or "gesture" in f.lower()]
            if dvs_files:
                check(True, "DVS128 Gesture checkpoints found")
                for f in dvs_files:
                    full = os.path.join(args.ckpt_dir, f)
                    sz_mb = os.path.getsize(full) / 1024 / 1024
                    print(f"           {f}  ({sz_mb:.1f} MB)")
            else:
                print(f"  [FAIL] No DVS128 Gesture checkpoints in "
                      f"{args.ckpt_dir}")
                print(f"         Available checkpoints (other models):")
                for f in files[:10]:
                    if f.endswith(".pth") or f.endswith(".pt"):
                        full = os.path.join(args.ckpt_dir, f)
                        sz_mb = os.path.getsize(full) / 1024 / 1024
                        print(f"           {f}  ({sz_mb:.1f} MB)")
        except Exception as e:
            print(f"  [FAIL] Cannot list ckpt dir: {e}")

    section("4. Build DVSGestureNet and report shape")
    if sj_pln:
        try:
            from spikingjelly.activation_based.model.parametric_lif_net import (
                DVSGestureNet)
            from spikingjelly.activation_based import (
                functional, neuron, surrogate)
            net = DVSGestureNet(
                channels=128,
                spiking_neuron=neuron.LIFNode,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
            )
            functional.set_step_mode(net, "m")
            n_params = sum(p.numel() for p in net.parameters())
            print(f"  [PASS] DVSGestureNet builds")
            print(f"         channels=128 (SJ tutorial default)")
            print(f"         params: {n_params/1e6:.2f} M")
            print(f"         expected input: [T, B, 2, 128, 128]")
            print(f"         expected output: [T, B, 11]  (11 gesture classes)")
        except Exception as e:
            print(f"  [FAIL] DVSGestureNet build: {type(e).__name__}: {e}")

    section("5. Check CUDA availability")
    try:
        import torch
        check(torch.cuda.is_available(), "CUDA available",
              torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "no CUDA")
        if torch.cuda.is_available():
            print(f"         CUDA: {torch.version.cuda}")
            print(f"         PyTorch: {torch.__version__}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"         GPU memory: {mem_gb:.1f} GB")
    except Exception as e:
        check(False, "PyTorch import", f"{type(e).__name__}: {e}")

    section("6. Recommendation")
    print("""
  Based on the above checks, recommended next step:

  A) If checkpoint EXISTS and dataset EXISTS:
       → Run inference-only (Suite B Variant 1, ~1 hr V100)
       Tool: p9_12_dvs_inference.py (Phase 2 Exp 12)

  B) If checkpoint MISSING but dataset EXISTS (or downloadable):
       → Train DVSGestureNet first, then run inference
       SJ baseline reaches ~96% accuracy in ~50-100 epochs
       Time: ~6-12 hr V100 train + ~1 hr inference

  C) If dataset MISSING and download fails:
       → Fall back to .aedat microbenchmark (Variant 2)
       Tool: p9_13_aedat_microbench.py (no checkpoint needed)
       Time: ~1 day V100 (download single .aedat sample, build pipeline)

  Paste this output back to Claude for decision.
""")


if __name__ == "__main__":
    main()