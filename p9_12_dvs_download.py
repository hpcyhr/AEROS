#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — DVS128 Gesture dataset download.

Tries SpikingJelly's auto-download for DVS128 Gesture. If it succeeds we
proceed to training; if it fails (network restrictions, IBM Research
mirror down, etc.) we fall back to .aedat microbenchmark.

Auto-download path:
  - SJ creates {root}/download/ and pulls 11 .aedat files from
    IBM Research mirror (~3 GB total)
  - Then extracts to {root}/extract/
  - Then converts to {root}/events_np/ at first __init__ with
    data_type='event' or to {root}/frames_number_T/ with
    data_type='frame'

We pre-create the dir, request frame conversion at T=16 (which is the
SJ tutorial default for DVS128Gesture), and let it run.

Usage:
    python p9_12_dvs_download.py --root /data/yhr/AEROS/DVS128Gesture \\
        --T 16 --timeout 1800
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--T", type=int, default=16,
                        help="Frame integration window (SJ default 16)")
    parser.add_argument("--split_by", default="number",
                        choices=["number", "time"])
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Total wall-clock timeout in seconds")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    print(f"=== AEROS Phase 2 Exp 12 — DVS128 Gesture download ===")
    print(f"  Root: {args.root}")
    print(f"  Frame T: {args.T}")
    print(f"  Split-by: {args.split_by}")
    print(f"  Timeout: {args.timeout}s")
    print()

    t0 = time.time()
    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    except Exception as e:
        print(f"[FATAL] SJ DVS128Gesture import failed: {e}")
        sys.exit(1)

    print(f"[step 1/3] Constructing train split (this triggers download "
          f"+ extract + frame conversion)...")
    print(f"           Watch for IBM Research mirror connect lines below.")
    print(f"           If it stalls > {args.timeout//60} min on download "
          f"step, kill and run aedat microbench fallback.")
    print()

    try:
        train_set = DVS128Gesture(
            root=args.root,
            train=True,
            data_type="frame",
            frames_number=args.T,
            split_by=args.split_by,
        )
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Killed after {time.time()-t0:.0f}s")
        sys.exit(2)
    except Exception as e:
        print(f"\n[FAIL] Train split construction failed at "
              f"{time.time()-t0:.0f}s")
        print(f"       Error: {type(e).__name__}: {e}")
        print()
        traceback.print_exc()
        print()
        print("Likely causes:")
        print("  1. IBM Research mirror unreachable (network / firewall)")
        print("  2. Disk full")
        print("  3. SJ version mismatch in API")
        print()
        print("Fallback: run .aedat microbenchmark instead "
              "(p9_13_aedat_microbench.py). Tell Claude.")
        sys.exit(3)

    elapsed_train = time.time() - t0
    print(f"[step 1/3] Train split ready ({elapsed_train:.0f}s, "
          f"{len(train_set)} samples)")

    t1 = time.time()
    print(f"[step 2/3] Constructing test split...")
    try:
        test_set = DVS128Gesture(
            root=args.root,
            train=False,
            data_type="frame",
            frames_number=args.T,
            split_by=args.split_by,
        )
    except Exception as e:
        print(f"[FAIL] Test split: {type(e).__name__}: {e}")
        sys.exit(4)
    elapsed_test = time.time() - t1
    print(f"[step 2/3] Test split ready ({elapsed_test:.0f}s, "
          f"{len(test_set)} samples)")

    print(f"[step 3/3] Sample one item from each split to verify shape...")
    try:
        x_tr, y_tr = train_set[0]
        x_te, y_te = test_set[0]
        print(f"  Train sample 0: x.shape={tuple(x_tr.shape)}  y={y_tr}")
        print(f"  Test sample 0:  x.shape={tuple(x_te.shape)}  y={y_te}")
    except Exception as e:
        print(f"[WARN] Sample fetch failed: {type(e).__name__}: {e}")

    # Disk usage
    print()
    print("[summary] Disk usage:")
    try:
        import subprocess
        result = subprocess.run(
            ["du", "-sh", args.root],
            capture_output=True, text=True, timeout=30)
        print(f"  {result.stdout.strip()}")
    except Exception:
        print(f"  (du unavailable)")

    total_elapsed = time.time() - t0
    print()
    print(f"=== DOWNLOAD + CONVERT COMPLETE ===")
    print(f"  Total wall-clock: {total_elapsed:.0f}s")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Test samples:  {len(test_set)}")
    print(f"  Frame T: {args.T}")
    print()
    print(f"Next step: train DVSGestureNet checkpoint, then run AEROS "
          f"inference. Tell Claude this completed cleanly.")


if __name__ == "__main__":
    main()