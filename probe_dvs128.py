#!/usr/bin/env python
"""
AEROS — probe DVS128 Gesture data structure on V100.

Locates the DVS128 Gesture dataset cache (where SpikingJelly stores
events_np / frames_np) and reports what's available. We want to confirm
event-level (raw .aedat-decoded) data is present so the streaming
microbench can rebuild the temporal pipeline from raw events instead
of pre-binned frames.

Output: console report of paths found + sample data structure.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


CANDIDATE_ROOTS = [
    "/data/yhr/AEROS/data_dvs",
    "/data/yhr/AEROS/datasets",
    "/data/yhr/datasets",
    "/data/yhr/data",
    "/root/datasets",
    "/data/yhr/AEROS/checkpoints_dvs",  # in case data is colocated
    "/data/datasets",
]

DVS_NAMES = ["DVS128Gesture", "dvs128_gesture", "dvs_gesture", "DVSGesture"]


def find_dvs_root():
    """Walk common locations to find the DVS128 Gesture root directory."""
    found = []
    for base in CANDIDATE_ROOTS:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if not os.path.isdir(full):
                continue
            for name in DVS_NAMES:
                if name.lower() in entry.lower():
                    found.append(full)
        # Also recurse one level
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if not os.path.isdir(full):
                continue
            for sub in os.listdir(full):
                fullsub = os.path.join(full, sub)
                if not os.path.isdir(fullsub):
                    continue
                for name in DVS_NAMES:
                    if name.lower() in sub.lower():
                        found.append(fullsub)
    return list(set(found))


def inspect_root(root):
    """Inspect a candidate DVS128 root directory and return what's there."""
    info = {"root": root, "subdirs": [], "events_np_root": None,
            "frames_np_root": None, "raw_aedat": [],
            "events_np_samples": [], "frames_np_samples": []}
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            info["subdirs"].append(entry)
            if "events_np" in entry.lower():
                info["events_np_root"] = full
            if "frames_np" in entry.lower():
                info["frames_np_root"] = full
            if entry.lower() in ("download", "extract", "raw", "aedat"):
                # Look for .aedat files inside
                aedat = glob.glob(os.path.join(full, "**/*.aedat"),
                                   recursive=True)
                aedat += glob.glob(os.path.join(full, "**/*.aedat3"),
                                    recursive=True)
                if aedat:
                    info["raw_aedat"] = aedat[:5]  # first 5

    if info["events_np_root"]:
        # Collect a couple sample paths
        npz = sorted(glob.glob(os.path.join(info["events_np_root"],
                                              "**/*.npz"), recursive=True))
        info["events_np_samples"] = npz[:3]
        info["events_np_count"] = len(npz)
    if info["frames_np_root"]:
        npz = sorted(glob.glob(os.path.join(info["frames_np_root"],
                                              "**/*.npz"), recursive=True))
        info["frames_np_samples"] = npz[:3]
        info["frames_np_count"] = len(npz)
    return info


def inspect_npz(path):
    """Open one npz file and report its keys + shapes."""
    import numpy as np
    try:
        d = np.load(path, allow_pickle=True)
        keys = list(d.keys())
        report = []
        for k in keys[:8]:
            try:
                arr = d[k]
                report.append(f"      {k}: shape={getattr(arr, 'shape', None)}  "
                              f"dtype={getattr(arr, 'dtype', None)}")
                if k.lower() in ("t", "x", "y", "p", "polarity") \
                        and hasattr(arr, "__len__") and len(arr) > 0:
                    report.append(f"        first 3 values: {arr[:3]}")
                    if k.lower() == "t" and len(arr) > 1:
                        report.append(f"        time range: {arr.min()} ~ "
                                      f"{arr.max()} (n={len(arr)})")
            except Exception as e:
                report.append(f"      {k}: <error reading: {e}>")
        d.close()
        return report
    except Exception as e:
        return [f"      <error opening: {e}>"]


def main():
    print("=" * 78)
    print("AEROS — DVS128 Gesture data structure probe")
    print("=" * 78)

    print("\n[1/3] Searching for DVS128 root in common locations...")
    candidates = find_dvs_root()
    if not candidates:
        print(f"  No DVS128-like directory found.")
        print(f"  Searched roots: {CANDIDATE_ROOTS}")
        print(f"  Searched name patterns: {DVS_NAMES}")
        print()
        print("Try: find / -name 'DVS128Gesture' 2>/dev/null | head -10")
        print("     find / -name '*.aedat*' 2>/dev/null | head -10")
        print("     find / -name 'events_np' -type d 2>/dev/null | head -10")
        sys.exit(1)

    print(f"  Found {len(candidates)} candidate(s):")
    for c in candidates:
        print(f"    {c}")

    print("\n[2/3] Inspecting each candidate...")
    for c in candidates:
        print(f"\n--- {c} ---")
        info = inspect_root(c)
        print(f"  Subdirs: {info['subdirs']}")
        if info["events_np_root"]:
            print(f"  events_np_root: {info['events_np_root']}")
            print(f"    sample count: {info.get('events_np_count', '?')}")
            for s in info["events_np_samples"]:
                print(f"    sample: {s}")
        else:
            print(f"  events_np_root: NOT FOUND")

        if info["frames_np_root"]:
            print(f"  frames_np_root: {info['frames_np_root']}")
            print(f"    sample count: {info.get('frames_np_count', '?')}")
            for s in info["frames_np_samples"]:
                print(f"    sample: {s}")
        else:
            print(f"  frames_np_root: NOT FOUND")

        if info["raw_aedat"]:
            print(f"  Raw .aedat files: found")
            for a in info["raw_aedat"]:
                size_mb = os.path.getsize(a) / (1024 * 1024)
                print(f"    {a}  ({size_mb:.1f} MB)")
        else:
            print(f"  Raw .aedat files: not in standard subdirs")

    print("\n[3/3] Inspecting structure of one sample...")
    for c in candidates:
        info = inspect_root(c)
        if info["events_np_samples"]:
            sample = info["events_np_samples"][0]
            print(f"\n  events_np sample: {sample}")
            for line in inspect_npz(sample):
                print(line)
            break
        elif info["frames_np_samples"]:
            sample = info["frames_np_samples"][0]
            print(f"\n  frames_np sample: {sample}")
            for line in inspect_npz(sample):
                print(line)
            break

    print("\n" + "=" * 78)
    print("Summary for streaming microbench design:")
    print("=" * 78)

    has_events = any(inspect_root(c).get("events_np_root")
                     for c in candidates)
    has_frames = any(inspect_root(c).get("frames_np_root")
                     for c in candidates)
    has_raw = any(inspect_root(c).get("raw_aedat") for c in candidates)

    if has_raw:
        print("  ✓ raw .aedat available — can decode from binary directly")
    else:
        print("  - raw .aedat: not found (either deleted or non-standard path)")

    if has_events:
        print("  ✓ events_np available — pre-decoded raw events ready for")
        print("    streaming pipeline (RECOMMENDED: this is .aedat-derived")
        print("    data with original timestamps, no pre-binning)")
    else:
        print("  - events_np: not found")

    if has_frames and not has_events:
        print("  ⚠ only frames_np: pre-binned, NOT suitable for true")
        print("    .aedat streaming microbench")

    if not (has_raw or has_events):
        print("\n  [ACTION] No event-level data found. Either:")
        print("    1. Run SpikingJelly DVS128Gesture init once to populate")
        print("       events_np cache, OR")
        print("    2. Locate raw .aedat files manually with `find`")


if __name__ == "__main__":
    main()