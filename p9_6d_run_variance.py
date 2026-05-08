#!/usr/bin/env python
"""
AEROS — B3 variance bands: re-run p9_6d disjoint-fold compliance with
multiple fold-split seeds and aggregate.

Runs p9_6d_disjoint_calibration.py with N seeds (default 3) for the
fold split, then aggregates the in-sample and held-out compliance
numbers across the N runs and reports mean ± std.

Output:
  p9_6d_seed{S}_{ts}.json      one per seed run (raw)
  p9_6d_variance_aggregate.json  the cross-seed aggregate

Usage:
  python p9_6d_run_variance.py \\
      --coeffs p9_1a_full16.json \\
      --seeds 42 123 7 \\
      --Ts 128 1024 4096 \\
      --Ms 4 8 16 24 \\
      --output_prefix p9_6d_variance
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_one_seed(coeffs, seed, Ts, Ms, modes, nets, batch_size, H, C,
                  delta, eps, output_path, p9_6d_script):
    """Invoke p9_6d_disjoint_calibration.py with a given seed."""
    cmd = [
        "python", "-u", p9_6d_script,
        "--coeffs", coeffs,
        "--output", str(output_path),
        "--seed", str(seed),
        "--batch_size", str(batch_size),
        "--H", str(H), "--C", str(C),
        "--delta", str(delta),
        "--epsilon_safety", str(eps),
    ]
    if Ts:
        cmd += ["--Ts"] + [str(t) for t in Ts]
    if Ms:
        cmd += ["--Ms"] + [str(m) for m in Ms]
    if modes:
        cmd += ["--modes"] + [str(m) for m in modes]
    if nets:
        cmd += ["--nets"] + nets

    print(f"\n{'='*78}")
    print(f"Running seed={seed}: {' '.join(cmd)}")
    print(f"{'='*78}\n", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"  [FAIL] seed={seed} exited with code {proc.returncode}",
              flush=True)
        return None
    print(f"  [ok] seed={seed} done in {dt/60:.1f} min", flush=True)
    return output_path.with_suffix(".json")


def extract_summary(json_path):
    """Pull the headline compliance numbers from a p9_6d output JSON.

    The schema (per p9_6d_disjoint_calibration.py) contains:
      - results_in_sample: list of cells with 'compliant' boolean
      - results_held_out: list of cells with 'compliant' boolean
      - summary_in_sample: {compliance_pct, oom_pct, ...}
      - summary_held_out:  {compliance_pct, oom_pct, ...}
    """
    with open(json_path) as f:
        d = json.load(f)
    result = {}
    for tag in ("in_sample", "held_out"):
        key = f"summary_{tag}"
        if key not in d:
            continue
        s = d[key]
        result[tag] = {
            "compliance_pct": s.get("compliance_pct", float("nan")),
            "oom_pct": s.get("oom_pct", float("nan")),
            "violation_pct": s.get("violation_pct", float("nan")),
            "n_total": s.get("n_total", 0),
            "n_compliant": s.get("n_compliant", 0),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeffs", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[128, 1024, 4096])
    parser.add_argument("--Ms", type=float, nargs="+",
                        default=[4.0, 8.0, 16.0, 24.0])
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--nets", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon_safety", type=float, default=0.02)
    parser.add_argument("--output_prefix", default="p9_6d_variance")
    parser.add_argument("--p9_6d_script",
                        default="p9_6d_disjoint_calibration.py")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 78)
    print("AEROS B3 — disjoint-fold compliance variance bands")
    print("=" * 78)
    print(f"  seeds: {args.seeds}")
    print(f"  Ts: {args.Ts}, Ms: {args.Ms}, modes: {args.modes}")
    print(f"  output prefix: {args.output_prefix}")
    print(f"  timestamp: {ts}")

    per_seed_summaries = []
    overall_t0 = time.time()
    for seed in args.seeds:
        out_path = Path(f"{args.output_prefix}_seed{seed}_{ts}")
        result_path = run_one_seed(
            args.coeffs, seed, args.Ts, args.Ms, args.modes, args.nets,
            args.batch_size, args.H, args.C,
            args.delta, args.epsilon_safety, out_path, args.p9_6d_script)
        if result_path is None or not result_path.exists():
            print(f"  [skip] seed={seed} — no output file")
            continue
        s = extract_summary(result_path)
        s["seed"] = seed
        s["json_path"] = str(result_path)
        per_seed_summaries.append(s)
        print(f"  seed={seed}: in_sample={s.get('in_sample', {}).get('compliance_pct')}%, "
              f"held_out={s.get('held_out', {}).get('compliance_pct')}%")

    overall_wall = time.time() - overall_t0
    print(f"\nAll seeds done in {overall_wall/60:.1f} min")

    if not per_seed_summaries:
        print("[FATAL] no successful seed runs")
        sys.exit(1)

    # Aggregate
    print("\n" + "=" * 78)
    print(f"Cross-seed aggregate (n_seeds={len(per_seed_summaries)})")
    print("=" * 78)

    aggregate = {"per_seed": per_seed_summaries, "aggregate": {}}
    for tag in ("in_sample", "held_out"):
        compliance = [s.get(tag, {}).get("compliance_pct")
                      for s in per_seed_summaries
                      if tag in s and s[tag].get("compliance_pct") is not None]
        oom = [s.get(tag, {}).get("oom_pct") for s in per_seed_summaries
               if tag in s and s[tag].get("oom_pct") is not None]
        viol = [s.get(tag, {}).get("violation_pct") for s in per_seed_summaries
                if tag in s and s[tag].get("violation_pct") is not None]
        if not compliance:
            continue
        agg = {
            "n_seeds": len(compliance),
            "compliance_pct_mean": float(np.mean(compliance)),
            "compliance_pct_std": float(np.std(compliance)),
            "compliance_per_seed": compliance,
            "oom_pct_mean": float(np.mean(oom)) if oom else 0.0,
            "oom_pct_std": float(np.std(oom)) if oom else 0.0,
            "violation_pct_mean": float(np.mean(viol)) if viol else 0.0,
            "violation_pct_std": float(np.std(viol)) if viol else 0.0,
        }
        aggregate["aggregate"][tag] = agg
        print(f"\n  {tag:<11s}: compliance = "
              f"{agg['compliance_pct_mean']:.2f}% "
              f"± {agg['compliance_pct_std']:.2f}  "
              f"(per-seed: {[f'{x:.2f}%' for x in compliance]})")
        print(f"             OOM = {agg['oom_pct_mean']:.2f}% "
              f"± {agg['oom_pct_std']:.2f}")
        print(f"             violation = {agg['violation_pct_mean']:.2f}% "
              f"± {agg['violation_pct_std']:.2f}")

    out_agg = f"{args.output_prefix}_aggregate_{ts}.json"
    with open(out_agg, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved aggregate: {out_agg}")


if __name__ == "__main__":
    main()