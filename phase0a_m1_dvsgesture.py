"""
AEROS Phase 0A — Measurement M1: N_A(s) / T_train on DVS Gesture.

For each test sample s:
  Δt_train(s) = sample_duration(s) / T_train
  N_A(s) = number of bins j ∈ [0, T_train) with zero events
  Report: per-sample N_A/T_train + cross-sample mean / p25 / p50 / p75

R10 trigger: if mean(N_A/T_train) < 0.10, Mode A exact skip on the
training grid has near-zero headroom on this dataset. Paper must
reframe to Mode B.
"""
import argparse, os
import numpy as np
import torch
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/dvs128_gesture')
    p.add_argument('--T_train', type=int, default=20,
                   help='match the trained ckpt T (we used T=20 for 96.5% ckpt)')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--out', default='m1_dvsg_NA.npz')
    args = p.parse_args()

    print(f'Loading DVS128 Gesture {args.split} (raw events)...')
    # data_type='event' returns raw events, not frame-integrated
    ds = DVS128Gesture(args.data_dir, train=(args.split == 'train'),
                       data_type='event')
    print(f'  {len(ds)} samples')

    NA_per_sample = []        # N_A(s) / T_train
    duration_per_sample = []  # microseconds
    nevents_per_sample = []
    bins_with_events = []     # for histogram of bin event-count

    for i in range(len(ds)):
        events, label = ds[i]
        # SJ returns events as dict {'t': ..., 'x': ..., 'y': ..., 'p': ...}
        # or as structured numpy array, depending on SJ version
        t = events['t']

        if len(t) == 0:
            continue
        t_min, t_max = int(t.min()), int(t.max())
        duration = t_max - t_min  # microseconds
        if duration <= 0:
            continue

        dt_train_us = duration / args.T_train  # per-sample Δt_train

        # Bin events into T_train uniform bins on the trained grid
        # bin index = floor((t - t_min) / dt_train_us), clipped to T_train-1
        bin_idx = np.minimum(
            ((t - t_min) / dt_train_us).astype(np.int64),
            args.T_train - 1
        )
        # Count events per bin
        bin_count = np.bincount(bin_idx, minlength=args.T_train)
        # N_A = number of bins with zero events
        N_A = int((bin_count == 0).sum())
        NA_per_sample.append(N_A / args.T_train)
        duration_per_sample.append(duration)
        nevents_per_sample.append(len(t))
        bins_with_events.append(bin_count)

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(ds)}  '
                  f'dur={duration/1e6:.2f}s  events={len(t)}  '
                  f'N_A/T={N_A/args.T_train:.3f}')

    NA = np.array(NA_per_sample)
    durations_s = np.array(duration_per_sample) / 1e6
    nevents = np.array(nevents_per_sample)

    # Save
    np.savez(args.out,
             NA_per_T=NA,
             duration_s=durations_s,
             nevents=nevents,
             bin_counts=np.stack(bins_with_events),
             T_train=args.T_train)
    print(f'\nSaved {args.out}')

    # Report
    print(f'\n=== M1 results: N_A/T_train on {args.split} ({len(NA)} samples) ===')
    print(f'  mean   : {NA.mean():.3f}')
    print(f'  p25    : {np.percentile(NA, 25):.3f}')
    print(f'  p50    : {np.percentile(NA, 50):.3f}')
    print(f'  p75    : {np.percentile(NA, 75):.3f}')
    print(f'  max    : {NA.max():.3f}')
    print(f'  min    : {NA.min():.3f}')
    print(f'\n  mean sample duration: {durations_s.mean():.2f} s  (Δt_train = {durations_s.mean()/args.T_train:.3f} s)')
    print(f'  mean events per sample: {nevents.mean():.0f}')

    # R10 trigger check
    print(f'\n=== R10 check ===')
    if NA.mean() < 0.10:
        print('RED: Mode A exact-skip headroom < 10%. R10 triggered.')
        print('     Paper must reframe to Mode B-dominant.')
    elif NA.mean() < 0.30:
        print('YELLOW: Mode A headroom 10-30%. Marginal; rely on Mode B for')
        print('        substantial speedup.')
    else:
        print(f'GREEN: Mode A headroom {NA.mean()*100:.1f}%. Exact-skip viable.')


if __name__ == '__main__':
    main()