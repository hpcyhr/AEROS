"""
AEROS Phase 0A — extended M1 with multi-scale occupancy.

For each test sample, compute non-empty bin fraction at multiple bin widths:
  100µs, 250µs, 500µs, 1ms, 2ms, 5ms, 10ms, 50ms, 100ms, plus Δt_train

This reveals whether burstiness exists at executable scales below Δt_train.
- If non-empty fraction is < 0.5 at fine scales (1-10ms): there ARE
  empty intervals at fine scale; AEROS could exploit them via finer-grained
  training (larger T_train) or Mode B.
- If non-empty fraction stays ~1.0 at all scales: the event stream is
  truly continuous; AEROS has no skip opportunity on this dataset.
"""
import argparse
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/dvs128_gesture')
    p.add_argument('--T_train', type=int, default=20)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--out', default='m1_dvsg_multiscale.npz')
    args = p.parse_args()

    print(f'Loading DVS128 Gesture {args.split} (raw events)...')
    ds = DVS128Gesture(args.data_dir, train=(args.split == 'train'),
                       data_type='event')
    print(f'  {len(ds)} samples')

    # bin widths in microseconds
    bin_widths_us = [100, 250, 500, 1000, 2000, 5000,
                     10000, 50000, 100000]

    occupancy_by_scale = {bw: [] for bw in bin_widths_us}
    occupancy_at_dt_train = []

    # Also collect bin event-count distribution at one moderate scale
    # for later analysis (use 1ms)
    bin_counts_1ms_all = []

    for i in range(len(ds)):
        events, _ = ds[i]
        # SJ data_type='event' returns NpzFile; access via key
        t = events['t']
        if len(t) == 0:
            continue
        t = t.astype(np.int64)
        t_min, t_max = int(t.min()), int(t.max())
        dur = t_max - t_min
        if dur <= 0:
            continue

        for bw in bin_widths_us:
            n_bins = max(1, int(dur // bw))
            bin_idx = np.minimum((t - t_min) // bw, n_bins - 1).astype(np.int64)
            bin_count = np.bincount(bin_idx, minlength=n_bins)
            non_empty_frac = (bin_count > 0).mean()
            occupancy_by_scale[bw].append(non_empty_frac)

            # Save bin_count distribution at 1ms scale (one sample's worth)
            if bw == 1000 and len(bin_counts_1ms_all) < 50:
                # Subsample to keep size manageable
                bin_counts_1ms_all.append(bin_count[:5000])

        # Per-sample Δt_train measurement
        dt_train_us = dur / args.T_train
        bin_idx = np.minimum(((t - t_min) / dt_train_us).astype(np.int64),
                             args.T_train - 1)
        bin_count = np.bincount(bin_idx, minlength=args.T_train)
        occupancy_at_dt_train.append((bin_count > 0).mean())

        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(ds)}')

    # Save all data
    save_dict = {
        'bin_widths_us': np.array(bin_widths_us),
        'occ_dt_train': np.array(occupancy_at_dt_train),
        'T_train': args.T_train,
    }
    for bw in bin_widths_us:
        save_dict[f'occ_{bw}us'] = np.array(occupancy_by_scale[bw])
    np.savez(args.out, **save_dict)
    print(f'\nSaved {args.out}')

    # Print summary table
    print(f'\n=== Multi-scale non-empty bin fraction on {args.split} '
          f'({len(occupancy_at_dt_train)} samples) ===')
    print(f'{"scale":>12s}  {"mean":>8s}  {"p10":>8s}  '
          f'{"p50":>8s}  {"p90":>8s}  {"min":>8s}  {"max":>8s}')
    print('-' * 70)
    for bw in bin_widths_us:
        a = np.array(occupancy_by_scale[bw])
        if bw < 1000:
            label = f'{bw}us'
        elif bw < 1000000:
            label = f'{bw // 1000}ms'
        else:
            label = f'{bw // 1000000}s'
        print(f'{label:>12s}  {a.mean():.4f}  {np.percentile(a, 10):.4f}  '
              f'{np.percentile(a, 50):.4f}  {np.percentile(a, 90):.4f}  '
              f'{a.min():.4f}  {a.max():.4f}')
    a = np.array(occupancy_at_dt_train)
    print(f'{"Δt_train":>12s}  {a.mean():.4f}  {np.percentile(a, 10):.4f}  '
          f'{np.percentile(a, 50):.4f}  {np.percentile(a, 90):.4f}  '
          f'{a.min():.4f}  {a.max():.4f}')

    # Interpretation
    print(f'\n=== Interpretation guide ===')
    occ_1ms = np.array(occupancy_by_scale[1000])
    occ_5ms = np.array(occupancy_by_scale[5000])
    occ_10ms = np.array(occupancy_by_scale[10000])

    print(f'1ms scale  : mean non-empty frac = {occ_1ms.mean():.3f}  '
          f'(empty bin frac = {1 - occ_1ms.mean():.3f})')
    print(f'5ms scale  : mean non-empty frac = {occ_5ms.mean():.3f}  '
          f'(empty bin frac = {1 - occ_5ms.mean():.3f})')
    print(f'10ms scale : mean non-empty frac = {occ_10ms.mean():.3f}  '
          f'(empty bin frac = {1 - occ_10ms.mean():.3f})')

    print('\nDecision tree:')
    if occ_1ms.mean() < 0.5 or occ_5ms.mean() < 0.5:
        print('  GREEN(ish): substantial burstiness exists at 1-5ms scale.')
        print('    AEROS could exploit it via:')
        print('    1) larger T_train training (e.g., T=100-200, Δt~30-70ms)')
        print('    2) Mode B at fine sub-Δt_train scale')
    elif occ_10ms.mean() < 0.85:
        print('  YELLOW: some burstiness at 10ms+ scale, marginal headroom.')
        print('    AEROS Mode A would need very large T_train to capture this.')
        print('    Mode B is the more realistic path.')
    else:
        print('  RED: event stream is essentially continuous at all scales')
        print('    above 1ms. DVS Gesture has no temporal-skip opportunity.')
        print('    AEROS must depend on N-CARS / Gen1 for its main story.')


if __name__ == '__main__':
    main()