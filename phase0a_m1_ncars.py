"""
AEROS Phase 0A — Measurement M1 on N-CARS (multi-scale occupancy).

N-CARS samples are 100ms each, raw event recordings from ATIS sensor
behind a car windshield. Parser based on Prophesee .dat format spec.
"""
import argparse, os, glob
import numpy as np


def read_prophesee_dat(path):
    """Read Prophesee .dat file → 1D ndarray of timestamps (microseconds).

    Format (from Prophesee SDK docs):
      - ASCII header lines starting with '%' (variable length)
      - Then 2 bytes: ev_type (uint8) + ev_size (uint8)
      - Then 8 bytes per event:
          uint32 timestamp (little-endian)
          uint32 packed: bits 0-13=x, 14-27=y, 28=polarity, 29-31=padding
    """
    with open(path, 'rb') as f:
        # Skip ASCII header (% comments)
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'%'):
                f.seek(pos)
                break
        # 2-byte event type + size header
        _ = f.read(2)
        # Raw events
        data = np.fromfile(f, dtype=[('t', '<u4'), ('packed', '<u4')])
    return data['t']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',
                   default='/data/yhr/datasets/n_cars/Prophesee_Dataset_n_cars')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--T_train', type=int, default=20)
    p.add_argument('--max_samples', type=int, default=2000,
                   help='cap to keep runtime reasonable')
    p.add_argument('--out', default='m1_ncars_multiscale.npz')
    args = p.parse_args()

    split_dir = f'n-cars_{args.split}'
    cars_path = os.path.join(args.data_dir, split_dir, 'cars')
    bg_path = os.path.join(args.data_dir, split_dir, 'background')

    # Collect file list
    files = []
    for d, label in [(cars_path, 'cars'), (bg_path, 'background')]:
        if not os.path.isdir(d):
            print(f'  WARN: {d} not found')
            continue
        fs = sorted(glob.glob(os.path.join(d, '*.dat')))
        files.extend([(f, label) for f in fs])
    print(f'Found {len(files)} .dat files in {split_dir}')

    # Subsample if too many
    if len(files) > args.max_samples:
        np.random.seed(0)
        idx = np.random.choice(len(files), args.max_samples, replace=False)
        files = [files[i] for i in idx]
        print(f'  Subsampling to {len(files)} samples')

    bin_widths_us = [100, 250, 500, 1000, 2000, 5000,
                     10000, 50000, 100000]
    occupancy_by_scale = {bw: [] for bw in bin_widths_us}
    occupancy_at_dt_train = []
    durations_us = []
    nevents = []
    parse_errors = 0

    for i, (f, _label) in enumerate(files):
        try:
            t = read_prophesee_dat(f)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                print(f'  parse error on {f}: {e}')
            continue

        if len(t) == 0:
            continue
        t = t.astype(np.int64)
        t_min, t_max = int(t.min()), int(t.max())
        dur = t_max - t_min
        if dur <= 0:
            continue

        durations_us.append(dur)
        nevents.append(len(t))

        for bw in bin_widths_us:
            n_bins = max(1, int(dur // bw))
            bin_idx = np.minimum((t - t_min) // bw, n_bins - 1).astype(np.int64)
            bin_count = np.bincount(bin_idx, minlength=n_bins)
            occupancy_by_scale[bw].append((bin_count > 0).mean())

        dt_train_us = dur / args.T_train
        bin_idx = np.minimum(((t - t_min) / dt_train_us).astype(np.int64),
                             args.T_train - 1)
        bin_count = np.bincount(bin_idx, minlength=args.T_train)
        occupancy_at_dt_train.append((bin_count > 0).mean())

        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(files)}')

    # Save
    save_dict = {
        'bin_widths_us': np.array(bin_widths_us),
        'occ_dt_train': np.array(occupancy_at_dt_train),
        'durations_us': np.array(durations_us),
        'nevents': np.array(nevents),
        'T_train': args.T_train,
    }
    for bw in bin_widths_us:
        save_dict[f'occ_{bw}us'] = np.array(occupancy_by_scale[bw])
    np.savez(args.out, **save_dict)
    print(f'\nSaved {args.out}  (parse errors: {parse_errors})')

    print(f'\nMean sample duration: {np.mean(durations_us) / 1000:.1f} ms')
    print(f'Mean events / sample : {np.mean(nevents):.0f}')
    print(f'Mean event rate      : {np.mean(nevents) / (np.mean(durations_us) / 1e6):.0f} ev/s')

    print(f'\n=== Multi-scale non-empty bin fraction on N-CARS {args.split} '
          f'({len(occupancy_at_dt_train)} samples) ===')
    print(f'{"scale":>12s}  {"mean":>8s}  {"p10":>8s}  '
          f'{"p50":>8s}  {"p90":>8s}  {"min":>8s}  {"max":>8s}')
    print('-' * 70)
    for bw in bin_widths_us:
        a = np.array(occupancy_by_scale[bw])
        if bw < 1000:
            label = f'{bw}us'
        else:
            label = f'{bw // 1000}ms'
        print(f'{label:>12s}  {a.mean():.4f}  {np.percentile(a, 10):.4f}  '
              f'{np.percentile(a, 50):.4f}  {np.percentile(a, 90):.4f}  '
              f'{a.min():.4f}  {a.max():.4f}')
    a = np.array(occupancy_at_dt_train)
    print(f'{"Δt_train":>12s}  {a.mean():.4f}  {np.percentile(a, 10):.4f}  '
          f'{np.percentile(a, 50):.4f}  {np.percentile(a, 90):.4f}  '
          f'{a.min():.4f}  {a.max():.4f}')

    print(f'\n=== Quick verdict ===')
    occ_1ms = np.array(occupancy_by_scale[1000])
    occ_5ms = np.array(occupancy_by_scale[5000])

    if occ_5ms.mean() < 0.7:
        print(f'GREEN: 5ms-scale empty bin frac = {1 - occ_5ms.mean():.2f}.')
        print(f'  AEROS has substantial fine-scale skip headroom.')
    elif occ_5ms.mean() < 0.95:
        print(f'YELLOW: 5ms-scale empty bin frac = {1 - occ_5ms.mean():.2f}.')
        print(f'  Marginal headroom; depends on w_amortized.')
    else:
        print(f'RED: 5ms-scale empty bin frac = {1 - occ_5ms.mean():.3f}.')
        print(f'  Same pattern as DVS Gesture. AEROS depends on Gen1.')


if __name__ == '__main__':
    main()