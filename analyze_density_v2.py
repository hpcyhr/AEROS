"""Per-layer verdict for AEROS density study."""
import argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--inp', default='density_nmnist_trained.npz')
p.add_argument('--sparse_thresh', type=float, default=0.10)
p.add_argument('--dense_thresh',  type=float, default=0.30)
args = p.parse_args()

d = np.load(args.inp)
print(f'\n{"layer":8s} {"mean":>8s} {"std":>8s} {"regime":>10s}')
print('-' * 40)
n_sparse, n_mid, n_dense = 0, 0, 0
for k in d.files:
    a = d[k]
    m = a.mean()
    s = a.std()
    if m < args.sparse_thresh:
        regime = 'SPARSE'; n_sparse += 1
    elif m < args.dense_thresh:
        regime = 'MID';    n_mid += 1
    else:
        regime = 'DENSE';  n_dense += 1
    print(f'{k:8s} {m:8.4f} {s:8.4f} {regime:>10s}')

print(f'\nLayers: sparse={n_sparse}, mid={n_mid}, dense={n_dense}')
print(f'\n=== Verdict ===')
if n_sparse == 0:
    print('RED: no layer is sparse enough (<10% mean) for event-driven path.')
    print('AEROS event path has no headroom. Reconsider direction.')
elif n_dense == 0 and n_mid == 0:
    print('YELLOW: all layers sparse. AEROS reduces to "event-driven everywhere",')
    print('no density-adaptive story. Still publishable but loses key novelty axis.')
else:
    print(f'GREEN: layer-heterogeneous distribution detected.')
    print(f'  {n_sparse} sparse layer(s) -> event/bitset path beneficial')
    print(f'  {n_mid + n_dense} non-sparse layer(s) -> dense path mandatory')
    print('Density-adaptive dispatch has headroom. Proceed.')