"""
Phase 1.3b — 双变量 memory model + layer-aware non-uniform planner.

Refit:  peak_mem = M_const + alpha_T * T + alpha_K * K_max
        across all 36 sweep configs simultaneously.
"""
import numpy as np


def fit_2var_model_per_BH(results):
    """
    Per (B, H) group, fit:
      peak_mem(T, K) = M_const + alpha_T * T + alpha_K * K
    Stack all (T, K, mem) points for that (B, H), 3-param linear regression.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if 'skip' in r:
            continue
        B, H = r['B'], r['H']
        T = r['T']
        Kr = r.get('K_results', [])
        for kr in Kr:
            if kr['status'] != 'ok':
                continue
            groups[(B, H)].append((T, kr['K'], kr['peak_mem_GB']))

    fits = {}
    for (B, H), pts in groups.items():
        if len(pts) < 4:
            continue
        Ts = np.array([t for t, _, _ in pts], dtype=np.float64)
        Ks = np.array([k for _, k, _ in pts], dtype=np.float64)
        mems = np.array([m for _, _, m in pts], dtype=np.float64)
        A = np.stack([Ts, Ks, np.ones_like(Ts)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, mems, rcond=None)
        a_T, a_K, M_const = float(coef[0]), float(coef[1]), float(coef[2])
        pred = a_T * Ts + a_K * Ks + M_const
        ss_res = ((mems - pred) ** 2).sum()
        ss_tot = ((mems - mems.mean()) ** 2).sum()
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        fits[(B, H)] = {
            'a_T': a_T, 'a_K': a_K, 'M_const': M_const,
            'R2': R2, 'n_points': len(pts),
        }
    return fits


def main():
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']

    fits = fit_2var_model_per_BH(results)
    print(f'=== 2-variable memory model per (B, H) ===')
    print(f'  peak_mem = M_const + alpha_T * T + alpha_K * K')
    print()
    print(f'{"B":>3} {"H":>4}  {"alpha_T":>8} {"alpha_K":>8} {"M_const":>8} '
          f'{"R2":>6} {"n_pts":>5}')
    print('-' * 60)
    for (B, H), f in sorted(fits.items()):
        print(f'{B:>3} {H:>4}   '
              f'{f["a_T"]:>7.5f}  {f["a_K"]:>7.5f}  {f["M_const"]:>7.3f}  '
              f'{f["R2"]:>5.3f}  {f["n_points"]:>4}')

    # Compare alpha_T to expected input tensor cost
    print(f'\n=== alpha_T vs theoretical input tensor cost ===')
    print(f'{"B":>3} {"H":>4}  {"alpha_T_meas":>11} {"alpha_T_theory":>14} {"ratio":>6}')
    for (B, H), f in sorted(fits.items()):
        # Input tensor adds B*3*H*H*4 bytes per T step = B*3*H*H*4/1e9 GB/step
        theory = B * 3 * H * H * 4 / 1e9
        ratio = f['a_T'] / theory if theory > 0 else 0
        print(f'{B:>3} {H:>4}   {f["a_T"]:>10.6f}    {theory:>12.6f}   {ratio:>5.2f}x')

    # Validate: predict on Test 3 data (T variation at K_max=16)
    print(f'\n=== Validation: predict mem for (B=32, H=128, K_max=16) at varying T ===')
    f = fits.get((32, 128))
    if f:
        print(f'  Using fit: a_T={f["a_T"]:.5f}, a_K={f["a_K"]:.5f}, M_const={f["M_const"]:.3f}')
        actuals = {64: 1.81, 100: 2.06, 128: 2.24, 200: 2.72, 256: 3.09}
        print(f'{"T":>4}  {"actual_GB":>10} {"predict_GB":>11} {"err":>7}')
        for T, actual in actuals.items():
            pred = f['a_T'] * T + f['a_K'] * 16 + f['M_const']
            err = abs(pred - actual)
            print(f'{T:>4}   {actual:>9.2f}    {pred:>10.2f}   {err:>6.3f}')


if __name__ == '__main__':
    main()
