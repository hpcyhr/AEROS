"""
Phase 1.2 — Budget-aware K selector with closed-form derivation.

Memory model:
  peak_mem(K) = M_static + alpha * K

  where:
    M_static  ~  weights + LIF state + framework  (independent of K)
    alpha     ~  per-step activation footprint

Selector:
  K* = max K such that K | T and M_static + alpha * K <= budget
       (smaller K => more memory savings; we want the smallest K that fits budget)
       
  Actually we want the LARGEST K that still fits, because larger K = less slowdown.
  Smaller K = more memory savings but more slowdown.
"""
import numpy as np


# -----------------------------------------------------------------------------
# Step 1: Fit alpha and M_static from sweep data
# -----------------------------------------------------------------------------
def fit_memory_model(K_results):
    """
    K_results: list of {'K': int, 'peak_mem_GB': float, 'status': 'ok'|...}
    Returns (alpha, M_static, fit_R2)
    """
    valid = [(r['K'], r['peak_mem_GB']) for r in K_results
             if r['status'] == 'ok' and r['peak_mem_GB'] is not None]
    if len(valid) < 2:
        return None, None, None
    Ks = np.array([k for k, _ in valid], dtype=np.float64)
    mems = np.array([m for _, m in valid], dtype=np.float64)
    # Linear regression: mem = alpha * K + M_static
    A = np.stack([Ks, np.ones_like(Ks)], axis=1)
    coef, residuals, rank, _ = np.linalg.lstsq(A, mems, rcond=None)
    alpha, M_static = float(coef[0]), float(coef[1])
    pred = alpha * Ks + M_static
    ss_res = ((mems - pred) ** 2).sum()
    ss_tot = ((mems - mems.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return alpha, M_static, R2


# -----------------------------------------------------------------------------
# Step 2: Selector — given budget and T, return largest K that fits and K | T
# -----------------------------------------------------------------------------
def select_K(T, alpha, M_static, budget_GB, valid_Ks=None):
    """
    Returns largest K such that K | T and M_static + alpha*K <= budget_GB.
    valid_Ks defaults to all divisors of T.
    """
    if alpha is None or alpha <= 0:
        return None  # cannot fit
    # Max K from budget
    K_max_budget = (budget_GB - M_static) / alpha
    if K_max_budget < 1:
        return None  # not enough budget even for K=1
    if valid_Ks is None:
        valid_Ks = [d for d in range(1, T + 1) if T % d == 0]
    # Pick largest K <= K_max_budget
    feasible = [K for K in valid_Ks if K <= K_max_budget]
    return max(feasible) if feasible else None


# -----------------------------------------------------------------------------
# Step 3: Validate against 1.1d sweep
# -----------------------------------------------------------------------------
def validate_selector():
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']
    print(f'Validating selector on {len(results)} configs from 1.1d sweep...')
    print()
    print(f'{"T":>4s} {"B":>3s} {"HW":>9s}  {"alpha":>7s} {"M_st":>6s} {"R2":>6s}  '
          f'{"sweet_K":>7s} {"pred_K@sweet_mem":>15s} {"match?":>7s}')
    print('-' * 90)
    n_correct = 0
    n_total = 0
    n_within_one = 0
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']; B = r['B']; H = r['H']
        K_results = r.get('K_results', [])
        alpha, M_static, R2 = fit_memory_model(K_results)
        sweet_K = r.get('sweet_K')
        if sweet_K is None or alpha is None:
            print(f'{T:>4d} {B:>3d} {H}x{H:<5d}  -- (insufficient data)')
            continue
        sweet_mem = r.get('sweet_mem')
        # Predict K given sweet_mem as budget
        pred_K = select_K(T, alpha, M_static, sweet_mem)
        match = '✓' if pred_K == sweet_K else '✗'
        within_one = (
            pred_K is not None
            and sweet_K is not None
            and abs(np.log2(max(pred_K, 1)) - np.log2(max(sweet_K, 1))) <= 1.01
        )
        if pred_K == sweet_K:
            n_correct += 1
        if within_one:
            n_within_one += 1
        n_total += 1
        print(f'{T:>4d} {B:>3d} {H}x{H:<5d}  '
              f'{alpha:>7.4f} {M_static:>6.2f} {R2:>6.3f}  '
              f'K*={sweet_K:<3d}  pred K={pred_K!s:<5s}  {match}')
    if n_total > 0:
        print(f'\n=== Selector accuracy ===')
        print(f'  Exact match:   {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)')
        print(f'  Within 1 octave: {n_within_one}/{n_total} ({n_within_one/n_total*100:.0f}%)')


# -----------------------------------------------------------------------------
# Step 4: alpha / M_static cross-config relationship — does alpha scale with B*C*H*W?
# -----------------------------------------------------------------------------
def explore_alpha_scaling():
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']
    rows = []
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']; B = r['B']; H = r['H']; W = r['W']
        K_results = r.get('K_results', [])
        alpha, M_static, R2 = fit_memory_model(K_results)
        if alpha is None or R2 < 0.9:
            continue
        rows.append({
            'T': T, 'B': B, 'H': H,
            'BHW': B * H * W,
            'BCHW_GB': B * 3 * H * W * 4 / 1e9,
            'alpha': alpha,
            'M_static': M_static,
            'R2': R2,
        })

    print(f'\n=== alpha scaling analysis ===')
    print(f'{"T":>4s} {"B":>3s} {"H":>4s}  {"input_BCHW_GB":>13s}  '
          f'{"alpha":>8s}  {"alpha/input":>12s}  {"M_static":>8s}')
    for row in sorted(rows, key=lambda r: r['BHW']):
        print(f'{row["T"]:>4d} {row["B"]:>3d} {row["H"]:>4d}   '
              f'{row["BCHW_GB"]:>12.4f}   '
              f'{row["alpha"]:>7.4f}   '
              f'{row["alpha"]/row["BCHW_GB"]:>11.2f}   '
              f'{row["M_static"]:>7.3f}')

    # Linear fit: alpha ~ k * input_size
    if rows:
        inputs = np.array([r['BCHW_GB'] for r in rows])
        alphas = np.array([r['alpha'] for r in rows])
        # alpha = k * input_BCHW
        k = (alphas * inputs).sum() / (inputs * inputs).sum()
        print(f'\nFit: alpha ≈ {k:.2f} × input_BCHW_GB')
        print(f'(meaning: per-K activation footprint is ~{k:.1f}× the raw input size,')
        print(f' roughly the network activation expansion factor at peak intermediate layer)')


if __name__ == '__main__':
    validate_selector()
    explore_alpha_scaling()
