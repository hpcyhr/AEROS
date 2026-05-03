"""
Phase 1.2b — K selector with corrected validation:
  budget = sweet_mem * (1 + margin) to account for fit noise + real-world headroom.
"""
import numpy as np


def fit_memory_model(K_results):
    valid = [(r['K'], r['peak_mem_GB']) for r in K_results
             if r['status'] == 'ok' and r['peak_mem_GB'] is not None]
    if len(valid) < 2:
        return None, None, None
    Ks = np.array([k for k, _ in valid], dtype=np.float64)
    mems = np.array([m for _, m in valid], dtype=np.float64)
    A = np.stack([Ks, np.ones_like(Ks)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, mems, rcond=None)
    alpha, M_static = float(coef[0]), float(coef[1])
    pred = alpha * Ks + M_static
    ss_res = ((mems - pred) ** 2).sum()
    ss_tot = ((mems - mems.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return alpha, M_static, R2


def select_K(T, alpha, M_static, budget_GB):
    if alpha is None or alpha <= 0:
        return None
    K_max_budget = (budget_GB - M_static) / alpha
    if K_max_budget < 1:
        return None
    valid_Ks = [d for d in range(1, T + 1) if T % d == 0]
    feasible = [K for K in valid_Ks if K <= K_max_budget]
    return max(feasible) if feasible else None


def validate(margin):
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']
    n_total, n_correct, n_within = 0, 0, 0
    print(f'\n=== Validation with budget = sweet_mem * {1+margin:.2f} ===')
    print(f'{"T":>4} {"B":>3} {"HW":>9}  {"sweet_K":>7} {"budget":>7} {"pred_K":>6} {"match":>6}')
    print('-' * 65)
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']; B = r['B']; H = r['H']
        K_results = r.get('K_results', [])
        alpha, M_static, R2 = fit_memory_model(K_results)
        sweet_K = r.get('sweet_K')
        sweet_mem = r.get('sweet_mem')
        if sweet_K is None or alpha is None or sweet_mem is None:
            continue
        budget = sweet_mem * (1 + margin)
        pred_K = select_K(T, alpha, M_static, budget)
        match = '✓' if pred_K == sweet_K else '✗'
        within = (
            pred_K is not None
            and abs(np.log2(max(pred_K, 1)) - np.log2(max(sweet_K, 1))) <= 1.01
        )
        if pred_K == sweet_K: n_correct += 1
        if within: n_within += 1
        n_total += 1
        print(f'{T:>4} {B:>3} {H}x{H:<5}  K*={sweet_K:<3} {budget:>6.2f}  '
              f'pred={pred_K!s:<5} {match}')
    if n_total > 0:
        print(f'\n  Exact:    {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)')
        print(f'  Within 1: {n_within}/{n_total} ({n_within/n_total*100:.0f}%)')
    return n_correct / n_total if n_total > 0 else 0


# Try several margins
print('=== Margin sensitivity sweep ===')
for m in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    rate = validate(m)


# More principled approach: select K such that 
# predicted_mem(K) <= budget, BUT K should be smaller of [floor result, 
# next K up if mem(K_up) < budget]
def select_K_robust(T, alpha, M_static, budget_GB, mem_safety=0.05):
    """Robust selector: include 5% mem safety margin INSIDE the formula,
    so user-supplied budget doesn't need to be inflated."""
    if alpha is None or alpha <= 0:
        return None
    # Effective ceiling: budget * (1 - safety)
    effective = budget_GB * (1 - mem_safety)
    K_max = (effective - M_static) / alpha
    if K_max < 1:
        return None
    valid_Ks = [d for d in range(1, T + 1) if T % d == 0]
    feasible = [K for K in valid_Ks if K <= K_max]
    return max(feasible) if feasible else None


# Validate robust version using budget = baseline_mem (the realistic upper bound)
def validate_robust():
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']
    n_total, n_correct, n_within = 0, 0, 0
    print(f'\n=== Robust selector: budget = baseline_mem (realistic) ===')
    print(f'{"T":>4} {"B":>3} {"HW":>9}  {"sweet_K":>7} {"baseline":>9} {"pred":>5} {"match":>6}')
    print('-' * 65)
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']; B = r['B']; H = r['H']
        K_results = r.get('K_results', [])
        alpha, M_static, R2 = fit_memory_model(K_results)
        sweet_K = r.get('sweet_K')
        if sweet_K is None or alpha is None:
            continue
        # Use baseline_mem as budget (realistic: user has whatever memory baseline used)
        budget = r.get('baseline_mem')
        if budget is None:
            # baseline OOM, use V100 max
            budget = 30.0  # V100 32GB practical max
        pred_K = select_K_robust(T, alpha, M_static, budget, mem_safety=0.0)
        match = '✓' if pred_K == sweet_K else '✗'
        within = (
            pred_K is not None
            and abs(np.log2(max(pred_K, 1)) - np.log2(max(sweet_K, 1))) <= 1.01
        )
        if pred_K == sweet_K: n_correct += 1
        if within: n_within += 1
        n_total += 1
        print(f'{T:>4} {B:>3} {H}x{H:<5}  K*={sweet_K:<3}  '
              f'{budget:>7.2f}  K={pred_K!s:<4} {match}')
    print(f'\n  Exact:    {n_correct}/{n_total}')
    print(f'  Within 1: {n_within}/{n_total}')


validate_robust()
