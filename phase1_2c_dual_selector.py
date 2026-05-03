"""
Phase 1.2c — Dual K selector with Problem A (budget) and Problem B (slowdown).

Problem A (budget):     K_A* = max{K | T, M_static + alpha*K <= budget * (1 - safety)}
Problem B (slowdown):   K_B* = min{K | T, wall(K) / wall_max <= slowdown_max}

For B, we fit a wall-clock model:
    wall(K) ~ wall_min + beta * (T / K)   [more chunks = more overhead]
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


def fit_wall_model(K_results, T):
    """Fit: wall(K) = wall_min + beta * (T/K - 1)
                    ≈ wall_min when K=T, increases as K shrinks."""
    valid = [(r['K'], r['wall_ms']) for r in K_results
             if r['status'] == 'ok' and r['wall_ms'] is not None]
    if len(valid) < 2:
        return None, None, None
    Ks = np.array([k for k, _ in valid], dtype=np.float64)
    walls = np.array([w for _, w in valid], dtype=np.float64)
    # x = T/K - 1 (zero when K=T, large when K small)
    x = T / Ks - 1
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, walls, rcond=None)
    beta, wall_min = float(coef[0]), float(coef[1])
    pred = beta * x + wall_min
    ss_res = ((walls - pred) ** 2).sum()
    ss_tot = ((walls - walls.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return beta, wall_min, R2


def selector_A(T, alpha, M_static, budget_GB, safety=0.05):
    """Largest K satisfying memory budget."""
    if alpha is None or alpha <= 0:
        return None
    eff = budget_GB * (1 - safety)
    K_max = (eff - M_static) / alpha
    if K_max < 1:
        return None
    valid_Ks = [d for d in range(1, T + 1) if T % d == 0]
    feasible = [K for K in valid_Ks if K <= K_max]
    return max(feasible) if feasible else None


def selector_B(T, beta, wall_min, slowdown_max):
    """Smallest K satisfying slowdown <= slowdown_max."""
    if beta is None or wall_min is None or beta <= 0:
        return None
    # wall(K) / wall(T) <= slowdown_max
    # wall(T) = wall_min + beta * 0 = wall_min
    # wall(K) = wall_min + beta * (T/K - 1)
    # => (wall_min + beta * (T/K - 1)) / wall_min <= slowdown_max
    # => beta * (T/K - 1) <= (slowdown_max - 1) * wall_min
    # => T/K <= 1 + (slowdown_max - 1) * wall_min / beta
    # => K >= T / (1 + (slowdown_max - 1) * wall_min / beta)
    if slowdown_max <= 1:
        return T  # only K=T fits
    K_min = T / (1 + (slowdown_max - 1) * wall_min / beta)
    valid_Ks = [d for d in range(1, T + 1) if T % d == 0]
    feasible = [K for K in valid_Ks if K >= K_min]
    return min(feasible) if feasible else T


# --------------------------- Validation ---------------------------
def main():
    data = np.load('phase1_1d_sweep.npz', allow_pickle=True)
    results = data['results']

    # Step 1: check wall-clock model R² across all configs
    print('=== Wall-clock model fit quality ===')
    print(f'{"T":>4} {"B":>3} {"HW":>9}  {"beta":>7} {"wall_min":>9} {"R2":>6}')
    print('-' * 60)
    wall_R2_list = []
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']
        Kr = r.get('K_results', [])
        beta, wmin, R2 = fit_wall_model(Kr, T)
        if R2 is None:
            continue
        wall_R2_list.append(R2)
        print(f'{T:>4} {r["B"]:>3} {r["H"]}x{r["H"]:<5}  '
              f'{beta:>6.2f}  {wmin:>8.1f}  {R2:>6.3f}')
    print(f'\nMean wall R²: {np.mean(wall_R2_list):.3f}, '
          f'min: {np.min(wall_R2_list):.3f}')

    # Step 2: Problem A — budget-constrained validation
    print('\n=== Problem A: budget-constrained (3 budget scenarios) ===')
    print(f'{"T":>4} {"B":>3} {"HW":>9}  '
          f'{"5GB→K":>6} {"actual_GB":>9} {"slowdown":>8}  '
          f'{"16GB→K":>7} {"actual_GB":>9} {"slowdown":>8}')
    print('-' * 95)

    n_a_correct_5GB, n_a_correct_16GB, n_a_total = 0, 0, 0
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']
        Kr = r.get('K_results', [])
        alpha, M_static, _ = fit_memory_model(Kr)
        if alpha is None:
            continue

        cells = []
        for budget in [5.0, 16.0]:
            K_pred = selector_A(T, alpha, M_static, budget)
            if K_pred is None:
                cells.append((None, None, None, False))
                continue
            # Look up actual measured mem and wall for this K
            actual = next((kr for kr in Kr if kr['K'] == K_pred), None)
            if actual is None or actual['status'] != 'ok':
                cells.append((K_pred, None, None, False))
                continue
            mem_K = actual['peak_mem_GB']
            wall_K = actual['wall_ms']
            # Reference wall: largest successful K
            wall_ref = next((kr['wall_ms'] for kr in Kr
                            if kr['status'] == 'ok' and kr['wall_ms'] is not None), None)
            slowdown = wall_K / wall_ref if wall_ref else None
            within_budget = mem_K <= budget * 1.05  # allow 5% slack
            cells.append((K_pred, mem_K, slowdown, within_budget))

        # Print row
        row = f'{T:>4} {r["B"]:>3} {r["H"]}x{r["H"]:<5}  '
        for K_pred, mem_K, sd, ok in cells:
            if K_pred is None:
                row += f'{"—":>6}  {"—":>9}  {"—":>8}'
            else:
                row += f'{K_pred:>6}  {mem_K:>8.2f}  {sd if sd else 0:>7.2f}x'
        print(row)
        n_a_total += 1
        if cells[0][3]: n_a_correct_5GB += 1
        if cells[1][3]: n_a_correct_16GB += 1

    print(f'\n  5GB: {n_a_correct_5GB}/{n_a_total} configs respected budget')
    print(f' 16GB: {n_a_correct_16GB}/{n_a_total} configs respected budget')

    # Step 3: Problem B — slowdown-constrained validation
    print('\n=== Problem B: slowdown-constrained (3 tolerance scenarios) ===')
    print(f'{"T":>4} {"B":>3} {"HW":>9}  '
          f'{"S=1.2→K":>9} {"actual_sd":>10} {"savings":>8}  '
          f'{"S=1.5→K":>9} {"actual_sd":>10} {"savings":>8}')
    print('-' * 100)

    n_b_correct_12, n_b_correct_15, n_b_total = 0, 0, 0
    for r in results:
        if 'skip' in r:
            continue
        T = r['T']
        Kr = r.get('K_results', [])
        beta, wmin, _ = fit_wall_model(Kr, T)
        if beta is None or wmin is None:
            continue

        # Reference: largest successful K wall and mem
        ref = next((kr for kr in Kr if kr['status'] == 'ok'), None)
        if ref is None:
            continue
        wall_ref = ref['wall_ms']
        mem_ref = ref['peak_mem_GB']

        cells = []
        for S in [1.2, 1.5]:
            K_pred = selector_B(T, beta, wmin, S)
            if K_pred is None:
                cells.append((None, None, None, False))
                continue
            actual = next((kr for kr in Kr if kr['K'] == K_pred and kr['status'] == 'ok'), None)
            if actual is None:
                cells.append((K_pred, None, None, False))
                continue
            sd = actual['wall_ms'] / wall_ref
            savings = mem_ref / actual['peak_mem_GB']
            within = sd <= S * 1.10  # allow 10% slack
            cells.append((K_pred, sd, savings, within))

        row = f'{T:>4} {r["B"]:>3} {r["H"]}x{r["H"]:<5}  '
        for K_pred, sd, sv, ok in cells:
            if K_pred is None:
                row += f'{"—":>9}  {"—":>10}  {"—":>8}'
            else:
                row += f'{K_pred:>9}  {sd:>9.2f}x  {sv:>7.2f}x'
        print(row)
        n_b_total += 1
        if cells[0][3]: n_b_correct_12 += 1
        if cells[1][3]: n_b_correct_15 += 1

    print(f'\n  S=1.2: {n_b_correct_12}/{n_b_total} configs respected slowdown bound')
    print(f'  S=1.5: {n_b_correct_15}/{n_b_total} configs respected slowdown bound')


if __name__ == '__main__':
    main()
