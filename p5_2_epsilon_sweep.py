"""
P5-2: Calibrated-ε sweep for Problem A (budget-constrained selector).

Goal: directly close GPT P4-1's residual concern. v5 paper §5.3 reports 30/33
compliant at M=5GB with ε=0.05, with 3 violations falling in 1.09–1.14× over
budget. This script re-runs the selector at ε ∈ {0.05, 0.10, 0.15, 0.20} on
the same 36-config sweep and reports the compliance frontier, demonstrating
that ε is a tunable safety knob.

Method:
1) Load cached (T, b, H, κ, peak_mem_GB) from phase1_1d_sweep.npz
2) Per (b, H), fit AEROS memory model: M(T,κ) = M_0 + α_in*T + α_K*κ
3) For each ε and each (T, b, H), compute κ_A*(ε, M) analytically
4) Snap κ_A* to nearest valid grid κ ∈ {1,2,4,8,16,32,64,128,256}
5) Look up actual peak_mem from cache OR run AEROS to measure if not cached
6) Record compliance: actual ≤ M

For each ε ∈ {0.05, 0.10, 0.15, 0.20} report:
  - infeasible: count where predicted min(κ=1) > M*(1-ε)
  - compliant: count where actual ≤ M
  - savings: mean savings vs baseline

Usage:
    python p5_2_epsilon_sweep.py
    python p5_2_epsilon_sweep.py --budget 5.0    # 5 GB tight budget
    python p5_2_epsilon_sweep.py --budget 16.0   # 16 GB realistic budget
"""
import argparse, os, time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18

GRID_K = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# -----------------------------------------------------------------------------
def load_cached_sweep(path='phase1_1d_sweep.npz'):
    """Load cached (T, b, H, κ, peak_mem) from previous sweep.
    Returns dict (T, b, H) -> {K: peak_mem_GB} and {(T, b, H): baseline_mem}."""
    if not os.path.exists(path):
        print(f'  NOTE: {path} not found; will run all measurements fresh.')
        return None, None
    d = np.load(path, allow_pickle=True)
    results = d['results']
    cache = {}
    base = {}
    for r in results:
        key = (int(r['T']), int(r['B']), int(r['H']))
        base[key] = r['baseline_mem']
        cache[key] = {}
        for kr in r['K_results']:
            if kr.get('peak_mem_GB') is not None:
                cache[key][int(kr['K'])] = float(kr['peak_mem_GB'])
    print(f'  Loaded cache: {len(cache)} configs, '
          f'{sum(len(v) for v in cache.values())} (T,b,H,κ) measurements')
    return cache, base


def fit_per_bh(cache, b, H):
    """Fit (M_0, α_in, α_K) using only cached configs at this (b, H)."""
    pts = []
    for (T, b_, H_), kdict in cache.items():
        if b_ != b or H_ != H:
            continue
        for k, mem in kdict.items():
            pts.append((T, k, mem))
    if len(pts) < 4:
        return None
    A = np.array([[1, T, k] for T, k, _ in pts])
    y = np.array([m for _, _, m in pts])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeffs
    R2 = 1 - ((y - pred)**2).sum() / max(((y - y.mean())**2).sum(), 1e-9)
    return tuple(coeffs.tolist()) + (R2,)


def kA_selector(M_budget, eps, M_0, a_in, a_K, T):
    """κ_A* = floor((M*(1-ε) - M_0 - α_in*T) / α_K), clamped to [1, T]."""
    headroom = M_budget * (1.0 - eps) - M_0 - a_in * T
    if headroom <= 0:
        return None  # infeasible: even predicted κ=1 exceeds budget
    k_real = headroom / a_K
    k = int(np.floor(k_real))
    k = max(1, min(k, T))
    return k


def snap_to_grid(k, T):
    """Snap selector output to nearest valid power-of-2 κ ≤ T (and present in grid)."""
    candidates = [g for g in GRID_K if g <= T]
    if not candidates:
        return 1
    # Snap DOWN (more conservative on memory)
    valid = [g for g in candidates if g <= k]
    return max(valid) if valid else min(candidates)


# -----------------------------------------------------------------------------
@torch.no_grad()
def measure_mem_for_config(net, T, b, H, k, device):
    """Run a single (T, b, H, κ) and return peak memory in GB.
    Falls back to None on OOM."""
    try:
        functional.reset_net(net)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.rand(T, b, 3, H, H, device=device)
        i = 0
        while i < T:
            sz = min(k, T - i)
            _ = net(x[i:i + sz])
            i += sz
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del x
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, nargs='+', default=[5.0, 16.0],
                    help='memory budget(s) in GB to test')
    ap.add_argument('--epsilons', type=float, nargs='+',
                    default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument('--cache', type=str, default='phase1_1d_sweep.npz')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    cache, _ = load_cached_sweep(args.cache)

    if cache is None:
        print('Cache missing; this script needs phase1_1d_sweep.npz to be efficient.')
        return

    # Build SpikingResNet-18 once (used only when cache misses)
    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    # Sweep grid (must match phase1_1d): T ∈ {32,64,128,256}, b ∈ {16,32,64}, H ∈ {64,128,224}
    Ts = [32, 64, 128, 256]
    bs = [16, 32, 64]
    Hs = [64, 128, 224]

    # Fit per (b, H) once
    fits = {}
    for b in bs:
        for H in Hs:
            f = fit_per_bh(cache, b, H)
            if f is None:
                print(f'  WARN: insufficient cache for (b={b}, H={H}), skipping')
                continue
            fits[(b, H)] = f
    print(f'\n=== Per-(b,H) fits ===')
    print(f'{"(b,H)":<10} {"M_0":>7} {"α_in":>9} {"α_K":>9} {"R²":>7}')
    for (b, H), (M_0, a_in, a_K, R2) in sorted(fits.items()):
        print(f'  ({b:>2},{H:>3}) {M_0:>7.3f} {a_in:>9.5f} {a_K:>9.5f} {R2:>7.4f}')

    # ε sweep
    new_runs = 0
    rows = []
    for budget in args.budget:
        print(f'\n=== Budget M = {budget} GB ===')
        print(f'{"ε":>6} {"infeasible":>11} {"feasible":>9} {"compliant":>10} '
              f'{"max_violation":>14} {"mean_savings":>13}')

        for eps in args.epsilons:
            n_infeasible = 0
            n_feasible = 0
            n_compliant = 0
            max_viol = 1.0
            savings = []

            for T in Ts:
                for b in bs:
                    for H in Hs:
                        if (b, H) not in fits:
                            continue
                        M_0, a_in, a_K, _ = fits[(b, H)]
                        k_an = kA_selector(budget, eps, M_0, a_in, a_K, T)
                        if k_an is None:
                            n_infeasible += 1
                            continue
                        k = snap_to_grid(k_an, T)
                        # Look up actual peak memory
                        actual = cache.get((T, b, H), {}).get(k)
                        if actual is None:
                            actual = measure_mem_for_config(net, T, b, H, k, device)
                            new_runs += 1
                            if actual is None:
                                continue  # OOM, skip
                        n_feasible += 1
                        if actual <= budget + 1e-6:
                            n_compliant += 1
                        else:
                            ratio = actual / budget
                            max_viol = max(max_viol, ratio)
                        # Savings vs baseline
                        baseline = cache.get((T, b, H), {}).get(max([k for k in cache.get((T,b,H),{}).keys() if k <= T], default=1))
                        if baseline:
                            savings.append(baseline / actual)

                        rows.append({
                            'budget_GB': budget, 'eps': eps,
                            'T': T, 'b': b, 'H': H,
                            'k_analytical': k_an, 'k_snapped': k,
                            'actual_GB': actual, 'compliant': actual <= budget + 1e-6,
                        })

            mean_sav = np.mean(savings) if savings else 0.0
            print(f'{eps:>6.2f} {n_infeasible:>11} {n_feasible:>9} '
                  f'{n_compliant:>10} {max_viol:>13.3f}× {mean_sav:>12.2f}×')

    np.savez('p5_2_results.npz', rows=rows, fits=fits,
             budgets=args.budget, epsilons=args.epsilons)
    print(f'\n  New V100 runs needed: {new_runs}')
    print('  Saved to p5_2_results.npz')

    # Markdown-ready summary
    print('\n=== Markdown summary (paste into v6 §5.3) ===')
    print('| Budget | ε | Infeasible | Feasible | Compliant | Max viol |')
    print('|---|---|---|---|---|---|')
    seen = set()
    for r in rows:
        key = (r['budget_GB'], r['eps'])
        if key in seen: continue
        seen.add(key)
        # Recount per (budget, eps)
        sub = [x for x in rows if x['budget_GB'] == r['budget_GB'] and x['eps'] == r['eps']]
        nc = sum(1 for x in sub if x['compliant'])
        mv = max([x['actual_GB']/x['budget_GB'] for x in sub if not x['compliant']],
                 default=1.0)
        print(f'| {r["budget_GB"]} GB | {r["eps"]:.2f} | — | {len(sub)} | {nc}/{len(sub)} | {mv:.3f}× |')


if __name__ == '__main__':
    main()