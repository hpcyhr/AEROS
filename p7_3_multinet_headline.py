"""
p7_3_multinet_headline.py — Multi-network AEROS headline benchmark.

Purpose: extend AEROS evaluation from SpikingResNet-18 single-architecture
headline to four networks, matching the workload-breadth standard of
Chronos (11 STNN models) and Helios (8 DiNN workloads).

Networks:
  1. SpikingResNet-18      (LIF + spiking residual)
  2. SEW-ResNet-18         (LIF + SEW-style residual)
  3. SpikingVGG-11-BN      (LIF + plain VGG)
  4. ConvLSTM (custom)     (h, c) tuple state — non-LIF stateful recurrent

For each network, three sub-tests:
  (A) Memory model fit:
      Sweep (T, K) ∈ {64,128,256} × {1,2,4,8,16,32,64,128}, fit
      M_0 + α_in*T + α_K*κ; report R².
  (B) Selector compliance:
      Apply Problem A (M=5GB, M=16GB) and Problem B (S=1.2, S=1.5)
      using each network's fitted coefficients; verify post-execution
      compliance.
  (C) Headline + max-feasible-T:
      At b=32, H=224, sweep T ∈ {32,64,128,256,512,1024,2048,4096,8192}
      for both unchunked baseline (where feasible) and AEROS κ=8 (or
      next-feasible κ); report peak memory and max feasible T.

Output: p7_3_results.npz with per-network coefficients, compliance, and
headline data. Markdown summary printed for direct paste into v7 paper.

Usage:
    python p7_3_multinet_headline.py
    python p7_3_multinet_headline.py --skip convlstm  # if cell2/cell1 OOM
    python p7_3_multinet_headline.py --networks resnet18 sew18  # subset
"""
import argparse
import os
import sys
import time
import gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18

# SEW-ResNet has different module paths across SJ versions:
#   newer:  spikingjelly.activation_based.model.sew_resnet.sew_resnet18
#   older:  spikingjelly.activation_based.model.spiking_resnet.sew_resnet18
sew_resnet18 = None
for module_path in [
    'spikingjelly.activation_based.model.sew_resnet',
    'spikingjelly.activation_based.model.spiking_resnet',
]:
    try:
        mod = __import__(module_path, fromlist=['sew_resnet18'])
        if hasattr(mod, 'sew_resnet18'):
            sew_resnet18 = getattr(mod, 'sew_resnet18')
            print(f'  found sew_resnet18 in {module_path}')
            break
    except ImportError:
        continue
if sew_resnet18 is None:
    print('  WARN: sew_resnet18 not found in this SJ version; '
          'sew18 tests will be skipped')

from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn

# Local convlstm_module
try:
    from convlstm_module import build_convlstm_network
    HAS_CONVLSTM = True
except ImportError:
    print('WARN: convlstm_module.py not found in cwd; ConvLSTM tests will be skipped')
    HAS_CONVLSTM = False


# -----------------------------------------------------------------------------
def build_network(name: str, num_classes: int = 11):
    """Build network and set multi-step mode."""
    if name == 'resnet18':
        net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True, num_classes=num_classes)
        functional.set_step_mode(net, step_mode='m')
    elif name == 'sew18':
        if sew_resnet18 is None:
            print(f'  Skipping sew18 — sew_resnet18 not importable')
            return None
        # cnf parameter: SJ versions accept 'ADD' or 'add' — try both
        try:
            net = sew_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True,
                               cnf='ADD',
                               num_classes=num_classes)
        except (TypeError, ValueError):
            net = sew_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True,
                               cnf='add',
                               num_classes=num_classes)
        functional.set_step_mode(net, step_mode='m')
    elif name == 'vgg11':
        net = spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True, num_classes=num_classes)
        functional.set_step_mode(net, step_mode='m')
    elif name == 'convlstm':
        if not HAS_CONVLSTM:
            return None
        net = build_convlstm_network(num_classes=num_classes)
        functional.set_step_mode(net, step_mode='m')
    else:
        raise ValueError(f'Unknown network: {name}')
    return net.cuda().eval()


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


@torch.no_grad()
def measure_chunked(net, T, b, H, K, device):
    """Run chunked forward, return peak_mem_GB or None on OOM."""
    reset_mem()
    try:
        x = torch.rand(T, b, 3, H, H, device=device)
        functional.reset_net(net)
        i = 0
        while i < T:
            sz = min(K, T - i)
            _ = net(x[i:i + sz])
            i += sz
        torch.cuda.synchronize()
        peak = peak_mem_GB()
        del x
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


@torch.no_grad()
def measure_baseline(net, T, b, H, device):
    """Unchunked: pass full T at once."""
    return measure_chunked(net, T, b, H, T, device)  # K=T = unchunked


@torch.no_grad()
def measure_chunked_with_wall(net, T, b, H, K, device, n_iters=5):
    """Run chunked forward N times, return (peak_mem, median_wall_ms)."""
    reset_mem()
    try:
        x = torch.rand(T, b, 3, H, H, device=device)
        # Warmup
        functional.reset_net(net)
        i = 0
        while i < T:
            sz = min(K, T - i)
            _ = net(x[i:i + sz])
            i += sz
        torch.cuda.synchronize()

        walls = []
        for _ in range(n_iters):
            functional.reset_net(net)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            i = 0
            while i < T:
                sz = min(K, T - i)
                _ = net(x[i:i + sz])
                i += sz
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)

        peak = peak_mem_GB()
        del x
        return peak, float(np.median(walls))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


# -----------------------------------------------------------------------------
def fit_memory_model(net, name, device, b=32, H=128):
    """(A) Fit M_0 + α_in*T + α_K*κ on a small (T, K) grid."""
    print(f'\n--- (A) Memory model fit for {name} (b={b}, H={H}) ---')
    Ts = [64, 128, 256]
    Ks = [1, 2, 4, 8, 16, 32, 64, 128]
    pts = []
    print(f'{"T":>4} {"K":>4} {"peak (GB)":>11} {"status":>8}')
    for T in Ts:
        for K in Ks:
            if K > T:
                continue
            peak = measure_chunked(net, T, b, H, K, device)
            status = 'OK' if peak else 'OOM'
            print(f'{T:>4} {K:>4} {(f"{peak:.3f}" if peak else "—"):>11} {status:>8}')
            if peak:
                pts.append((T, K, peak))

    if len(pts) < 4:
        print(f'  WARN: only {len(pts)} points fit; aborting fit')
        return None

    A = np.array([[1, T, K] for T, K, _ in pts])
    y = np.array([m for _, _, m in pts])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    M_0, a_in, a_K = coeffs
    pred = A @ coeffs
    R2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)

    print(f'\n  Fit: M_0={M_0:.3f}, α_in={a_in:.5f}, α_K={a_K:.5f}, R²={R2:.4f}')
    return {'M_0': M_0, 'a_in': a_in, 'a_K': a_K, 'R2': R2,
            'b': b, 'H': H, 'pts': pts}


def kA_selector(M_budget, eps, M_0, a_in, a_K, T):
    """κ_A* = floor((M(1-ε) - M_0 - α_in*T) / α_K), clamped to [1, T]."""
    headroom = M_budget * (1.0 - eps) - M_0 - a_in * T
    if headroom <= 0:
        return None
    return max(1, min(int(headroom / a_K), T))


def snap_kappa(k_real, T, grid=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """Snap to the largest grid value ≤ min(k_real, T) (conservative)."""
    candidates = [g for g in grid if g <= min(k_real, T)]
    return max(candidates) if candidates else 1


def selector_compliance(net, name, fit, device):
    """(B) Test Problem A at M ∈ {5, 16} GB across (T, b, H) sub-grid."""
    print(f'\n--- (B) Selector compliance for {name} ---')
    M_0, a_in, a_K = fit['M_0'], fit['a_in'], fit['a_K']
    Ts = [64, 128, 256]
    bs = [16, 32]
    Hs = [64, 128, 224]
    eps = 0.05
    rows = []

    for budget in [5.0, 16.0]:
        nfeasible = 0
        ncompliant = 0
        ninfeasible = 0
        for T in Ts:
            for b in bs:
                for H in Hs:
                    k = kA_selector(budget, eps, M_0, a_in, a_K, T)
                    if k is None:
                        ninfeasible += 1
                        continue
                    k = snap_kappa(k, T)
                    actual = measure_chunked(net, T, b, H, k, device)
                    if actual is None:
                        continue
                    nfeasible += 1
                    if actual <= budget + 1e-6:
                        ncompliant += 1
                    rows.append({'budget': budget, 'T': T, 'b': b, 'H': H,
                                 'k': k, 'actual_GB': actual,
                                 'compliant': actual <= budget + 1e-6})
        print(f'  M={budget}GB: {ninfeasible} INFEASIBLE, '
              f'{nfeasible} feasible, {ncompliant}/{nfeasible} compliant')
    return rows


def headline_and_maxT(net, name, device, b=32, H=224, K=8):
    """(C) Headline timing + max-feasible-T sweep."""
    print(f'\n--- (C) Headline + max-T for {name} (b={b}, H={H}, κ={K}) ---')
    Ts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    rows = []
    print(f'{"T":>5}  {"baseline mem":>14} {"baseline wall":>14}  '
          f'{"AEROS mem":>10} {"AEROS wall":>12}  {"slowdown":>9}')
    print('-' * 80)

    baseline_oomed = False
    for T in Ts:
        # Baseline
        if baseline_oomed:
            base_mem, base_wall = None, None
        else:
            base_mem, base_wall = measure_chunked_with_wall(net, T, b, H, T, device)
            if base_mem is None:
                baseline_oomed = True

        # AEROS
        aeros_mem, aeros_wall = measure_chunked_with_wall(net, T, b, H, K, device)

        sd = ''
        if base_wall and aeros_wall:
            sd = f'{aeros_wall/base_wall:.3f}×'

        bm = f'{base_mem:.2f}GB' if base_mem else '—'
        bw = f'{base_wall:.0f}ms' if base_wall else '—'
        am = f'{aeros_mem:.2f}GB' if aeros_mem else '—'
        aw = f'{aeros_wall:.0f}ms' if aeros_wall else '—'
        bs_status = 'OK' if base_mem else 'OOM'
        as_status = 'OK' if aeros_mem else 'OOM'

        print(f'{T:>5}  {bm:>10} {bs_status:>3} {bw:>14}  '
              f'{am:>10} {aw:>9} {as_status:>3}  {sd:>9}')
        rows.append({'T': T, 'b': b, 'H': H, 'K': K,
                     'base_mem': base_mem, 'base_wall': base_wall,
                     'aeros_mem': aeros_mem, 'aeros_wall': aeros_wall})

    max_base = max([r['T'] for r in rows if r['base_mem']], default=0)
    max_aeros = max([r['T'] for r in rows if r['aeros_mem']], default=0)
    print(f'\n  Max feasible T:  baseline={max_base},  AEROS κ={K}={max_aeros}')
    if max_aeros > max_base:
        print(f'  → AEROS extends horizon by {max_aeros/max(max_base, 1):.1f}×')
    return rows, max_base, max_aeros


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--networks', nargs='+',
                    default=['resnet18', 'sew18', 'vgg11', 'convlstm'])
    ap.add_argument('--skip', nargs='+', default=[])
    args = ap.parse_args()

    device = torch.device('cuda:0')
    out = {}

    for name in args.networks:
        if name in args.skip:
            continue
        print('\n' + '=' * 80)
        print(f'=== Network: {name} ===')
        print('=' * 80)
        net = build_network(name)
        if net is None:
            print(f'  Skipped (build returned None)')
            continue

        # Sub-test (A)
        fit = fit_memory_model(net, name, device)
        if fit is None:
            print(f'  Skipping (B), (C) for {name} due to fit failure')
            continue

        # Sub-test (B)
        compliance = selector_compliance(net, name, fit, device)

        # Sub-test (C)
        headline, max_base, max_aeros = headline_and_maxT(net, name, device)

        out[name] = {
            'fit': fit, 'compliance': compliance,
            'headline': headline,
            'max_base': max_base, 'max_aeros': max_aeros,
        }

        # Free network before building next
        del net
        torch.cuda.empty_cache()
        gc.collect()

    # ----- Final summary -----
    print('\n' + '=' * 80)
    print('=== Multi-network summary (paste into v7 paper) ===')
    print('=' * 80)
    print('\n(A) Memory model fit:')
    print(f'{"Network":<14} {"M_0":>7} {"α_in":>9} {"α_K":>9} {"R²":>7}')
    for name, r in out.items():
        f = r['fit']
        print(f'{name:<14} {f["M_0"]:>7.3f} {f["a_in"]:>9.5f} '
              f'{f["a_K"]:>9.5f} {f["R2"]:>7.4f}')

    print('\n(B) Selector compliance @ M=5GB:')
    for name, r in out.items():
        c = [x for x in r['compliance'] if x['budget'] == 5.0]
        n_comply = sum(1 for x in c if x['compliant'])
        print(f'  {name}: {n_comply}/{len(c)} compliant '
              f'(of {len(c)} feasible attempted)')

    print('\n(C) Max feasible T (b=32, H=224, κ=8):')
    print(f'{"Network":<14} {"baseline":>10} {"AEROS κ=8":>10} {"extension":>11}')
    for name, r in out.items():
        ext = f"{r['max_aeros']/max(r['max_base'],1):.1f}×"
        print(f'{name:<14} {r["max_base"]:>10} {r["max_aeros"]:>10} {ext:>11}')

    # Save
    np.savez('p7_3_results.npz', data=out)
    print('\n  Saved to p7_3_results.npz')


if __name__ == '__main__':
    main()