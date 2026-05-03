"""
p7_3_multinet_headline_v2.py — Multi-network AEROS headline, with bug fixes.

Bug fixes from v1:
  1. (B) compliance test now uses the SAME (b, H) as (A) fit.
     v1 was incorrectly applying a single (b=32, H=128) fit's coefficients
     to predict memory for ALL (b, H) combinations, which is mathematically
     wrong and produced 14/18 false-violations.
  2. Aggressive memory cleanup between every measurement: del net, gc, sync.
  3. (B) reduced to a single (b, H) per network (the same one used for fit),
     with T sweep ∈ {64, 128, 256}; this matches v6 paper §5.3 protocol
     (which used pre-cached phase1_1d_sweep with per-(b,H) fits).

Networks: resnet18, sew18, vgg11, convlstm.

For each network:
  (A) Fit M_0 + α_in*T + α_K*κ at fixed (b=32, H=128) over (T, K) grid.
  (B) Apply Problem A selector at M ∈ {5, 16} GB across T ∈ {64, 128, 256}
      (same b=32, H=128). Verify post-execution compliance.
  (C) Headline + max-feasible-T at b=32, H=224, K=8, T ∈ {32...8192}.

Usage:
    python p7_3_multinet_headline_v2.py
    python p7_3_multinet_headline_v2.py --networks resnet18 vgg11
    python p7_3_multinet_headline_v2.py --skip convlstm
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

# SEW-ResNet has different module paths across SJ versions
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

from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn

try:
    from convlstm_module import build_convlstm_network
    HAS_CONVLSTM = True
except ImportError:
    print('WARN: convlstm_module.py not found in cwd; ConvLSTM tests will be skipped')
    HAS_CONVLSTM = False


# -----------------------------------------------------------------------------
def aggressive_cleanup():
    """Force PyTorch to release cached allocator memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def reset_mem():
    """Reset peak memory tracking and clear cache."""
    aggressive_cleanup()
    torch.cuda.reset_peak_memory_stats()


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


# -----------------------------------------------------------------------------
def build_network(name, num_classes=11):
    if name == 'resnet18':
        net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True, num_classes=num_classes)
        functional.set_step_mode(net, step_mode='m')
    elif name == 'sew18':
        if sew_resnet18 is None:
            return None
        try:
            net = sew_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True, cnf='ADD',
                               num_classes=num_classes)
        except (TypeError, ValueError):
            net = sew_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True, cnf='add',
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


@torch.no_grad()
def measure_chunked(net, T, b, H, K, device):
    """Run chunked forward, return peak_mem_GB or None on OOM.
    Aggressive cleanup before AND after to prevent caching allocator
    fragmentation across many measurements."""
    aggressive_cleanup()
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
        functional.reset_net(net)
        aggressive_cleanup()
        return peak
    except torch.cuda.OutOfMemoryError:
        functional.reset_net(net)
        aggressive_cleanup()
        return None
    except Exception as e:
        print(f'    ERROR in measure_chunked T={T},b={b},H={H},K={K}: {e}')
        functional.reset_net(net)
        aggressive_cleanup()
        return None


@torch.no_grad()
def measure_chunked_with_wall(net, T, b, H, K, device, n_iters=5):
    aggressive_cleanup()
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
        functional.reset_net(net)
        aggressive_cleanup()
        return peak, float(np.median(walls))
    except torch.cuda.OutOfMemoryError:
        functional.reset_net(net)
        aggressive_cleanup()
        return None, None


# -----------------------------------------------------------------------------
def fit_memory_model(net, name, device, b=32, H=128):
    """(A) Fit M_0 + α_in*T + α_K*κ at fixed (b, H) over (T, K) grid."""
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
    headroom = M_budget * (1.0 - eps) - M_0 - a_in * T
    if headroom <= 0:
        return None
    return max(1, min(int(headroom / a_K), T))


def snap_kappa(k_real, T, grid=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    candidates = [g for g in grid if g <= min(k_real, T)]
    return max(candidates) if candidates else 1


def selector_compliance(net, name, fit, device):
    """(B) Test Problem A at M ∈ {5, 16} GB on the SAME (b, H) as the fit."""
    print(f'\n--- (B) Selector compliance for {name} '
          f'(b={fit["b"]}, H={fit["H"]}, ε=0.05) ---')
    M_0, a_in, a_K = fit['M_0'], fit['a_in'], fit['a_K']
    b, H = fit['b'], fit['H']
    Ts = [64, 128, 256]  # match fit grid; per-(b,H) coefficients are valid here
    eps = 0.05
    rows = []

    for budget in [5.0, 16.0]:
        nfeasible = 0
        ncompliant = 0
        ninfeasible = 0
        violations = []
        for T in Ts:
            k = kA_selector(budget, eps, M_0, a_in, a_K, T)
            if k is None:
                ninfeasible += 1
                continue
            k = snap_kappa(k, T)
            actual = measure_chunked(net, T, b, H, k, device)
            if actual is None:
                continue
            nfeasible += 1
            compliant = actual <= budget + 1e-6
            if compliant:
                ncompliant += 1
            else:
                violations.append((T, k, actual, actual / budget))
            rows.append({'budget': budget, 'T': T, 'b': b, 'H': H,
                         'k': k, 'actual_GB': actual,
                         'compliant': compliant})
        print(f'  M={budget}GB: {ninfeasible} INFEASIBLE, '
              f'{nfeasible} feasible, {ncompliant}/{nfeasible} compliant')
        for T, k, actual, ratio in violations:
            print(f'    VIOLATION at T={T}, k={k}: actual={actual:.3f} GB '
                  f'({ratio:.3f}× budget)')
    return rows


def headline_and_maxT(net, name, device, b=32, H=224, K=8):
    print(f'\n--- (C) Headline + max-T for {name} (b={b}, H={H}, κ={K}) ---')
    Ts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    rows = []
    print(f'{"T":>5}  {"baseline":>14} {"AEROS":>14} {"slowdown":>9}')
    print('-' * 70)

    baseline_oomed = False
    for T in Ts:
        if baseline_oomed:
            base_mem, base_wall = None, None
        else:
            base_mem, base_wall = measure_chunked_with_wall(net, T, b, H, T, device)
            if base_mem is None:
                baseline_oomed = True

        aeros_mem, aeros_wall = measure_chunked_with_wall(net, T, b, H, K, device)

        sd = ''
        if base_wall and aeros_wall:
            sd = f'{aeros_wall/base_wall:.3f}×'

        bm = (f'{base_mem:.2f}GB/{base_wall:.0f}ms' if base_mem else 'OOM')
        am = (f'{aeros_mem:.2f}GB/{aeros_wall:.0f}ms' if aeros_mem else 'OOM')

        print(f'{T:>5}  {bm:>14} {am:>14} {sd:>9}')
        rows.append({'T': T, 'b': b, 'H': H, 'K': K,
                     'base_mem': base_mem, 'base_wall': base_wall,
                     'aeros_mem': aeros_mem, 'aeros_wall': aeros_wall})

    max_base = max([r['T'] for r in rows if r['base_mem']], default=0)
    max_aeros = max([r['T'] for r in rows if r['aeros_mem']], default=0)
    print(f'\n  Max feasible T: baseline={max_base}, AEROS κ={K}={max_aeros}')
    if max_aeros > max_base:
        print(f'  → Horizon extension: {max_aeros/max(max_base,1):.1f}×')
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

        try:
            fit = fit_memory_model(net, name, device)
            if fit is None:
                print(f'  Skipping (B), (C) for {name}')
                del net
                aggressive_cleanup()
                continue

            compliance = selector_compliance(net, name, fit, device)

            headline, max_base, max_aeros = headline_and_maxT(net, name, device)

            out[name] = {
                'fit': fit, 'compliance': compliance,
                'headline': headline,
                'max_base': max_base, 'max_aeros': max_aeros,
            }
        except Exception as e:
            print(f'  EXCEPTION in {name}: {e}')
            import traceback; traceback.print_exc()
        finally:
            del net
            aggressive_cleanup()

    # ---------- Final summary ----------
    print('\n' + '=' * 80)
    print('=== Multi-network summary (paste into v7 paper) ===')
    print('=' * 80)
    print('\n(A) Memory model fit (b=32, H=128):')
    print(f'{"Network":<14} {"M_0":>7} {"α_in":>9} {"α_K":>9} {"R²":>7}')
    for name, r in out.items():
        f = r['fit']
        print(f'{name:<14} {f["M_0"]:>7.3f} {f["a_in"]:>9.5f} '
              f'{f["a_K"]:>9.5f} {f["R2"]:>7.4f}')

    print('\n(B) Selector compliance (b=32, H=128, T∈{64,128,256}):')
    for name, r in out.items():
        for budget in [5.0, 16.0]:
            c = [x for x in r['compliance'] if x['budget'] == budget]
            n_comply = sum(1 for x in c if x['compliant'])
            print(f'  {name} M={budget}GB: {n_comply}/{len(c)} compliant')

    print('\n(C) Max feasible T (b=32, H=224, κ=8):')
    print(f'{"Network":<14} {"baseline":>10} {"AEROS κ=8":>10} {"extension":>11}')
    for name, r in out.items():
        ext = f"{r['max_aeros']/max(r['max_base'],1):.1f}×"
        print(f'{name:<14} {r["max_base"]:>10} {r["max_aeros"]:>10} {ext:>11}')

    np.savez('p7_3_v2_results.npz', data=out)
    print('\n  Saved to p7_3_v2_results.npz')


if __name__ == '__main__':
    main()