"""
p7_5_regime_transfer.py — coefficient transfer between (b, H) regimes.

Purpose: validate the AEROS paper claim that the AEROS memory model
fit at one (b, H) "transfers across regimes within an architecture",
i.e., the same coefficients (M_0, α_in, α_K) can predict memory at
other (b, H) settings without re-fitting.

This is a CATFuse-style sensitivity analysis. The hypothesis is that:
  - α_in scales as O(b * H * H) — input cost depends on (b, H)
  - α_K scales as O(b * H * H) — chunk activation cost depends on (b, H)
  - M_0 stays roughly constant (it's parameter-tied)

If we fit at reference (b_0, H_0) = (32, 128), the predicted memory
at any (b, H) should be:

    M(T, κ; b, H) ≈ M_0 + α_in * (b·H²)/(b_0·H_0²) * T
                       + α_K * (b·H²)/(b_0·H_0²) * κ

This script fits at (32, 128), predicts at (16, 64), (16, 224),
(64, 128), (64, 224), measures actual vs predicted, reports
prediction error percentile.

If max error < 15%, "fit transfers across regimes" is well-supported.
If error > 30% in some regime, paper should report regime-specific
fit instead.

Usage:
    python p7_5_regime_transfer.py
"""
import argparse
import gc
import time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


@torch.no_grad()
def safe_measure(net, T, b, H, K, device):
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
        gc.collect()
        torch.cuda.empty_cache()
        return peak
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        try:
            functional.reset_net(net)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        return None


def fit_at(net, b, H, device, Ts=(64, 128), Ks=(1, 2, 4, 8, 16, 32, 64)):
    pts = []
    for T in Ts:
        for K in Ks:
            if K > T:
                continue
            peak = safe_measure(net, T, b, H, K, device)
            if peak:
                pts.append((T, K, peak))
    if len(pts) < 4:
        return None
    A = np.array([[1, T, K] for T, K, _ in pts])
    y = np.array([m for _, _, m in pts])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    M_0, a_in, a_K = coeffs
    pred = A @ coeffs
    R2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    return {'M_0': float(M_0), 'a_in': float(a_in), 'a_K': float(a_K),
            'R2': float(R2), 'b': b, 'H': H, 'pts': pts}


def predict_with_scaling(fit_ref, b_target, H_target, T, K):
    """Predict mem at (b_target, H_target) using ref fit at (b_ref, H_ref).

    Scaling: α_in and α_K scale as (b * H²) / (b_ref * H_ref²).
    M_0 stays constant.
    """
    b_ref, H_ref = fit_ref['b'], fit_ref['H']
    scale = (b_target * H_target * H_target) / (b_ref * H_ref * H_ref)
    a_in_pred = fit_ref['a_in'] * scale
    a_K_pred = fit_ref['a_K'] * scale
    return fit_ref['M_0'] + a_in_pred * T + a_K_pred * K


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print('=== P7-5: Coefficient regime transfer ===')
    print('Reference fit at (b=32, H=128); predict at other (b, H)\n')

    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).cuda().eval()
    functional.set_step_mode(net, step_mode='m')

    # Reference fit
    print('\n--- Reference fit at (b=32, H=128) ---')
    fit_ref = fit_at(net, 32, 128, device)
    if fit_ref is None:
        print('  Reference fit failed; aborting'); return
    print(f'  M_0={fit_ref["M_0"]:.3f}, α_in={fit_ref["a_in"]:.5f}, '
          f'α_K={fit_ref["a_K"]:.5f}, R²={fit_ref["R2"]:.4f}')

    # Test regimes — predict and measure
    target_regimes = [
        (16, 64), (16, 224),
        (32, 64), (32, 224),
        (64, 128),
    ]
    Ts = [64, 128]
    Ks = [1, 4, 16]

    print('\n--- Predict-vs-measure across regimes ---')
    print(f'{"(b,H)":<10} {"T":>4} {"K":>4} {"predicted":>10} {"actual":>10} '
          f'{"abs err":>9} {"rel err":>9}')
    print('-' * 70)
    rows = []
    for (b, H) in target_regimes:
        for T in Ts:
            for K in Ks:
                if K > T:
                    continue
                pred = predict_with_scaling(fit_ref, b, H, T, K)
                actual = safe_measure(net, T, b, H, K, device)
                if actual is None:
                    print(f'  ({b:>2},{H:>3}) {T:>4} {K:>4} {pred:>10.3f} {"OOM":>10}')
                    rows.append({'b': b, 'H': H, 'T': T, 'K': K,
                                 'pred': pred, 'actual': None,
                                 'abs_err': None, 'rel_err': None})
                    continue
                abs_err = abs(pred - actual)
                rel_err = abs_err / actual * 100
                print(f'  ({b:>2},{H:>3}) {T:>4} {K:>4} {pred:>10.3f} '
                      f'{actual:>10.3f} {abs_err:>9.3f} {rel_err:>8.1f}%')
                rows.append({'b': b, 'H': H, 'T': T, 'K': K,
                             'pred': pred, 'actual': actual,
                             'abs_err': abs_err, 'rel_err': rel_err})

    # Summary statistics
    valid = [r for r in rows if r['rel_err'] is not None]
    if valid:
        rel_errs = [r['rel_err'] for r in valid]
        print('\n=== Summary ===')
        print(f'  Configurations tested:    {len(rows)} '
              f'(of which {len(valid)} non-OOM)')
        print(f'  Mean relative error:      {np.mean(rel_errs):.1f}%')
        print(f'  Median relative error:    {np.median(rel_errs):.1f}%')
        print(f'  Max relative error:       {np.max(rel_errs):.1f}%')
        print(f'  P95 relative error:       {np.percentile(rel_errs, 95):.1f}%')

        if np.max(rel_errs) < 15:
            print(f'\n  ✓ Coefficient transfer well-supported (<15% max error)')
        elif np.max(rel_errs) < 30:
            print(f'\n  ◐ Coefficient transfer acceptable (15-30% max error)')
            print(f'    Tight-budget deployments should re-fit per regime')
        else:
            print(f'\n  ✗ Coefficient transfer significantly degraded (>30%)')
            print(f'    Paper should retract "transfers across regimes" claim')

    np.savez('p7_5_results.npz', fit_ref=fit_ref, rows=rows)
    print('\n  Saved to p7_5_results.npz')


if __name__ == '__main__':
    main()