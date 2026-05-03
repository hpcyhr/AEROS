"""
p7_2_v2_real_network_statecarry.py — fix v1's "no divergence" issue.

v1 finding: SR-18 + DVS Gesture density (b=16, H=128, T=128, K=8)
showed all three methods identical (max_err = 0). This is because:
  - Default LIF tau makes per-step v decay rapidly (mean |v| at boundary ≈ 0.006)
  - Most LIF neurons are sub-threshold and not firing in any chunk
  - Reset between chunks doesn't change the (mostly-zero) firing pattern

To genuinely expose state-carry necessity, we need configurations where:
  1. Many neurons are NEAR threshold at chunk boundaries (so reset
     destroys queued spikes).
  2. Chunks are SHORT enough that recovery from v=0 cannot reach
     threshold within one chunk.
  3. Input drives v consistently HIGH so accumulation matters.

v2 strategy:
  - Use HIGH-density input (rate 0.7-0.95, not 0.1-0.5)
  - Use SHORTER chunks (K=4)
  - Use SHALLOWER cumulative time T=128 with K=4 → 32 chunk boundaries
  - Use HIGHER batch size to get more samples
  - Diagnose v statistics MORE carefully (per-layer report)

If state-carry is truly necessary in this regime, max_err should be
substantial (> 0.1) and prediction agreement should drop noticeably.

Usage:
    python p7_2_v2_real_network_statecarry.py
    python p7_2_v2_real_network_statecarry.py --rate_min 0.5 --rate_max 0.95
"""
import argparse
import os
import gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def make_input(T, b, H, device, rate_min, rate_max, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    # Linearly varying rate in [rate_min, rate_max] across timesteps
    rates = torch.linspace(rate_min, rate_max, T, device=device).view(T, 1, 1, 1, 1)
    u = torch.rand(T, b, 3, H, H, generator=g, device=device)
    return (u < rates).float()


@torch.no_grad()
def boundary_v_stats(net, x, K):
    """v statistics at first chunk boundary (after K steps), per LIF layer."""
    functional.reset_net(net)
    _ = net(x[:K])
    stats = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            v = m.v
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                v_flat = v.detach().flatten()
                v_th = m.v_threshold
                stats.append({
                    'name': name,
                    'v_th': float(v_th),
                    'frac_above_half_th': float((v_flat.abs() > 0.5 * v_th).float().mean()),
                    'frac_above_quarter_th': float((v_flat.abs() > 0.25 * v_th).float().mean()),
                    'mean_abs': float(v_flat.abs().mean()),
                    'max_abs': float(v_flat.abs().max()),
                })
    return stats


@torch.no_grad()
def run_baseline(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def run_chunked(net, x, K, do_state_carry):
    T = x.shape[0]
    functional.reset_net(net)
    outs = []
    i = 0
    while i < T:
        if (not do_state_carry) and i > 0:
            functional.reset_net(net)
        sz = min(K, T - i)
        outs.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--b', type=int, default=16)
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--rate_min', type=float, default=0.5)
    ap.add_argument('--rate_max', type=float, default=0.95)
    ap.add_argument('--num_classes', type=int, default=11)
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print(f'\n=== P7-2 v2: aggressive state-carry ablation ===')
    print(f'  T={args.T}, K={args.K}, b={args.b}, H={args.H}')
    print(f'  Input rate range: [{args.rate_min:.2f}, {args.rate_max:.2f}]')
    print(f'  ({args.T // args.K} chunk boundaries)\n')

    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True,
                           num_classes=args.num_classes).cuda().eval()
    functional.set_step_mode(net, step_mode='m')

    x = make_input(args.T, args.b, args.H, device, args.rate_min, args.rate_max)

    # ---- Pre-test diagnostic ----
    v_stats = boundary_v_stats(net, x, args.K)
    if v_stats:
        fracs_half = [s['frac_above_half_th'] for s in v_stats]
        fracs_quarter = [s['frac_above_quarter_th'] for s in v_stats]
        means = [s['mean_abs'] for s in v_stats]
        maxes = [s['max_abs'] for s in v_stats]
        print(f'Pre-test v-at-boundary diagnostic ({len(v_stats)} LIF layers):')
        print(f'  Mean |v| across layers:                 {np.mean(means):.4f}')
        print(f'  Max  |v| across layers:                 {np.max(maxes):.4f}')
        print(f'  Mean fraction |v|>0.25·V_th per layer:  {np.mean(fracs_quarter)*100:.1f}%')
        print(f'  Mean fraction |v|>0.50·V_th per layer:  {np.mean(fracs_half)*100:.1f}%')
        if np.mean(fracs_quarter) < 0.05:
            print(f'  ⚠ Few neurons near threshold — state-carry may not matter much')
        elif np.mean(fracs_quarter) > 0.15:
            print(f'  ✓ Many neurons near threshold — state-carry should matter')

    # ---- Run all three ----
    print(f'\n--- Running baseline (unchunked T={args.T}) ---')
    y_base = run_baseline(net, x)
    pred_base = y_base.sum(dim=0).argmax(dim=1).cpu().numpy()
    spikes_base = float(y_base.sum())
    print(f'  output mass: {spikes_base:.2f}')
    print(f'  pred (first {min(args.b, 8)}): {pred_base[:8]}')

    print(f'\n--- Running AEROS κ={args.K} (with state-carry) ---')
    y_carry = run_chunked(net, x, args.K, do_state_carry=True)
    err_carry = (y_carry - y_base).abs().max().item()
    diff_carry = float((y_carry != y_base).float().mean())
    spikes_carry = float(y_carry.sum())
    pred_carry = y_carry.sum(dim=0).argmax(dim=1).cpu().numpy()
    pred_match_carry = float((pred_carry == pred_base).mean()) * 100
    print(f'  max_err: {err_carry:.2e}, diff_frac: {diff_carry*100:.2f}%, '
          f'pred_match: {pred_match_carry:.1f}%')

    print(f'\n--- Running AEROS κ={args.K} (NO state-carry) ---')
    y_nocarry = run_chunked(net, x, args.K, do_state_carry=False)
    err_nocarry = (y_nocarry - y_base).abs().max().item()
    diff_nocarry = float((y_nocarry != y_base).float().mean())
    spikes_nocarry = float(y_nocarry.sum())
    pred_nocarry = y_nocarry.sum(dim=0).argmax(dim=1).cpu().numpy()
    pred_match_nocarry = float((pred_nocarry == pred_base).mean()) * 100
    print(f'  max_err: {err_nocarry:.2e}, diff_frac: {diff_nocarry*100:.2f}%, '
          f'pred_match: {pred_match_nocarry:.1f}%')

    # ---- Summary ----
    print('\n' + '=' * 80)
    print(f'Summary — SpikingResNet-18 state-carry necessity, real network')
    print('=' * 80)
    print(f'{"Method":<35} {"max_err":>10} {"diff %":>8} {"out Δ%":>8} {"pred match":>11}')
    print('-' * 80)
    print(f'{"Baseline (unchunked)":<35} {0.0:>10.2e} {0.0:>7.2f}% '
          f'{0.0:>7.2f}% {100.0:>10.2f}%')
    print(f'{"AEROS (state-carry)":<35} {err_carry:>10.2e} {diff_carry*100:>7.2f}% '
          f'{(spikes_carry/spikes_base-1)*100:>7.2f}% {pred_match_carry:>10.2f}%')
    print(f'{"AEROS (no state-carry)":<35} {err_nocarry:>10.2e} '
          f'{diff_nocarry*100:>7.2f}% {(spikes_nocarry/spikes_base-1)*100:>7.2f}% '
          f'{pred_match_nocarry:>10.2f}%')

    print('\nVerdict:')
    if err_carry == 0.0 and err_nocarry > 1e-3:
        print(f'  ✓ STATE-CARRY VERIFIED NECESSARY on real SpikingResNet-18')
        print(f'    Without state-carry on T={args.T}, K={args.K}:')
        print(f'      max_err = {err_nocarry:.2e}')
        print(f'      {diff_nocarry*100:.2f}% of output elements differ')
        print(f'      Spike-count delta: {(spikes_nocarry/spikes_base-1)*100:+.2f}%')
        print(f'      Top-1 prediction agreement: {pred_match_nocarry:.2f}% '
              f'(vs {pred_match_carry:.2f}% w/ state-carry)')
    elif err_carry > 0.0:
        print(f'  ✗ Implementation issue: state-carry should be bit-exact, '
              f'got max_err={err_carry:.2e}')
    else:
        print(f'  ⚠ No divergence even in this aggressive regime')
        print(f'  → Suggests SR-18 + LIF naturally tolerates inter-chunk reset:')
        print(f'    LIF tau likely makes |v| at chunk boundary << V_th in most layers')
        print(f'  → TinyLIF microbenchmark remains the principled demonstration')

    np.savez('p7_2_v2_results.npz',
             T=args.T, K=args.K, b=args.b, H=args.H,
             rate_min=args.rate_min, rate_max=args.rate_max,
             v_stats=v_stats,
             err_carry=err_carry, err_nocarry=err_nocarry,
             diff_carry=diff_carry, diff_nocarry=diff_nocarry,
             spikes_base=spikes_base, spikes_carry=spikes_carry,
             spikes_nocarry=spikes_nocarry,
             pred_match_carry=pred_match_carry,
             pred_match_nocarry=pred_match_nocarry)
    print('\n  Saved to p7_2_v2_results.npz')


if __name__ == '__main__':
    main()