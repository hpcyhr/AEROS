"""
P5-3 v3.2: ablation (d) ONLY (c is already clean from v3.1).

v3 had wrong LIF dynamics math. SJ LIFNode actual equation:
    v_new = v + (x - (v - v_reset)) / tau
          = v*(1 - 1/tau) + x/tau
Steady-state v_ss = I (input value), NOT I/(1-decay).
With τ=20, I=0.6 → v_ss=0.6, never reaches V_th=10. No spikes ever.

v3.2 corrected: V_th=0.5, I=0.8, τ=20, K=16, T=128.
  v_ss = 0.8 > V_th = 0.5 → fires periodically
  v at K=16 from v=0: 0.8*(1-0.951^16) = 0.44 (below V_th, no fire yet)
  baseline first fire at t≈20
  state-carry: continues from v=0.44 → fires at t=20 (matches baseline)
  no-state-carry: resets to 0 → fires at t=36 (16-step delay)
  → guaranteed divergence

Pre-test diagnostic: print m.v statistics after K=16 steps to confirm
the regime is in the right operating range BEFORE drawing conclusions.

Usage:
    python p5_3_v3_2.py
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional


class TinyLIF(nn.Module):
    def __init__(self, v_threshold=0.5, tau=20.0):
        super().__init__()
        self.lif = neuron.LIFNode(
            tau=tau, decay_input=True,
            v_threshold=v_threshold, v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, step_mode='m',
        )
    def forward(self, x):
        return self.lif(x)


@torch.no_grad()
def baseline(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked(net, x, K, do_state_carry):
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
    device = torch.device('cuda:0')

    print('=' * 78)
    print('=== Ablation (d) v3.2: state-carry necessity ===')
    print('=== TinyLIF (single-layer): τ=20, V_th=0.5, constant I=0.8 ===')
    print('=' * 78)

    V_th = 0.5
    tau = 20.0
    I = 0.8
    T, b, n, K = 128, 4, 1024, 16

    # Math sanity
    decay = 1 - 1/tau
    v_at_K = I * (1 - decay**K)  # closed-form for v(K) starting from v=0
    fire_time = -np.log(1 - V_th/I) / np.log(1/decay)
    print(f'\nLIF dynamics: v_new = {decay}*v + x/{tau}')
    print(f'  steady-state v_ss = I = {I:.2f}')
    print(f'  v at chunk boundary K={K} (from v=0): {v_at_K:.3f} '
          f'(V_th={V_th}, ratio {v_at_K/V_th:.2f})')
    print(f'  expected first fire at t ≈ {fire_time:.1f} from v=0')

    print(f'\nSetup: T={T}, b={b}, n={n} neurons, K={K} (so {T//K} chunks)')

    net = TinyLIF(v_threshold=V_th, tau=tau).to(device)
    net.eval()

    x = torch.full((T, b, n), I, device=device)

    # ---- Pre-test diagnostic: capture m.v after first chunk ----
    functional.reset_net(net)
    _ = net(x[:K])
    v_mass = net.lif.v.detach()
    print(f'\nPre-test diagnostic — m.v statistics after K={K} steps from v=0:')
    print(f'  shape: {tuple(v_mass.shape)}')
    print(f'  mean |v|: {v_mass.abs().mean().item():.4f}')
    print(f'  max  |v|: {v_mass.abs().max().item():.4f}')
    print(f'  fraction >= 0.3*V_th: '
          f'{(v_mass.abs() >= 0.3*V_th).float().mean().item()*100:.1f}%')

    if v_mass.abs().mean().item() < 0.1 * V_th:
        print('  WARN: v is << V_th, regime may not be discriminating.')

    # ---- Run baseline + with-carry + without-carry ----
    y_base = baseline(net, x)
    y_carry = chunked(net, x, K, do_state_carry=True)
    y_nocarry = chunked(net, x, K, do_state_carry=False)

    err_carry = (y_carry - y_base).abs().max().item()
    err_nocarry = (y_nocarry - y_base).abs().max().item()
    diff_carry = float((y_carry != y_base).float().mean())
    diff_nocarry = float((y_nocarry != y_base).float().mean())
    spikes_base = float(y_base.sum())
    spikes_carry = float(y_carry.sum())
    spikes_nocarry = float(y_nocarry.sum())

    print(f'\n{"Method":<35} {"total spikes":>14} {"max_err":>10} {"differing %":>13}')
    print('-' * 78)
    print(f'{"Baseline (unchunked)":<35} {spikes_base:>14.0f} '
          f'{0.0:>10.2e} {0.0:>12.2f}%')
    print(f'{"AEROS κ=16 (state-carry)":<35} {spikes_carry:>14.0f} '
          f'{err_carry:>10.2e} {diff_carry*100:>12.2f}%')
    print(f'{"AEROS κ=16 (NO state-carry)":<35} {spikes_nocarry:>14.0f} '
          f'{err_nocarry:>10.2e} {diff_nocarry*100:>12.2f}%')

    print('\n' + '=' * 78)
    if err_carry == 0.0 and err_nocarry > 0.0:
        print('✓ STATE-CARRY VERIFIED NECESSARY')
        print(f'  With state-carry:    bit-exact (max_err = 0.0)')
        print(f'  Without state-carry: divergence at {diff_nocarry*100:.2f}% '
              f'of output elements')
        print(f'  Spike-count delta:   {spikes_carry-spikes_nocarry:+.0f} '
              f'({(spikes_nocarry/spikes_carry - 1)*100:+.1f}%)')
    elif err_carry > 0.0:
        print(f'✗ Implementation issue: state-carry should be bit-exact, '
              f'got max_err={err_carry:.2e}')
    else:
        print(f'✗ Even controlled regime did not show divergence — investigating')
        print(f'  baseline spikes = {spikes_base:.0f}; '
              f'this should be >0 for a meaningful test')

    np.savez('p5_3_v3_2_results.npz', rows=[{
        'V_th': V_th, 'tau': tau, 'I': I, 'T': T, 'b': b, 'n': n, 'K': K,
        'v_mean': v_mass.abs().mean().item(),
        'v_max': v_mass.abs().max().item(),
        'spikes_base': spikes_base,
        'spikes_carry': spikes_carry, 'err_carry': err_carry,
        'spikes_nocarry': spikes_nocarry, 'err_nocarry': err_nocarry,
        'diff_nocarry': diff_nocarry,
    }])
    print('\n  Saved to p5_3_v3_2_results.npz')


if __name__ == '__main__':
    main()