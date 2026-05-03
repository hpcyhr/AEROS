"""
P5-3 v3.1: same as v3, with no_grad bug fix in ablation (c).

The bug in v3: full_input_mode / streaming_mode forwarded the network
WITHOUT @torch.no_grad(), so autograd retained all intermediate activations
across every LIFNode in SpikingResNet-18. Memory blew up at T=128 due to
graph accumulation. Fix: add @torch.no_grad() decorator (matches v1/v2).

(c) Real streaming with no_grad: full-input vs segment-by-segment.
(d) Single-layer LIF with τ=20, V_th=10, I=0.6, K=16 (unchanged from v3).

Usage:
    python p5_3_v3_1.py
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Ablation (c) v3.1: full-input vs *real* streaming, with no_grad
# =============================================================================
@torch.no_grad()
def full_input_mode(net, T, b, H, K, device, seed=42):
    """Full-input: pre-materialize x[T, ...], then chunk-process."""
    reset_mem()
    torch.manual_seed(seed)
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
def streaming_mode(net, T, b, H, K, device, seed=42):
    """Real streaming: generate each segment fresh; never materialize full x.
    Output not retained — we measure peak memory only."""
    reset_mem()
    torch.manual_seed(seed)
    try:
        functional.reset_net(net)
        i = 0
        while i < T:
            sz = min(K, T - i)
            seg = torch.rand(sz, b, 3, H, H, device=device)
            _ = net(seg)
            del seg
            i += sz
        torch.cuda.synchronize()
        peak = peak_mem_GB()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def ablation_c_v3(device):
    print('\n' + '=' * 78)
    print('=== Ablation (c) v3.1: full-input (pre-materialized) vs '
          'real streaming (generated) ===')
    print('=== at b=32, H=224, K=8 — input alone is 0.0192 GB/step ===')
    print('=' * 78)
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    Ts = [128, 256, 512, 1024, 1536, 2048, 4096, 8192]
    b, H, K = 32, 224, 8

    print(f'\n{"T":>5}  {"full-input":>14} {"full status":>13}  '
          f'{"streaming":>11} {"stream status":>14}  {"savings":>8}')
    print('-' * 78)

    rows = []
    full_oomed = False
    for T in Ts:
        if full_oomed:
            full_mem = None
            full_str = 'skipped'
        else:
            full_mem = full_input_mode(net, T, b, H, K, device)
            full_str = f'{full_mem:.2f} GB' if full_mem else 'OOM'
            if full_mem is None:
                full_oomed = True

        stream_mem = streaming_mode(net, T, b, H, K, device)
        stream_str = f'{stream_mem:.2f} GB' if stream_mem else 'OOM'

        savings = ''
        if full_mem and stream_mem:
            savings = f'{full_mem/stream_mem:.2f}×'

        print(f'{T:>5}  {full_str:>14} {("OK" if full_mem else "—"):>13}  '
              f'{stream_str:>11} {("OK" if stream_mem else "OOM"):>14}  {savings:>8}')
        rows.append({'T': T, 'b': b, 'H': H, 'K': K,
                     'full_mem': full_mem, 'stream_mem': stream_mem})

    max_full = max([r['T'] for r in rows if r['full_mem']], default=0)
    max_stream = max([r['T'] for r in rows if r['stream_mem']], default=0)
    print(f'\n  Max feasible T (full-input):  {max_full}')
    print(f'  Max feasible T (streaming):   {max_stream}')
    if max_stream > max_full:
        print(f'  → Streaming extends feasible horizon by {max_stream/max(max_full,1):.1f}×')
    return rows


# =============================================================================
# Ablation (d) v3.1: single-layer LIF integrator (unchanged from v3)
# =============================================================================
class TinyLIF(nn.Module):
    def __init__(self, v_threshold=10.0, tau=20.0):
        super().__init__()
        self.lif = neuron.LIFNode(
            tau=tau, decay_input=True,
            v_threshold=v_threshold, v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, step_mode='m',
        )
    def forward(self, x):
        return self.lif(x)


def ablation_d_v3(device):
    print('\n' + '=' * 78)
    print('=== Ablation (d) v3.1: state-carry necessity ===')
    print('=== TinyLIF (single-layer): τ=20, V_th=10, constant I=0.6 ===')
    print('=== Expected: v at chunk boundary K=16 is ~6.8; reset clobbers this ===')
    print('=== → spike timing differs by ~16 steps → guaranteed divergence ===')
    print('=' * 78)

    net = TinyLIF(v_threshold=10.0, tau=20.0).to(device)
    net.eval()

    T, b, n, K = 64, 4, 1024, 16
    I = 0.6
    x = torch.full((T, b, n), I, device=device)
    print(f'\nSetup: T={T}, b={b}, n={n} neurons, K={K}, I={I}')
    print(f'  decay factor exp(-1/τ) = {np.exp(-1/20):.4f}')
    print(f'  steady-state v (no fire) = I / (1-decay) = '
          f'{I / (1 - np.exp(-1/20)):.2f}')
    print(f'  v at t=K=16 (no fire, from v=0): '
          f'{I * (1 - np.exp(-1/20)**16) / (1 - np.exp(-1/20)):.3f}')
    print(f'  Expected: baseline first fires at ~t=33; with state-carry t≈33; '
          f'no-state-carry t≈49')

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

    y_base = baseline(net, x)
    y_carry = chunked(net, x, K, do_state_carry=True)
    y_nocarry = chunked(net, x, K, do_state_carry=False)

    err_carry = (y_carry - y_base).abs().max().item()
    err_nocarry = (y_nocarry - y_base).abs().max().item()
    diff_nocarry = float((y_nocarry != y_base).float().mean())
    spikes_base = float(y_base.sum())
    spikes_carry = float(y_carry.sum())
    spikes_nocarry = float(y_nocarry.sum())

    print(f'\n  Baseline (unchunked):     total spikes = {spikes_base:.0f}')
    print(f'  AEROS κ=16 (state-carry): total spikes = {spikes_carry:.0f}, '
          f'max_err = {err_carry:.2e}')
    print(f'  AEROS κ=16 (NO carry):    total spikes = {spikes_nocarry:.0f}, '
          f'max_err = {err_nocarry:.2e}, '
          f'differing elements = {diff_nocarry*100:.2f}%')

    if err_carry == 0.0 and err_nocarry > 0.0:
        print('\n  ✓ STATE-CARRY VERIFIED NECESSARY:')
        print(f'    With state-carry: bit-exact (max_err = 0.0)')
        print(f'    Without state-carry: divergence at {diff_nocarry*100:.2f}% '
              f'of output elements')
    elif err_carry > 0.0:
        print(f'\n  WARN: state-carry DID NOT achieve bit-exact (err = {err_carry:.2e})')
    else:
        print(f'\n  WARN: even controlled regime did not show divergence.')

    return [{
        'T': T, 'b': b, 'n': n, 'K': K, 'I': I, 'tau': 20.0, 'v_th': 10.0,
        'spikes_base': spikes_base,
        'spikes_carry': spikes_carry, 'err_carry': err_carry,
        'spikes_nocarry': spikes_nocarry, 'err_nocarry': err_nocarry,
        'diff_nocarry_frac': diff_nocarry,
    }]


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', nargs='+', default=[], choices=['c', 'd'])
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print(f'Running on {device}\n')

    out = {}
    if 'c' not in args.skip:
        out['c'] = ablation_c_v3(device)
    if 'd' not in args.skip:
        out['d'] = ablation_d_v3(device)

    np.savez('p5_3_v3_1_results.npz', **{f'ablation_{k}': v for k, v in out.items()})
    print('\n  Saved to p5_3_v3_1_results.npz')


if __name__ == '__main__':
    main()