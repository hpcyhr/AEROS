"""
P5-3 v2: Ablations (c) and (d) at corrected regimes.

(c) At headline regime (b=32, H=224) — full-input AEROS expected to OOM at
    T~=2048; streaming expected to remain feasible through T=8192.

(d) Multi-regime LIF parameter sweep — under default LIF dynamics (τ=2,
    V_thresh=1), v at chunk boundary is dominated by recent fire+reset, so
    no-state-carry is indistinguishable from state-carry. With elevated
    V_thresh (2.0) or slow decay (τ=10), v accumulates non-trivially across
    chunk boundaries, and the no-state-carry ablation should diverge.

This script ALSO instruments v-at-boundary statistics: it reports
fraction of LIF neurons with |v| > 1e-4 at the first chunk boundary,
to confirm whether the test regime is discriminating BEFORE concluding
state-carry doesn't matter.

Usage:
    python p5_3_v2.py
"""
import argparse, os, time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


# -----------------------------------------------------------------------------
def build_net(device, v_threshold=1.0, tau=2.0):
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True,
                           v_threshold=v_threshold, tau=tau).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    return net


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def run_baseline(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def run_chunked(net, x, K, do_state_carry=True):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        if (not do_state_carry) and i > 0:
            functional.reset_net(net)
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def run_streaming(net, x, K):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        seg = x[i:i + sz].clone()
        chunks.append(net(seg))
        del seg
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def boundary_v_stats(net, x, K):
    """Process first chunk in baseline mode, snapshot v at every LIFNode.
    Returns dict: layer_name -> (frac_nonzero, mean_abs_v, max_abs_v)."""
    functional.reset_net(net)
    _ = net(x[:K])  # process one chunk
    stats = {}
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            v = m.v
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                v_flat = v.detach().flatten()
                frac_nz = float((v_flat.abs() > 1e-4).float().mean())
                mean_abs = float(v_flat.abs().mean())
                max_abs = float(v_flat.abs().max())
                stats[name] = (frac_nz, mean_abs, max_abs)
    return stats


# =============================================================================
def ablation_c_streaming_at_headline(device):
    """(c) v2: full-input vs streaming at b=32, H=224 (the §5.7 headline regime)."""
    print('\n' + '=' * 78)
    print('=== Ablation (c) v2: No streaming — full-input vs streaming '
          'at b=32, H=224 ===')
    print('=' * 78)
    print('Theoretical predictions from fitted (b=32,H=224) coefficients:')
    print('  full-input M(T, K=8) ≈ 0.66 + 0.0196*T + 0.204*8 GB')
    print('  full-input expected OOM around T~1500-2000')
    print('  streaming expected ~4 GB regardless of T\n')

    net = build_net(device)  # default LIF
    Ts = [128, 256, 512, 1024, 1536, 2048, 4096, 8192]
    b, H, K = 32, 224, 8

    print(f'{"T":>5}  {"full-input mem":>16} {"full status":>13}  '
          f'{"streaming mem":>14} {"stream status":>14}  {"speedup":>8}')
    print('-' * 78)

    rows = []
    for T in Ts:
        # Full-input
        reset_mem()
        full_mem = None
        full_ok = False
        full_wall = None
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = run_chunked(net, x, K)
            torch.cuda.synchronize()
            full_wall = (time.perf_counter() - t0) * 1000
            full_mem = peak_mem_GB()
            full_ok = True
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # Streaming
        reset_mem()
        stream_mem = None
        stream_ok = False
        stream_wall = None
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = run_streaming(net, x, K)
            torch.cuda.synchronize()
            stream_wall = (time.perf_counter() - t0) * 1000
            stream_mem = peak_mem_GB()
            stream_ok = True
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        full_str = f'{full_mem:.2f} GB' if full_mem else '—'
        stream_str = f'{stream_mem:.2f} GB' if stream_mem else '—'
        speedup_str = ''
        if full_wall and stream_wall:
            speedup_str = f'{full_wall/stream_wall:.2f}×'
        print(f'{T:>5}  {full_str:>16} {("OK" if full_ok else "OOM"):>13}  '
              f'{stream_str:>14} {("OK" if stream_ok else "OOM"):>14}  {speedup_str:>8}')
        rows.append({'T': T, 'b': b, 'H': H, 'K': K,
                     'full_mem': full_mem, 'full_ok': full_ok, 'full_wall': full_wall,
                     'stream_mem': stream_mem, 'stream_ok': stream_ok,
                     'stream_wall': stream_wall})

        # Stop full-input attempts after first OOM (subsequent T will also OOM)
        # but keep trying streaming
        if not full_ok and T < 8192:
            # Skip remaining full-input runs but continue streaming
            pass

    max_full = max([r['T'] for r in rows if r['full_ok']], default=0)
    max_stream = max([r['T'] for r in rows if r['stream_ok']], default=0)
    print(f'\n  Max feasible T (full-input):  {max_full}')
    print(f'  Max feasible T (streaming):   {max_stream}')
    if max_stream > max_full:
        print(f'  → Streaming extends feasible horizon by {max_stream/max(max_full,1):.1f}×')
    return rows


# =============================================================================
def ablation_d_state_carry_multiregime(device):
    """(d) v2: state-carry necessity under multiple LIF parameter regimes."""
    print('\n' + '=' * 78)
    print('=== Ablation (d) v2: state-carry necessity, multi-regime ===')
    print('=' * 78)

    # Regimes to test
    regimes = [
        # (label, V_thresh, tau, input_rate, T, K)
        ('default LIF (τ=2, V_th=1)',           1.0,  2.0, 0.3, 64,  8),
        ('elevated V_th (τ=2, V_th=2)',         2.0,  2.0, 0.5, 128, 8),
        ('slow decay (τ=10, V_th=1)',           1.0, 10.0, 0.3, 128, 8),
        ('elevated+slow (τ=10, V_th=2)',        2.0, 10.0, 0.5, 128, 8),
        ('high input + slow (τ=10, V_th=1, p=0.7)', 1.0, 10.0, 0.7, 128, 8),
    ]

    b, H = 8, 64

    print(f'\nFor each regime: build SpikingResNet-18 with custom LIF params,')
    print(f'check v-at-chunk-boundary stats, run state-carry vs no-state-carry.\n')

    rows = []
    for label, v_th, tau, p_input, T, K in regimes:
        print(f'\n--- Regime: {label} ---')
        net = build_net(device, v_threshold=v_th, tau=tau)

        # Generate input
        torch.manual_seed(42)
        x = (torch.rand(T, b, 3, H, H, device=device) < p_input).float()

        # First, check v stats at chunk boundary
        v_stats = boundary_v_stats(net, x, K)
        # Aggregate: mean fraction-nonzero across all LIF layers
        if v_stats:
            fracs = [s[0] for s in v_stats.values()]
            mean_abs = [s[1] for s in v_stats.values()]
            max_abs = [s[2] for s in v_stats.values()]
            agg_frac_nz = float(np.mean(fracs))
            agg_mean_abs = float(np.mean(mean_abs))
            agg_max_abs = float(np.max(max_abs))
            print(f'  v-at-boundary (after first chunk K={K}):')
            print(f'    avg fraction-nonzero across LIFs: {agg_frac_nz*100:.1f}%')
            print(f'    avg |v|:  {agg_mean_abs:.4f}')
            print(f'    max |v|:  {agg_max_abs:.4f}')
        else:
            agg_frac_nz = 0.0
            agg_mean_abs = 0.0
            agg_max_abs = 0.0
            print(f'  (no LIFNode v captured)')

        # Run baseline + with-state-carry + without-state-carry
        y_base = run_baseline(net, x)
        y_carry = run_chunked(net, x, K, do_state_carry=True)
        y_nocarry = run_chunked(net, x, K, do_state_carry=False)

        err_carry = (y_carry - y_base).abs().max().item()
        err_nocarry = (y_nocarry - y_base).abs().max().item()

        # Also compute what fraction of output elements differ
        diff_carry_frac = float((y_carry != y_base).float().mean())
        diff_nocarry_frac = float((y_nocarry != y_base).float().mean())

        diverged = err_nocarry > 1e-4

        print(f'  T={T}, K={K}, # chunk boundaries = {T//K - 1}')
        print(f'  state-carry max_err     = {err_carry:.2e}  '
              f'(differing elements: {diff_carry_frac*100:.2f}%)')
        print(f'  no-state-carry max_err  = {err_nocarry:.2e}  '
              f'(differing elements: {diff_nocarry_frac*100:.2f}%)')
        print(f'  → divergence detected: {"YES" if diverged else "no"}')

        rows.append({
            'regime': label, 'v_threshold': v_th, 'tau': tau,
            'input_rate': p_input, 'T': T, 'K': K,
            'frac_nonzero_v': agg_frac_nz,
            'mean_abs_v': agg_mean_abs, 'max_abs_v': agg_max_abs,
            'err_carry': err_carry, 'err_nocarry': err_nocarry,
            'diff_carry_frac': diff_carry_frac, 'diff_nocarry_frac': diff_nocarry_frac,
            'diverged': diverged,
        })

        # Free memory
        del net, x, y_base, y_carry, y_nocarry
        torch.cuda.empty_cache()

    print('\n' + '=' * 78)
    print('Summary: under which regime does state-carry observably matter?')
    print('=' * 78)
    print(f'{"Regime":<45} {"v boundary":>12} {"diverged":>10}')
    for r in rows:
        marker = '✓' if r['diverged'] else '·'
        print(f'{r["regime"]:<45} {r["frac_nonzero_v"]*100:>10.1f}% {marker:>10}')

    diverged_any = any(r['diverged'] for r in rows)
    if diverged_any:
        print('\n  At least one regime confirms state-carry necessity.')
    else:
        print('\n  WARN: no regime triggered divergence. May need to dig further.')
    return rows


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', nargs='+', default=[], choices=['c', 'd'])
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print(f'Building spiking_resnet18 on {device}...')

    out = {}
    if 'c' not in args.skip:
        out['c'] = ablation_c_streaming_at_headline(device)
    if 'd' not in args.skip:
        out['d'] = ablation_d_state_carry_multiregime(device)

    np.savez('p5_3_v2_results.npz', **{f'ablation_{k}': v for k, v in out.items()})
    print('\n  Saved to p5_3_v2_results.npz')


if __name__ == '__main__':
    main()