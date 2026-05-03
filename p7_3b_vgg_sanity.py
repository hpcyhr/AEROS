"""
p7_3b_vgg_sanity.py — VGG-11-BN sanity check at H=128.

Purpose: Doris P7 review noted that v7's VGG-11-BN "rescue" claim
(baseline OOMs at every T at H=224) is easy to attack as cherry-picked.
She recommended a companion experiment showing AEROS still wins at
H=128 where the baseline DOES fit — converting "pathological rescue"
into "same trend across feasible and infeasible regimes."

This script runs VGG-11-BN at H=128, b=32 (a regime where baseline
fits at small T), measuring:
  - Baseline max feasible T
  - AEROS κ=8 max feasible T
  - Headline timing at the largest T where baseline fits

If AEROS extends max-T meaningfully (≥ 4x) at H=128 too, the rescue
case at H=224 is a stress test, not cherry-picking.

Usage:
    python p7_3b_vgg_sanity.py
    python p7_3b_vgg_sanity.py --b 16 --H 224
"""
import argparse
import time
import gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn


def aggressive_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def reset_mem():
    aggressive_cleanup()
    torch.cuda.reset_peak_memory_stats()


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


@torch.no_grad()
def safe_measure_with_wall(net, T, b, H, K, device, n_iters=4):
    aggressive_cleanup()
    reset_mem()
    try:
        x = torch.rand(T, b, 3, H, H, device=device)
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
        walls = sorted(walls)[:-1]  # drop slowest as warmup
        return peak, float(np.median(walls))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        try:
            functional.reset_net(net)
        except Exception:
            pass
        aggressive_cleanup()
        return None, None
    except Exception as e:
        print(f'    Unexpected: {e}')
        aggressive_cleanup()
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=32)
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--num_classes', type=int, default=11)
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print(f'\n=== P7-3b: VGG-11-BN sanity check (b={args.b}, H={args.H}) ===\n')

    print('Building spiking_vgg11_bn...')
    net = spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True,
                           num_classes=args.num_classes).cuda().eval()
    functional.set_step_mode(net, step_mode='m')

    # Sweep T
    Ts = [32, 64, 128, 256, 512, 1024, 2048]
    K = 8
    rows = []
    print(f'{"T":>5}  {"baseline":>22} {"AEROS κ=8":>22} {"slowdown":>9}')
    print('-' * 70)

    baseline_oomed = False
    for T in Ts:
        if baseline_oomed:
            base_mem, base_wall = None, None
        else:
            base_mem, base_wall = safe_measure_with_wall(net, T, args.b, args.H, T, device)
            if base_mem is None:
                baseline_oomed = True

        aeros_mem, aeros_wall = safe_measure_with_wall(net, T, args.b, args.H, K, device)

        sd = ''
        if base_wall and aeros_wall:
            sd = f'{aeros_wall/base_wall:.3f}×'

        bm = (f'{base_mem:.2f}GB/{base_wall:.0f}ms' if base_mem else 'OOM')
        am = (f'{aeros_mem:.2f}GB/{aeros_wall:.0f}ms' if aeros_mem else 'OOM')
        print(f'{T:>5}  {bm:>22} {am:>22} {sd:>9}')
        rows.append({'T': T, 'b': args.b, 'H': args.H, 'K': K,
                     'base_mem': base_mem, 'base_wall': base_wall,
                     'aeros_mem': aeros_mem, 'aeros_wall': aeros_wall})

    max_base = max([r['T'] for r in rows if r['base_mem']], default=0)
    max_aeros = max([r['T'] for r in rows if r['aeros_mem']], default=0)

    print('\n' + '=' * 70)
    print(f'Summary — VGG-11-BN at b={args.b}, H={args.H}')
    print('=' * 70)
    print(f'  Baseline max feasible T:  {max_base}')
    print(f'  AEROS κ=8 max feasible T: {max_aeros}')
    if max_base > 0 and max_aeros > 0:
        ext = max_aeros / max_base
        print(f'  Horizon extension:        {ext:.1f}×')
        # Slowdown at largest baseline-feasible T
        last_both = [r for r in rows if r['T'] <= max_base
                     and r['base_wall'] and r['aeros_wall']]
        if last_both:
            r = last_both[-1]
            sd = r['aeros_wall'] / r['base_wall']
            print(f'  Slowdown at T={r["T"]}: {sd:.3f}× '
                  f'(baseline {r["base_wall"]:.0f}ms, AEROS {r["aeros_wall"]:.0f}ms)')
        if ext >= 4.0:
            print(f'\n  ✓ AEROS extends max-T by {ext:.1f}× even at H={args.H} where '
                  f'baseline fits at small T')
            print(f'    → VGG rescue at H=224 is a stress test, not cherry-picking')
        else:
            print(f'\n  ◐ Extension only {ext:.1f}× at H={args.H}; '
                  f'rescue framing should be regime-specific')
    elif max_base == 0:
        print(f'  ⚠ Baseline OOMs even at H={args.H}; this regime is too tight too')
    elif max_aeros == 0:
        print(f'  ✗ AEROS OOMs at H={args.H} — implementation issue')

    np.savez('p7_3b_vgg_sanity_results.npz',
             b=args.b, H=args.H, K=K, rows=rows,
             max_base=max_base, max_aeros=max_aeros)
    print('\n  Saved to p7_3b_vgg_sanity_results.npz')


if __name__ == '__main__':
    main()