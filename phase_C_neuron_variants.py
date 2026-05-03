"""
Direction C sanity check: how slow are non-standard neuron variants vs LIF?
"""
import time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional


def bench(neuron_cls, neuron_kwargs, x, name, n_warmup=5, n_iters=30,
          step_mode='m'):
    n = neuron_cls(**neuron_kwargs).to(x.device)
    functional.set_step_mode(n, step_mode=step_mode)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = n(x)
            functional.reset_net(n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = n(x)
            functional.reset_net(n)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / n_iters * 1000
    return wall_ms


def main():
    device = 'cuda:0'
    T, B, C, H, W = 32, 16, 256, 14, 14
    x = torch.randn(T, B, C, H, W, device=device)

    print(f'Input shape: T={T} B={B} C={C} H={H} W={W}\n')

    # Discover available neuron variants in SpikingJelly
    candidates = [
        ('LIFNode', neuron.LIFNode, {'detach_reset': True}),
        ('ParametricLIFNode', getattr(neuron, 'ParametricLIFNode', None),
         {'detach_reset': True}),
        ('IFNode', getattr(neuron, 'IFNode', None), {'detach_reset': True}),
        ('EIFNode', getattr(neuron, 'EIFNode', None), {'detach_reset': True}),
        ('QIFNode', getattr(neuron, 'QIFNode', None), {'detach_reset': True}),
        ('IzhikevichNode', getattr(neuron, 'IzhikevichNode', None),
         {'detach_reset': True}),
        ('LIAFNode', getattr(neuron, 'LIAFNode', None),
         {'act': torch.nn.ReLU(), 'detach_reset': True}),
        ('KLIFNode', getattr(neuron, 'KLIFNode', None), {'detach_reset': True}),
    ]

    print(f'{"neuron":<25s}  {"step":<10s}  {"wall ms":>9s}  '
          f'{"vs LIF":>8s}  {"backend":<10s}')
    print('-' * 75)

    results = {}
    lif_baseline = None
    for name, cls, kwargs in candidates:
        if cls is None:
            print(f'{name:<25s}  -- not in this SJ version --')
            continue
        # Try multi-step first
        for step_mode in ['m', 's']:
            try:
                wall = bench(cls, kwargs, x, name, step_mode=step_mode)
                if name == 'LIFNode' and step_mode == 'm':
                    lif_baseline = wall
                ratio = wall / lif_baseline if lif_baseline else 1.0
                # Try cupy backend for LIF as comparison
                backend = 'torch'
                if step_mode == 'm' and name == 'LIFNode':
                    try:
                        n2 = cls(**kwargs).to(device)
                        functional.set_step_mode(n2, step_mode='m')
                        functional.set_backend(n2, backend='cupy')
                        for _ in range(5):
                            with torch.no_grad():
                                _ = n2(x); functional.reset_net(n2)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(30):
                            with torch.no_grad():
                                _ = n2(x); functional.reset_net(n2)
                        torch.cuda.synchronize()
                        wall_cupy = (time.perf_counter() - t0) / 30 * 1000
                        print(f'{"LIFNode (cupy)":<25s}  {"m":<10s}  '
                              f'{wall_cupy:>8.2f}  '
                              f'{wall_cupy/lif_baseline:>7.2f}x  cupy')
                    except Exception as e:
                        pass
                print(f'{name:<25s}  {step_mode:<10s}  {wall:>8.2f}  '
                      f'{ratio:>7.2f}x  {backend:<10s}')
                results[(name, step_mode)] = wall
                if step_mode == 'm':
                    break  # if multi-step works, skip single-step
            except Exception as e:
                if step_mode == 's':
                    print(f'{name:<25s}  {step_mode:<10s}  '
                          f'FAIL: {str(e)[:40]}')

    print(f'\n=== Verdict ===')
    if lif_baseline:
        # Find slowest non-LIF
        non_lif = [(k, v) for k, v in results.items()
                   if k[0] != 'LIFNode' and k[1] == 'm']
        if non_lif:
            slowest_name, slowest_t = max(non_lif, key=lambda kv: kv[1])
            ratio = slowest_t / lif_baseline
            print(f'Slowest non-LIF (multi-step): {slowest_name[0]} = '
                  f'{ratio:.2f}× LIF baseline')
            if ratio > 5:
                print('GREEN: large gap. Codegen could 5× speedup non-LIF.')
            elif ratio > 2:
                print('YELLOW: 2-5× gap, marginal benefit.')
            else:
                print('RED: < 2× gap. SJ already optimizes non-LIF well.')


if __name__ == '__main__':
    main()