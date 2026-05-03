"""
Direction A sanity check: SNN inference memory pressure as T grows.

Profile peak GPU memory + wall-clock for SNN models at varying T.
Decompose memory into: weights, activations, LIF state.

Output:
  GREEN if memory exceeds 80% of 32GB at T=64+
  RED   if memory stays below 50% at T=128
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.activation_based.model import spiking_resnet


def build_model(name, channels_in=3):
    """Build SNN model. Default ResNet18-SNN."""
    if name == 'sresnet18':
        net = spiking_resnet.spiking_resnet18(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            num_classes=1000,
        )
        # First conv expects 3 channels
    elif name == 'sresnet34':
        net = spiking_resnet.spiking_resnet34(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            num_classes=1000,
        )
    else:
        raise ValueError(name)
    functional.set_step_mode(net, step_mode='m')
    return net


def measure_memory(model, T, B, H, W, C_in, device, n_warmup=2, n_iters=3):
    """Run forward, measure peak memory + wall-clock."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Track baseline (weights + book-keeping)
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # Input: [T, B, C, H, W]
    x = torch.randn(T, B, C_in, H, W, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x).mean(0)
            functional.reset_net(model)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Measure
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            y = model(x).mean(0)
            functional.reset_net(model)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / n_iters

    peak_mem = torch.cuda.max_memory_allocated(device)

    return {
        'T': T, 'B': B,
        'peak_mem_GB': peak_mem / 1e9,
        'weight_mem_MB': weight_mem / 1e6,
        'wall_ms': t * 1000,
        'peak_frac_32GB': peak_mem / (32 * 1024**3),
    }


def measure_state_buffers(model):
    """Estimate LIF state memory by running 1 forward and counting v stored."""
    state_bytes = 0
    for m in model.modules():
        if isinstance(m, neuron.LIFNode):
            if hasattr(m, 'v') and isinstance(m.v, torch.Tensor):
                state_bytes += m.v.numel() * m.v.element_size()
    return state_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='sresnet18',
                   choices=['sresnet18', 'sresnet34'])
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--H', type=int, default=224)
    p.add_argument('--W', type=int, default=224)
    p.add_argument('--C_in', type=int, default=3)
    p.add_argument('--T_list', type=int, nargs='+',
                   default=[4, 8, 16, 32, 64, 96, 128])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='phaseA_mem_sresnet18.npz')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'Building {args.model}...')
    net = build_model(args.model, channels_in=args.C_in).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  params: {n_params / 1e6:.2f}M')

    # Total GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory
    print(f'  GPU total memory: {total_mem / 1e9:.1f} GB')

    print(f'\n{"T":>4s}  {"B":>3s}  {"peak GB":>8s}  {"% 32GB":>7s}  '
          f'{"weight MB":>10s}  {"wall ms":>9s}')
    print('-' * 60)

    results = []
    for T in args.T_list:
        try:
            r = measure_memory(net, T, args.B, args.H, args.W, args.C_in, device)
            results.append(r)
            print(f'{r["T"]:>4d}  {r["B"]:>3d}  {r["peak_mem_GB"]:>7.2f}  '
                  f'{r["peak_frac_32GB"]*100:>6.1f}%  {r["weight_mem_MB"]:>9.1f}  '
                  f'{r["wall_ms"]:>8.1f}')
        except torch.cuda.OutOfMemoryError as e:
            print(f'{T:>4d}  {args.B:>3d}  OOM')
            results.append({'T': T, 'B': args.B,
                            'peak_mem_GB': float('nan'),
                            'peak_frac_32GB': float('nan'),
                            'wall_ms': float('nan'),
                            'OOM': True})
            torch.cuda.empty_cache()
            break

    # Save
    np.savez(args.out, results=results)
    print(f'\nSaved {args.out}')

    # Verdict
    finite = [r for r in results if not r.get('OOM', False)]
    if not finite:
        print('All T values OOMed — strong GREEN, but probably needs smaller B.')
        return

    max_frac = max(r['peak_frac_32GB'] for r in finite)
    max_T = max(r['T'] for r in finite if not r.get('OOM', False))

    print(f'\n=== Direction A verdict ===')
    if any(r.get('OOM', False) for r in results) or max_frac > 0.80:
        print(f'GREEN: peak {max_frac*100:.1f}% of 32GB at T={max_T}.')
        print(f'  Memory pressure is real, system pain point exists.')
        print(f'  Selective recompute / streaming could trade memory for time.')
    elif max_frac > 0.50:
        print(f'YELLOW: peak {max_frac*100:.1f}% of 32GB at T={max_T}.')
        print(f'  Some pressure, but not dramatic. May still be paper-worthy')
        print(f'  on A100 80GB or larger inputs / models.')
    else:
        print(f'RED: peak {max_frac*100:.1f}% of 32GB at T={max_T}.')
        print(f'  No memory pressure. Direction A unlikely to yield speedups.')


if __name__ == '__main__':
    main()