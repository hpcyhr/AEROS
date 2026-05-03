"""
Direction A extended: full memory decomposition + larger setup sweep.

For each (model, B, T), measure:
  - weight memory
  - LIF state memory (membrane V across all layers)
  - activation memory (peak - weight - state)
  - wall-clock
"""
import argparse, time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model import spiking_resnet


def build(name, num_classes=1000):
    if name == 'sresnet18':
        net = spiking_resnet.spiking_resnet18(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
            detach_reset=True, num_classes=num_classes)
    elif name == 'sresnet34':
        net = spiking_resnet.spiking_resnet34(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
            detach_reset=True, num_classes=num_classes)
    elif name == 'sresnet50':
        net = spiking_resnet.spiking_resnet50(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
            detach_reset=True, num_classes=num_classes)
    else:
        raise ValueError(name)
    functional.set_step_mode(net, step_mode='m')
    return net


def state_bytes(model):
    s = 0
    for m in model.modules():
        if isinstance(m, neuron.LIFNode):
            if hasattr(m, 'v') and isinstance(m.v, torch.Tensor):
                s += m.v.numel() * m.v.element_size()
    return s


def measure(model, T, B, H, W, device, n_warmup=2, n_iters=3):
    torch.cuda.empty_cache()
    weight_b = sum(p.numel() * p.element_size() for p in model.parameters())
    x = torch.randn(T, B, 3, H, W, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x).mean(0)
            functional.reset_net(model)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x).mean(0)
            # Capture state size right after forward, before reset
            state_b = state_bytes(model)
            functional.reset_net(model)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / n_iters
    peak_b = torch.cuda.max_memory_allocated(device)

    return {
        'peak_GB': peak_b / 1e9,
        'weight_MB': weight_b / 1e6,
        'state_MB': state_b / 1e6,
        'activation_MB': max(0, (peak_b - weight_b - state_b) / 1e6),
        'wall_ms': t * 1000,
        'frac_32GB': peak_b / (32 * 1024**3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='phaseA_ext.npz')
    args = p.parse_args()

    device = torch.device(args.device)

    # (model, B, T_list)
    configs = [
        ('sresnet18', 32, [4, 8, 16, 32, 64, 96, 128]),
        ('sresnet18', 64, [4, 8, 16, 32, 64, 96]),
        ('sresnet34', 16, [4, 8, 16, 32, 64, 96, 128]),
        ('sresnet34', 32, [4, 8, 16, 32, 64]),
    ]

    print(f'{"model":>10s}  {"B":>3s}  {"T":>4s}  {"peak GB":>8s}  '
          f'{"% 32GB":>7s}  {"weight":>8s}  {"state":>9s}  '
          f'{"act":>9s}  {"wall ms":>9s}')
    print('-' * 92)

    all_results = []
    for model_name, B, T_list in configs:
        net = build(model_name).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        for T in T_list:
            try:
                r = measure(net, T, B, 224, 224, device)
                r.update({'model': model_name, 'B': B, 'T': T,
                          'params_M': n_params / 1e6})
                all_results.append(r)
                print(f'{model_name:>10s}  {B:>3d}  {T:>4d}  '
                      f'{r["peak_GB"]:>7.2f}  {r["frac_32GB"]*100:>6.1f}%  '
                      f'{r["weight_MB"]:>7.1f}M  {r["state_MB"]:>8.1f}M  '
                      f'{r["activation_MB"]:>8.1f}M  {r["wall_ms"]:>8.1f}')
            except torch.cuda.OutOfMemoryError:
                print(f'{model_name:>10s}  {B:>3d}  {T:>4d}  OOM')
                all_results.append({'model': model_name, 'B': B, 'T': T,
                                    'OOM': True})
                torch.cuda.empty_cache()
                break  # Skip larger T after OOM
        del net
        torch.cuda.empty_cache()

    np.savez(args.out, results=all_results)
    print(f'\nSaved {args.out}')

    # Verdict
    oom_configs = [r for r in all_results if r.get('OOM', False)]
    if oom_configs:
        print(f'\n=== Verdict: GREEN ===')
        print(f'Hit OOM in {len(oom_configs)} configurations:')
        for r in oom_configs:
            print(f'  {r["model"]}, B={r["B"]}, T={r["T"]} → OOM')
        print('Real memory pressure exists. Selective recompute / streaming')
        print('would unblock these configurations.')
    else:
        max_frac = max(r.get('frac_32GB', 0) for r in all_results)
        if max_frac > 0.85:
            print(f'\n=== Verdict: GREEN-borderline ===')
            print(f'Max {max_frac*100:.1f}% of 32GB. Close to OOM in extreme cases.')
        elif max_frac > 0.50:
            print(f'\n=== Verdict: YELLOW ===')
            print(f'Max {max_frac*100:.1f}% of 32GB.')
            print('Some pressure but not dramatic. A100 (80GB) would have even less.')
        else:
            print(f'\n=== Verdict: RED ===')

    # Report state vs activation breakdown
    print('\n=== Memory composition (last successful T per config) ===')
    print(f'{"model":>10s}  {"B":>3s}  {"T":>4s}  {"weight%":>8s}  '
          f'{"state%":>8s}  {"act%":>8s}')
    seen = set()
    for r in reversed(all_results):
        if r.get('OOM', False):
            continue
        key = (r['model'], r['B'])
        if key in seen:
            continue
        seen.add(key)
        total = r['weight_MB'] + r['state_MB'] + r['activation_MB']
        if total > 0:
            print(f'{r["model"]:>10s}  {r["B"]:>3d}  {r["T"]:>4d}  '
                  f'{r["weight_MB"]/total*100:>7.1f}%  '
                  f'{r["state_MB"]/total*100:>7.1f}%  '
                  f'{r["activation_MB"]/total*100:>7.1f}%')


if __name__ == '__main__':
    main()