"""
Phase 1.4c — Long-T motivation: verify ANN2SNN-typical T=64-512 configs
that motivate paper's "long-T inference" framing.

Goal: show realistic ANN2SNN deployment configs OOM without AEROS.
For T in [64, 128, 256, 512] × ResNet18-SNN at ImageNet-scale 224x224, B=32.
"""
import time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def chunked_forward(net, x, K_max):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K_max, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


def measure(fn, *args, n_warmup=1, n_iters=2):
    torch.cuda.synchronize()
    torch.cuda.empty_cache(); gc.collect()
    try:
        for _ in range(n_warmup):
            _ = fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fn(*args)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated()/1e9,
                (time.perf_counter()-t0)/n_iters*1000, 'ok')
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, 'OOM'


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    B = 32
    H = 224  # ImageNet scale

    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    # Motivation T values from ANN2SNN literature
    Ts = [64, 128, 256, 512]
    results = []

    print(f'\n=== Long-T inference: B={B}, H={H} (ImageNet scale) ===')
    print(f'{"T":>4}  {"baseline":>10}  {"K=8 chunked":>13}  '
          f'{"savings":>9}  {"slowdown":>9}')
    print('-' * 65)

    for T in Ts:
        # Allocate input
        try:
            x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
            input_gb = x.element_size() * x.numel() / 1e9
        except torch.cuda.OutOfMemoryError:
            print(f'{T:>4}  input alloc OOM (T*B*3*H*H*4={T*B*3*H*H*4/1e9:.1f} GB)')
            continue

        # Baseline
        mem_b, wall_b, st_b = measure(chunked_forward, net, x, T)
        # Chunked at K=8 (sweet spot from prior phases)
        mem_K, wall_K, st_K = measure(chunked_forward, net, x, 8)

        # Format
        b_str = f'{mem_b:.1f}GB/{wall_b:.0f}ms' if st_b == 'ok' else 'OOM'
        if st_K == 'ok':
            K_str = f'{mem_K:.1f}GB/{wall_K:.0f}ms'
            if st_b == 'ok':
                savings = mem_b / mem_K
                slowdown = wall_K / wall_b
                sv = f'{savings:.2f}x'
                sd = f'{slowdown:.2f}x'
            else:
                sv = 'OOM→OK'
                sd = 'rescued'
        else:
            K_str = 'OOM'
            sv = '—'
            sd = '—'

        print(f'{T:>4}  {b_str:>10}  {K_str:>13}  {sv:>9}  {sd:>9}')
        results.append({'T': T, 'baseline': (mem_b, wall_b, st_b),
                        'K8': (mem_K, wall_K, st_K)})
        del x
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f'\n=== Motivation summary ===')
    n_baseline_oom = sum(1 for r in results if r['baseline'][2] != 'ok')
    n_chunked_ok = sum(1 for r in results if r['K8'][2] == 'ok')
    n_rescued = sum(1 for r in results
                    if r['baseline'][2] != 'ok' and r['K8'][2] == 'ok')
    print(f'Baseline runs:    {len(results) - n_baseline_oom}/{len(results)}')
    print(f'AEROS K=8 runs:   {n_chunked_ok}/{len(results)}')
    print(f'OOM configs rescued: {n_rescued}')
    print(f'\nPaper claim: at ImageNet scale (224x224, B=32),')
    print(f'AEROS extends feasible inference T from ~64 baseline to ≥{max(r["T"] for r in results if r["K8"][2] == "ok")}.')


if __name__ == '__main__':
    main()
