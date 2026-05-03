"""Phase 1.1d sweep with verbose progress."""
import time
import gc
import itertools
import sys
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def baseline_forward(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_forward(net, x, K):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    for i in range(T // K):
        chunks.append(net(x[i * K:(i + 1) * K]))
    return torch.cat(chunks, dim=0)


def measure(fn, *args, n_warmup=1, n_iters=3):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    try:
        for _ in range(n_warmup):
            _ = fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fn(*args)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / n_iters * 1000
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        return peak_mem, wall_ms, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None, None, 'OOM'
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return None, None, f'FAIL:{str(e)[:40]}'


def get_K_grid(T):
    Ks = []
    K = T
    while K >= 1:
        Ks.append(K)
        K //= 2
    return Ks


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    Ts = [32, 64, 128, 256]
    Bs = [16, 32, 64]
    HWs = [(64, 64), (128, 128), (224, 224)]

    print('Building net...', flush=True)
    net = spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    ).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    print(f'  params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M', flush=True)

    results = []
    configs = list(itertools.product(Ts, Bs, HWs))
    n_total = len(configs)
    t_global_start = time.perf_counter()

    for idx, (T, B, (H, W)) in enumerate(configs):
        elapsed = time.perf_counter() - t_global_start
        input_gb = T * B * 3 * H * W * 4 / 1e9
        print(f'\n[{idx+1}/{n_total}] (elapsed {elapsed:.0f}s) T={T} B={B} HW={H}x{W} '
              f'input={input_gb:.2f}GB', flush=True)

        # Allocate input
        try:
            x = (torch.rand(T, B, 3, H, W, device=device) > 0.7).float()
        except torch.cuda.OutOfMemoryError:
            print(f'  input alloc OOM, skip', flush=True)
            results.append({'T':T,'B':B,'H':H,'W':W,'skip':'input_oom'})
            continue

        # Baseline
        print(f'  baseline (K={T})...', end=' ', flush=True)
        t0 = time.perf_counter()
        mem_base, wall_base, status_base = measure(baseline_forward, net, x)
        dt = time.perf_counter() - t0
        if status_base == 'ok':
            print(f'mem={mem_base:.2f}GB wall={wall_base:.0f}ms ({dt:.0f}s)', flush=True)
        else:
            print(f'{status_base} ({dt:.0f}s)', flush=True)

        # Chunked sweep
        Ks = get_K_grid(T)
        K_results = []
        for K in Ks:
            if K == T:
                # already measured as baseline
                K_results.append({
                    'K': K, 'peak_mem_GB': mem_base, 'wall_ms': wall_base,
                    'status': status_base
                })
                continue
            t0 = time.perf_counter()
            mem_K, wall_K, status_K = measure(chunked_forward, net, x, K)
            dt = time.perf_counter() - t0
            if status_K == 'ok':
                print(f'    K={K:>3d}: mem={mem_K:.2f}GB wall={wall_K:.0f}ms ({dt:.0f}s)',
                      flush=True)
            else:
                print(f'    K={K:>3d}: {status_K} ({dt:.0f}s)', flush=True)
            K_results.append({
                'K': K, 'peak_mem_GB': mem_K, 'wall_ms': wall_K, 'status': status_K
            })

        # Sweet spot
        if status_base == 'ok':
            ref_wall, ref_mem, ref_K = wall_base, mem_base, T
            base_label = 'OK'
        else:
            ref_wall = ref_mem = ref_K = None
            for r in K_results:
                if r['status'] == 'ok':
                    ref_wall = r['wall_ms']; ref_mem = r['peak_mem_GB']; ref_K = r['K']
                    break
            base_label = 'baseOOM'

        sweet = None
        if ref_wall is not None:
            for r in K_results:
                if r['status'] == 'ok' and r['wall_ms'] / ref_wall < 1.5:
                    if sweet is None or r['K'] < sweet['K']:
                        sweet = r
            if sweet is not None:
                savings = ref_mem / sweet['peak_mem_GB']
                slowdown = sweet['wall_ms'] / ref_wall
                print(f'  → {base_label} | sweet K={sweet["K"]} '
                      f'mem={sweet["peak_mem_GB"]:.2f}GB '
                      f'savings={savings:.2f}x slowdown={slowdown:.2f}x', flush=True)
            else:
                print(f'  → {base_label} | no sweet K found', flush=True)
        else:
            print(f'  → all OOM', flush=True)

        results.append({
            'T':T,'B':B,'H':H,'W':W,
            'baseline_mem': mem_base, 'baseline_wall': wall_base,
            'baseline_status': status_base,
            'ref_K': ref_K, 'ref_mem': ref_mem, 'ref_wall': ref_wall,
            'sweet_K': sweet['K'] if sweet else None,
            'sweet_mem': sweet['peak_mem_GB'] if sweet else None,
            'sweet_wall': sweet['wall_ms'] if sweet else None,
            'K_results': K_results,
        })

        del x
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    np.savez('phase1_1d_sweep.npz', results=results)
    print(f'\n\n=== Summary === total elapsed: {time.perf_counter()-t_global_start:.0f}s')
    n_base_ok = sum(1 for r in results if r.get('baseline_status') == 'ok')
    n_base_oom = sum(1 for r in results if r.get('baseline_status') == 'OOM')
    n_saved = sum(1 for r in results
                  if r.get('baseline_status') == 'OOM' and r.get('sweet_K') is not None)
    n_skip = sum(1 for r in results if r.get('skip') == 'input_oom')
    print(f'Total configs: {len(results)}')
    print(f'Baseline OK:   {n_base_ok}')
    print(f'Baseline OOM:  {n_base_oom}  (chunked saved: {n_saved})')
    print(f'Input OOM:     {n_skip}')

    sweet_Ks = [r['sweet_K'] for r in results if r.get('sweet_K') is not None]
    if sweet_Ks:
        from collections import Counter
        c = Counter(sweet_Ks)
        print(f'Sweet K dist:  {dict(sorted(c.items()))}')


if __name__ == '__main__':
    main()
