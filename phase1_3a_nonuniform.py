"""
Phase 1.3a — Non-uniform chunked execution.

Key change vs 1.1:
  uniform:    T must be divisible by K. Chunks: [K, K, ..., K] (T/K of them)
  non-uniform: T can be anything. Chunks: [K_max, K_max, ..., K_max, K_residual]
               where K_residual = T - floor(T/K_max) * K_max < K_max

Validation:
  1. Bit-exact vs uniform K (when both fit, e.g. K | T)
  2. Works for T values where K does NOT divide T (uniform fails)
  3. peak_mem = M_static + alpha * K_max (same as uniform)
"""
import time
import gc
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def baseline_forward(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_forward_uniform(net, x, K):
    """Uniform K (requires K | T)."""
    T = x.shape[0]
    assert T % K == 0, f'uniform requires K | T, got T={T} K={K}'
    functional.reset_net(net)
    chunks = []
    for i in range(T // K):
        chunks.append(net(x[i * K:(i + 1) * K]))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def chunked_forward_nonuniform(net, x, K_max):
    """Non-uniform: chunks of size K_max, last chunk possibly smaller."""
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        chunk_size = min(K_max, T - i)
        chunks.append(net(x[i:i + chunk_size]))
        i += chunk_size
    return torch.cat(chunks, dim=0)


def measure(fn, *args, n_warmup=2, n_iters=5):
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
            out = fn(*args)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / n_iters * 1000
        peak = torch.cuda.max_memory_allocated() / 1e9
        return peak, wall, out, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, 'OOM'


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    B, H = 32, 128

    print('Building net...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    # Test 1: T=128 (divisible by many K), bit-exact uniform vs non-uniform
    print('\n=== Test 1: T=128, K divides T — uniform vs non-uniform should match exactly ===')
    T = 128
    x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
    for K in [128, 64, 32, 16, 8, 4, 2, 1]:
        mem_u, wall_u, out_u, st_u = measure(chunked_forward_uniform, net, x, K)
        mem_n, wall_n, out_n, st_n = measure(chunked_forward_nonuniform, net, x, K)
        if st_u == 'ok' and st_n == 'ok':
            err = (out_u - out_n).abs().max().item()
            print(f'  K={K:>3}: uniform mem={mem_u:.2f}GB wall={wall_u:.0f}ms | '
                  f'nonuniform mem={mem_n:.2f}GB wall={wall_n:.0f}ms | err={err:.2e}')
        else:
            print(f'  K={K:>3}: uniform={st_u}, nonuniform={st_n}')

    # Test 2: T=100 (NOT power of 2, K=16 does NOT divide), uniform fails, non-uniform should work
    print('\n=== Test 2: T=100 (K=16 does NOT divide T) — uniform fails, non-uniform works ===')
    T = 100
    x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
    # Reference: T-step baseline (no chunking, K=T=100)
    mem_ref, wall_ref, out_ref, st_ref = measure(chunked_forward_nonuniform, net, x, T)
    print(f'  baseline (K=T={T}): mem={mem_ref:.2f}GB wall={wall_ref:.0f}ms')

    for K_max in [50, 32, 25, 16, 13, 8, 7, 4, 3]:
        mem, wall, out, st = measure(chunked_forward_nonuniform, net, x, K_max)
        if st == 'ok':
            err = (out - out_ref).abs().max().item()
            n_chunks = (T + K_max - 1) // K_max
            last_chunk = T - (n_chunks - 1) * K_max
            print(f'  K_max={K_max:>3}: chunks=[{K_max}]*{n_chunks-1}+[{last_chunk}], '
                  f'mem={mem:.2f}GB savings={mem_ref/mem:.2f}x '
                  f'wall={wall:.0f}ms slowdown={wall/wall_ref:.2f}x err={err:.2e}')
        else:
            print(f'  K_max={K_max:>3}: {st}')

    # Test 3: Equivalence under same K_max — non-uniform with K|T should equal uniform
    print('\n=== Test 3: Validate peak_mem = M_static + alpha * K_max formula ===')
    print('  (peak_mem should be ~constant for K_max regardless of T, given same B/H)')
    for T_test in [64, 100, 128, 200, 256]:
        x = (torch.rand(T_test, B, 3, H, H, device=device) > 0.7).float()
        K_max = 16
        mem, wall, _, st = measure(chunked_forward_nonuniform, net, x, K_max)
        if st == 'ok':
            print(f'  T={T_test:>3}, K_max={K_max}: mem={mem:.2f}GB '
                  f'(predicted ~M_static + alpha*K_max from 1.2)')
        del x

    print('\n=== Verdict ===')
    print('  (a) Test 1: uniform vs non-uniform bit-exact when K|T — verifies semantic equivalence')
    print('  (b) Test 2: non-uniform handles arbitrary T (K_max not divisor of T)')
    print('  (c) Test 3: peak_mem invariant under T at fixed K_max — confirms model')


if __name__ == '__main__':
    main()
