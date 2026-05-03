"""
P0-6: Batch microbatching baseline vs AEROS temporal chunking.

Design:
  Effective batch = 32, H = 224
  
  Method 1 — Baseline: full [T, 32, ...] in one forward (often OOM)
  Method 2 — AEROS K=8: full b=32, temporal chunks of 8
  Method 3 — Batch micro b'=16: split into 2 sequential [T, 16, ...] forwards
  Method 4 — Batch micro b'=8: split into 4 sequential [T, 8, ...] forwards
  Method 5 — Batch micro b'=4: split into 8 sequential [T, 4, ...] forwards

For each T ∈ {64, 128, 256, 512}, compare memory + wall.

Expectation:
  - Batch micro saves memory but kills throughput (GPU underutil at small b)
  - AEROS preserves batch-level throughput while saving memory
"""
import time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def baseline_forward(net, x):
    """Full [T, b, ...] one forward."""
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_T_forward(net, x, K):
    """AEROS: temporal chunks."""
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def batch_microbatch_forward(net, x, b_micro):
    """
    Split batch axis into chunks of b_micro.
    Each microbatch: full T forward + reset_net before.
    """
    T, b = x.shape[0], x.shape[1]
    assert b % b_micro == 0, f'b={b} not div by b_micro={b_micro}'
    outs = []
    for i in range(0, b, b_micro):
        functional.reset_net(net)  # fresh state per microbatch
        x_micro = x[:, i:i + b_micro]
        outs.append(net(x_micro))
    return torch.cat(outs, dim=1)


def measure(fn, *args, n_warmup=5, n_iters=10):
    torch.cuda.synchronize()
    torch.cuda.empty_cache(); gc.collect()
    try:
        for _ in range(n_warmup):
            _ = fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        walls = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn(*args)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1e9
        med = float(np.median(walls))
        p25 = float(np.percentile(walls, 25))
        p75 = float(np.percentile(walls, 75))
        return peak, med, p25, p75, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, None, 'OOM'


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    b = 32
    H = 224

    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    Ts = [64, 128, 256, 512]

    # Method labels
    methods = [
        ('Baseline (full T, b=32)', None, None),
        ('AEROS K=8 (chunk T)', 'aeros', 8),
        ('B-micro b\'=16 (split b)', 'bmicro', 16),
        ('B-micro b\'=8 (split b)', 'bmicro', 8),
        ('B-micro b\'=4 (split b)', 'bmicro', 4),
    ]

    print(f'\n=== P0-6 Batch microbatching vs AEROS (b={b}, H={H}) ===')
    print(f'{"T":>4}  {"Method":<28s}  {"Mem GB":>8} {"Wall ms":>14} {"vs base mem":>12} {"vs base wall":>14}')
    print('-' * 100)

    rows = []
    for T in Ts:
        try:
            x = (torch.rand(T, b, 3, H, H, device=device) > 0.7).float()
        except torch.cuda.OutOfMemoryError:
            print(f'{T:>4}  input alloc OOM, skip')
            continue

        # Baseline first (for ratio reference)
        m_base, med_base, _, _, st_base = measure(baseline_forward, net, x)
        if st_base == 'ok':
            print(f'{T:>4}  {"Baseline (full T, b=32)":<28s}  {m_base:>7.2f}  '
                  f'{med_base:>9.0f} ms  {"1.00x":>12} {"1.00x":>14}')
        else:
            m_base, med_base = None, None
            print(f'{T:>4}  {"Baseline (full T, b=32)":<28s}  {"OOM":>8s}')

        # All other methods
        for label, kind, param in methods[1:]:
            if kind == 'aeros':
                m, med, p25, p75, st = measure(chunked_T_forward, net, x, param)
            elif kind == 'bmicro':
                m, med, p25, p75, st = measure(batch_microbatch_forward, net, x, param)
            else:
                continue

            if st == 'ok':
                if m_base is not None:
                    mem_ratio = f'{m_base/m:.2f}x save'
                    wall_ratio = f'{med/med_base:.2f}x'
                else:
                    mem_ratio = 'rescued'
                    wall_ratio = 'rescued'
                print(f'{T:>4}  {label:<28s}  {m:>7.2f}  {med:>9.0f} ms  '
                      f'{mem_ratio:>12} {wall_ratio:>14}')
            else:
                print(f'{T:>4}  {label:<28s}  {st:>8s}')

        del x
        torch.cuda.empty_cache(); gc.collect()

    # Final analysis: throughput comparison
    print(f'\n=== Throughput analysis ===')
    print(f'(rate = T*b / wall_ms; higher = better)')


if __name__ == '__main__':
    main()
