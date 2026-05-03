"""
Phase 1 prototype: chunked timestep execution.

Compare three modes on a minimal SNN (conv + LIF + conv + LIF):
  Mode "full"     : SJ multi-step (T at once)              ← OOM at large T/B
  Mode "chunked"  : process K timesteps at a time, save state across chunks
  Mode "single"   : K=1 (fully sequential)

Sweep K ∈ {1, 2, 4, 8, 16, 32, T} and report:
  - peak memory
  - wall-clock
  - output equivalence (must match Mode 'full' bit-for-bit)
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional


class MiniSNN(nn.Module):
    """Minimal SNN: conv→LIF→conv→LIF→pool→fc. Multi-step mode."""
    def __init__(self, channels=64, num_classes=10):
        super().__init__()
        self.conv1 = layer.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn1   = layer.BatchNorm2d(channels)
        self.lif1  = neuron.LIFNode(detach_reset=True,
                                    surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = layer.BatchNorm2d(channels)
        self.lif2  = neuron.LIFNode(detach_reset=True,
                                    surrogate_function=surrogate.ATan())
        self.pool  = layer.AdaptiveAvgPool2d(1)
        self.fc    = layer.Linear(channels, num_classes)

    def forward(self, x):
        # x: [T, B, C, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.pool(x)        # [T, B, C, 1, 1]
        x = torch.flatten(x, start_dim=2)
        return self.fc(x)       # [T, B, num_classes]


def run_full(net, x):
    """Mode 'full': SJ multi-step, all T at once."""
    functional.set_step_mode(net, step_mode='m')
    functional.reset_net(net)
    return net(x).mean(0)


def run_chunked(net, x, K):
    """Mode 'chunked': K timesteps at a time, propagate LIF state.
    Output: same shape as run_full output, [B, num_classes]."""
    functional.set_step_mode(net, step_mode='m')
    functional.reset_net(net)
    T = x.shape[0]
    out_accum = None
    n_steps_done = 0
    for chunk_start in range(0, T, K):
        chunk_end = min(chunk_start + K, T)
        x_chunk = x[chunk_start:chunk_end]
        # Forward only this chunk; LIF state persists across chunks
        # because we DON'T call reset_net between chunks
        y_chunk = net(x_chunk)        # [K, B, num_classes]
        if out_accum is None:
            out_accum = y_chunk.sum(0)
        else:
            out_accum = out_accum + y_chunk.sum(0)
        n_steps_done += y_chunk.shape[0]
        # Crucial: free intermediate tensors. PyTorch should auto-free
        # since y_chunk is not referenced beyond .sum(0)
    return out_accum / n_steps_done    # mean across timesteps


def measure(fn, *args, n_warmup=2, n_iters=3, device='cuda:0'):
    torch.cuda.empty_cache()
    # Warmup
    for _ in range(n_warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n_iters * 1000
    peak = torch.cuda.max_memory_allocated(device)
    return out, wall, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=128)
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--H', type=int, default=64)
    p.add_argument('--W', type=int, default=64)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)
    net = MiniSNN(channels=args.channels).to(device)
    net.eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)

    x = torch.randn(args.T, args.B, 3, args.H, args.W, device=device)

    print(f'MiniSNN: T={args.T} B={args.B} C={args.channels} H={args.H} W={args.W}')
    print(f'Total elements per timestep: {args.B * args.channels * args.H * args.W:_}\n')

    # Baseline: full multi-step
    print(f'{"mode":<12s}  {"K":>4s}  {"peak GB":>9s}  {"wall ms":>9s}  '
          f'{"vs full":>8s}  {"output match":>14s}')
    print('-' * 75)

    try:
        with torch.no_grad():
            ref_out, ref_wall, ref_peak = measure(run_full, net, x, device=str(device))
        print(f'{"full":<12s}  {args.T:>4d}  {ref_peak/1e9:>8.2f}  '
              f'{ref_wall:>8.2f}  {1.0:>7.2f}x  {"baseline":>14s}')
        full_failed = False
    except torch.cuda.OutOfMemoryError:
        print(f'{"full":<12s}  {args.T:>4d}  {"OOM":>9s}')
        ref_out, ref_wall, ref_peak = None, None, None
        full_failed = True
        torch.cuda.empty_cache()

    # Chunked sweep
    K_list = [1, 2, 4, 8, 16, 32]
    if args.T not in K_list:
        K_list.append(args.T)
    K_list = sorted(set(K_list))

    for K in K_list:
        if K > args.T:
            continue
        try:
            with torch.no_grad():
                out, wall, peak = measure(run_chunked, net, x, K, device=str(device))
            speedup = ref_wall / wall if ref_wall else float('nan')
            if ref_out is not None:
                err = (out - ref_out).abs().max().item()
                match = f'err={err:.2e}'
            else:
                match = 'no ref'
            print(f'{"chunked":<12s}  {K:>4d}  {peak/1e9:>8.2f}  '
                  f'{wall:>8.2f}  {speedup:>7.2f}x  {match:>14s}')
        except torch.cuda.OutOfMemoryError:
            print(f'{"chunked":<12s}  {K:>4d}  OOM')
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()