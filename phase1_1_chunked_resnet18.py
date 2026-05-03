"""
Phase 1.1 — M1 chunked execution on production-scale ResNet18-SNN.

Goal: verify M1 wins (memory reduction, low slowdown, bit-exact output)
hold on real network, not just toy MiniSNN.

Strategy:
  - Build SpikingResNet18 (SJ stable 0.0.0.0.14 has it)
  - Run baseline forward (T=64 or 128, full-T at once)
  - Run chunked forward (split T into K chunks; reset state at start, propagate
    state across chunks, free activation per chunk)
  - Compare:
    * peak GPU memory (torch.cuda.max_memory_allocated)
    * wall-clock (perf_counter, after warmup)
    * output equality (bit-exact for fp32 expected)
"""
import time
import gc
import torch
import torch.nn as nn
from spikingjelly.activation_based import (
    neuron, surrogate, functional, layer
)
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


# -----------------------------------------------------------------------------
# Chunked forward: feed (T_full, B, C, H, W); split T_full into K chunks,
# reset state once at start; for each chunk run model with internal LIF state
# carried over (SJ's reset_net only at start, NOT between chunks).
# -----------------------------------------------------------------------------
@torch.no_grad()
def baseline_forward(net, x_TBCHW):
    """Standard SJ multi-step forward: feed full T at once."""
    functional.reset_net(net)
    out = net(x_TBCHW)  # shape: [T, B, num_classes]
    return out


@torch.no_grad()
def chunked_forward(net, x_TBCHW, K):
    """
    Split T axis into chunks of K, run sequentially, carry LIF state.
    Critical: only reset_net ONCE at start. State is carried in module attrs.
    """
    T = x_TBCHW.shape[0]
    assert T % K == 0, f"T={T} must be divisible by K={K}"
    n_chunks = T // K
    functional.reset_net(net)
    chunks_out = []
    for i in range(n_chunks):
        x_chunk = x_TBCHW[i * K : (i + 1) * K]  # [K, B, C, H, W]
        out_chunk = net(x_chunk)                # [K, B, num_classes]
        chunks_out.append(out_chunk)
        # Free activation tensors NOT needed for next chunk
        # (LIF state is in module.v, persists across chunks automatically)
    return torch.cat(chunks_out, dim=0)         # [T, B, num_classes]


def measure(fn, *args, n_warmup=2, n_iters=5):
    """Returns (peak_mem_GB, wall_ms, output_tensor)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Warmup
    for _ in range(n_warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / n_iters * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    return peak_mem, wall_ms, out


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    # Config
    B = 8       # batch (try smaller if OOM)
    T = 64      # timesteps
    C, H, W = 3, 64, 64  # input shape (use 64x64 to save memory; later try 224)
    Ks = [64, 32, 16, 8, 4, 2, 1]   # K=64 means baseline (1 chunk = full T)

    # Build SpikingResNet18 with single-step inference mode (m mode)
    print(f'Building SpikingResNet18 (T={T}, B={B}, input={C}x{H}x{W})...')
    net = spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    ).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    # Don't set cupy backend yet — Phase 1.1 is correctness + memory check
    # Add cupy backend in Phase 1.4 after baseline numbers are clear
    print(f'  Total params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M')

    # Generate input: spike-like binary {0,1}, fp32
    x = (torch.rand(T, B, C, H, W, device=device) > 0.7).float()

    # Reference output (full T at once, K=T, no chunking)
    print(f'\nMeasuring baseline (K=T={T}, no chunking)...')
    try:
        mem_ref, wall_ref, out_ref = measure(baseline_forward, net, x)
        print(f'  baseline: peak_mem={mem_ref:.3f} GB, wall={wall_ref:.2f} ms')
    except torch.cuda.OutOfMemoryError as e:
        print(f'  baseline OOM at B={B}, T={T}, input={C}x{H}x{W}')
        print(f'  This is *expected* for large config — confirms memory bottleneck.')
        print(f'  Try smaller B or T, or skip baseline and only measure chunked.')
        return

    # Chunked: K = T means 1 chunk = baseline (sanity), K=1 is fully sequential
    print(f'\nMeasuring chunked execution:')
    print(f'{"K":>4s}  {"peak_mem_GB":>12s}  {"mem_reduce":>11s}  '
          f'{"wall_ms":>9s}  {"slowdown":>9s}  {"max_err":>10s}')
    print('-' * 75)

    for K in Ks:
        if T % K != 0:
            continue
        try:
            mem_K, wall_K, out_K = measure(chunked_forward, net, x, K)
            err = (out_K - out_ref).abs().max().item()
            print(f'{K:>4d}  {mem_K:>11.3f}   {mem_ref/mem_K:>9.2f}x  '
                  f'{wall_K:>8.2f}   {wall_K/wall_ref:>8.2f}x  {err:>10.2e}')
        except torch.cuda.OutOfMemoryError:
            print(f'{K:>4d}  OOM')
        except Exception as e:
            print(f'{K:>4d}  FAIL: {str(e)[:80]}')

    print(f'\n=== Verdict ===')
    print(f'M1 PASS conditions:')
    print(f'  (a) some K achieves >= 5x memory reduction')
    print(f'  (b) wall-clock slowdown <= 30% at the same K')
    print(f'  (c) max output error < 1e-5 (bit-exact for fp32 expected)')
    print(f'M1 FAIL: any of above broken on ResNet18 (vs MiniSNN GREEN).')


if __name__ == '__main__':
    main()