"""
P1: Streaming-input AEROS variant.

Mechanism: instead of materializing full [T, b, C, H, W] input tensor in
GPU memory, accept a generator that yields one [κ, b, C, H, W] segment
at a time. Per Eq. 6:

    M_peak^stream(κ) = M_0 + (α_in + α_K) · κ      (no T dependence)

Verifies:
  1. bit-exact output vs full-input AEROS (same seed → same x_seg sequence)
  2. mem(T) at fixed κ approximately constant across T (Eq. 6)
  3. extends feasible T further than full-input AEROS (T=1024+)
"""
import time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


# -----------------------------------------------------------------------------
# Standard full-input chunked (baseline for comparison)
# -----------------------------------------------------------------------------
@torch.no_grad()
def full_input_chunked(net, x, K):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


# -----------------------------------------------------------------------------
# Streaming-input AEROS: generator-based, no full-T materialization
# -----------------------------------------------------------------------------
def stream_input_generator(T, b, C, H, W, K, device, seed=42):
    """
    Yields [K_seg, b, C, H, W] tensors on demand.
    Uses fixed seed so we can compare bit-exact against pre-materialized x.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    i = 0
    while i < T:
        sz = min(K, T - i)
        # Generate this segment fresh on GPU (simulating decode-and-feed)
        # Use same seed-state continuity as if generated all at once
        x_seg = (torch.rand(sz, b, C, H, H, generator=g, device=device) > 0.7).float()
        yield x_seg
        i += sz


@torch.no_grad()
def stream_chunked(net, stream_gen, return_output=True):
    """
    Run AEROS forward over a generator. State persists across segments.
    """
    functional.reset_net(net)
    outs = []
    for x_seg in stream_gen:
        y_seg = net(x_seg)
        if return_output:
            outs.append(y_seg)
        # x_seg is freed after this iteration (out of scope)
    if return_output and outs:
        return torch.cat(outs, dim=0)
    return None


def materialize_full_input(T, b, C, H, W, K, device, seed=42):
    """Generate the SAME tensor sequence as the streamer, but all-at-once.
    Used for bit-exact reference."""
    g = torch.Generator(device=device).manual_seed(seed)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        x_seg = (torch.rand(sz, b, C, H, H, generator=g, device=device) > 0.7).float()
        chunks.append(x_seg)
        i += sz
    return torch.cat(chunks, dim=0)


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------
def measure_full_input(net, T, b, C, H, K, device, seed=42, n_warmup=2, n_iters=5):
    """Measure full-input chunked: x materialized first, then chunked forward."""
    torch.cuda.empty_cache(); gc.collect()
    try:
        x = materialize_full_input(T, b, C, H, H, K, device, seed=seed)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, 'input alloc OOM'

    try:
        for _ in range(n_warmup):
            _ = full_input_chunked(net, x, K)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        walls = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = full_input_chunked(net, x, K)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1e9
        wall_med = float(np.median(walls))
        del x
        torch.cuda.empty_cache(); gc.collect()
        return peak, wall_med, out, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, 'forward OOM'


def measure_stream(net, T, b, C, H, K, device, seed=42, n_warmup=2, n_iters=5):
    """Measure streaming-input chunked: generator yields segments on demand."""
    torch.cuda.empty_cache(); gc.collect()
    try:
        for _ in range(n_warmup):
            gen = stream_input_generator(T, b, C, H, H, K, device, seed=seed)
            _ = stream_chunked(net, gen)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        walls = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = stream_input_generator(T, b, C, H, H, K, device, seed=seed)
            out = stream_chunked(net, gen)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1e9
        wall_med = float(np.median(walls))
        return peak, wall_med, out, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, 'OOM'


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    b = 32
    H = 224
    K = 8

    # ------------------------------------------------------------------
    # Test 1: Bit-exact equivalence at moderate T
    # ------------------------------------------------------------------
    print('\n=== Test 1: bit-exact full-input vs streaming (T=64) ===')
    T = 64
    peak_f, wall_f, out_f, st_f = measure_full_input(net, T, b, 3, H, K, device, n_warmup=1, n_iters=2)
    peak_s, wall_s, out_s, st_s = measure_stream(net, T, b, 3, H, K, device, n_warmup=1, n_iters=2)
    if st_f == 'ok' and st_s == 'ok':
        err = (out_f - out_s).abs().max().item()
        print(f'  full-input:  mem={peak_f:.3f}GB wall={wall_f:.0f}ms')
        print(f'  streaming:   mem={peak_s:.3f}GB wall={wall_s:.0f}ms')
        print(f'  output max_err: {err:.4e}  '
              f'{"✓ bit-exact" if err < 1e-5 else "✗ MISMATCH"}')
    else:
        print(f'  full-input: {st_f}, streaming: {st_s}')

    # ------------------------------------------------------------------
    # Test 2: Memory invariance under T (streaming should be flat)
    # ------------------------------------------------------------------
    print(f'\n=== Test 2: mem(T) at fixed κ=8, streaming vs full-input ===')
    print(f'{"T":>5}  {"full-input mem":>15} {"stream mem":>12} '
          f'{"full-input wall":>16} {"stream wall":>13}')
    print('-' * 75)
    rows = []
    for T in [64, 128, 256, 512, 1024, 2048]:
        peak_f, wall_f, _, st_f = measure_full_input(net, T, b, 3, H, K, device, n_warmup=1, n_iters=2)
        peak_s, wall_s, _, st_s = measure_stream(net, T, b, 3, H, K, device, n_warmup=1, n_iters=2)
        f_str = f'{peak_f:.2f}GB' if st_f == 'ok' else st_f
        s_str = f'{peak_s:.2f}GB' if st_s == 'ok' else st_s
        wf_str = f'{wall_f:.0f}ms' if st_f == 'ok' else '—'
        ws_str = f'{wall_s:.0f}ms' if st_s == 'ok' else '—'
        print(f'{T:>5}  {f_str:>15} {s_str:>12} {wf_str:>16} {ws_str:>13}')
        rows.append({'T': T, 'full': (peak_f, wall_f, st_f),
                     'stream': (peak_s, wall_s, st_s)})

    # ------------------------------------------------------------------
    # Test 3: Verify streaming mem ≈ M_0 + (α_in + α_K)·κ (Eq. 6)
    # ------------------------------------------------------------------
    print(f'\n=== Test 3: streaming mem vs Eq. 6 prediction ===')
    # Theoretical: M_0 + (α_in + α_K)·κ
    # From Phase 1: at b=32 H=224, M_0 ≈ 0.66, α_K ≈ 0.204, α_T (per-step input cost) ≈ 0.0196
    # Stream mode: M_0 + (α_T_inputonly + α_K) · κ ≈ 0.66 + (0.0196 + 0.204) × 8 = 0.66 + 1.79 = 2.45 GB
    # We expect streaming mem to be near 2.45 GB regardless of T
    M_0 = 0.656  # from §5.2 fit
    alpha_T_per_step = 0.01957  # per-step input cost coefficient
    alpha_K = 0.20421
    K = 8
    pred_stream_mem = M_0 + (alpha_T_per_step + alpha_K) * K
    print(f'  Theoretical Eq. 6 prediction (κ=8): M_0 + (α_in + α_K)·κ')
    print(f'                                      = 0.656 + (0.01957 + 0.20421) × 8')
    print(f'                                      = {pred_stream_mem:.2f} GB (T-independent)')
    print(f'  Measured streaming mem across T:')
    for r in rows:
        if r['stream'][2] == 'ok':
            T = r['T']
            mem = r['stream'][0]
            err = abs(mem - pred_stream_mem) / pred_stream_mem * 100
            print(f'    T={T:>4}: {mem:.2f} GB  (vs pred {pred_stream_mem:.2f}, err {err:.1f}%)')

    # ------------------------------------------------------------------
    # Test 4: How far can streaming push T?
    # ------------------------------------------------------------------
    print(f'\n=== Test 4: extreme T with streaming (κ=8) ===')
    for T in [4096, 8192]:
        peak_s, wall_s, _, st_s = measure_stream(net, T, b, 3, H, K, device, n_warmup=1, n_iters=1)
        if st_s == 'ok':
            print(f'  T={T:>5}: mem={peak_s:.2f}GB wall={wall_s:.0f}ms ✓')
        else:
            print(f'  T={T:>5}: {st_s}')


if __name__ == '__main__':
    main()
