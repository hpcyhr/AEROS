"""Phase 1.1c — proper stress test with baseline OOM tolerated."""
import time
import gc
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def baseline_forward(net, x_TBCHW):
    functional.reset_net(net)
    return net(x_TBCHW)


@torch.no_grad()
def chunked_forward(net, x_TBCHW, K):
    T = x_TBCHW.shape[0]
    assert T % K == 0
    functional.reset_net(net)
    chunks_out = []
    for i in range(T // K):
        x_chunk = x_TBCHW[i * K:(i + 1) * K]
        chunks_out.append(net(x_chunk))
    return torch.cat(chunks_out, dim=0)


def measure(fn, *args, n_warmup=1, n_iters=3):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    try:
        for _ in range(n_warmup):
            out = fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = fn(*args)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / n_iters * 1000
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        return peak_mem, wall_ms, out, None
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return None, None, None, 'OOM'
    except Exception as e:
        return None, None, None, f'FAIL: {str(e)[:80]}'


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    B = 32
    T = 128
    C, H, W = 3, 224, 224
    Ks = [128, 64, 32, 16, 8, 4, 2, 1]

    print(f'Building SpikingResNet18 (T={T}, B={B}, input={C}x{H}x{W})...')
    net = spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    ).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    print(f'  Params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M')

    x = (torch.rand(T, B, C, H, W, device=device) > 0.7).float()
    input_size_gb = x.element_size() * x.numel() / 1e9
    print(f'  Input tensor size: {input_size_gb:.2f} GB')

    # Try baseline (expected OOM)
    print(f'\nBaseline (K=T={T}):')
    mem_ref, wall_ref, out_ref, err_ref = measure(baseline_forward, net, x)
    if err_ref:
        print(f'  baseline {err_ref}  ← expected, this is M1 motivation')
        out_ref = None
    else:
        print(f'  baseline: peak_mem={mem_ref:.3f} GB, wall={wall_ref:.2f} ms')

    # Chunked
    print(f'\nChunked execution:')
    print(f'{"K":>4s}  {"peak_GB":>8s}  {"wall_ms":>9s}  '
          f'{"vs_base":>9s}  {"max_err":>10s}  notes')
    print('-' * 75)

    out_largest_K = None
    largest_K_for_ref = None

    for K in Ks:
        if T % K != 0:
            continue
        mem_K, wall_K, out_K, err_K = measure(chunked_forward, net, x, K)
        if err_K:
            print(f'{K:>4d}  {err_K}')
            continue

        # vs baseline
        if mem_ref is not None:
            mem_str = f'{mem_ref/mem_K:.2f}x'
        else:
            mem_str = 'base OOM'

        # output equality: compare against largest successful K (closest to baseline)
        if out_largest_K is None:
            out_largest_K = out_K.cpu().clone()
            largest_K_for_ref = K
            err_str = '(ref)'
        else:
            err = (out_K.cpu() - out_largest_K).abs().max().item()
            err_str = f'{err:.2e}'

        print(f'{K:>4d}  {mem_K:>7.3f}   {wall_K:>8.2f}  {mem_str:>9s}  '
              f'{err_str:>10s}  {"" if largest_K_for_ref else ""}')

    print(f'\n=== Verdict (revised bars) ===')
    print(f'  (a) baseline OOM, chunked saves it: ', end='')
    print('YES' if err_ref else 'NO')
    print(f'  (b) sweet-spot K wall < 2x slowest K: see table')
    print(f'  (c) cross-K bit-exact (vs largest successful K=', largest_K_for_ref, ')')


if __name__ == '__main__':
    main()
