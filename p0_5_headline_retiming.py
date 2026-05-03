"""
P0-5: Re-time headline configurations with 10 warmup + 30 iters,
report median + IQR (5th, 95th percentile).

Headline configs from §5.3:
  ResNet18-SNN, b=32, 224x224, T ∈ {64, 128, 256, 512}, K ∈ {T (baseline), 8}
"""
import time, gc
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
    i = 0
    while i < T:
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


def measure_robust(fn, *args, n_warmup=10, n_iters=30):
    """Returns (peak_GB, median_ms, iqr_lo_ms, iqr_hi_ms, all_walls, status)."""
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
        walls = np.array(walls)
        med = float(np.median(walls))
        p25 = float(np.percentile(walls, 25))
        p75 = float(np.percentile(walls, 75))
        return peak, med, p25, p75, walls, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, None, None, 'OOM'


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

    print(f'\n=== Headline retiming (b={b}, H={H}, 10 warmup + 30 iters) ===')
    print(f'{"T":>4}  {"baseline (med [IQR])":>30s}  {"K=8 (med [IQR])":>30s}  '
          f'{"savings":>9}  {"slowdown":>16}')
    print('-' * 110)

    rows = []
    for T in Ts:
        try:
            x = (torch.rand(T, b, 3, H, H, device=device) > 0.7).float()
        except torch.cuda.OutOfMemoryError:
            print(f'{T:>4}  input alloc OOM')
            continue

        # Baseline
        m_b, med_b, p25_b, p75_b, w_b, st_b = measure_robust(baseline_forward, net, x)
        # K=8
        m_k, med_k, p25_k, p75_k, w_k, st_k = measure_robust(chunked_forward, net, x, 8)

        if st_b == 'ok':
            base_str = f'{m_b:.2f}GB / {med_b:.0f}[{p25_b:.0f}–{p75_b:.0f}]ms'
        else:
            base_str = 'OOM'
        if st_k == 'ok':
            k_str = f'{m_k:.2f}GB / {med_k:.0f}[{p25_k:.0f}–{p75_k:.0f}]ms'
            if st_b == 'ok':
                savings = m_b / m_k
                slowdown_med = med_k / med_b
                # Slowdown distribution: ratio of medians, plus 95% CI from bootstrap
                ratios = w_k.reshape(-1, 1) / w_b.reshape(1, -1)
                ratio_med = float(np.median(ratios.flatten()))
                ratio_p25 = float(np.percentile(ratios.flatten(), 25))
                ratio_p75 = float(np.percentile(ratios.flatten(), 75))
                sv_str = f'{savings:.2f}x'
                sd_str = f'{ratio_med:.3f}x[{ratio_p25:.3f}–{ratio_p75:.3f}]'
            else:
                sv_str = 'rescued'
                sd_str = 'rescued'
        else:
            k_str = 'OOM'
            sv_str = '—'
            sd_str = '—'

        print(f'{T:>4}  {base_str:>30s}  {k_str:>30s}  {sv_str:>9}  {sd_str:>16}')
        rows.append({
            'T': T,
            'baseline': (m_b, med_b, p25_b, p75_b, w_b),
            'K8': (m_k, med_k, p25_k, p75_k, w_k),
        })
        del x
        torch.cuda.empty_cache(); gc.collect()

    # Save raw walls for paper figure
    np.savez('p0_5_headline_retiming.npz', rows=rows, allow_pickle=True)
    print(f'\nSaved to p0_5_headline_retiming.npz')

    # Summary stats for paper Section 5.3
    print(f'\n=== Paper Section 5.3 update ===')
    for r in rows:
        T = r['T']
        m_b, med_b, p25_b, p75_b, w_b = r['baseline']
        m_k, med_k, p25_k, p75_k, w_k = r['K8']
        if w_b is not None and w_k is not None:
            sd_arr = w_k[:, None] / w_b[None, :]
            sd_med = float(np.median(sd_arr))
            sd_lo = float(np.percentile(sd_arr, 25))
            sd_hi = float(np.percentile(sd_arr, 75))
            print(f'  T={T}: slowdown median {sd_med:.3f}x, IQR [{sd_lo:.3f}, {sd_hi:.3f}]')


if __name__ == '__main__':
    main()
