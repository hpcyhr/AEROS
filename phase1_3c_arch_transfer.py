"""
Phase 1.3c — Cross-architecture transfer of 2-variable memory model.

Architectures: spiking_resnet18, sew_resnet18, spiking_vgg11_bn
Hypothesis:
  α_T ≈ 1.04 × input_BCHW (architecture-agnostic — input tensor cost)
  α_K depends on architecture (activation expansion factor)
"""
import time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


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


def measure(fn, *args, n_warmup=1, n_iters=3):
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
        return torch.cuda.max_memory_allocated()/1e9, (time.perf_counter()-t0)/n_iters*1000, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, 'OOM'


def build_net(arch_name):
    """Build with consistent kwargs across architectures."""
    common = {
        'spiking_neuron': neuron.LIFNode,
        'surrogate_function': surrogate.ATan(),
        'detach_reset': True,
    }
    if arch_name == 'spiking_resnet18':
        return spiking_resnet18(**common)
    elif arch_name == 'sew_resnet18':
        # SEW needs cnf parameter ('ADD' is standard)
        return sew_resnet18(cnf='ADD', **common)
    elif arch_name == 'spiking_vgg11_bn':
        return spiking_vgg11_bn(**common)
    else:
        raise ValueError(arch_name)


def fit_2var(points):
    """Fit peak_mem = M_const + a_T*T + a_K*K from list of (T, K, mem)."""
    if len(points) < 3:
        return None
    Ts = np.array([t for t,_,_ in points], dtype=np.float64)
    Ks = np.array([k for _,k,_ in points], dtype=np.float64)
    mems = np.array([m for _,_,m in points], dtype=np.float64)
    A = np.stack([Ts, Ks, np.ones_like(Ts)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, mems, rcond=None)
    a_T, a_K, M_const = float(coef[0]), float(coef[1]), float(coef[2])
    pred = a_T * Ts + a_K * Ks + M_const
    ss_res = ((mems - pred)**2).sum()
    ss_tot = ((mems - mems.mean())**2).sum()
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0
    return a_T, a_K, M_const, R2


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    B = 32
    H = 128
    Ts = [64, 128]
    K_grid = lambda T: [T, T//2, T//4, T//8, T//16, T//32, 1]   # 7 K values per T

    archs = ['spiking_resnet18', 'sew_resnet18', 'spiking_vgg11_bn']
    all_data = {}

    for arch in archs:
        print(f'\n=== {arch} (B={B}, H={H}) ===')
        try:
            net = build_net(arch).to(device)
            net.eval()
            functional.set_step_mode(net, step_mode='m')
            n_params = sum(p.numel() for p in net.parameters()) / 1e6
            print(f'  params: {n_params:.2f}M')
        except Exception as e:
            print(f'  build failed: {e}')
            continue

        points = []
        for T in Ts:
            x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
            Ks = sorted(set(K for K in K_grid(T) if K >= 1), reverse=True)
            for K in Ks:
                mem, wall, st = measure(chunked_forward, net, x, K)
                if st == 'ok':
                    points.append((T, K, mem))
                    print(f'    T={T:>3} K={K:>3}: mem={mem:.3f}GB wall={wall:.0f}ms')
                else:
                    print(f'    T={T:>3} K={K:>3}: {st}')
            del x
            torch.cuda.empty_cache(); gc.collect()
        all_data[arch] = points
        del net
        torch.cuda.empty_cache(); gc.collect()

    # Fit each architecture
    print(f'\n\n=== 2-var model fit per architecture ===')
    print(f'  peak_mem = M_const + α_T * T + α_K * K')
    print()
    print(f'{"Arch":<22}  {"α_T":>10} {"α_K":>10} {"M_const":>9} {"R²":>6}')
    print('-' * 65)
    
    input_cost = B * 3 * H * H * 4 / 1e9  # GB per timestep
    print(f'  (input tensor theoretical cost: {input_cost:.5f} GB/step)\n')
    
    fits = {}
    for arch, pts in all_data.items():
        f = fit_2var(pts)
        if f is None:
            print(f'{arch:<22}  insufficient data')
            continue
        a_T, a_K, M_const, R2 = f
        fits[arch] = f
        print(f'{arch:<22}  {a_T:>9.5f}  {a_K:>9.5f}  {M_const:>8.3f}  {R2:>5.3f}')
        print(f'    α_T / input_cost = {a_T/input_cost:.2f}x  '
              f'α_K / input_cost = {a_K/input_cost:.1f}x')

    # Compare across architectures
    print(f'\n=== Cross-architecture comparison ===')
    if 'spiking_resnet18' in fits:
        a_T_ref, a_K_ref, _, _ = fits['spiking_resnet18']
        print(f'Relative to spiking_resnet18:')
        print(f'{"Arch":<22}  {"α_T_rel":>9} {"α_K_rel":>9}')
        for arch, (a_T, a_K, _, _) in fits.items():
            if arch == 'spiking_resnet18':
                print(f'{arch:<22}  {"1.00x":>9} {"1.00x":>9} (ref)')
            else:
                print(f'{arch:<22}  {a_T/a_T_ref:>8.2f}x {a_K/a_K_ref:>8.2f}x')

    print(f'\n=== Verdict ===')
    print(f'Hypothesis 1 (α_T architecture-agnostic ≈ 1.04 × input_cost):')
    if fits:
        ratios = [f[0]/input_cost for f in fits.values()]
        print(f'  Measured α_T / input_cost: {[f"{r:.2f}" for r in ratios]}')
        if max(ratios) - min(ratios) < 0.20:
            print(f'  → CONFIRMED: α_T tightly clustered, transferable')
        else:
            print(f'  → REJECTED: α_T varies by {max(ratios)-min(ratios):.2f}x across archs')
    print(f'Hypothesis 2 (α_K is architecture signature):')
    if fits:
        a_K_vals = [f[1] for f in fits.values()]
        print(f'  Measured α_K values: {[f"{v:.4f}" for v in a_K_vals]}')
        if max(a_K_vals) / min(a_K_vals) > 1.5:
            print(f'  → CONFIRMED: α_K varies {max(a_K_vals)/min(a_K_vals):.2f}x — architecture-specific')
        else:
            print(f'  → INCONCLUSIVE: α_K too similar across these archs')


if __name__ == '__main__':
    main()
