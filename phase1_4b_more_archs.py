"""
Phase 1.4b — Extend 1.3c to more architectures.

Goal: validate alpha_T architecture-stable / alpha_K architecture-signature
hypothesis on a broader set:
  - spiking_resnet50 (deeper ResNet)
  - spiking_vgg19_bn (deeper VGG)
  - sew_resnet50 (deep SEW)
"""
import time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import (
    spiking_resnet18, spiking_resnet50)
from spikingjelly.activation_based.model.spiking_vgg import (
    spiking_vgg11_bn, spiking_vgg19_bn)
from spikingjelly.activation_based.model.sew_resnet import (
    sew_resnet18, sew_resnet50)


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


def build_net(name):
    common = {
        'spiking_neuron': neuron.LIFNode,
        'surrogate_function': surrogate.ATan(),
        'detach_reset': True,
    }
    if name == 'spiking_resnet18':  return spiking_resnet18(**common)
    if name == 'spiking_resnet50':  return spiking_resnet50(**common)
    if name == 'sew_resnet18':      return sew_resnet18(cnf='ADD', **common)
    if name == 'sew_resnet50':      return sew_resnet50(cnf='ADD', **common)
    if name == 'spiking_vgg11_bn':  return spiking_vgg11_bn(**common)
    if name == 'spiking_vgg19_bn':  return spiking_vgg19_bn(**common)
    raise ValueError(name)


def fit_2var(points):
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
    K_grid = lambda T: sorted(set(K for K in [T, T//2, T//4, T//8, T//16, T//32, 1] if K >= 1), reverse=True)

    archs = [
        'spiking_resnet18',  # ref baseline (already known)
        'spiking_resnet50',
        'sew_resnet18',
        'sew_resnet50',
        'spiking_vgg11_bn',
        'spiking_vgg19_bn',
    ]

    all_data = {}
    print(f'Sweep: B={B} H={H}, Ts={Ts}, K from divisors\n')

    for arch in archs:
        print(f'=== {arch} ===', flush=True)
        try:
            net = build_net(arch).to(device)
            net.eval()
            functional.set_step_mode(net, step_mode='m')
            n_params = sum(p.numel() for p in net.parameters()) / 1e6
            print(f'  params: {n_params:.2f}M', flush=True)
        except Exception as e:
            print(f'  build failed: {str(e)[:100]}', flush=True)
            continue

        points = []
        for T in Ts:
            try:
                x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
            except torch.cuda.OutOfMemoryError:
                print(f'  T={T} input alloc OOM', flush=True)
                continue
            for K in K_grid(T):
                mem, wall, st = measure(chunked_forward, net, x, K)
                if st == 'ok':
                    points.append((T, K, mem))
                    print(f'    T={T:>3} K={K:>3}: mem={mem:.3f}GB wall={wall:.0f}ms',
                          flush=True)
                else:
                    print(f'    T={T:>3} K={K:>3}: {st}', flush=True)
            del x
            torch.cuda.empty_cache(); gc.collect()
        all_data[arch] = (n_params, points)
        del net
        torch.cuda.empty_cache(); gc.collect()

    # Fit + report
    print(f'\n\n=== 2-var model fit summary ===')
    print(f'  peak_mem = M_const + α_T·T + α_K·K')
    print(f'  input cost theoretical: {B*3*H*H*4/1e9:.5f} GB/step\n')
    print(f'{"Arch":<22} {"params":>9} {"α_T":>10} {"α_K":>10} {"M_const":>9} {"R²":>6}')
    print('-' * 75)
    fits = {}
    input_cost = B * 3 * H * H * 4 / 1e9
    for arch, (n_params, pts) in all_data.items():
        f = fit_2var(pts)
        if f is None:
            print(f'{arch:<22} insufficient data ({len(pts)} points)')
            continue
        a_T, a_K, M_const, R2 = f
        fits[arch] = f
        print(f'{arch:<22} {n_params:>7.1f}M  {a_T:>9.5f}  {a_K:>9.5f}  '
              f'{M_const:>8.3f}  {R2:>5.3f}')
        print(f'    α_T/input = {a_T/input_cost:.2f}x   '
              f'α_K/input = {a_K/input_cost:.1f}x')

    # Hypothesis verdict
    print(f'\n=== Hypothesis verdict ===')
    if fits:
        a_T_ratios = [f[0]/input_cost for f in fits.values()]
        a_K_ratios = [f[1]/input_cost for f in fits.values()]
        print(f'α_T/input range: {min(a_T_ratios):.2f}x – {max(a_T_ratios):.2f}x  '
              f'(spread {max(a_T_ratios)/min(a_T_ratios):.2f}x)')
        print(f'α_K/input range: {min(a_K_ratios):.1f}x – {max(a_K_ratios):.1f}x   '
              f'(spread {max(a_K_ratios)/min(a_K_ratios):.2f}x)')

        # Group by family
        families = {
            'spiking_resnet': ['spiking_resnet18', 'spiking_resnet50'],
            'sew_resnet':     ['sew_resnet18', 'sew_resnet50'],
            'spiking_vgg':    ['spiking_vgg11_bn', 'spiking_vgg19_bn'],
        }
        print(f'\n=== Within-family α_K variance ===')
        for fam, members in families.items():
            vals = [fits[m][1] for m in members if m in fits]
            if len(vals) >= 2:
                print(f'  {fam}: α_K = {[f"{v:.4f}" for v in vals]}  '
                      f'(spread {max(vals)/min(vals):.2f}x)')

    np.savez('phase1_4b_archs.npz', data=all_data)
    print(f'\nSaved to phase1_4b_archs.npz')


if __name__ == '__main__':
    main()
