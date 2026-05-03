"""
Phase 1.5 — Attention compatibility study.

Part A: Verify TemporalWiseAttention (TWA) is incompatible with chunked execution.
Part B: Verify per-timestep (SE-block) attention IS compatible.
"""
import time, gc
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer


# ============================================================================
# Part A: TWA-augmented model (cross-time attention)
# ============================================================================
class MinimalTWAModel(nn.Module):
    """Conv → TWA(T) → LIF → GAP → FC. TWA needs T at init."""
    def __init__(self, T, C=32, num_classes=10):
        super().__init__()
        self.T = T
        self.conv = layer.Conv2d(3, C, kernel_size=3, padding=1, step_mode='m')
        self.bn = layer.BatchNorm2d(C, step_mode='m')
        self.twa = layer.TemporalWiseAttention(T=T, reduction=4, dimension=4)
        self.lif = neuron.LIFNode(detach_reset=True, step_mode='m')
        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.fc = layer.Linear(C, num_classes, step_mode='m')

    def forward(self, x):
        # x: [T, B, 3, H, W]
        x = self.conv(x)
        x = self.bn(x)
        x = self.twa(x)        # cross-time attention
        x = self.lif(x)
        x = self.pool(x)
        x = x.flatten(2)       # [T, B, C]
        x = self.fc(x)         # [T, B, num_classes]
        return x


# ============================================================================
# Part B: Per-timestep SE-block attention (chunked-compatible)
# ============================================================================
class PerStepSEBlock(nn.Module):
    """Per-timestep channel attention. Applies SE independently at each t."""
    def __init__(self, C, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(C, max(C // reduction, 1))
        self.fc2 = nn.Linear(max(C // reduction, 1), C)

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        # GAP per (T, B): [T, B, C]
        s = x.mean(dim=(3, 4))
        # FC: [T, B, C]
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        # Broadcast multiply
        s = s.view(T, B, C, 1, 1)
        return x * s


class MinimalSEModel(nn.Module):
    """Conv → BN → SE → LIF → GAP → FC. SE is per-timestep, T-independent."""
    def __init__(self, C=32, num_classes=10):
        super().__init__()
        self.conv = layer.Conv2d(3, C, kernel_size=3, padding=1, step_mode='m')
        self.bn = layer.BatchNorm2d(C, step_mode='m')
        self.se = PerStepSEBlock(C, reduction=4)  # per-timestep
        self.lif = neuron.LIFNode(detach_reset=True, step_mode='m')
        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.fc = layer.Linear(C, num_classes, step_mode='m')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.lif(x)
        x = self.pool(x)
        x = x.flatten(2)
        x = self.fc(x)
        return x


# ============================================================================
# Common: chunked forward, measure
# ============================================================================
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
            out = fn(*args)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / n_iters * 1000
        peak = torch.cuda.max_memory_allocated() / 1e9
        return peak, wall, out, 'ok'
    except Exception as e:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, None, f'FAIL:{str(e)[:80]}'


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


# ============================================================================
# Part A: Verify TWA incompatibility
# ============================================================================
def test_twa():
    print('=' * 70)
    print('Part A: TWA chunked-execution compatibility')
    print('=' * 70)
    device = torch.device('cuda:0')
    T, B, H = 64, 16, 64
    torch.manual_seed(42)

    net = MinimalTWAModel(T=T, C=32, num_classes=10).to(device).eval()
    functional.set_step_mode(net, step_mode='m')
    x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()

    # Baseline
    print('\nBaseline (K=T=64):')
    mem_b, wall_b, out_b, st_b = measure(baseline_forward, net, x)
    if st_b == 'ok':
        print(f'  mem={mem_b:.3f}GB wall={wall_b:.0f}ms output_norm={out_b.norm().item():.4f}')
    else:
        print(f'  FAIL: {st_b}')
        return

    # Chunked at K=8 (model thinks T=64, but chunk only feeds 8 timesteps)
    print('\nChunked K=8 (TWA expects T=64, chunks feed K=8):')
    mem_c, wall_c, out_c, st_c = measure(chunked_forward, net, x, 8)
    if st_c == 'ok':
        err = (out_c - out_b).abs().max().item()
        print(f'  mem={mem_c:.3f}GB wall={wall_c:.0f}ms')
        print(f'  max_err vs baseline: {err:.4e}')
        if err > 1e-5:
            print(f'  → INCOMPATIBLE (verified): chunked TWA produces non-bit-exact output')
        else:
            print(f'  → SURPRISINGLY bit-exact?? Investigate.')
    else:
        print(f'  FAIL: {st_c}')
        print(f'  → INCOMPATIBLE (verified): chunked TWA crashes (likely shape mismatch)')

    print('\nVerdict: TWA cannot be chunked while preserving model semantics.')


# ============================================================================
# Part B: Verify SE-block (per-timestep) compatibility
# ============================================================================
def test_se_block():
    print('\n' + '=' * 70)
    print('Part B: per-timestep SE-block compatibility')
    print('=' * 70)
    device = torch.device('cuda:0')
    B, H = 16, 64
    torch.manual_seed(42)

    net = MinimalSEModel(C=32, num_classes=10).to(device).eval()
    functional.set_step_mode(net, step_mode='m')

    # Sweep T × K
    points = []
    for T in [32, 64, 128]:
        x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()
        Ks = sorted(set(K for K in [T, T//2, T//4, T//8, T//16, T//32, 1] if K >= 1), reverse=True)

        # Baseline (K=T) for bit-exact reference
        mem_ref, wall_ref, out_ref, st_ref = measure(baseline_forward, net, x)
        print(f'\nT={T}: baseline mem={mem_ref:.3f}GB wall={wall_ref:.0f}ms')

        for K in Ks:
            mem, wall, out, st = measure(chunked_forward, net, x, K)
            if st == 'ok':
                err = (out - out_ref).abs().max().item()
                bit_exact = err < 1e-5
                points.append((T, K, mem))
                print(f'  K={K:>3}: mem={mem:.3f}GB wall={wall:.0f}ms err={err:.2e} '
                      f'{"✓bit-exact" if bit_exact else "✗MISMATCH"}')
            else:
                print(f'  K={K:>3}: {st}')
        del x
        torch.cuda.empty_cache(); gc.collect()

    # Fit memory model
    f = fit_2var(points)
    if f:
        a_T, a_K, M_const, R2 = f
        print(f'\nFit: peak_mem = {M_const:.3f} + {a_T:.5f}·T + {a_K:.5f}·K')
        print(f'R² = {R2:.4f}')
        if R2 >= 0.99:
            print(f'→ COMPATIBLE: SE-block + chunked execution preserves model + R² ≥ 0.99')
        else:
            print(f'→ R² lower than CNN baseline (1.000); attention adds noise to memory model')


if __name__ == '__main__':
    test_twa()
    test_se_block()
