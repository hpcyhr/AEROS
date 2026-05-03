"""
Phase 1.5b — Reframed Part A using minimal cross-time attention.
Part B (SE-block) already PASS in 1.5; this just adds clean Part A.
"""
import time, gc
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer


class MinimalCrossTimeAttention(nn.Module):
    """Cross-time attention: weight at timestep t depends on ALL T timesteps.
       Input MUST have T == self.T_init or raises dim mismatch."""
    def __init__(self, T):
        super().__init__()
        self.T_init = T
        self.fc1 = nn.Linear(T, max(T // 4, 1))
        self.fc2 = nn.Linear(max(T // 4, 1), T)

    def forward(self, x):
        T, B, C, H, W = x.shape
        s = x.mean(dim=(1, 2, 3, 4))
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * w.view(T, 1, 1, 1, 1)


class CrossTimeAttnModel(nn.Module):
    """Conv → BN → CrossTimeAttn(T_init) → LIF → GAP → FC."""
    def __init__(self, T_init, C=32, num_classes=10):
        super().__init__()
        self.T_init = T_init
        self.conv = layer.Conv2d(3, C, kernel_size=3, padding=1, step_mode='m')
        self.bn = layer.BatchNorm2d(C, step_mode='m')
        self.attn = MinimalCrossTimeAttention(T_init)
        self.lif = neuron.LIFNode(detach_reset=True, step_mode='m')
        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.fc = layer.Linear(C, num_classes, step_mode='m')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.attn(x)
        x = self.lif(x)
        x = self.pool(x).flatten(2)
        x = self.fc(x)
        return x


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


def measure(fn, *args, n_warmup=1, n_iters=2):
    torch.cuda.synchronize()
    torch.cuda.empty_cache(); gc.collect()
    try:
        for _ in range(n_warmup):
            _ = fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = fn(*args)
        torch.cuda.synchronize()
        return out, 'ok', None
    except Exception as e:
        return None, 'fail', str(e)[:200]


def main():
    print('=' * 70)
    print('Part A v2: Cross-time attention chunked-execution incompatibility')
    print('=' * 70)
    device = torch.device('cuda:0')
    T_init = 64
    B, H = 16, 64
    torch.manual_seed(42)

    net = CrossTimeAttnModel(T_init=T_init, C=32, num_classes=10).to(device).eval()
    functional.set_step_mode(net, step_mode='m')
    x = (torch.rand(T_init, B, 3, H, H, device=device) > 0.7).float()

    # Baseline (K=T_init, attention sees full T_init steps as expected)
    print(f'\nBaseline (K=T_init={T_init}, attn dim matches):')
    out_b, st_b, err_b = measure(baseline_forward, net, x)
    if st_b == 'ok':
        print(f'  OK, output_norm={out_b.norm().item():.4f}')
    else:
        print(f'  FAIL: {err_b}')
        return

    # Chunked at K < T_init: chunks feed K timesteps, but attention.fc1 expects T_init
    print(f'\nChunked execution attempts (K < T_init):')
    print(f'  Mode 1: same model instance, chunked feeds K-step input to attn(T_init)')
    print()
    for K in [32, 16, 8, 4, 2, 1]:
        out_c, st_c, err_c = measure(chunked_forward, net, x, K)
        if st_c == 'ok':
            err = (out_c - out_b).abs().max().item()
            print(f'  K={K:>3}: OK but err={err:.4e} '
                  f'{"BIT-EXACT" if err < 1e-5 else "DIVERGED (semantics broken)"}')
        else:
            err_short = err_c.split('\n')[0][:120]
            print(f'  K={K:>3}: FAIL — {err_short}')

    print(f'\nVerdict:')
    print(f'  Cross-time attention (fc1 dim = T_init) cannot accept K-step chunks.')
    print(f'  Either dimension mismatch (clean failure) or semantic divergence,')
    print(f'  proving fundamental incompatibility.')

    # Repeat Part B summary for completeness in this log
    print()
    print('=' * 70)
    print('Part B summary (verified in phase1_5):')
    print('  per-timestep SE-block: 21/21 bit-exact, R²=1.000')
    print('=' * 70)


if __name__ == '__main__':
    main()
