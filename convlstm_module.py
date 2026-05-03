"""
convlstm_module.py v3 — selective batching: only non-recurrent paths.

v2 → v3 fix:
  v2 batched conv_x INSIDE the recurrent cell, materializing
  [T, b, 4*hidden_C, H, W] gates_x tensor at once. That tensor is
  ~26 GB at b=32, H=224, T=8, hidden_C=128 — destroying AEROS's whole
  memory-savings story (α_K went from 0 to 1.65, baseline went OOM).

v3 strategy:
  - The STEM (Conv2D + BN + ReLU) is a per-step pure function. Batch
    it: fold T into batch dim, single launch. This eliminates Python
    launch overhead WITHOUT materializing per-step gate tensors.
  - The CLASSIFIER (AdaptiveAvgPool + Linear) is also per-step pure.
    Batch the same way.
  - The two ConvLSTMCells STAY single-step Python-loop (one iteration
    per timestep). Their inner gate computation does NOT materialize
    a T-step tensor; (h, c) state flows step-by-step as the model
    intends. The α_K coefficient stays small.

Net result:
  - Memory: matches v1 (α_K small, no T-step gate materialization).
  - Wallclock: stem and classifier each save (T-1) launches, which
    is the bulk of CPU-side overhead at large T. The cell's own
    sequential overhead is unavoidable (it's the recurrence).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, functional


class ConvLSTMCell(base.MemoryModule):
    """Standard ConvLSTM cell — single-step forward only.

    Memory: one [b, 4*hidden_C, H, W] gates tensor per step (released
    each step), plus persistent (h, c) state.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Single combined conv on cat([x, h])
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size=kernel_size, padding=padding,
                              bias=True)

        self.register_memory('h', None)
        self.register_memory('c', None)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        if self.h is None:
            self.h = torch.zeros(b, self.hidden_channels, H, W,
                                 device=device, dtype=dtype)
            self.c = torch.zeros(b, self.hidden_channels, H, W,
                                 device=device, dtype=dtype)

        gates = self.conv(torch.cat([x, self.h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        new_c = f * self.c + i * g
        new_h = o * torch.tanh(new_c)

        self.h = new_h
        self.c = new_c
        return new_h


class ConvLSTMNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: tuple = (32, 64, 128),
                 num_classes: int = 11):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Conv2d(in_channels, hidden_channels[0],
                              kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels[0])
        self.cell1 = ConvLSTMCell(hidden_channels[0], hidden_channels[1])
        self.cell2 = ConvLSTMCell(hidden_channels[1], hidden_channels[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, b, 3, H, W] -> y: [T, b, num_classes]

        Stem and classifier are batched across T (single launch each).
        ConvLSTMCells iterate sequentially (their (h, c) state requires it),
        but the cells' single_step_forward only allocates per-step
        gate tensors, which are freed at next step boundary.
        """
        T, b, in_C, H, W = x.shape

        # Batched stem: [T*b, in_C, H, W] -> [T*b, hidden_0, H, W] -> [T, b, ...]
        x_flat = x.reshape(T * b, in_C, H, W)
        x_flat = F.relu(self.bn(self.stem(x_flat)))
        x_seq = x_flat.reshape(T, b, -1, H, W)
        del x_flat  # free intermediate

        # Sequential ConvLSTM through cells
        # Process step-by-step but keep all per-step h2 outputs to fold
        # into a single batched classifier call at the end.
        h2_steps = []
        for t in range(T):
            h1 = self.cell1.single_step_forward(x_seq[t])
            h2 = self.cell2.single_step_forward(h1)
            h2_steps.append(h2)
        del x_seq

        # Stack [T, b, hidden_2, H, W]
        h2_seq = torch.stack(h2_steps, dim=0)
        del h2_steps

        # Batched pool + classifier
        h2_flat = h2_seq.reshape(T * b, -1, H, W)
        del h2_seq
        pooled = self.pool(h2_flat).flatten(1)
        y_flat = self.fc(pooled)
        return y_flat.reshape(T, b, -1)


def build_convlstm_network(num_classes: int = 11):
    return ConvLSTMNetwork(in_channels=3, hidden_channels=(32, 64, 128),
                           num_classes=num_classes)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--b', type=int, default=4)
    ap.add_argument('--H', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Reference: per-step Python loop (no batching at all)
    class ReferenceConvLSTMNetwork(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.stem = src.stem
            self.bn = src.bn
            self.cell1 = src.cell1
            self.cell2 = src.cell2
            self.pool = src.pool
            self.fc = src.fc
        def forward(self, x):
            T = x.shape[0]
            outs = []
            for t in range(T):
                xt = F.relu(self.bn(self.stem(x[t])))
                ht1 = self.cell1.single_step_forward(xt)
                ht2 = self.cell2.single_step_forward(ht1)
                outs.append(self.fc(self.pool(ht2).flatten(1)))
            return torch.stack(outs, dim=0)

    net = build_convlstm_network(num_classes=11).to(device)
    net.eval()
    x = torch.rand(args.T, args.b, 3, args.H, args.H, device=device)

    # Test 1: batched-stem vs reference equivalence
    ref = ReferenceConvLSTMNetwork(net)
    functional.reset_net(net)
    with torch.no_grad():
        y_ref = ref(x)
    functional.reset_net(net)
    with torch.no_grad():
        y_batched = net(x)
    err = (y_ref - y_batched).abs().max().item()
    print(f'Reference vs v3 max_err: {err:.2e}')
    if err < 1e-5:
        print('  ✓ v3 is numerically equivalent to reference')
    else:
        print(f'  ✗ Numerical divergence — investigate')

    # Test 2: chunked AEROS bit-exactness
    K = 4
    functional.reset_net(net)
    with torch.no_grad():
        y_full = net(x)
    functional.reset_net(net)
    with torch.no_grad():
        outs, i = [], 0
        while i < args.T:
            sz = min(K, args.T - i)
            outs.append(net(x[i:i + sz]))
            i += sz
        y_chunk = torch.cat(outs, dim=0)
    err = (y_full - y_chunk).abs().max().item()
    print(f'Full vs chunked κ={K} max_err: {err:.2e}')
    if err == 0.0:
        print('  ✓ Bit-exact under AEROS chunking')
    elif err < 1e-5:
        print(f'  ≈ Within float-reorder noise ({err:.2e})')
    else:
        print(f'  WARN: max_err = {err:.2e}')

    # Test 3: timing — also test at larger T to see batching benefit emerge
    if device.type == 'cuda':
        # Warmup
        for _ in range(2):
            functional.reset_net(net); _ = ref(x)
            functional.reset_net(net); _ = net(x)
        torch.cuda.synchronize()

        functional.reset_net(net)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            functional.reset_net(net)
            with torch.no_grad():
                _ = ref(x)
        torch.cuda.synchronize()
        ref_ms = (time.perf_counter() - t0) / 5 * 1000

        functional.reset_net(net)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            functional.reset_net(net)
            with torch.no_grad():
                _ = net(x)
        torch.cuda.synchronize()
        v3_ms = (time.perf_counter() - t0) / 5 * 1000

        print(f'\nTiming (T={args.T}, b={args.b}, H={args.H}):')
        print(f'  Reference (per-step Python):  {ref_ms:.1f} ms')
        print(f'  v3 (batched stem+classifier): {v3_ms:.1f} ms')
        print(f'  Speedup:                       {ref_ms/v3_ms:.2f}×')

        # Memory comparison at this small case
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        functional.reset_net(net)
        with torch.no_grad():
            _ = ref(x)
        torch.cuda.synchronize()
        ref_peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        functional.reset_net(net)
        with torch.no_grad():
            _ = net(x)
        torch.cuda.synchronize()
        v3_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'\nPeak memory (T={args.T}, b={args.b}, H={args.H}):')
        print(f'  Reference:                    {ref_peak:.3f} GB')
        print(f'  v3:                            {v3_peak:.3f} GB '
              f'(diff: {(v3_peak/ref_peak-1)*100:+.1f}%)')

    n_params = sum(p.numel() for p in net.parameters())
    print(f'\nParameters: {n_params:,} ({n_params/1e6:.2f}M)')