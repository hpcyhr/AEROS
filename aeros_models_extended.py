"""
AEROS extended architecture suite — non-SNN stateful sequence networks.

Provides 6 new architectures (in addition to the SNN suite from
spikingjelly):

    ConvLSTM-2L     Convolutional recurrent
    ConvGRU-2L      Convolutional recurrent
    LSTM-4L         Recurrent (sequence-level, 1D)
    GRU-4L          Recurrent (sequence-level, 1D)
    CausalTCN-8L    Dilated causal conv with finite RF halo
    MinimalSSM-2L   Diagonal-A state-space layer (S4-style reference)

All accept input of shape [T, B, C, H, W] for spatial models
(ConvLSTM, ConvGRU, CausalTCN), or [T, B, F] for sequence models
(LSTM, GRU, MinimalSSM). Output is the same shape (or [T, B, num_classes]
for the classification head when applicable).

All are pure PyTorch reference implementations (no fused kernels),
so segment-boundary carry behavior is directly observable.

Each model exposes:
  - forward(x): standard
  - reset_state(): zero hidden state buffers
  - The model's hidden state is held in module attributes (e.g., self.h)
    and persists between forward() calls until reset_state() is called,
    enabling carry-stream segmentation tests.

Usage:
    from aeros_models_extended import build_extended_net, EXTENDED_NETS
    net = build_extended_net("ConvLSTM-2L")
    # net is in eval(), step_mode-equivalent for [T, B, ...] inputs
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EXTENDED_NETS = [
    "ConvLSTM-2L",
    "ConvGRU-2L",
    "LSTM-4L",
    "GRU-4L",
    "CausalTCN-8L",
    "MinimalSSM-2L",
]


# ============================================================================
# 1) ConvLSTM
# ============================================================================

class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell. Holds h, c as buffer attributes."""

    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hid_ch = hid_ch
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size,
                              padding=pad)
        self.h: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        # x: [B, C, H, W]
        B, _, H, W = x.shape
        if self.h is None:
            self.h = torch.zeros(B, self.hid_ch, H, W, device=x.device,
                                 dtype=x.dtype)
            self.c = torch.zeros(B, self.hid_ch, H, W, device=x.device,
                                 dtype=x.dtype)
        comb = torch.cat([x, self.h], dim=1)
        gates = self.conv(comb)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * self.c + i * g
        h_new = o * torch.tanh(c_new)
        self.h = h_new
        self.c = c_new
        return h_new


class ConvLSTM2L(nn.Module):
    """Two-layer ConvLSTM video model with classification head.
    Input: [T, B, C_in, H, W]
    Output: [T, B, num_classes]
    """

    def __init__(self, in_ch: int = 3, hid_ch: int = 32, num_classes: int = 10):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_ch, hid_ch)
        self.cell2 = ConvLSTMCell(hid_ch, hid_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hid_ch, num_classes)

    def reset_state(self):
        self.cell1.reset_state()
        self.cell2.reset_state()

    def forward(self, x):
        # x: [T, B, C, H, W]
        T = x.shape[0]
        outs = []
        for t in range(T):
            h1 = self.cell1(x[t])
            h2 = self.cell2(h1)
            pooled = self.pool(h2).flatten(1)
            outs.append(self.fc(pooled))
        return torch.stack(outs, dim=0)  # [T, B, num_classes]


# ============================================================================
# 2) ConvGRU
# ============================================================================

class ConvGRUCell(nn.Module):
    """ConvGRU. Holds h as buffer attribute."""

    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hid_ch = hid_ch
        pad = kernel_size // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size,
                                 padding=pad)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size,
                                padding=pad)
        self.h: Optional[torch.Tensor] = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        B, _, H, W = x.shape
        if self.h is None:
            self.h = torch.zeros(B, self.hid_ch, H, W, device=x.device,
                                 dtype=x.dtype)
        comb = torch.cat([x, self.h], dim=1)
        zr = self.conv_zr(comb)
        z, r = zr.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        comb2 = torch.cat([x, r * self.h], dim=1)
        h_tilde = torch.tanh(self.conv_h(comb2))
        h_new = (1 - z) * self.h + z * h_tilde
        self.h = h_new
        return h_new


class ConvGRU2L(nn.Module):
    def __init__(self, in_ch: int = 3, hid_ch: int = 32, num_classes: int = 10):
        super().__init__()
        self.cell1 = ConvGRUCell(in_ch, hid_ch)
        self.cell2 = ConvGRUCell(hid_ch, hid_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hid_ch, num_classes)

    def reset_state(self):
        self.cell1.reset_state()
        self.cell2.reset_state()

    def forward(self, x):
        T = x.shape[0]
        outs = []
        for t in range(T):
            h1 = self.cell1(x[t])
            h2 = self.cell2(h1)
            pooled = self.pool(h2).flatten(1)
            outs.append(self.fc(pooled))
        return torch.stack(outs, dim=0)


# ============================================================================
# 3) Stacked LSTM (sequence-level, 1D feature input)
#
# Adapter: takes [T, B, C, H, W], pools spatially each step to [T, B, F],
# runs LSTM, returns [T, B, num_classes].
# ============================================================================

class LSTM4L(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 256, num_layers: int = 4,
                 num_classes: int = 10, H: int = 32):
        super().__init__()
        self.feat_dim = in_ch * H * H if H > 0 else in_ch
        # Use a single Linear projector to keep hidden dim fixed
        self.in_proj = nn.Linear(self.feat_dim, hidden)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden,
                            num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden, num_classes)
        self._h = None
        self._c = None

    def reset_state(self):
        self._h = None
        self._c = None

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B = x.shape[0], x.shape[1]
        flat = x.reshape(T, B, -1)  # [T, B, C*H*W]
        proj = self.in_proj(flat)   # [T, B, hidden]
        if self._h is None:
            out, (h, c) = self.lstm(proj)
        else:
            out, (h, c) = self.lstm(proj, (self._h, self._c))
        self._h = h.detach()
        self._c = c.detach()
        return self.fc(out)  # [T, B, num_classes]


# ============================================================================
# 4) Stacked GRU
# ============================================================================

class GRU4L(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 256, num_layers: int = 4,
                 num_classes: int = 10, H: int = 32):
        super().__init__()
        self.feat_dim = in_ch * H * H
        self.in_proj = nn.Linear(self.feat_dim, hidden)
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden,
                          num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden, num_classes)
        self._h = None

    def reset_state(self):
        self._h = None

    def forward(self, x):
        T, B = x.shape[0], x.shape[1]
        flat = x.reshape(T, B, -1)
        proj = self.in_proj(flat)
        if self._h is None:
            out, h = self.gru(proj)
        else:
            out, h = self.gru(proj, self._h)
        self._h = h.detach()
        return self.fc(out)


# ============================================================================
# 5) Causal TCN (dilated 1D conv stack with finite-RF temporal halo)
# ============================================================================

class CausalConv1D(nn.Module):
    """1D causal dilated conv along the T dimension."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.k = kernel_size
        self.d = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=0)
        # halo buffer: last (kernel-1)*dilation timesteps from previous segment
        self.halo: Optional[torch.Tensor] = None  # [B*F_spatial, in_ch, H_pad]

    def reset_state(self):
        self.halo = None

    def forward(self, x):
        # x: [T, B*F_spatial, in_ch]
        # → permute to [BS, in_ch, T] for conv1d
        T, BS, C = x.shape
        xt_orig = x.permute(1, 2, 0).contiguous()  # [BS, C, T] — the input segment
        # Prepend halo (left context) for causal conv
        if self.halo is not None and self.halo.shape[2] > 0:
            xt = torch.cat([self.halo, xt_orig], dim=2)
        else:
            # First segment after reset: pad with zeros for causal start
            xt = F.pad(xt_orig, (self.left_pad, 0))
        out = self.conv(xt)  # [BS, out_ch, T_seg]
        # Update halo from the original input only (not the prepended halo).
        # Halo for next call = last left_pad timesteps of input we just saw.
        if self.left_pad > 0:
            if T >= self.left_pad:
                self.halo = xt_orig[:, :, -self.left_pad:].detach().clone()
            else:
                # input shorter than left_pad — concat previous halo+input,
                # keep last left_pad
                if self.halo is not None:
                    combined = torch.cat([self.halo, xt_orig], dim=2)
                else:
                    combined = F.pad(xt_orig, (self.left_pad, 0))
                self.halo = combined[:, :, -self.left_pad:].detach().clone()
        return out.permute(2, 0, 1).contiguous()  # [T_seg, BS, out_ch]


class CausalTCN8L(nn.Module):
    """8-layer dilated causal TCN. Input [T, B, C, H, W] → flatten spatial,
    run 1D causal TCN along T, project + reshape."""

    def __init__(self, in_ch: int = 3, hidden: int = 64, num_classes: int = 10,
                 H: int = 32):
        super().__init__()
        self.H = H
        self.feat_dim = in_ch * H * H
        self.in_proj = nn.Linear(self.feat_dim, hidden)
        # 8 layers of dilated causal conv with dilations 1,2,4,8,16,32,64,128
        self.layers = nn.ModuleList([
            CausalConv1D(hidden, hidden, kernel_size=3, dilation=2 ** i)
            for i in range(8)
        ])
        self.fc = nn.Linear(hidden, num_classes)

    def reset_state(self):
        for l in self.layers:
            l.reset_state()

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B = x.shape[0], x.shape[1]
        flat = x.reshape(T, B, -1)
        h = self.in_proj(flat)  # [T, B, hidden]
        for l in self.layers:
            h = F.relu(l(h))
        return self.fc(h)


# ============================================================================
# 6) Minimal SSM (S4-style, diagonal A)
#
# h_t = A * h_{t-1} + B * x_t   (diagonal, real-valued)
# y_t = C * h_t
#
# Reference loop implementation (no fused selective scan kernel) so that
# segment-boundary carry is directly testable.
# ============================================================================

class DiagonalSSMCell(nn.Module):
    """h_t = A·h_{t-1} + B·x_t, y_t = C·h_t. Diagonal A so it's stable
    and decoupled along feature dim."""

    def __init__(self, dim: int):
        super().__init__()
        # Parameterize log_A so A = exp(log_A) ∈ (0, ∞), and use sigmoid form
        # for unit-bounded decay
        self.log_neg_A = nn.Parameter(torch.zeros(dim))  # decay rate
        self.B = nn.Parameter(torch.randn(dim) * 0.1)
        self.C = nn.Parameter(torch.randn(dim) * 0.1)
        self.dim = dim
        self.h: Optional[torch.Tensor] = None  # [B, dim]

    def reset_state(self):
        self.h = None

    def forward(self, x):
        # x: [T, B, dim]
        T, B, _ = x.shape
        if self.h is None:
            self.h = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
        # Diagonal A: vector [dim], 0 < a < 1
        a = torch.sigmoid(-self.log_neg_A)  # decay in (0,1)
        outs = []
        h = self.h
        for t in range(T):
            h = a * h + self.B * x[t]
            outs.append(self.C * h)
        self.h = h
        return torch.stack(outs, dim=0)


class MinimalSSM2L(nn.Module):
    """Two-layer SSM with linear projector. Input [T, B, C, H, W] →
    flatten spatial → SSM along T → linear → output."""

    def __init__(self, in_ch: int = 3, hidden: int = 256, num_classes: int = 10,
                 H: int = 32):
        super().__init__()
        self.feat_dim = in_ch * H * H
        self.in_proj = nn.Linear(self.feat_dim, hidden)
        self.ssm1 = DiagonalSSMCell(hidden)
        self.ssm2 = DiagonalSSMCell(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def reset_state(self):
        self.ssm1.reset_state()
        self.ssm2.reset_state()

    def forward(self, x):
        T, B = x.shape[0], x.shape[1]
        flat = x.reshape(T, B, -1)
        h = self.in_proj(flat)        # [T, B, hidden]
        h = self.ssm1(h)
        h = self.norm(h)
        h = self.ssm2(h)
        return self.fc(h)


# ============================================================================
# Builder + utility
# ============================================================================

def build_extended_net(name: str, num_classes: int = 10, H: int = 32,
                        in_ch: int = 3) -> nn.Module:
    """Build one of the 6 extended-suite architectures by name.

    H is the spatial size (assumed H == W). For SNN suite p9_1a uses H=128
    by default; for the extended suite we default to H=32 to keep parameter
    counts reasonable across all 6 archs while still demonstrating
    [T,B,C,H,W] -> stateful -> [T,B,num_classes] correctness.
    """
    if name == "ConvLSTM-2L":
        net = ConvLSTM2L(in_ch=in_ch, hid_ch=32, num_classes=num_classes)
    elif name == "ConvGRU-2L":
        net = ConvGRU2L(in_ch=in_ch, hid_ch=32, num_classes=num_classes)
    elif name == "LSTM-4L":
        net = LSTM4L(in_ch=in_ch, hidden=256, num_layers=4,
                     num_classes=num_classes, H=H)
    elif name == "GRU-4L":
        net = GRU4L(in_ch=in_ch, hidden=256, num_layers=4,
                    num_classes=num_classes, H=H)
    elif name == "CausalTCN-8L":
        net = CausalTCN8L(in_ch=in_ch, hidden=64,
                          num_classes=num_classes, H=H)
    elif name == "MinimalSSM-2L":
        net = MinimalSSM2L(in_ch=in_ch, hidden=256,
                           num_classes=num_classes, H=H)
    else:
        raise ValueError(f"unknown extended net: {name}")
    net.eval()
    return net


def reset_state_extended(net: nn.Module) -> None:
    """Reset all stateful submodules in an extended-suite net."""
    if hasattr(net, "reset_state"):
        net.reset_state()
        return
    for m in net.modules():
        if hasattr(m, "reset_state") and m is not net:
            m.reset_state()


# ============================================================================
# Sanity check (run as: python aeros_models_extended.py)
# ============================================================================

if __name__ == "__main__":
    import time
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Extended suite sanity check (device={device}) ===")
    T, B, C, H = 16, 4, 3, 32
    x_5d = torch.randn(T, B, C, H, H, device=device)
    print(f"Input shape: {tuple(x_5d.shape)}")
    print()

    for name in EXTENDED_NETS:
        net = build_extended_net(name, num_classes=10, H=H, in_ch=C).to(device)
        reset_state_extended(net)
        n_params = sum(p.numel() for p in net.parameters())
        t0 = time.time()
        with torch.no_grad():
            y = net(x_5d)
        t = time.time() - t0
        print(f"  {name:<14s}  params={n_params:>11,}  out={tuple(y.shape)}"
              f"  time={t*1000:.1f}ms")