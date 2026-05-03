"""
AEROS Phase 1b: micro-benchmark naive event-driven conv vs cuDNN dense conv.

This is NOT designed to beat cuDNN. It establishes:
  1. cross-over density: where naive event matches cuDNN
  2. baseline scaling: how naive performance varies with density
  3. correctness: numerical equivalence to F.conv2d
Phase 1c (Triton) targets pushing the cross-over toward measured DVS densities.
"""
import argparse, time
import torch
import torch.nn.functional as F


def event_conv2d(spike, weight, padding=1):
    """Naive event-driven 2D conv via nonzero + index_add.
    spike:  [B, Cin, H, W]  any dtype, treated by nonzero
    weight: [Cout, Cin, K, K]
    Returns:[B, Cout, H, W]  matches F.conv2d(spike, weight, padding=padding)
    """
    B, Cin, H, W = spike.shape
    Cout, _, K, _ = weight.shape
    pad = padding

    nz = spike.nonzero(as_tuple=False)            # [N, 4]
    N = nz.size(0)
    if N == 0:
        return spike.new_zeros(B, Cout, H, W, dtype=weight.dtype)

    bs, cs, hs, ws = nz.unbind(dim=1)
    vals = spike[bs, cs, hs, ws].to(weight.dtype)  # [N]

    out_flat = spike.new_zeros(B * H * W, Cout, dtype=weight.dtype)

    for dh in range(K):
        for dw in range(K):
            # output position contributed to: (h - dh + pad, w - dw + pad)
            h_o = hs - dh + pad
            w_o = ws - dw + pad
            valid = (h_o >= 0) & (h_o < H) & (w_o >= 0) & (w_o < W)
            if not valid.any():
                continue
            v = valid.nonzero(as_tuple=True)[0]
            b_v = bs[v]; c_v = cs[v]
            h_v = h_o[v]; w_v = w_o[v]
            val_v = vals[v]                                  # [Nv]

            # weight[:, c_v, dh, dw]  shape [Cout, Nv]; scale by spike value
            w_slice = weight[:, c_v, dh, dw] * val_v.unsqueeze(0)  # [Cout, Nv]

            idx = b_v * (H * W) + h_v * W + w_v              # [Nv]
            out_flat.index_add_(0, idx, w_slice.t())         # [Nv, Cout]

    return out_flat.view(B, H, W, Cout).permute(0, 3, 1, 2).contiguous()


def benchmark(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', default='conv1',
                   choices=['conv1', 'conv2', 'conv3', 'conv4'])
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--iters', type=int, default=50)
    args = p.parse_args()

    # DVS Gesture network conv-layer shapes
    shapes = {
        'conv1': (128, 128,  64,  64, 3),  # mass-dominant layer, 1208M dense FLOPs
        'conv2': (128, 128,  32,  32, 3),
        'conv3': (128, 128,  16,  16, 3),
        'conv4': (128, 128,   8,   8, 3),
    }
    Cin, Cout, H, W, K = shapes[args.shape]
    B = args.B

    print(f'\nShape: {args.shape}  B={B} Cin={Cin} Cout={Cout} H={H} W={W} K={K}')
    print(f'Dense FLOPs/step: {Cin*Cout*H*W*K*K*2/1e6:.1f}M')

    torch.backends.cudnn.benchmark = True
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)

    densities = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50]

    print(f'\n{"density":>9s}  {"cuDNN ms":>9s}  {"event ms":>9s}  '
          f'{"speedup":>8s}  {"FLOP ceil":>10s}  {"err":>8s}')
    print('-' * 70)

    for d in densities:
        spike = (torch.rand(B, Cin, H, W, device='cuda') < d).float()
        actual_d = spike.mean().item()

        # correctness check
        out_dense = F.conv2d(spike, weight, padding=1)
        out_event = event_conv2d(spike, weight, padding=1)
        max_err = (out_dense - out_event).abs().max().item()

        t_dense = benchmark(F.conv2d, spike, weight, None, 1, 1, iters=args.iters)
        t_event = benchmark(event_conv2d, spike, weight, 1, iters=args.iters)

        flop_ceil = 1.0 / max(actual_d, 1e-6)
        speedup = t_dense / t_event

        print(f'{actual_d:9.4f}  {t_dense*1000:8.3f}  {t_event*1000:8.3f}  '
              f'{speedup:7.2f}x  {flop_ceil:8.1f}x  {max_err:8.1e}')

    print('\nLegend:')
    print('  speedup  = event vs cuDNN (>1 means event wins)')
    print('  FLOP ceil= 1/density, theoretical max if event kernel had cuDNN-level utilization')
    print('  err      = max abs diff between event and cuDNN outputs (should be < 1e-3)')


if __name__ == '__main__':
    main()