"""
AEROS Phase 1c Mini: Triton event-driven conv kernel for one specific shape.
Hardcoded: B=16, Cin=Cout=128, H=W=64, K=3, stride=1, padding=1, fp32.

Design:
  Output-stationary, no atomics. Each program block computes a tile of output:
    [BLOCK_H * BLOCK_W * BLOCK_COUT] = [8 * 8 * 64] elements.
  Reads its (BLOCK_H+2) x (BLOCK_W+2) input neighborhood, streams over Cin
  in chunks of BLOCK_CIN=16, accumulates in registers, writes once.

  Sparsity exploit: NOT data-skip (we still read all input, dense load).
  The mul-by-zero in fp32 short-circuits the FMA contribution naturally.
  Wins come from (a) better launch / pipelining than naive PyTorch,
  (b) memory bandwidth reduction is bounded — bound to be modest on V100.

Goal on V100:
  density 0.5%: event ≤ cuDNN
  density 2%:   event ≤ 2x cuDNN
  density >=10%: event allowed up to 5x cuDNN
"""
import argparse, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Hard-coded kernel parameters (Mini scope)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Hard-coded kernel parameters (Mini scope) — V100 / Triton 2.1 conservative
# ---------------------------------------------------------------------
B, CIN, COUT = 16, 128, 128
H, W = 64, 64
K = 3
PAD = 1
STRIDE = 1

# Smaller blocks, no tl.dot/view path (Triton 2.1 + V100 bug-prone otherwise).
BLOCK_H = 4
BLOCK_W = 4
BLOCK_COUT = 32
BLOCK_CIN = 16
GROUPS_H = H // BLOCK_H        # 16
GROUPS_W = W // BLOCK_W        # 16
GROUPS_COUT = COUT // BLOCK_COUT  # 4


@triton.jit
def event_conv_kernel(
    spike_ptr, weight_ptr, output_ptr,
    s_b, s_cin, s_h, s_w,
    w_cout, w_cin, w_kh, w_kw,
    o_b, o_cout, o_h, o_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_COUT: tl.constexpr, BLOCK_CIN: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    CIN: tl.constexpr, COUT: tl.constexpr,
    K: tl.constexpr, PAD: tl.constexpr,
):
    pid_bnh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_co = tl.program_id(2)

    n_h_tiles = H // BLOCK_H
    b_idx = pid_bnh // n_h_tiles
    h_tile = pid_bnh % n_h_tiles

    h_offs = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)         # [BH]
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)          # [BW]
    co_offs = pid_co * BLOCK_COUT + tl.arange(0, BLOCK_COUT)  # [BCO]

    # accumulator [BH, BW, BCO]
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_COUT), dtype=tl.float32)

    for cin_start in range(0, CIN, BLOCK_CIN):
        cin_offs = cin_start + tl.arange(0, BLOCK_CIN)        # [BCI]

        for kh in tl.static_range(K):
            for kw in tl.static_range(K):
                h_in = h_offs[:, None] + kh - PAD             # [BH, 1]
                w_in = w_offs[None, :] + kw - PAD             # [1, BW]
                in_mask_hw = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

                h_safe = tl.where(in_mask_hw, h_in, 0)
                w_safe = tl.where(in_mask_hw, w_in, 0)

                # spike[b, cin_offs, h_safe, w_safe] -> [BH, BW, BCI]
                spike_addr = (
                    spike_ptr
                    + b_idx * s_b
                    + cin_offs[None, None, :] * s_cin
                    + h_safe[:, :, None] * s_h
                    + w_safe[:, :, None] * s_w
                )
                load_mask = in_mask_hw[:, :, None] & (cin_offs[None, None, :] < CIN)
                spike_tile = tl.load(spike_addr, mask=load_mask, other=0.0)
                # spike_tile: [BH, BW, BCI]

                # weight[co_offs, cin_offs, kh, kw] -> [BCO, BCI]
                weight_addr = (
                    weight_ptr
                    + co_offs[:, None] * w_cout
                    + cin_offs[None, :] * w_cin
                    + kh * w_kh
                    + kw * w_kw
                )
                weight_mask = (co_offs[:, None] < COUT) & (cin_offs[None, :] < CIN)
                weight_tile = tl.load(weight_addr, mask=weight_mask, other=0.0)
                # weight_tile: [BCO, BCI]

                # broadcast multiply + sum along Cin
                # spike[BH, BW, 1, BCI] * weight[1, 1, BCO, BCI] -> [BH, BW, BCO, BCI]
                # sum axis=3 -> [BH, BW, BCO]
                prod = spike_tile[:, :, None, :] * weight_tile[None, None, :, :]
                contrib = tl.sum(prod, axis=3)                # [BH, BW, BCO]
                acc = acc + contrib

    # Write output[b, co_offs, h_offs, w_offs]
    out_addr = (
        output_ptr
        + b_idx * o_b
        + co_offs[None, None, :] * o_cout
        + h_offs[:, None, None] * o_h
        + w_offs[None, :, None] * o_w
    )
    out_mask = (
        (co_offs[None, None, :] < COUT)
        & (h_offs[:, None, None] < H)
        & (w_offs[None, :, None] < W)
    )
    tl.store(out_addr, acc, mask=out_mask)


def event_conv_triton(spike, weight):
    assert spike.shape == (B, CIN, H, W)
    assert weight.shape == (COUT, CIN, K, K)
    assert spike.is_contiguous() and weight.is_contiguous()

    output = torch.empty((B, COUT, H, W), device=spike.device, dtype=torch.float32)

    grid = (B * GROUPS_H, GROUPS_W, GROUPS_COUT)
    event_conv_kernel[grid](
        spike, weight, output,
        spike.stride(0), spike.stride(1), spike.stride(2), spike.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_COUT=BLOCK_COUT, BLOCK_CIN=BLOCK_CIN,
        H=H, W=W, CIN=CIN, COUT=COUT, K=K, PAD=PAD,
        num_warps=2, num_stages=1,
    )
    return output
# ---------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------
def benchmark(fn, *args, warmup=20, iters=100):
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
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--check_only', action='store_true',
                   help='only run correctness, skip benchmark')
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    weight = torch.randn(COUT, CIN, K, K, device='cuda', dtype=torch.float32).contiguous()

    print(f'\nMini Triton conv (V100). Shape: B={B} Cin={CIN} Cout={COUT} '
          f'H={H} W={W} K={K} pad={PAD}')
    print(f'Grid: ({B*GROUPS_H}, {GROUPS_W}, {GROUPS_COUT}) = '
          f'{B*GROUPS_H*GROUPS_W*GROUPS_COUT} blocks')

    # Correctness check at three densities
    print('\n--- correctness ---')
    for d in [0.005, 0.05, 0.5]:
        spike = (torch.rand(B, CIN, H, W, device='cuda') < d).float().contiguous()
        out_ref = F.conv2d(spike, weight, padding=PAD)
        out_triton = event_conv_triton(spike, weight)
        max_err = (out_ref - out_triton).abs().max().item()
        rel_err = max_err / (out_ref.abs().max().item() + 1e-9)
        ok = '✓' if max_err < 1e-3 else '✗'
        print(f'  density {d:.3f}: max_abs_err = {max_err:.2e}  rel = {rel_err:.2e}  {ok}')

    if args.check_only:
        return

    # Density sweep benchmark
    print('\n--- benchmark ---')
    print(f'{"density":>9s}  {"cuDNN ms":>9s}  {"triton ms":>10s}  {"speedup":>8s}')
    print('-' * 50)
    for d in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        spike = (torch.rand(B, CIN, H, W, device='cuda') < d).float().contiguous()
        actual_d = spike.mean().item()
        t_dense = benchmark(F.conv2d, spike, weight, None, 1, PAD, iters=args.iters)
        t_triton = benchmark(event_conv_triton, spike, weight, iters=args.iters)
        print(f'{actual_d:9.4f}  {t_dense*1000:8.3f}  {t_triton*1000:9.3f}  '
              f'{t_dense/t_triton:6.2f}x')


if __name__ == '__main__':
    main()