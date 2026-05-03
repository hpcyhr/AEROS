"""
AEROS Phase 1c — sanity check: can PyTorch sparse SpMM beat cuDNN at low density?
30-minute test, 1-2 hour debug if errors.

Method: implicit-GEMM-style conv via im2col + sparse @ dense matmul.
  1. F.unfold(spike) -> [B, Cin*K*K, H*W]  (dense, mostly zero at low density)
  2. convert to sparse_coo
  3. weight.view(Cout, Cin*K*K)  (dense)
  4. sparse @ dense.T -> output [B, Cout, H*W]
  5. reshape to [B, Cout, H, W]

This is NOT optimized — torch.sparse.mm is a generic CSR/COO SpMM kernel,
not designed for the "sparsity per-batch + im2col-induced sparsity pattern"
we're feeding it. The point of this test is to learn whether ANY existing
sparse kernel on V100 gets within 2x of cuDNN at measured density. If yes,
a custom Triton implementation can plausibly do better. If no, V100 is hostile
to sparse paths in general — bigger AEROS strategy decision needed.
"""
import argparse, time
import torch
import torch.nn.functional as F


# Conv1 shape, hardcoded
B, CIN, COUT = 16, 128, 128
H, W = 64, 64
K = 3
PAD = 1


def sparse_conv2d_via_unfold(spike, weight):
    """spike: [B, Cin, H, W] dense fp32 (binary content)
       weight: [Cout, Cin, K, K] dense fp32
       Returns [B, Cout, H, W] fp32"""
    Bs, Cin_, Hs, Ws = spike.shape
    Cout_, _, Kh, Kw = weight.shape

    # 1. unfold -> [B, Cin*K*K, H*W] dense
    x_unfold = F.unfold(spike, kernel_size=K, padding=PAD)
    # x_unfold shape: [B, Cin*K*K, L] where L = H_out*W_out = H*W (with pad=1, stride=1)

    # 2. weight as 2D [Cout, Cin*K*K]
    w_flat = weight.view(Cout_, Cin_ * Kh * Kw)

    # 3-4. for each batch, sparse SpMM
    # We need: out_flat[b] = w_flat @ x_unfold[b]  shape [Cout, L]
    # x_unfold[b] is [Cin*K*K, L] mostly-zero
    # Convert x_unfold[b] to sparse_coo
    L = Hs * Ws
    out = torch.empty(Bs, Cout_, L, device=spike.device, dtype=torch.float32)
    for b in range(Bs):
        x_b = x_unfold[b]  # [Cin*K*K, L]
        # to sparse coo
        x_b_sp = x_b.to_sparse()
        # sparse @ dense.T: torch.sparse.mm requires (sparse, dense) -> dense
        # we want w_flat @ x_b -> [Cout, L]
        # = (x_b.T @ w_flat.T).T
        out_b = torch.sparse.mm(x_b_sp.t(), w_flat.t()).t()  # [Cout, L]
        out[b] = out_b

    return out.view(Bs, Cout_, Hs, Ws)


def sparse_conv2d_batched_unfold(spike, weight):
    """Variant: build one big sparse matrix for the whole batch, single SpMM.
    May be faster than per-batch loop due to amortized overhead."""
    Bs, Cin_, Hs, Ws = spike.shape
    Cout_, _, Kh, Kw = weight.shape
    L = Hs * Ws

    x_unfold = F.unfold(spike, kernel_size=K, padding=PAD)
    # [B, Cin*K*K, L] -> reshape to [Cin*K*K, B*L]
    x_unfold = x_unfold.permute(1, 0, 2).contiguous().view(Cin_ * Kh * Kw, Bs * L)

    w_flat = weight.view(Cout_, Cin_ * Kh * Kw)

    x_sp = x_unfold.to_sparse()
    # w_flat @ x_unfold -> [Cout, B*L]
    out = torch.sparse.mm(x_sp.t(), w_flat.t()).t()  # [Cout, B*L]

    return out.view(Cout_, Bs, L).permute(1, 0, 2).contiguous().view(Bs, Cout_, Hs, Ws)


def benchmark(fn, *args, warmup=5, iters=20):
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
    p.add_argument('--iters', type=int, default=20)
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    weight = torch.randn(COUT, CIN, K, K, device='cuda', dtype=torch.float32).contiguous()

    print(f'\nSparse SpMM sanity. Conv1: B={B} Cin={CIN} Cout={COUT} '
          f'H={H} W={W} K={K} pad={PAD}')

    # Correctness check at 2% density
    spike = (torch.rand(B, CIN, H, W, device='cuda') < 0.02).float().contiguous()
    out_ref = F.conv2d(spike, weight, padding=PAD)
    out_v1 = sparse_conv2d_via_unfold(spike, weight)
    out_v2 = sparse_conv2d_batched_unfold(spike, weight)
    err1 = (out_ref - out_v1).abs().max().item()
    err2 = (out_ref - out_v2).abs().max().item()
    print(f'\n--- correctness at density 0.02 ---')
    print(f'  per-batch  max_err: {err1:.2e}  {"OK" if err1 < 1e-3 else "FAIL"}')
    print(f'  batched    max_err: {err2:.2e}  {"OK" if err2 < 1e-3 else "FAIL"}')

    print(f'\n--- benchmark ---')
    print(f'{"density":>9s}  {"cuDNN":>8s}  {"per-b":>8s}  {"batched":>8s}  '
          f'{"sp_pb/cuDNN":>12s}  {"sp_b/cuDNN":>12s}')
    print('-' * 70)
    for d in [0.005, 0.01, 0.02, 0.05, 0.10, 0.30]:
        spike = (torch.rand(B, CIN, H, W, device='cuda') < d).float().contiguous()
        ad = spike.mean().item()

        t_cudnn = benchmark(F.conv2d, spike, weight, None, 1, PAD, iters=args.iters)
        t_pb = benchmark(sparse_conv2d_via_unfold, spike, weight, iters=args.iters)
        t_b = benchmark(sparse_conv2d_batched_unfold, spike, weight, iters=args.iters)

        print(f'{ad:9.4f}  {t_cudnn*1000:6.2f}ms  {t_pb*1000:6.2f}ms  '
              f'{t_b*1000:6.2f}ms  {t_pb/t_cudnn:8.2f}x slower  '
              f'{t_b/t_cudnn:8.2f}x slower')

    print('\nVerdict guide:')
    print('  - any cell <2x slower at density 0.01 → AEROS viable, custom Triton can win')
    print('  - all cells >5x slower → V100 + sparse path is hostile')
    print('  - middle (2-5x) → marginal, need explicit indices implementation')


if __name__ == '__main__':
    main()