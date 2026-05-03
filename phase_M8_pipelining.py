"""
Sanity check M8: can multi-stream pipeline conv and LIF for next/prev timestep?

Current: stream A runs [conv_t, LIF_t, conv_t+1, LIF_t+1, ...] sequential.
Pipelined: stream A runs conv_t, LIF_t while stream B runs conv_t+1.

If wall-clock with 2 streams < 0.7× single-stream, M8 GREEN.
"""
import time
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional


def sequential(x_chunks, conv_w, lif, S, P):
    """Run all chunks on default stream, sequential."""
    for x in x_chunks:
        y = F.conv2d(x, conv_w, stride=S, padding=P)
        y_t = y.unsqueeze(0)  # add T dim for LIF
        functional.reset_net(lif)
        _ = lif(y_t)


def pipelined(x_chunks, conv_w, lif1, lif2, S, P):
    """Alternate between two CUDA streams."""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for i, x in enumerate(x_chunks):
        stream = s1 if i % 2 == 0 else s2
        lif = lif1 if i % 2 == 0 else lif2
        with torch.cuda.stream(stream):
            y = F.conv2d(x, conv_w, stride=S, padding=P)
            y_t = y.unsqueeze(0)
            functional.reset_net(lif)
            _ = lif(y_t)
    s1.synchronize()
    s2.synchronize()


def main():
    device = torch.device('cuda:0')
    T, B = 32, 16

    print(f'{"shape (Cin,Cout,H,W)":<25s}  {"seq ms":>9s}  {"pipe ms":>9s}  '
          f'{"speedup":>9s}')
    print('-' * 60)

    for Cin, Cout, H, W in [(64, 64, 56, 56), (128, 128, 28, 28),
                            (256, 256, 14, 14), (512, 512, 7, 7)]:
        K, S, P = 3, 1, 1
        x_chunks = [torch.randn(B, Cin, H, W, device=device) for _ in range(T)]
        conv_w = torch.randn(Cout, Cin, K, K, device=device)
        lif1 = neuron.LIFNode(detach_reset=True).to(device)
        lif2 = neuron.LIFNode(detach_reset=True).to(device)
        functional.set_step_mode(lif1, step_mode='m')
        functional.set_step_mode(lif2, step_mode='m')

        # Warmup
        for _ in range(3):
            sequential(x_chunks, conv_w, lif1, S, P)
            pipelined(x_chunks, conv_w, lif1, lif2, S, P)
        torch.cuda.synchronize()

        # Sequential
        t0 = time.perf_counter()
        for _ in range(10):
            sequential(x_chunks, conv_w, lif1, S, P)
        torch.cuda.synchronize()
        seq_ms = (time.perf_counter() - t0) / 10 * 1000

        # Pipelined
        t0 = time.perf_counter()
        for _ in range(10):
            pipelined(x_chunks, conv_w, lif1, lif2, S, P)
        torch.cuda.synchronize()
        pipe_ms = (time.perf_counter() - t0) / 10 * 1000

        speedup = seq_ms / pipe_ms
        shape_str = f'{Cin},{Cout},{H},{W}'
        print(f'{shape_str:<25s}  {seq_ms:>8.2f}  {pipe_ms:>8.2f}  '
              f'{speedup:>8.2f}x')

    print(f'\nIf speedup > 1.3× consistently, M8 GREEN.')
    print(f'If speedup ≤ 1.1× (stream overhead eats savings), M8 RED.')


if __name__ == '__main__':
    main()