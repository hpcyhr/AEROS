"""
Sanity check M5: Conv→LIF intermediate activation HBM traffic.

If the intermediate tensor between conv and LIF is large vs the LIF state,
fusing them eliminates major HBM round-trip. (Helios reports 75× speedup
on ConvBNAct via this kind of fusion.)
"""
import time
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional


def main():
    device = torch.device('cuda:0')
    T, B = 32, 16

    print(f'{"shape (B,Cin,Cout,H,W)":<28s}  {"interm MB":>11s}  '
          f'{"state MB":>10s}  {"ratio":>7s}  {"conv ms":>9s}  {"lif ms":>9s}  '
          f'{"sum":>7s}  {"f.lower":>9s}')
    print('-' * 105)

    # Test on conv1, conv2, ... shapes typical of ResNet18-SNN
    configs = [
        (B,   3,  64, 224, 224, 3, 2, 1),  # conv1
        (B,  64,  64,  56,  56, 3, 1, 1),  # layer1
        (B,  64, 128,  28,  28, 3, 2, 1),  # layer2 first conv
        (B, 128, 128,  28,  28, 3, 1, 1),  # layer2 inner
        (B, 256, 256,  14,  14, 3, 1, 1),  # layer3
        (B, 512, 512,   7,   7, 3, 1, 1),  # layer4
    ]

    for cfg in configs:
        Bs, Cin, Cout, H, W, K, S, P = cfg
        x = torch.randn(T, Bs, Cin, H, W, device=device)
        weight = torch.randn(Cout, Cin, K, K, device=device)
        # Output H,W
        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        # Per-timestep conv output size (intermediate)
        interm_bytes = Bs * Cout * H_out * W_out * 4
        # LIF state size (= same as conv output for simple LIF)
        state_bytes = interm_bytes
        ratio = interm_bytes / state_bytes  # always 1, but keep for general

        # Bench: conv only, T timesteps
        # Reshape to (T*B, Cin, H, W) for batched conv
        x_flat = x.view(T*Bs, Cin, H, W)
        for _ in range(3):
            _ = F.conv2d(x_flat, weight, stride=S, padding=P)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = F.conv2d(x_flat, weight, stride=S, padding=P)
        torch.cuda.synchronize()
        conv_ms = (time.perf_counter() - t0) / 20 * 1000

        # Bench: LIF only on conv output
        lif = neuron.LIFNode(detach_reset=True).to(device)
        functional.set_step_mode(lif, step_mode='m')
        y_for_lif = y.view(T, Bs, Cout, H_out, W_out)
        for _ in range(3):
            _ = lif(y_for_lif); functional.reset_net(lif)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = lif(y_for_lif); functional.reset_net(lif)
        torch.cuda.synchronize()
        lif_ms = (time.perf_counter() - t0) / 20 * 1000

        # Theoretical HBM-fused lower bound:
        # If conv+LIF were fused, intermediate doesn't write to HBM,
        # only weight + input + final spike read/written.
        # Saved bytes = T * 2 * interm_bytes (write conv out, read LIF in)
        # At 900 GB/s, the saved time is bytes / BW
        saved_bytes = T * 2 * interm_bytes
        saved_ms_at_peak = saved_bytes / 900e9 * 1000
        # Lower bound assuming only the saving (conservative)
        f_lower = max(0.1, (conv_ms + lif_ms) - saved_ms_at_peak)

        shape_str = f'{Bs},{Cin},{Cout},{H},{W}'
        print(f'{shape_str:<28s}  {interm_bytes/1e6:>10.2f}M  '
              f'{state_bytes/1e6:>9.2f}M  {ratio:>6.2f}  {conv_ms:>8.2f}  '
              f'{lif_ms:>8.2f}  {conv_ms+lif_ms:>6.2f}  {f_lower:>8.2f}')

    print(f'\nIf "f.lower" (theoretical fused-kernel time) is < 50% of "sum",')
    print(f'fusion has substantial wins → M5 GREEN.')
    print(f'If f.lower ≈ sum, fusion saves little → M5 less attractive.')


if __name__ == '__main__':
    main()