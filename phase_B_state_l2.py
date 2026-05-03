"""
Direction B sanity check: Is SNN state buffer fitting in V100 L2 (6MB)?
And: how much HBM traffic does the LIF kernel actually use vs theoretical minimum?
"""
import argparse, time
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model import spiking_resnet


V100_L2_MB = 6.0
V100_HBM_BW_GBs = 900  # GB/s nominal


def build(name, num_classes=1000):
    if name == 'sresnet18':
        return spiking_resnet.spiking_resnet18(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
            detach_reset=True, num_classes=num_classes)
    elif name == 'sresnet34':
        return spiking_resnet.spiking_resnet34(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
            detach_reset=True, num_classes=num_classes)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='sresnet18',
                   choices=['sresnet18', 'sresnet34'])
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--T', type=int, default=32)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='phaseB_state.npz')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'Building {args.model}...')
    net = build(args.model).to(device)
    functional.set_step_mode(net, step_mode='m')

    # Run one forward to allocate state
    x = torch.randn(args.T, args.B, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = net(x).mean(0)

    # === Inspect each LIF layer's state size ===
    print(f'\n=== Per-layer LIF state size (B={args.B}) ===')
    print(f'{"layer":<35s}  {"state shape":<25s}  {"size MB":>8s}  '
          f'{"fits L2?":>9s}')
    print('-' * 85)
    layer_sizes = []
    layer_total = 0
    n_layers = 0
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            if hasattr(m, 'v') and isinstance(m.v, torch.Tensor):
                v = m.v
                size_mb = v.numel() * v.element_size() / 1e6
                fits = 'YES' if size_mb < V100_L2_MB else 'no'
                shape_str = str(tuple(v.shape))
                print(f'{name:<35s}  {shape_str:<25s}  {size_mb:>7.2f}  '
                      f'{fits:>9s}')
                layer_sizes.append((name, size_mb, list(v.shape)))
                layer_total += size_mb
                n_layers += 1
    print(f'{"TOTAL":<35s}  {"":<25s}  {layer_total:>7.2f}  '
          f'(L2 = {V100_L2_MB} MB)')

    # Count layers fitting in L2
    fits_l2 = sum(1 for _, sz, _ in layer_sizes if sz < V100_L2_MB)
    print(f'\nLayers fitting in V100 L2 (6 MB): {fits_l2}/{n_layers}')

    # === Theoretical HBM traffic for LIF update ===
    # LIF: v ← decay*v + input; spike = v >= V_th; v = reset(v, spike)
    # Per timestep, per layer: read v, read input, write v, write spike
    # Optimal HBM traffic = 3 * state_size (read v, write v, write spike)
    #   (input is fused with conv output, doesn't count as separate traffic)
    # If state fits L2 and is reused across T, optimal = 1 read + 1 write per T
    #   = 2 * state_size  (front-loaded)
    # PyTorch / SJ: each timestep launches separate kernel, full HBM r/w
    #   = T * 3 * state_size (per timestep traffic)
    print(f'\n=== Theoretical HBM traffic (single forward, all layers, T={args.T}) ===')
    total_state_bytes = layer_total * 1e6  # MB → bytes
    sj_traffic_GB = args.T * 3 * total_state_bytes / 1e9
    optimal_traffic_GB = 2 * total_state_bytes / 1e9
    print(f'  SJ multi-step (eager per-step):  {sj_traffic_GB:>7.2f} GB')
    print(f'  Optimal (state stays in L2):     {optimal_traffic_GB:>7.2f} GB')
    print(f'  Theoretical reduction:           {sj_traffic_GB/optimal_traffic_GB:>7.1f}×')

    # === Wall-clock measurement of LIF-only path ===
    # Pure LIF micro-bench: 1 LIF layer, multiple timesteps
    print(f'\n=== Pure LIF kernel benchmark (V100 single layer) ===')
    print(f'{"shape (T,B,C,H,W)":<25s}  {"size MB":>8s}  '
          f'{"wall ms":>9s}  {"GB/s eff":>9s}')
    print('-' * 60)
    for shape in [(args.T, args.B, 64, 56, 56),
                  (args.T, args.B, 128, 28, 28),
                  (args.T, args.B, 256, 14, 14),
                  (args.T, args.B, 512, 7, 7)]:
        T_, B_, C_, H_, W_ = shape
        lif = neuron.LIFNode(detach_reset=True).to(device)
        functional.set_step_mode(lif, step_mode='m')
        x_lif = torch.randn(*shape, device=device)
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = lif(x_lif)
                functional.reset_net(lif)
        torch.cuda.synchronize()

        # Measure
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            with torch.no_grad():
                _ = lif(x_lif)
                functional.reset_net(lif)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / N * 1000

        state_bytes_per_timestep = B_ * C_ * H_ * W_ * 4  # fp32
        hbm_bytes = T_ * 3 * state_bytes_per_timestep  # SJ pattern
        gbs_eff = hbm_bytes / (wall / 1000) / 1e9
        size_mb = state_bytes_per_timestep / 1e6
        print(f'{str(shape):<25s}  {size_mb:>7.2f}  {wall:>8.2f}  '
              f'{gbs_eff:>8.1f}')

    print(f'\n=== Verdict ===')
    print(f'V100 HBM peak BW: {V100_HBM_BW_GBs} GB/s')
    print(f'If LIF kernel achieves > 600 GB/s (>67% peak), HBM-bound:')
    print(f'  → state buffer optimization could give 2-3× wins')
    print(f'If LIF kernel achieves < 200 GB/s, NOT HBM-bound:')
    print(f'  → other bottleneck (launch overhead, register), B less attractive')


if __name__ == '__main__':
    main()