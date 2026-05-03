"""
AEROS Phase 1a: theoretical FLOP saving from event-driven path.

For each conv/fc layer, compute:
  dense_FLOPs   = output_elements * input_channel_window * kernel_volume * 2 (mul+add)
  effective_FLOPs (event) = dense_FLOPs * input_density
  speedup_ceiling = 1 / input_density   (assuming perfect event kernel)

This is the upper bound. Real Triton/CUDA kernels won't hit this number
because of overhead (gather/scatter, irregular memory access, kernel
launch). Phase 1b measures how close a naive PyTorch implementation
gets; Phase 1c measures how close a Triton implementation gets.

Layer index convention (matches DVSGestureSpikingNet):
  conv0:  input frames (event camera)        -> lif00
  conv1:  lif00 output                       -> lif01
  conv2:  lif01 output                       -> lif02
  conv3:  lif02 output                       -> lif03
  conv4:  lif03 output                       -> lif04
  fc1  :  lif04 output (flattened)           -> lif05
  fc2  :  lif05 output                       -> lif06
"""
import argparse, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inp', default='density_dvsg_trained.npz')
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--C', type=int, default=128)
    p.add_argument('--H', type=int, default=128)
    p.add_argument('--input_density', type=float, default=0.02,
                   help='estimated density of input event frames; ~2% for DVS128')
    args = p.parse_args()

    d = np.load(args.inp)

    # input density per LAYER = density of LIF before it
    # for conv0 there's no LIF before, use input_density estimate
    layer_specs = [
        # name,    Cin, Cout, H,  W,  K,  in_density_source
        ('conv0',  2,   args.C,  128, 128, 3, ('input', args.input_density)),
        ('conv1',  args.C, args.C,    64,  64,  3, ('lif', 'lif00')),
        ('conv2',  args.C, args.C,    32,  32,  3, ('lif', 'lif01')),
        ('conv3',  args.C, args.C,    16,  16,  3, ('lif', 'lif02')),
        ('conv4',  args.C, args.C,     8,   8,  3, ('lif', 'lif03')),
        ('fc1',    args.C * 4 * 4, args.C * 4, 1, 1, 1, ('lif', 'lif04')),
        ('fc2',    args.C * 4, 11, 1, 1, 1, ('lif', 'lif05')),
    ]

    print(f'{"layer":7s} {"Cin":>5s} {"Cout":>5s} {"H":>4s} {"K":>3s} '
          f'{"in_den":>8s} {"FLOPs":>12s} {"eff_FLOPs":>12s} {"ceil×":>8s}')
    print('-' * 80)

    total_dense = 0
    total_effective = 0
    for (name, Cin, Cout, H, W, K, src) in layer_specs:
        if src[0] == 'input':
            in_density = src[1]
        else:
            in_density = float(d[src[1]].mean())

        # FLOPs per timestep, summed over spatial positions
        # = Cout * H * W * Cin * K * K * 2 (mul + add)
        dense_flops = Cout * H * W * Cin * K * K * 2
        # event-driven: only nonzero input positions contribute
        # for conv: each nonzero (Cin, h, w) generates K*K*Cout MACs
        # average nonzero count = Cin * H * W * in_density
        # each nonzero contributes K*K*Cout*2 FLOPs
        eff_flops = (Cin * H * W * in_density) * (K * K * Cout * 2)

        ceil_speedup = dense_flops / max(eff_flops, 1)

        total_dense += dense_flops * args.T
        total_effective += eff_flops * args.T

        print(f'{name:7s} {Cin:5d} {Cout:5d} {H:4d} {K:3d} '
              f'{in_density:8.4f} {dense_flops/1e6:10.2f}M '
              f'{eff_flops/1e6:10.2f}M {ceil_speedup:7.1f}x')

    print('-' * 80)
    print(f'Total over T={args.T}:')
    print(f'  Dense    : {total_dense/1e9:.2f} GFLOPs')
    print(f'  Event    : {total_effective/1e9:.2f} GFLOPs')
    print(f'  Ceiling  : {total_dense/total_effective:.2f}x end-to-end')
    print()
    print('=== Interpretation ===')
    overall = total_dense / total_effective
    if overall >= 5.0:
        print(f'STRONG: end-to-end ceiling {overall:.1f}x. Even at 30% kernel efficiency,')
        print(f'AEROS can deliver {overall*0.3:.1f}x. Worth pursuing Phase 1b.')
    elif overall >= 2.5:
        print(f'MODERATE: end-to-end ceiling {overall:.1f}x. Need >50% kernel efficiency')
        print(f'to beat dense baseline meaningfully. Phase 1b risky but viable.')
    else:
        print(f'WEAK: end-to-end ceiling {overall:.1f}x. Even perfect event kernel')
        print(f'barely beats cuDNN. Reconsider AEROS direction.')


if __name__ == '__main__':
    main()