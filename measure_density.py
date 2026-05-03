"""
AEROS Phase 0: per-layer per-(t,b) spike density on DVS Gesture.

Goal: decide whether AEROS's density-adaptive execution has room to
beat densify→cuDNN. If most (layer, timestep) pairs have density
below ~10%, event-driven path has theoretical headroom. If density
is uniformly high, AEROS bet fails and we kill before further work.
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


class SpikingConvNet(nn.Module):
    """Small 4-conv SNN for DVS128 (2x128x128 → 11 classes).
    Architecture matches SJ's classify_dvsg.py closely enough that
    its trained checkpoint will load with state_dict matching."""
    def __init__(self, num_classes=11, channels=128):
        super().__init__()
        c = channels
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(c, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(c, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(c, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Linear(c * 8 * 8, c * 4, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(c * 4, num_classes, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        functional.set_step_mode(self, step_mode='m')
    def forward(self, x):
        # x: [T, B, 2, 128, 128]
        return self.conv_fc(x).mean(0)  # [B, num_classes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/dvs128_gesture')
    p.add_argument('--ckpt', default=None, help='trained checkpoint; STRONGLY recommended')
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_batches', type=int, default=64)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--out', default='aeros_density.npz')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)

    print(f'Loading DVS128 Gesture (test split, T={args.T})...')
    ds = DVS128Gesture(args.data_dir, train=False, data_type='frame',
                      frames_number=args.T, split_by='number')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    net = SpikingConvNet(num_classes=11, channels=args.channels).to(device)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        sd = sd.get('net', sd)  # SJ saves under 'net' key in some scripts
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f'Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}')
    else:
        print('WARNING: no checkpoint loaded. Random init density is NOT a valid '
              'proxy for trained SNN — verdict will be biased toward false negative.')
    net.eval()

    # Hooks on each LIF node
    records = {}  # layer_name -> list of [T, B] numpy arrays
    handles = []
    lif_idx = 0
    for name, mod in net.named_modules():
        if isinstance(mod, neuron.LIFNode):
            key = f'lif{lif_idx:02d}'
            records[key] = []
            def make_hook(k):
                def hook(module, inp, out):
                    # out: [T, B, ...] in multi-step mode
                    with torch.no_grad():
                        T_, B_ = out.shape[0], out.shape[1]
                        d = out.float().reshape(T_, B_, -1).mean(dim=2)  # [T, B]
                        records[k].append(d.cpu().numpy())
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))
            lif_idx += 1
    print(f'Hooked {len(handles)} LIF layers')

    # Profile
    with torch.no_grad():
        for i, (frames, _) in enumerate(loader):
            if i >= args.num_batches: break
            frames = frames.to(device).float()
            # SJ DVS128 frame loader returns [B, T, C, H, W]; permute to [T, B, C, H, W]
            frames = frames.permute(1, 0, 2, 3, 4).contiguous()
            _ = net(frames)
            functional.reset_net(net)
            if (i+1) % 10 == 0:
                print(f'  batch {i+1}/{args.num_batches}')

    # Save: each layer becomes a [num_batches, T, B] array → flatten last 2 dims
    out = {k: np.concatenate(v, axis=1) for k, v in records.items()}  # [T, total_B]
    np.savez(args.out, **out)
    print(f'Saved {args.out}')

    # Quick summary
    print(f'\n{"layer":8s} {"mean":>8s} {"p10":>8s} {"p50":>8s} {"p90":>8s} {"<10%":>7s}')
    print('-' * 60)
    for k, a in out.items():
        f10 = (a < 0.10).mean()
        print(f'{k:8s} {a.mean():8.4f} {np.percentile(a,10):8.4f} '
              f'{np.percentile(a,50):8.4f} {np.percentile(a,90):8.4f} '
              f'{f10*100:6.1f}%')

if __name__ == '__main__':
    main()