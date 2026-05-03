"""
AEROS Phase 0 SMOKE TEST on N-MNIST.

This is for verifying the measure→analyze pipeline runs end-to-end.
DO NOT use the density numbers from this run to decide AEROS go/no-go:
  - N-MNIST is sparser than typical event-camera workloads
  - Without a trained ckpt, LIF firing rates are biased
The real verdict comes from DVS Gesture (or another event-camera dataset)
with a trained checkpoint.
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.datasets.n_mnist import NMNIST


class NMNISTSpikingNet(nn.Module):
    """3-conv + 2-fc SNN for N-MNIST [T, B, 2, 34, 34] -> 10 classes."""
    def __init__(self, num_classes=10, channels=64):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            layer.Conv2d(2, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),                                    # 34 -> 17

            layer.Conv2d(c, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),                                    # 17 -> 8

            layer.Conv2d(c, c * 2, 3, padding=1, bias=False),
            layer.BatchNorm2d(c * 2),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),                                    # 8 -> 4

            layer.Flatten(),
            layer.Linear(c * 2 * 4 * 4, c * 2, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(c * 2, num_classes, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # x: [T, B, 2, 34, 34]
        return self.net(x).mean(0)  # [B, num_classes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/n_mnist')
    p.add_argument('--ckpt', default=None, help='trained checkpoint (optional)')
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_batches', type=int, default=32)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--out', default='density_nmnist.npz')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)

    print(f'Loading N-MNIST (test split, T={args.T})...')
    ds = NMNIST(args.data_dir, train=False, data_type='frame',
                frames_number=args.T, split_by='number')
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True)

    net = NMNISTSpikingNet(num_classes=10, channels=args.channels).to(device)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        sd = sd.get('net', sd)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f'Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}')
    else:
        print('WARNING: no checkpoint loaded. Random-init density is BIASED.')
        print('         This run is for pipeline verification only.')
    net.eval()

    # Hook every LIF
    records = {}
    handles = []
    lif_idx = 0
    for name, mod in net.named_modules():
        if isinstance(mod, neuron.LIFNode):
            key = f'lif{lif_idx:02d}'
            records[key] = []

            def make_hook(k):
                def hook(module, inp, out):
                    with torch.no_grad():
                        T_, B_ = out.shape[0], out.shape[1]
                        d = out.float().reshape(T_, B_, -1).mean(dim=2)  # [T, B]
                        records[k].append(d.cpu().numpy())
                return hook

            handles.append(mod.register_forward_hook(make_hook(key)))
            lif_idx += 1
    print(f'Hooked {len(handles)} LIF layers')

    with torch.no_grad():
        for i, (frames, _) in enumerate(loader):
            if i >= args.num_batches:
                break
            frames = frames.to(device).float()
            # SJ NMNIST frame loader returns [B, T, C, H, W]; permute to [T, B, C, H, W]
            frames = frames.permute(1, 0, 2, 3, 4).contiguous()
            _ = net(frames)
            functional.reset_net(net)
            if (i + 1) % 8 == 0:
                print(f'  batch {i + 1}/{args.num_batches}')

    out = {k: np.concatenate(v, axis=1) for k, v in records.items()}  # [T, total_B]
    np.savez(args.out, **out)
    print(f'Saved {args.out}')

    print(f'\n{"layer":8s} {"mean":>8s} {"p10":>8s} {"p50":>8s} {"p90":>8s} {"<10%":>7s}')
    print('-' * 60)
    for k, a in out.items():
        f10 = (a < 0.10).mean()
        print(f'{k:8s} {a.mean():8.4f} {np.percentile(a, 10):8.4f} '
              f'{np.percentile(a, 50):8.4f} {np.percentile(a, 90):8.4f} '
              f'{f10 * 100:6.1f}%')


if __name__ == '__main__':
    main()