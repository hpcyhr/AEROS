# measure_density_dvsg_v2.py — 与 train_dvsg_v2.py 架构对齐
import argparse, os
import numpy as np
import torch
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.activation_based.model.parametric_lif_net import DVSGestureNet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/dvs128_gesture')
    p.add_argument('--ckpt', default='./train_out_dvsg_v2/checkpoint_max.pth')
    p.add_argument('--T', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_batches', type=int, default=16)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--out', default='density_dvsg_v2.npz')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)

    print(f'Loading DVS128 Gesture (test, T={args.T})...')
    ds = DVS128Gesture(args.data_dir, train=False, data_type='frame',
                      frames_number=args.T, split_by='number')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    net = DVSGestureNet(channels=args.channels,
                        spiking_neuron=neuron.LIFNode,
                        surrogate_function=surrogate.ATan(),
                        detach_reset=True).to(device)
    functional.set_step_mode(net, step_mode='m')
    sd = torch.load(args.ckpt, map_location=device)
    sd = sd.get('net', sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f'Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}, '
          f'test_acc(at save)={sd.get("test_acc", "n/a")}')
    net.eval()

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
                        d = out.float().reshape(T_, B_, -1).mean(dim=2)
                        records[k].append(d.cpu().numpy())
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))
            lif_idx += 1
    print(f'Hooked {len(handles)} LIF layers')

    with torch.no_grad():
        for i, (frames, _) in enumerate(loader):
            if i >= args.num_batches: break
            frames = frames.to(device).float()
            frames = frames.permute(1, 0, 2, 3, 4).contiguous()
            _ = net(frames)
            functional.reset_net(net)
            if (i+1) % 4 == 0:
                print(f'  batch {i+1}/{args.num_batches}')

    out = {k: np.concatenate(v, axis=1) for k, v in records.items()}
    np.savez(args.out, **out)
    print(f'Saved {args.out}')

    print(f'\n{"layer":8s} {"mean":>8s} {"p10":>8s} {"p50":>8s} {"p90":>8s} {"<10%":>7s}')
    print('-' * 60)
    for k, a in out.items():
        f10 = (a < 0.10).mean()
        print(f'{k:8s} {a.mean():8.4f} {np.percentile(a,10):8.4f} '
              f'{np.percentile(a,50):8.4f} {np.percentile(a,90):8.4f} '
              f'{f10*100:6.1f}%')


if __name__ == '__main__':
    main()