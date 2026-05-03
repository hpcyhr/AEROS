"""
AEROS Phase 0 — train DVS Gesture SNN to literature-grade accuracy (95%+).
Replaces 80% baseline. Architecture matches SJ classify_dvsg.py reference.
"""
import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame  # for sanity
from spikingjelly.activation_based.model.parametric_lif_net import DVSGestureNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/dvs128_gesture')
    p.add_argument('--out_dir',  default='./train_out_dvsg_v2')
    p.add_argument('--T', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='use mixed precision')
    p.add_argument('--cupy', action='store_true',
                   help='use SpikingJelly cupy backend (faster forward on V100)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f'Loading DVS128 Gesture (T={args.T})...')
    train_set = DVS128Gesture(args.data_dir, train=True,  data_type='frame',
                              frames_number=args.T, split_by='number')
    test_set  = DVS128Gesture(args.data_dir, train=False, data_type='frame',
                              frames_number=args.T, split_by='number')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f'  train: {len(train_set)}, test: {len(test_set)}')

    # Use SJ's reference DVSGestureNet (8 conv blocks, parametric-LIF, well-tuned).
    # This architecture has been documented to reach ~97% on DVS Gesture in
    # SpikingJelly examples.
    net = DVSGestureNet(channels=args.channels,
                        spiking_neuron=neuron.LIFNode,
                        surrogate_function=surrogate.ATan(),
                        detach_reset=True).to(device)
    functional.set_step_mode(net, step_mode='m')
    if args.cupy:
        functional.set_backend(net, backend='cupy')
        print('  using SJ cupy backend')

    print(f'  params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M')

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_acc = 0.0
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        tot, correct, loss_sum = 0, 0, 0.0
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True).float()
            frames = frames.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,...] -> [T,B,...]
            labels = labels.to(device, non_blocking=True)
            label_onehot = F.one_hot(labels, 11).float()

            opt.zero_grad()
            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = net(frames).mean(0)        # [B, 11]
                    loss = F.mse_loss(logits, label_onehot)  # SJ reference uses MSE
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = net(frames).mean(0)
                loss = F.mse_loss(logits, label_onehot)
                loss.backward()
                opt.step()
            functional.reset_net(net)

            loss_sum += loss.item() * labels.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            tot      += labels.size(0)

        train_loss = loss_sum / tot
        train_acc  = correct / tot

        # eval
        net.eval()
        tot_e, correct_e = 0, 0
        with torch.no_grad():
            for frames, labels in test_loader:
                frames = frames.to(device, non_blocking=True).float()
                frames = frames.permute(1, 0, 2, 3, 4).contiguous()
                labels = labels.to(device, non_blocking=True)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        logits = net(frames).mean(0)
                else:
                    logits = net(frames).mean(0)
                functional.reset_net(net)
                correct_e += (logits.argmax(1) == labels).sum().item()
                tot_e     += labels.size(0)
        test_acc = correct_e / tot_e
        sched.step()

        dt = time.time() - t0
        print(f'epoch {ep+1:3d}/{args.epochs}  '
              f'train loss={train_loss:.4f} acc={train_acc*100:.2f}%  '
              f'test acc={test_acc*100:.2f}%  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  ({dt:.1f}s)')

        torch.save({'net': net.state_dict(), 'epoch': ep, 'test_acc': test_acc},
                   os.path.join(args.out_dir, 'checkpoint_latest.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'net': net.state_dict(), 'epoch': ep, 'test_acc': test_acc},
                       os.path.join(args.out_dir, 'checkpoint_max.pth'))
            print(f'  >> new best: {best_acc*100:.2f}%')

    print(f'\nDone. Best test acc: {best_acc*100:.2f}%')


if __name__ == '__main__':
    main()