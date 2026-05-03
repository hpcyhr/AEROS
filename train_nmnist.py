"""
AEROS Phase 0 — train a small SNN on N-MNIST.
Purpose: produce a non-degenerate ckpt so measure_density_nmnist.py
runs against a real trained network. The resulting density numbers
are still NOT a valid AEROS verdict (N-MNIST is too sparse and
not representative of event-camera workloads).
"""
import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from spikingjelly.datasets.n_mnist import NMNIST


class NMNISTSpikingNet(nn.Module):
    """Must match the architecture in measure_density_nmnist.py exactly."""
    def __init__(self, num_classes=10, channels=64):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            layer.Conv2d(2, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(c, c, 3, padding=1, bias=False),
            layer.BatchNorm2d(c),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(c, c * 2, 3, padding=1, bias=False),
            layer.BatchNorm2d(c * 2),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Linear(c * 2 * 4 * 4, c * 2, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(c * 2, num_classes, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        return self.net(x).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/yhr/datasets/n_mnist')
    p.add_argument('--out_dir', default='./train_out_nmnist')
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f'Loading N-MNIST (T={args.T})...')
    train_set = NMNIST(args.data_dir, train=True,  data_type='frame',
                       frames_number=args.T, split_by='number')
    test_set  = NMNIST(args.data_dir, train=False, data_type='frame',
                       frames_number=args.T, split_by='number')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f'  train: {len(train_set)}, test: {len(test_set)}')

    net = NMNISTSpikingNet(num_classes=10, channels=args.channels).to(device)
    print(f'  params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M')

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for ep in range(args.epochs):
        # train
        net.train()
        t0 = time.time()
        tot, correct, loss_sum = 0, 0, 0.0
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True).float()
            frames = frames.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,...] -> [T,B,...]
            labels = labels.to(device, non_blocking=True)

            logits = net(frames)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
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
                logits = net(frames)
                functional.reset_net(net)
                correct_e += (logits.argmax(1) == labels).sum().item()
                tot_e     += labels.size(0)
        test_acc = correct_e / tot_e
        sched.step()

        dt = time.time() - t0
        print(f'epoch {ep+1:2d}/{args.epochs}  '
              f'train loss={train_loss:.4f} acc={train_acc*100:.2f}%  '
              f'test acc={test_acc*100:.2f}%  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  ({dt:.1f}s)')

        # save
        torch.save({'net': net.state_dict(), 'epoch': ep, 'test_acc': test_acc},
                   os.path.join(args.out_dir, 'checkpoint_latest.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'net': net.state_dict(), 'epoch': ep, 'test_acc': test_acc},
                       os.path.join(args.out_dir, 'checkpoint_max.pth'))
            print(f'  >> new best: {best_acc*100:.2f}%')

    print(f'\nDone. Best test acc: {best_acc*100:.2f}%')
    print(f'Best ckpt: {os.path.join(args.out_dir, "checkpoint_max.pth")}')


if __name__ == '__main__':
    main()