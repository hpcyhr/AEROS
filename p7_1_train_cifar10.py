"""
p7_1_train_cifar10.py — generic CIFAR-10 trainer for AEROS S-3 (trained accuracy).

Purpose: GPT P7/P8 review identifies "no trained-checkpoint top-1
accuracy" as a submission-critical gap (S-3). This trainer produces
trained checkpoints for any of the 17 networks listed in
p7_3c_extended_multinet.py, so that bit-exact AEROS preservation can
be reported on real (non-random-weight) inference.

Recipe is borrowed from CATFuse phaseC1_train_sew_cifar10.py:
  - SGD + momentum=0.9 + weight_decay=5e-4
  - cosine LR schedule
  - AMP via torch.cuda.amp.GradScaler
  - rate encoding (replicate static image across T timesteps)
  - mean over T → cross-entropy
  - functional.reset_net after each batch
  - CIFAR-10 32x32, T=4 (matches CATFuse / Chronos / Helios baseline)

Architectures supported:
  Imported from /data/yhr/CATFuse/models/ + SJ stdlib. The same
  registry as p7_3c_extended_multinet.py.

Usage (single net):
    python p7_1_train_cifar10.py --arch SR-18 --epochs 100 -b 128 \\
        --T 4 --lr 0.1 --amp --output-dir checkpoints/

Usage (all nets, sequential):
    bash run_p7_1_all.sh

Time per net (V100, T=4, b=128, AMP): ~2-3 min/epoch
  - 50 epochs: ~2 hours (sufficient for ~85-90% top-1)
  - 100 epochs: ~4 hours (~90-92% top-1)
  - 200 epochs: ~8 hours (best)

For 17 networks at 50 epochs each: ~34 hours total. Doable over 1-2 nights.
For 17 networks at 100 epochs: ~70 hours. Use --epochs 50 for first pass.
"""
import argparse
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# ---- CATFuse path setup ----
CATFUSE_PATH = '/data/yhr/CATFuse'
if CATFUSE_PATH not in sys.path:
    sys.path.insert(0, CATFUSE_PATH)

from spikingjelly.activation_based import (
    functional, layer, neuron, surrogate)
from spikingjelly.activation_based.model import (
    spiking_resnet, sew_resnet, spiking_vgg)


# =============================================================================
# Network builder registry — same as p7_3c
# =============================================================================
def build_network(name, num_classes=10, input_size=32, cifar10_stem=True,
                  v_threshold=1.0, tau=2.0):
    """Build a network by name. Applies CIFAR-10 stem adaptation
    (3x3 stride=1 conv1 + Identity maxpool) for ResNet/SEW families,
    matching CATFuse phaseC1 protocol.

    Surrogate function: ATan (SJ default). Sigmoid causes gradient
    vanishing in deep sequential SNNs (AlexNet/ZFNet/VGG without
    residual connections).
    """
    common_lif = dict(spiking_neuron=neuron.LIFNode,
                      surrogate_function=surrogate.ATan(),
                      detach_reset=True,
                      tau=tau,
                      v_threshold=v_threshold,
                      num_classes=num_classes)

    # === ResNet family ===
    if name == 'SR-18':
        m = spiking_resnet.spiking_resnet18(**common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
    elif name == 'SR-34':
        m = spiking_resnet.spiking_resnet34(**common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
    elif name == 'SR-50':
        m = spiking_resnet.spiking_resnet50(**common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()

    # === SEW-ResNet family ===
    elif name == 'SEW-18':
        m = sew_resnet.sew_resnet18(cnf='ADD', **common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
    elif name == 'SEW-50':
        m = sew_resnet.sew_resnet50(cnf='ADD', **common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
    elif name == 'SEW-101':
        m = sew_resnet.sew_resnet101(cnf='ADD', **common_lif)
        if cifar10_stem:
            m.conv1 = layer.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()

    # === Spiking VGG family (no stem adaptation needed; VGG architecture
    #     already CIFAR-friendly) ===
    elif name == 'VGG-11-BN':
        m = spiking_vgg.spiking_vgg11_bn(**common_lif)
    elif name == 'VGG-13-BN':
        m = spiking_vgg.spiking_vgg13_bn(**common_lif)
    elif name == 'VGG-16-BN':
        m = spiking_vgg.spiking_vgg16_bn(**common_lif)
    elif name == 'VGG-19-BN':
        m = spiking_vgg.spiking_vgg19_bn(**common_lif)

    # === Lightweight CNN (CATFuse wrappers, already CIFAR-aware via input_size) ===
    elif name == 'AlexNet':
        from models.spiking_alexnet import SpikingAlexNet
        m = SpikingAlexNet(num_classes=num_classes,
                           spiking_neuron=neuron.LIFNode,
                           tau=tau, surrogate_function=surrogate.ATan(),
                           detach_reset=True, v_threshold=v_threshold,
                           input_size=input_size)
    elif name == 'ZFNet':
        from models.spiking_zfnet import SpikingZFNet
        m = SpikingZFNet(num_classes=num_classes,
                         spiking_neuron=neuron.LIFNode,
                         tau=tau, surrogate_function=surrogate.ATan(),
                         detach_reset=True, v_threshold=v_threshold,
                         input_size=input_size)
    elif name == 'MobileNet-V1':
        from models.spiking_mobilenet import SpikingMobileNetV1
        m = SpikingMobileNetV1(num_classes=num_classes,
                               spiking_neuron=neuron.LIFNode,
                               tau=tau, surrogate_function=surrogate.ATan(),
                               detach_reset=True, v_threshold=v_threshold,
                               input_size=input_size)

    # === Spike Transformer (use CATFuse-style builders; their internal T is
    #     the model's T, while the trainer's outer T is for rate encoding —
    #     these are different concepts. We pass T=1 to the builder so it
    #     doesn't add its own time replication.) ===
    elif name == 'Spikformer-T':
        from models.spikformer_github import spikformer_cifar_tiny
        m = spikformer_cifar_tiny(spiking_neuron=neuron.LIFNode,
                                   num_classes=num_classes, T=1,
                                   v_threshold=v_threshold)
    elif name == 'Spikformer-S':
        from models.spikformer_github import spikformer_cifar_small
        m = spikformer_cifar_small(spiking_neuron=neuron.LIFNode,
                                    num_classes=num_classes, T=1,
                                    v_threshold=v_threshold)
    elif name == 'QKFormer-T':
        from models.qkformer_github import qkformer_cifar_tiny
        m = qkformer_cifar_tiny(spiking_neuron=neuron.LIFNode,
                                 num_classes=num_classes, T=1,
                                 v_threshold=v_threshold)
    elif name == 'SDTv1-T':
        from models.sdtv1_github import sdtv1_cifar_tiny
        m = sdtv1_cifar_tiny(spiking_neuron=neuron.LIFNode,
                              num_classes=num_classes, T=1,
                              v_threshold=v_threshold)

    else:
        raise ValueError(f'Unknown architecture: {name}')

    # SJ multi-step mode (most networks); transformers handle their own T
    try:
        functional.set_step_mode(m, 'm')
    except Exception:
        pass  # Some transformer architectures manage T internally

    return m


# =============================================================================
# Data
# =============================================================================
def get_dataloaders(data_path, batch_size, num_workers=4):
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        data_path, train=True, transform=train_tfm, download=True)
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=True)
    return train_loader, test_loader


def encode(x, T, arch):
    """Direct (rate) encoding: replicate static image across T time steps.
    [B, C, H, W] -> [T, B, C, H, W]

    Spike Transformers (Spikformer/QKFormer/SDTv1) typically expect
    [B, C, H, W] without explicit T (they replicate internally). We
    detect that pattern by arch name and handle accordingly.
    """
    if arch in ('Spikformer-T', 'Spikformer-S',
                'QKFormer-T', 'SDTv1-T'):
        # Transformer builders expect [B, C, H, W]; their forward replicates
        # internally if they need T. Just return the 4D tensor.
        return x
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


def model_forward(net, x_encoded, arch, T):
    """Run forward and return [B, num_classes] logits.

    For SJ multi-step models with [T, B, C, H, W] input → mean over T.
    For Spike Transformers with [B, C, H, W] input → direct output.
    """
    if arch in ('Spikformer-T', 'Spikformer-S',
                'QKFormer-T', 'SDTv1-T'):
        out = net(x_encoded)
        # Some transformer outputs are [T, B, num_classes], some [B, num_classes];
        # Try to handle both
        if out.dim() == 3:
            return out.mean(dim=0)
        return out
    out = net(x_encoded)  # [T, B, num_classes]
    return out.mean(dim=0)


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(net, loader, criterion, optimizer, scaler, T, device,
                    epoch, arch):
    net.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for batch_idx, (img, label) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            x = encode(img, T, arch)
            out = model_forward(net, x, arch, T)
            loss = criterion(out, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(net)
        total += label.size(0)
        correct += (out.argmax(1) == label).sum().item()
        loss_sum += loss.item() * label.size(0)

    elapsed = time.time() - t0
    print(f'  Ep {epoch:3d} train: loss={loss_sum/total:.4f} '
          f'acc={100*correct/total:.2f}% time={elapsed:.1f}s', flush=True)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(net, loader, T, device, arch):
    net.eval()
    total, correct = 0, 0
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        x = encode(img, T, arch)
        out = model_forward(net, x, arch, T)
        functional.reset_net(net)
        total += label.size(0)
        correct += (out.argmax(1) == label).sum().item()
    return correct / total


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, required=True,
                    help='Architecture name (SR-18, SEW-18, VGG-11-BN, ...)')
    ap.add_argument('--data-path', type=str,
                    default='/data/yhr/datasets/cifar10')
    ap.add_argument('--output-dir', type=str, default='checkpoints/')
    ap.add_argument('--epochs', type=int, default=50,
                    help='50 = quick (~2h, ~85-88% top-1); 100 = better; 200 = best')
    ap.add_argument('-b', '--batch-size', type=int, default=128)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--tau', type=float, default=2.0)
    ap.add_argument('--v-threshold', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--cifar10-stem', action='store_true', default=True)
    ap.add_argument('--no-cifar10-stem', dest='cifar10_stem',
                    action='store_false')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f'\n=== P7-1 Training: {args.arch} on CIFAR-10 ===')
    print(f'  Args: {vars(args)}')
    print(f'  Start: {datetime.datetime.now().isoformat()}', flush=True)

    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(
        args.data_path, args.batch_size, args.workers)

    # Build network
    net = build_network(args.arch,
                        num_classes=10,
                        input_size=32,
                        cifar10_stem=args.cifar10_stem,
                        v_threshold=args.v_threshold,
                        tau=args.tau).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'  Model: {args.arch}, params={n_params/1e6:.2f}M, T={args.T}',
          flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_acc = 0.0
    history = []
    arch_safe = args.arch.replace('-', '_').replace('/', '_')
    best_ckpt_path = Path(args.output_dir) / f'{arch_safe}_cifar10_best.pth'

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            net, train_loader, criterion, optimizer, scaler,
            args.T, device, epoch, args.arch)
        scheduler.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            test_acc = evaluate(net, test_loader, args.T, device, args.arch)
            print(f'  Ep {epoch:3d}  test={100*test_acc:.2f}% '
                  f'(best={100*best_acc:.2f}%)', flush=True)
            history.append({'epoch': epoch, 'train_acc': train_acc,
                           'test_acc': test_acc, 'lr': scheduler.get_last_lr()[0]})

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'args': vars(args),
                    'n_params': n_params,
                }, best_ckpt_path)
                print(f'           -> saved best checkpoint to {best_ckpt_path}',
                      flush=True)

    final_test_acc = evaluate(net, test_loader, args.T, device, args.arch)
    final_path = Path(args.output_dir) / f'{arch_safe}_cifar10_final.pth'
    torch.save({
        'epoch': args.epochs - 1,
        'arch': args.arch,
        'model_state_dict': net.state_dict(),
        'test_acc': final_test_acc,
        'history': history,
        'args': vars(args),
        'n_params': n_params,
    }, final_path)

    # Append to global summary JSON
    summary_path = Path(args.output_dir) / 'p7_1_summary.json'
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    summary[args.arch] = {
        'best_acc': float(best_acc),
        'final_acc': float(final_test_acc),
        'n_params': int(n_params),
        'epochs': args.epochs,
        'T': args.T,
        'best_ckpt': str(best_ckpt_path),
        'final_ckpt': str(final_path),
        'completed_at': datetime.datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f'\n=== {args.arch} training complete ===')
    print(f'  Best test acc:  {100*best_acc:.2f}%')
    print(f'  Final test acc: {100*final_test_acc:.2f}%')
    print(f'  End: {datetime.datetime.now().isoformat()}')


if __name__ == '__main__':
    main()