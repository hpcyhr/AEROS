#!/usr/bin/env python
"""
AEROS Phase 2 Exp 12 — DVSGestureNet training on DVS128 Gesture.

Trains the SpikingJelly DVSGestureNet baseline on DVS128 Gesture frames.
Based on SJ classify_dvsg tutorial. Single GPU, AMP, cosine LR.

Expected performance:
    - 50-100 epochs to reach ~95% test accuracy
    - V100: ~5-7 min per epoch at b=16, T=16 (1176 train samples)
    - Total: ~6-10 hr to convergence

Outputs:
    - /data/yhr/AEROS/checkpoints/dvs128_gesture_best.pth   (highest val acc)
    - /data/yhr/AEROS/checkpoints/dvs128_gesture_final.pth  (last epoch)

Usage:
    python p9_12_dvs_train.py \\
        --data_root /data/yhr/AEROS/DVS128Gesture \\
        --out_dir /data/yhr/AEROS/checkpoints \\
        --epochs 64 --batch_size 16 --T 16 --lr 0.1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/yhr/AEROS/DVS128Gesture")
    parser.add_argument("--out_dir", default="/data/yhr/AEROS/checkpoints")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    # Imports inside main so a missing SJ doesn't break --help
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.parametric_lif_net import (
        DVSGestureNet)
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    print(f"=== AEROS Phase 2 Exp 12 — DVS128 Gesture Training ===")
    print(f"  Data root: {args.data_root}")
    print(f"  Out dir:   {args.out_dir}")
    print(f"  Epochs={args.epochs}  batch={args.batch_size}  T={args.T}")
    print(f"  channels={args.channels}  lr={args.lr}  AMP={not args.no_amp}")

    # ------------------------ Data ------------------------
    print(f"\n[setup] Loading DVS128 Gesture frames at T={args.T}...")
    train_set = DVS128Gesture(
        root=args.data_root, train=True, data_type="frame",
        frames_number=args.T, split_by="number",
    )
    test_set = DVS128Gesture(
        root=args.data_root, train=False, data_type="frame",
        frames_number=args.T, split_by="number",
    )
    print(f"         train={len(train_set)}  test={len(test_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ------------------------ Model ------------------------
    net = DVSGestureNet(
        channels=args.channels,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    ).to(device)
    functional.set_step_mode(net, "m")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n[model] DVSGestureNet  channels={args.channels}  "
          f"params={n_params/1e6:.2f}M")

    # ------------------------ Optimizer ------------------------
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=args.momentum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=not args.no_amp)

    # ------------------------ Train loop ------------------------
    best_acc = 0.0
    log_path = os.path.join(args.out_dir, "dvs128_gesture_train.log")
    log_f = open(log_path, "a")
    log_f.write(f"\n=== Run @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        # Train
        net.train()
        train_loss_sum = 0.0
        train_acc_sum = 0
        train_n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float()  # [B, T, 2, 128, 128]
            y = y.to(device, non_blocking=True)
            x = x.transpose(0, 1)  # [T, B, 2, 128, 128]

            optimizer.zero_grad()
            functional.reset_net(net)
            with autocast(enabled=not args.no_amp):
                y_hat = net(x).mean(dim=0)  # [B, 11]  (mean over T)
                y_one_hot = F.one_hot(y, num_classes=11).float()
                loss = F.mse_loss(y_hat, y_one_hot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.shape[0]
            train_acc_sum += (y_hat.argmax(1) == y).sum().item()
            train_n += y.shape[0]

        lr_scheduler.step()

        # Eval
        net.eval()
        test_acc_sum = 0
        test_n = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True).float().transpose(0, 1)
                y = y.to(device, non_blocking=True)
                functional.reset_net(net)
                y_hat = net(x).mean(dim=0)
                test_acc_sum += (y_hat.argmax(1) == y).sum().item()
                test_n += y.shape[0]

        train_loss = train_loss_sum / train_n
        train_acc = train_acc_sum / train_n
        test_acc = test_acc_sum / test_n
        elapsed = time.time() - t0

        msg = (f"epoch {epoch:3d} | loss {train_loss:.4f}  "
               f"train_acc {train_acc*100:5.2f}%  test_acc {test_acc*100:5.2f}%  "
               f"lr {optimizer.param_groups[0]['lr']:.4f}  "
               f"({elapsed:.0f}s)")
        print(msg)
        log_f.write(msg + "\n"); log_f.flush()

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "test_acc": test_acc,
                "T": args.T,
                "channels": args.channels,
            }, os.path.join(args.out_dir, "dvs128_gesture_best.pth"))
            print(f"  [best] test_acc {test_acc*100:.2f}% -> saved")

    # Save final
    torch.save({
        "epoch": args.epochs - 1,
        "state_dict": net.state_dict(),
        "test_acc": test_acc,
        "T": args.T,
        "channels": args.channels,
    }, os.path.join(args.out_dir, "dvs128_gesture_final.pth"))

    log_f.write(f"=== Done. Best test_acc {best_acc*100:.2f}% ===\n")
    log_f.close()
    print(f"\n=== Training done. Best test_acc: {best_acc*100:.2f}% ===")
    print(f"Best ckpt: {args.out_dir}/dvs128_gesture_best.pth")
    print(f"Final ckpt: {args.out_dir}/dvs128_gesture_final.pth")


if __name__ == "__main__":
    main()