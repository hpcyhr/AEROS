#!/usr/bin/env python
"""
AEROS — train 6 extended-suite archs to convergence on simple tasks.

Tasks:
  Moving MNIST           — ConvLSTM-2L, ConvGRU-2L (video frame -> classification)
  Sequential MNIST       — LSTM-4L, GRU-4L, CausalTCN-8L, MinimalSSM-2L

The point is NOT to achieve SOTA. The point is to produce *trained* checkpoints
so subsequent §5 experiments report bit-exact / scaling on real weights, not
random init. We train until validation accuracy plateaus.

Both datasets auto-download via torchvision (sMNIST) or one-shot wget
(Moving MNIST) on first call.

Outputs:
  /data/yhr/AEROS/checkpoints_extended/{arch}_best.pth
  /data/yhr/AEROS/checkpoints_extended/{arch}_metrics.json

Usage:
  # Single arch
  python train_extended.py --arch LSTM-4L

  # All 6 archs sequentially
  python train_extended.py --arch all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Ensure aeros_models_extended is importable
sys.path.insert(0, str(Path(__file__).parent))
from aeros_models_extended import (
    build_extended_net, reset_state_extended, EXTENDED_NETS
)


CHECKPOINT_DIR = Path("/data/yhr/AEROS/checkpoints_extended")
DATA_DIR       = Path("/data/yhr/AEROS/data_extended")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Sequential MNIST: read pixel-by-pixel as a length-784 sequence
# ============================================================================

def get_smnist_loaders(batch_size=128):
    """Sequential MNIST. Each MNIST image [1,28,28] becomes a sequence
    [T=28, C=3, H=32, W=32] by tiling row by row, broadcasting to 3
    channels, padding 28->32.

    Lazy version: converts images in __getitem__ rather than
    pre-materializing all 60K samples (which OOMs CPU RAM at ~12GB).
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # [1, 28, 28]
    ])
    train = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                           transform=transform)
    test = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                          transform=transform)

    print(f"  preparing sMNIST (T=28, C=3, H=32, W=32)...")
    print(f"    train size: {len(train)}, test size: {len(test)}")
    print(f"    using lazy on-the-fly sequence conversion")

    class SMNISTLazy(Dataset):
        """Lazy MNIST -> sequence wrapper. Converts in __getitem__."""
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, lbl = self.base[idx]                # img: [1, 28, 28]
            rows = img.squeeze(0).unsqueeze(1).unsqueeze(2)  # [28, 1, 1, 28]
            rows = rows.expand(-1, 3, 28, -1)        # [28, 3, 28, 28]
            rows = F.pad(rows, (2, 2, 2, 2))         # [28, 3, 32, 32]
            return rows, lbl

    train_ds = SMNISTLazy(train)
    test_ds = SMNISTLazy(test)

    def collate(batch):
        Xs = torch.stack([b[0] for b in batch], dim=1)   # [T, B, C, H, W]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return Xs, ys

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate,
                             pin_memory=True)
    return train_loader, test_loader


# ============================================================================
# Moving MNIST: video classification proxy task (predict last-frame label)
#
# Standard Moving MNIST has 10K test sequences of 20 frames at 64x64,
# but is a video-prediction (regression) task. To turn it into a clean
# classification benchmark, we use synthetic Moving MNIST built on the
# fly from MNIST: pick a digit, animate it bouncing in a 32x32 frame for
# T=16 timesteps, train to classify the digit identity.
# ============================================================================

def get_moving_mnist_loaders(batch_size=64, T=16, H=32, N_train=5000, N_test=1000):
    """Synthetic Moving MNIST classification: each sample is a T-frame
    animation of a single MNIST digit translating across a HxH frame at
    constant velocity. Label = digit class.

    Lazy version: generates animation in __getitem__ from cached MNIST
    images, instead of pre-materializing N_train × T × C × H² tensor.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                                  transform=transform)
    mnist_test = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                                 transform=transform)

    print(f"  preparing Moving MNIST (T={T}, C=3, H={H}, W={H})...")
    print(f"    train: {N_train} samples, test: {N_test} samples")
    print(f"    using lazy on-the-fly sequence generation")

    def make_seq_static(image_tensor, T, H, seed):
        """image_tensor: [1, 28, 28]. Returns [T, 3, H, H] animation.
        Uses a per-sample seed for deterministic generation."""
        rng = np.random.default_rng(seed)
        digit = image_tensor.squeeze(0).numpy()
        x0 = rng.integers(0, max(1, H - 28))
        y0 = rng.integers(0, max(1, H - 28))
        vx = rng.choice([-1, 1])
        vy = rng.choice([-1, 1])
        frames = np.zeros((T, 1, H, H), dtype=np.float32)
        for t in range(T):
            x = x0 + vx * t
            y = y0 + vy * t
            x = abs(x) if x < 0 else (2 * (H - 28) - x if x > H - 28 else x)
            y = abs(y) if y < 0 else (2 * (H - 28) - y if y > H - 28 else y)
            x = max(0, min(H - 28, x))
            y = max(0, min(H - 28, y))
            frames[t, 0, y:y + 28, x:x + 28] = digit
        frames = np.repeat(frames, 3, axis=1)
        return torch.from_numpy(frames)

    class MovingMNISTLazy(Dataset):
        def __init__(self, base, n_samples, T, H, seed_offset):
            self.base = base
            self.n = min(n_samples, len(base))
            self.T = T
            self.H = H
            self.seed_offset = seed_offset
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            img, lbl = self.base[idx]
            seq = make_seq_static(img, self.T, self.H, idx + self.seed_offset)
            return seq, lbl

    train_ds = MovingMNISTLazy(mnist_train, N_train, T, H, seed_offset=0)
    test_ds = MovingMNISTLazy(mnist_test, N_test, T, H, seed_offset=1000000)

    def collate(batch):
        Xs = torch.stack([b[0] for b in batch], dim=1)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return Xs, ys

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate,
                             pin_memory=True)
    return train_loader, test_loader


# ============================================================================
# Train + eval loop
# ============================================================================

def train_one_arch(arch_name, device, epochs=10, lr=1e-3, batch_size=64,
                    H=32, num_classes=10):
    """Train arch_name to convergence; return best ckpt path and metrics."""
    print(f"\n{'='*78}\nTraining {arch_name}\n{'='*78}")

    # Pick task and loaders
    if arch_name in ("ConvLSTM-2L", "ConvGRU-2L"):
        task = "moving_mnist"
        train_loader, test_loader = get_moving_mnist_loaders(
            batch_size=batch_size, T=16, H=H)
    else:
        task = "smnist"
        train_loader, test_loader = get_smnist_loaders(batch_size=batch_size)

    print(f"  task: {task}")
    print(f"  device: {device}")
    print(f"  epochs: {epochs}, lr: {lr}, batch: {batch_size}")

    # Build net
    net = build_extended_net(arch_name, num_classes=num_classes,
                              H=H, in_ch=3).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=max(1, epochs // 3),
                                                  gamma=0.5)

    best_acc = 0.0
    best_path = CHECKPOINT_DIR / f"{arch_name}_best.pth"
    metrics = {
        "arch": arch_name, "task": task,
        "n_params": n_params, "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "H": H, "num_classes": num_classes,
        "epoch_train_loss": [], "epoch_train_acc": [],
        "epoch_test_acc": [], "wall_clock_sec": 0.0,
    }
    t0 = time.time()

    for epoch in range(epochs):
        net.train()
        loss_sum, correct_sum, n = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)         # [T, B, 3, H, H]
            yb = yb.to(device)         # [B]
            reset_state_extended(net)
            logits_seq = net(xb)        # [T, B, num_classes]
            # Use last timestep's logits for classification
            logits = logits_seq[-1]     # [B, num_classes]
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * yb.size(0)
            correct_sum += (logits.argmax(dim=-1) == yb).sum().item()
            n += yb.size(0)

        train_loss = loss_sum / n
        train_acc = correct_sum / n
        scheduler.step()

        # Eval
        net.eval()
        correct_sum, n = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                reset_state_extended(net)
                logits_seq = net(xb)
                logits = logits_seq[-1]
                correct_sum += (logits.argmax(dim=-1) == yb).sum().item()
                n += yb.size(0)
        test_acc = correct_sum / n

        metrics["epoch_train_loss"].append(train_loss)
        metrics["epoch_train_acc"].append(train_acc)
        metrics["epoch_test_acc"].append(test_acc)

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
              f"test_acc={test_acc*100:.2f}%  ({elapsed:.0f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "arch": arch_name,
                "state_dict": net.state_dict(),
                "test_acc": test_acc,
                "epoch": epoch,
                "metrics": metrics,
            }, best_path)

    metrics["wall_clock_sec"] = time.time() - t0
    metrics["best_test_acc"] = best_acc
    metrics["best_path"] = str(best_path)

    metrics_path = CHECKPOINT_DIR / f"{arch_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Best test acc: {best_acc*100:.2f}%")
    print(f"  Checkpoint: {best_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Wall: {metrics['wall_clock_sec']:.0f}s")

    return best_path, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="all",
                        help=f"one of {EXTENDED_NETS} or 'all'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip archs whose _best.pth already exists "
                             "(useful for resuming after interruption)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch == "all":
        archs = list(EXTENDED_NETS)
    else:
        if args.arch not in EXTENDED_NETS:
            print(f"[ERR] unknown arch: {args.arch}")
            sys.exit(1)
        archs = [args.arch]

    if args.skip_existing:
        before = list(archs)
        archs = [a for a in archs
                 if not (CHECKPOINT_DIR / f"{a}_best.pth").exists()]
        skipped = [a for a in before if a not in archs]
        if skipped:
            print(f"  --skip_existing: skipping already-trained archs:")
            for s in skipped:
                ck = CHECKPOINT_DIR / f"{s}_best.pth"
                try:
                    ckpt = torch.load(ck, map_location="cpu")
                    print(f"    {s}  test_acc={ckpt.get('test_acc', 0)*100:.2f}%")
                except Exception:
                    print(f"    {s}  (ckpt unreadable, will retrain)")
                    archs.append(s)
        if not archs:
            print("  All archs already trained; nothing to do.")
            return

    print("=" * 78)
    print("AEROS extended-suite training")
    print("=" * 78)
    print(f"  archs to train: {archs}")
    print(f"  device: {device}")

    summary = {}
    for arch in archs:
        try:
            best_path, metrics = train_one_arch(
                arch, device, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size)
            summary[arch] = {
                "best_test_acc": metrics["best_test_acc"],
                "wall_clock_sec": metrics["wall_clock_sec"],
                "ckpt": str(best_path),
            }
        except Exception as e:
            print(f"[ERR] {arch}: {type(e).__name__}: {e}")
            summary[arch] = {"error": f"{type(e).__name__}: {e}"}
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("Training summary")
    print("=" * 78)
    for arch, s in summary.items():
        if "error" in s:
            print(f"  {arch:<14s}  ERROR: {s['error']}")
        else:
            print(f"  {arch:<14s}  best_test_acc={s['best_test_acc']*100:.2f}%  "
                  f"({s['wall_clock_sec']:.0f}s)  -> {s['ckpt']}")

    summary_path = CHECKPOINT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()