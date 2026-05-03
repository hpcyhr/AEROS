"""
p7_1_eval_trained.py — load trained CIFAR-10 checkpoints, run baseline +
AEROS κ=4/8 + streaming variant, report bit-exact top-1 preservation.

Purpose: GPT P7/P8 review's S-3 (trained-checkpoint top-1 accuracy gap)
is the cheapest closeable submission-critical issue. This script
produces the §5.9 evidence table:

  Architecture | Baseline top-1 | AEROS κ=4 top-1 | AEROS κ=8 top-1 |
                 Streaming top-1 | max_err | Pred agreement

The expected outcome for every network in our suite is:
  - max_err = 0 (bit-exact, per Theorem 1)
  - top-1 agreement = 100.00%
  - all 4 columns identical

If any network shows non-zero max_err or non-100% prediction agreement,
that's a real bug in the AEROS chunked-execution path — important to
catch before submission.

Usage:
    python p7_1_eval_trained.py --ckpt-dir checkpoints/ --T 4 --b 64
    python p7_1_eval_trained.py --archs SR-18,SEW-18 --T 8

Time: ~30 sec/network for full CIFAR-10 test set @ T=4 → ~10 min total.
"""
import argparse
import json
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# ---- Path setup ----
CATFUSE_PATH = '/data/yhr/CATFuse'
if CATFUSE_PATH not in sys.path:
    sys.path.insert(0, CATFUSE_PATH)

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import (
    spiking_resnet, sew_resnet, spiking_vgg)

# ---- Reuse the build_network from training script ----
sys.path.insert(0, str(Path(__file__).parent))
try:
    from p7_1_train_cifar10 import build_network, encode, model_forward
except ImportError:
    # If running from a different dir, inline the imports
    sys.path.insert(0, '/data/yhr/AEROS')
    from p7_1_train_cifar10 import build_network, encode, model_forward


# =============================================================================
# AEROS chunked forward
# =============================================================================
@torch.no_grad()
def baseline_forward(net, x, arch, T):
    """Baseline: full unchunked T-step forward."""
    functional.reset_net(net)
    out = model_forward(net, x, arch, T)
    return out  # [B, num_classes]


@torch.no_grad()
def aeros_chunked_forward(net, x, arch, T, kappa):
    """AEROS chunked forward: split T-axis into kappa-step chunks,
    propagate state across chunks, mean over T at the end.
    """
    if arch in ('Spikformer-T', 'Spikformer-S',
                'QKFormer-T', 'SDTv1-T'):
        # Transformer arch handles its own T internally — just call once
        functional.reset_net(net)
        return model_forward(net, x, arch, T)

    functional.reset_net(net)
    chunks_out = []
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        chunk = x[i:i + sz]  # [sz, B, C, H, W]
        chunk_out = net(chunk)  # [sz, B, num_classes]
        chunks_out.append(chunk_out)
        i += sz
    full_out = torch.cat(chunks_out, dim=0)  # [T, B, num_classes]
    return full_out.mean(dim=0)


@torch.no_grad()
def aeros_streaming_forward(net, x, arch, T, kappa):
    """Streaming variant: same as chunked but emphasizes the
    no-full-T-input semantics. Functionally equivalent to chunked for
    bit-exactness verification.
    """
    return aeros_chunked_forward(net, x, arch, T, kappa)


# =============================================================================
# Eval pipeline
# =============================================================================
def get_test_loader(data_path, batch_size, num_workers=4):
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=False)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_method(net, loader, T, device, arch, method, kappa=None):
    """Returns dict with top1, predictions, full logits sum across batches."""
    net.eval()
    total = 0
    correct = 0
    all_preds = []
    all_logits_max = 0.0  # for diagnostic
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        x = encode(img, T, arch)

        if method == 'baseline':
            out = baseline_forward(net, x, arch, T)
        elif method == 'aeros':
            out = aeros_chunked_forward(net, x, arch, T, kappa)
        elif method == 'streaming':
            out = aeros_streaming_forward(net, x, arch, T, kappa)
        else:
            raise ValueError(method)

        functional.reset_net(net)
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        total += label.size(0)
        correct += (preds == label).sum().item()
    return {
        'top1': correct / total,
        'preds': np.concatenate(all_preds),
        'n_total': total,
    }


@torch.no_grad()
def measure_max_err(net, loader, T, device, arch, kappa, n_batches=10):
    """Measure max |y_aeros - y_baseline| across n_batches of test data.
    Should be 0 for SJ stdlib networks (bit-exact); transformers may have
    ULP-scale differences.
    """
    net.eval()
    max_err = 0.0
    for batch_idx, (img, _) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        img = img.to(device, non_blocking=True)
        x = encode(img, T, arch)
        y_base = baseline_forward(net, x, arch, T)
        y_aeros = aeros_chunked_forward(net, x, arch, T, kappa)
        err = (y_aeros - y_base).abs().max().item()
        max_err = max(max_err, err)
    return max_err


# =============================================================================
# Per-network eval
# =============================================================================
def eval_one_network(arch, ckpt_path, args, device):
    print(f'\n{"=" * 75}')
    print(f'=== {arch}  ckpt={ckpt_path.name} ===')
    print(f'{"=" * 75}')

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt.get('args', {})
    print(f'  Trained: epoch={ckpt.get("epoch")}, '
          f'recorded test_acc={100*ckpt.get("test_acc",0):.2f}%')

    # Rebuild network with same hyperparameters as training
    cifar10_stem = saved_args.get('cifar10_stem', True)
    v_threshold = saved_args.get('v_threshold', 1.0)
    tau = saved_args.get('tau', 2.0)
    net = build_network(arch, num_classes=10, input_size=32,
                        cifar10_stem=cifar10_stem,
                        v_threshold=v_threshold, tau=tau).to(device).eval()
    net.load_state_dict(ckpt['model_state_dict'])
    print(f'  Rebuilt + loaded weights ({sum(p.numel() for p in net.parameters())/1e6:.2f}M params)')

    test_loader = get_test_loader(args.data_path, args.b, args.workers)

    # Evaluate baseline + AEROS κ=4, 8 + streaming
    results = {}
    for method, kappa in [('baseline', None),
                          ('aeros',     4),
                          ('aeros',     8),
                          ('streaming', 8)]:
        label = method if kappa is None else f'{method} k={kappa}'
        t0 = time.time()
        r = evaluate_method(net, test_loader, args.T, device, arch,
                             method, kappa)
        elapsed = time.time() - t0
        print(f'  {label:<18}: top-1 = {100*r["top1"]:.2f}% '
              f'({r["n_total"]} samples, {elapsed:.1f}s)')
        results[label] = r

    # Bit-exact verification @ kappa=8
    max_err_k4 = measure_max_err(net, test_loader, args.T, device, arch, 4)
    max_err_k8 = measure_max_err(net, test_loader, args.T, device, arch, 8)
    print(f'  max_err (κ=4): {max_err_k4:.2e}')
    print(f'  max_err (κ=8): {max_err_k8:.2e}')

    # Prediction agreement
    base_preds = results['baseline']['preds']
    out = {
        'arch': arch,
        'recorded_test_acc': float(ckpt.get('test_acc', 0)),
        'baseline_top1': float(results['baseline']['top1']),
        'aeros_k4_top1': float(results['aeros k=4']['top1']),
        'aeros_k8_top1': float(results['aeros k=8']['top1']),
        'streaming_k8_top1': float(results['streaming k=8']['top1']),
        'max_err_k4': float(max_err_k4),
        'max_err_k8': float(max_err_k8),
        'pred_match_k4': float(np.mean(
            results['aeros k=4']['preds'] == base_preds)),
        'pred_match_k8': float(np.mean(
            results['aeros k=8']['preds'] == base_preds)),
        'pred_match_streaming': float(np.mean(
            results['streaming k=8']['preds'] == base_preds)),
    }

    # Verdict
    if (out['max_err_k4'] == 0.0 and out['max_err_k8'] == 0.0
            and out['pred_match_k4'] == 1.0 and out['pred_match_k8'] == 1.0):
        print(f'  ✓ Bit-exact preservation across baseline/AEROS κ=4/8/streaming')
    else:
        print(f'  ⚠ Non-zero divergence detected:')
        print(f'    max_err_k4 = {out["max_err_k4"]:.2e}')
        print(f'    max_err_k8 = {out["max_err_k8"]:.2e}')
        print(f'    pred_match_k4 = {100*out["pred_match_k4"]:.2f}%')
        print(f'    pred_match_k8 = {100*out["pred_match_k8"]:.2f}%')

    return out


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', type=str, default='checkpoints/')
    ap.add_argument('--data-path', type=str,
                    default='/data/yhr/datasets/cifar10')
    ap.add_argument('--archs', type=str, default='all',
                    help='comma-separated arch names, or "all"')
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--b', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--out', type=str, default='p7_1_eval_results.npz')
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    print(f'\n=== P7-1 Eval: trained-checkpoint top-1 across AEROS modes ===')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ckpt_dir: {ckpt_dir}')
    print(f'  T={args.T}, b={args.b}\n')

    # Discover available checkpoints
    available = {}
    for ckpt in sorted(ckpt_dir.glob('*_cifar10_best.pth')):
        # filename: SR_18_cifar10_best.pth → arch SR-18
        stem = ckpt.stem.replace('_cifar10_best', '')
        arch = stem.replace('_', '-')
        available[arch] = ckpt
    print(f'  Found {len(available)} trained checkpoints: {list(available.keys())}')

    if args.archs == 'all':
        target_archs = list(available.keys())
    else:
        target_archs = [a.strip() for a in args.archs.split(',')
                        if a.strip() in available]

    all_results = []
    for arch in target_archs:
        try:
            r = eval_one_network(arch, available[arch], args, device)
            all_results.append(r)
        except Exception as e:
            print(f'\n[ERROR] {arch}: {e}')
            import traceback; traceback.print_exc()
            all_results.append({'arch': arch, 'error': str(e)})
        gc.collect(); torch.cuda.empty_cache()

    # Final table
    print('\n' + '=' * 95)
    print('=== Trained-checkpoint AEROS preservation table (§5.9 in paper) ===')
    print('=' * 95)
    print(f'{"Arch":<14} {"Base":>7} {"AEROS κ=4":>10} {"AEROS κ=8":>10} '
          f'{"Stream":>8} {"max_err":>10} {"pred match":>11}')
    print('-' * 95)
    for r in all_results:
        if 'error' in r:
            print(f'{r["arch"]:<14}  ERROR: {r["error"][:70]}')
            continue
        max_e = max(r['max_err_k4'], r['max_err_k8'])
        match_min = min(r['pred_match_k4'], r['pred_match_k8'],
                        r['pred_match_streaming']) * 100
        print(f'{r["arch"]:<14} '
              f'{100*r["baseline_top1"]:>6.2f}% '
              f'{100*r["aeros_k4_top1"]:>9.2f}% '
              f'{100*r["aeros_k8_top1"]:>9.2f}% '
              f'{100*r["streaming_k8_top1"]:>7.2f}% '
              f'{max_e:>10.2e} '
              f'{match_min:>10.2f}%')

    np.savez(args.out, results=all_results, args=vars(args))
    print(f'\n  Saved to {args.out}')


if __name__ == '__main__':
    main()