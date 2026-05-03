"""
p7_3c_extended_multinet.py — extended 17-network memory model + selector + max-T sweep.

Purpose: AEROS v8.2 evaluates 7 networks; Chronos evaluates 11 STNN workloads
and Helios evaluates 8 DiNN workloads. Doris's directive is that AEROS's
workload count must exceed both. This driver adds 11 new networks (drawn
from CATFuse's existing wrapper set at /data/yhr/CATFuse/models/) on top
of the 6 SNN architectures already in v8.2, for a total of 17.

For each network, this script measures three artifacts:
  (1) Memory model coefficients (M_0, alpha_in, alpha_K, R^2) by sweeping
      (T, kappa) at fixed (b, H);
  (2) Selector compliance at M = {5 GB, 16 GB} with eps=0.05;
  (3) Max feasible T at fixed kappa = 8.

All measurements use random weights — this is the "AEROS works on the
architecture" sweep, separate from the trained-accuracy track.

Network list (17):
  - SpikingResNet-18 / 34 / 50          (3, ResNet family)
  - SEW-ResNet-18 / 50 / 101             (3, SEW family)
  - SpikingVGG-11-BN / 13-BN / 16-BN / 19-BN  (4, VGG family)
  - SpikingAlexNet, SpikingZFNet, SpikingMobileNet-V1  (3, lightweight CNN)
  - Spikformer-tiny / small, QKFormer-tiny, SDTv1-tiny  (4, Spike Transformer)
  - ConvLSTM (2-layer, 1.1M)             (1, recurrent — already in v8.2)

Usage:
    cd /data/yhr/AEROS/
    python p7_3c_extended_multinet.py --b 32 --H 32        # CIFAR-shape
    python p7_3c_extended_multinet.py --b 32 --H 224       # ImageNet-shape (some nets)
    python p7_3c_extended_multinet.py --skip-archs SR-50,VGG-19-BN   # for partial reruns

Outputs:
    p7_3c_results.npz — {arch_name: {coef, compliance, max_T}}
    p7_3c.log         — human-readable summary

Time estimate: 17 networks * ~10-15 min each = 3-4 hours total.
"""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- Path setup: import CATFuse wrappers ----
CATFUSE_PATH = '/data/yhr/CATFuse'
if CATFUSE_PATH not in sys.path:
    sys.path.insert(0, CATFUSE_PATH)

from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model import (
    spiking_resnet, sew_resnet, spiking_vgg)


# =============================================================================
# Network builder registry
# =============================================================================
def _build_lif():
    return neuron.LIFNode


def build_network_registry(num_classes=10, input_size=32):
    """Build a dict {name: lambda → nn.Module} for all 17 networks.

    All networks use LIFNode with tau=2.0, default v_threshold=1.0.
    Each network is constructed lazily (lambda) to avoid building all
    17 at once.
    """
    common_lif = dict(spiking_neuron=neuron.LIFNode, tau=2.0,
                      num_classes=num_classes)

    registry = {}

    # === ResNet family (3) ===
    registry['SR-18'] = lambda: spiking_resnet.spiking_resnet18(**common_lif)
    registry['SR-34'] = lambda: spiking_resnet.spiking_resnet34(**common_lif)
    registry['SR-50'] = lambda: spiking_resnet.spiking_resnet50(**common_lif)

    # === SEW-ResNet family (3) ===
    registry['SEW-18']  = lambda: sew_resnet.sew_resnet18(
        cnf='ADD', **common_lif)
    registry['SEW-50']  = lambda: sew_resnet.sew_resnet50(
        cnf='ADD', **common_lif)
    registry['SEW-101'] = lambda: sew_resnet.sew_resnet101(
        cnf='ADD', **common_lif)

    # === Spiking VGG family (4) ===
    registry['VGG-11-BN'] = lambda: spiking_vgg.spiking_vgg11_bn(**common_lif)
    registry['VGG-13-BN'] = lambda: spiking_vgg.spiking_vgg13_bn(**common_lif)
    registry['VGG-16-BN'] = lambda: spiking_vgg.spiking_vgg16_bn(**common_lif)
    registry['VGG-19-BN'] = lambda: spiking_vgg.spiking_vgg19_bn(**common_lif)

    # === Lightweight CNN (3) — from CATFuse wrappers ===
    try:
        from models.spiking_alexnet import SpikingAlexNet
        registry['AlexNet'] = lambda: SpikingAlexNet(
            num_classes=num_classes, spiking_neuron=neuron.LIFNode,
            tau=2.0, input_size=input_size)
    except ImportError as e:
        print(f'  [warn] SpikingAlexNet import failed: {e}')

    try:
        from models.spiking_zfnet import SpikingZFNet
        registry['ZFNet'] = lambda: SpikingZFNet(
            num_classes=num_classes, spiking_neuron=neuron.LIFNode,
            tau=2.0, input_size=input_size)
    except ImportError as e:
        print(f'  [warn] SpikingZFNet import failed: {e}')

    try:
        from models.spiking_mobilenet import SpikingMobileNetV1
        registry['MobileNet-V1'] = lambda: SpikingMobileNetV1(
            num_classes=num_classes, spiking_neuron=neuron.LIFNode,
            tau=2.0, input_size=input_size)
    except ImportError as e:
        print(f'  [warn] SpikingMobileNetV1 import failed: {e}')

    # === Spike Transformer (4) — from CATFuse wrappers ===
    try:
        from models.spikformer_github import (
            spikformer_cifar_tiny, spikformer_cifar_small)
        registry['Spikformer-T'] = lambda: spikformer_cifar_tiny(
            spiking_neuron=neuron.LIFNode, num_classes=num_classes)
        registry['Spikformer-S'] = lambda: spikformer_cifar_small(
            spiking_neuron=neuron.LIFNode, num_classes=num_classes)
    except ImportError as e:
        print(f'  [warn] Spikformer import failed: {e}')

    try:
        from models.qkformer_github import qkformer_cifar_tiny
        registry['QKFormer-T'] = lambda: qkformer_cifar_tiny(
            spiking_neuron=neuron.LIFNode, num_classes=num_classes)
    except ImportError as e:
        print(f'  [warn] QKFormer import failed: {e}')

    try:
        from models.sdtv1_github import sdtv1_cifar_tiny
        registry['SDTv1-T'] = lambda: sdtv1_cifar_tiny(
            spiking_neuron=neuron.LIFNode, num_classes=num_classes)
    except ImportError as e:
        print(f'  [warn] SDTv1 import failed: {e}')

    # ConvLSTM stays from AEROS local — try multiple known builder names
    try:
        sys.path.insert(0, '/data/yhr/AEROS')
        import convlstm_module as _cl
        if hasattr(_cl, 'build_convlstm_v1'):
            registry['ConvLSTM'] = lambda: _cl.build_convlstm_v1()
        elif hasattr(_cl, 'build_convlstm'):
            registry['ConvLSTM'] = lambda: _cl.build_convlstm()
        elif hasattr(_cl, 'ConvLSTM'):
            registry['ConvLSTM'] = lambda: _cl.ConvLSTM(
                input_dim=3, hidden_dim=64, kernel_size=(3, 3),
                num_layers=2, batch_first=False)
        else:
            print(f'  [info] convlstm_module loaded but no known builder; skipping')
    except (ImportError, Exception) as e:
        print(f'  [info] ConvLSTM not loaded ({e}); skipping')

    return registry


# =============================================================================
# Memory measurement helpers
# =============================================================================
def aggressive_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def reset_peak_mem():
    aggressive_cleanup()
    torch.cuda.reset_peak_memory_stats()


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


@torch.no_grad()
def is_multi_step_compatible(net):
    """Try to set step_mode='m'; some Spikformer-style nets don't support
    SJ multi-step API (they accept [T,B,...] directly without the
    set_step_mode call). Return whether we set it."""
    try:
        functional.set_step_mode(net, step_mode='m')
        return True
    except Exception:
        return False


@torch.no_grad()
def chunked_forward(net, x, kappa, multi_step):
    """Run net on x = [T, B, C, H, W] in chunks of width kappa."""
    T = x.shape[0]
    functional.reset_net(net)
    outs = []
    i = 0
    while i < T:
        sz = min(kappa, T - i)
        chunk = x[i:i + sz]
        try:
            out = net(chunk)
        except Exception:
            # Some Spikformer-style nets expect [T,B,C,H,W] only;
            # if it fails, retry without step_mode wrap
            functional.reset_net(net)
            out = net(chunk)
        outs.append(out)
        i += sz
    if isinstance(outs[0], torch.Tensor):
        return torch.cat(outs, dim=0) if outs[0].dim() >= 1 else outs[-1]
    return outs[-1]


@torch.no_grad()
def safe_measure_mem(net, T, b, C, H, kappa, device, multi_step):
    """Measure peak memory of a chunked forward at (T, b, H, kappa).

    Returns peak in GB, or None if OOM/error.
    """
    aggressive_cleanup()
    reset_peak_mem()
    try:
        x = torch.rand(T, b, C, H, H, device=device)
        _ = chunked_forward(net, x, kappa, multi_step)
        torch.cuda.synchronize()
        peak = peak_mem_GB()
        del x
        functional.reset_net(net)
        aggressive_cleanup()
        return peak
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if 'out of memory' in msg or 'cuda' in msg:
            try:
                functional.reset_net(net)
            except Exception:
                pass
            aggressive_cleanup()
            return None
        # Other RuntimeErrors are bugs — re-raise to surface
        raise


# =============================================================================
# Memory model fit
# =============================================================================
def fit_memory_model(samples):
    """Fit M = M_0 + alpha_in * T + alpha_K * kappa using least squares.

    samples: list of (T, kappa, peak_GB) tuples
    Returns dict with M_0, alpha_in, alpha_K, R^2, max_neg_residual_pct.
    """
    if len(samples) < 4:
        return None
    A = np.array([[1, T, K] for T, K, _ in samples])
    y = np.array([m for _, _, m in samples])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    M_0, a_in, a_K = coeffs
    pred = A @ coeffs
    resid = y - pred  # positive resid = under-prediction
    rel_resid = resid / np.maximum(y, 1e-9)
    R2 = 1 - (resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    return {
        'M_0': float(M_0),
        'alpha_in': float(a_in),
        'alpha_K': float(a_K),
        'R2': float(R2),
        'max_pos_residual_pct': float(rel_resid.max() * 100),
        'n_samples': len(samples),
    }


# =============================================================================
# Selector logic (mirrors §3.5 of paper)
# =============================================================================
def selector_kA(M_budget, eps, M_0, a_in, a_K, T, delta=1e-3):
    """Problem A: max kappa under memory budget."""
    if a_K <= delta:
        # Degenerate case (Eq. kA-degen): kappa doesn't affect mem
        if M_0 + a_in * T <= M_budget * (1 - eps):
            return T
        return None  # INFEASIBLE
    headroom = M_budget * (1.0 - eps) - M_0 - a_in * T
    if headroom <= 0:
        return None
    return max(1, min(int(headroom / a_K), T))


# =============================================================================
# Per-network experiment
# =============================================================================
def run_one_network(name, build_fn, args, device):
    """Run the full sweep for one network. Returns dict of results."""
    print(f'\n{"=" * 70}')
    print(f'=== {name} ===')
    print(f'{"=" * 70}')

    aggressive_cleanup()
    t0 = time.time()

    # ---- Build ----
    try:
        net = build_fn().to(device).eval()
    except Exception as e:
        print(f'  [ERROR] build failed: {e}')
        traceback.print_exc()
        return {'name': name, 'error': str(e)}

    multi_step = is_multi_step_compatible(net)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  Params: {n_params/1e6:.2f}M, multi-step: {multi_step}')

    # Determine input shape
    C, H = 3, args.H

    # ---- (1) Memory model sweep ----
    # T sweep depends on H — at H=128 we want T high enough that
    # alpha_in*T term dominates noise. Keep K sweep wide so we get the
    # alpha_K signal too.
    if args.H >= 128:
        Ts = [16, 32, 64, 128]
        Ks = [1, 4, 16, 32, 64]
    elif args.H >= 64:
        Ts = [16, 32, 64, 128]
        Ks = [1, 4, 16, 32]
    else:
        # H=32 — CIFAR-shape memory tiny; need very long T
        Ts = [64, 128, 256, 512]
        Ks = [1, 8, 32, 128]
    samples = []
    print(f'\n--- Phase 1/3: Memory model fit (T,kappa sweep) ---')
    print(f'    {"T":>4} {"K":>4}  {"peak (GB)":>10}')
    for T in Ts:
        for K in Ks:
            if K > T:
                continue
            peak = safe_measure_mem(net, T, args.b, C, H, K, device,
                                     multi_step)
            if peak is not None:
                samples.append((T, K, peak))
                print(f'    {T:>4} {K:>4}  {peak:>10.3f}')
            else:
                print(f'    {T:>4} {K:>4}  {"OOM":>10}')

    coef = fit_memory_model(samples)
    if coef is None:
        print(f'  [ERROR] insufficient samples for fit')
        return {'name': name, 'error': 'insufficient samples',
                'samples': samples}
    print(f'\n  Fit: M_0={coef["M_0"]:.3f} GB, '
          f'alpha_in={coef["alpha_in"]:.5f} GB/step, '
          f'alpha_K={coef["alpha_K"]:.5f} GB/chunk-step, '
          f'R^2={coef["R2"]:.4f}, '
          f'max_pos_resid={coef["max_pos_residual_pct"]:.2f}%')

    # ---- (2) Selector compliance @ {5 GB, 16 GB} ----
    print(f'\n--- Phase 2/3: Selector compliance ---')
    compliance = {}
    for M_budget in [5.0, 16.0]:
        comp_results = []
        for T in [64, 128, 256]:
            kappa = selector_kA(M_budget, 0.05,
                                coef['M_0'], coef['alpha_in'], coef['alpha_K'],
                                T)
            if kappa is None:
                comp_results.append({
                    'T': T, 'kappa': None, 'actual': None,
                    'compliant': True, 'note': 'INFEASIBLE'
                })
                print(f'    M={M_budget} GB, T={T}: INFEASIBLE (selector refused)')
                continue
            actual = safe_measure_mem(net, T, args.b, C, H, kappa, device,
                                       multi_step)
            if actual is None:
                comp_results.append({
                    'T': T, 'kappa': kappa, 'actual': None,
                    'compliant': False, 'note': 'OOM'
                })
                print(f'    M={M_budget} GB, T={T}, kappa={kappa}: '
                      f'measured OOM (selector wrong)')
            else:
                ok = actual <= M_budget
                comp_results.append({
                    'T': T, 'kappa': kappa, 'actual': float(actual),
                    'compliant': ok,
                    'overshoot': float(actual / M_budget) if not ok else 1.0,
                })
                tag = '✓' if ok else f'✗ ({actual/M_budget:.3f}x)'
                print(f'    M={M_budget} GB, T={T}, kappa={kappa}: '
                      f'actual={actual:.3f} GB {tag}')
        compliance[f'M={M_budget}GB'] = comp_results

    # ---- (3) Max feasible T at fixed kappa = 8 ----
    print(f'\n--- Phase 3/3: Max feasible T at kappa=8 ---')
    print(f'    {"T":>5}  {"baseline":>14}  {"AEROS κ=8":>14}')
    Ts_max = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    K_fixed = 8
    max_T = {'baseline': 0, 'aeros_k8': 0}
    rows = []
    base_oomed = False
    aeros_oomed = False
    for T in Ts_max:
        if not base_oomed:
            base = safe_measure_mem(net, T, args.b, C, H, T, device,
                                     multi_step)
            if base is None:
                base_oomed = True
        else:
            base = None

        if not aeros_oomed and K_fixed <= T:
            ae = safe_measure_mem(net, T, args.b, C, H, K_fixed, device,
                                   multi_step)
            if ae is None:
                aeros_oomed = True
        else:
            ae = None

        if base is not None and base < 32.0:
            max_T['baseline'] = T
        if ae is not None and ae < 32.0:
            max_T['aeros_k8'] = T

        bs = f'{base:.2f} GB' if base is not None else 'OOM'
        as_ = f'{ae:.2f} GB' if ae is not None else 'OOM'
        print(f'    {T:>5}  {bs:>14}  {as_:>14}')
        rows.append({'T': T, 'base': base, 'aeros_k8': ae})

        # Bail out once both modes OOM — no point continuing
        if base_oomed and aeros_oomed:
            print(f'    (both OOM at T={T}; stopping max-T sweep)')
            break

    elapsed = time.time() - t0
    print(f'\n  [done] {elapsed:.0f}s. baseline max-T={max_T["baseline"]}, '
          f'AEROS k=8 max-T={max_T["aeros_k8"]}, '
          f'extension={max_T["aeros_k8"] / max(max_T["baseline"], 1):.1f}x')

    return {
        'name': name,
        'n_params': n_params,
        'multi_step': multi_step,
        'b': args.b, 'H': H, 'C': C,
        'samples': samples,
        'coef': coef,
        'compliance': compliance,
        'max_T_rows': rows,
        'max_T_baseline': max_T['baseline'],
        'max_T_aeros': max_T['aeros_k8'],
        'elapsed_s': elapsed,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b', type=int, default=32, help='batch size')
    ap.add_argument('--H', type=int, default=128,
                    help='input H. 32 = CIFAR-shape (memory model fit '
                         'will be noise-dominated for small CNNs at small b); '
                         '128 = mid-shape (recommended for CNN memory fit, '
                         'but Spike Transformers crash since img_size=(32,32) '
                         'is hardcoded — use --skip-transformers); '
                         '224 = ImageNet-shape (lightweight nets crash).')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--skip-transformers', action='store_true',
                    help='Skip Spikformer/QKFormer/SDTv1 (use when H!=32)')
    ap.add_argument('--archs', type=str, default='all',
                    help='comma-separated list of arch names, or "all"')
    ap.add_argument('--skip-archs', type=str, default='',
                    help='comma-separated list to skip')
    ap.add_argument('--out', type=str, default='p7_3c_results.npz')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print(f'\n=== P7-3c: 17-network extended sweep ===')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  b={args.b}, H={args.H}, num_classes={args.num_classes}')
    print(f'  Estimated time: 17 networks * ~10-15 min = 3-4 hours\n')

    registry = build_network_registry(args.num_classes, args.H)
    print(f'  Loaded registry with {len(registry)} networks')

    if args.archs == 'all':
        target_archs = list(registry.keys())
    else:
        target_archs = args.archs.split(',')
    skip_set = set(s.strip() for s in args.skip_archs.split(',') if s.strip())
    target_archs = [a for a in target_archs
                    if a in registry and a not in skip_set]

    # Skip transformers if --skip-transformers (H != 32 case)
    transformer_names = {'Spikformer-T', 'Spikformer-S',
                         'QKFormer-T', 'SDTv1-T'}
    if args.skip_transformers:
        target_archs = [a for a in target_archs if a not in transformer_names]
        print(f'  [skip-transformers] Excluding {transformer_names & set(registry.keys())}')

    print(f'  Targets ({len(target_archs)}): {target_archs}\n')

    all_results = {}
    for name in target_archs:
        try:
            result = run_one_network(name, registry[name], args, device)
            all_results[name] = result
        except Exception as e:
            print(f'\n[FATAL] {name} crashed: {e}')
            traceback.print_exc()
            all_results[name] = {'name': name, 'error': str(e)}
        # Save incrementally
        np.savez(args.out, results=all_results, args=vars(args))

    # ---- Final summary table ----
    print('\n' + '=' * 90)
    print('=== Final summary table (suitable for §5.2 / §5.6 in paper) ===')
    print('=' * 90)
    print(f'{"Architecture":<18} {"Params":>9} {"alpha_in":>10} '
          f'{"alpha_K":>10} {"M_0":>8} {"R^2":>8} '
          f'{"base maxT":>10} {"AEROS maxT":>11} {"ext":>6}')
    print('-' * 90)
    for name in target_archs:
        r = all_results.get(name, {})
        if 'error' in r:
            print(f'{name:<18} ERROR: {r["error"][:60]}')
            continue
        c = r.get('coef', {})
        ext = (r['max_T_aeros'] / max(r['max_T_baseline'], 1)
               if r.get('max_T_baseline') else 0)
        print(f'{name:<18} {r["n_params"]/1e6:>8.2f}M '
              f'{c.get("alpha_in",0):>10.5f} '
              f'{c.get("alpha_K",0):>10.5f} '
              f'{c.get("M_0",0):>8.3f} '
              f'{c.get("R2",0):>8.4f} '
              f'{r.get("max_T_baseline",0):>10} '
              f'{r.get("max_T_aeros",0):>11} '
              f'{ext:>5.1f}x')

    print(f'\n  Saved to {args.out}')


if __name__ == '__main__':
    main()