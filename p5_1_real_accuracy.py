"""
P5-1: Real-data bit-exactness verification (and accuracy preservation if
      pretrained checkpoint provided).

[v2 — fixes broken f-string format spec from v1]
"""
import argparse, os, sys, time, gc
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


# -----------------------------------------------------------------------------
@torch.no_grad()
def baseline_forward(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_forward(net, x, K):
    """Standard AEROS full-input chunked with state propagation."""
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def streaming_forward(net, x, K):
    """AEROS streaming-input: simulate generator from materialized x."""
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        x_seg = x[i:i + sz].clone()
        chunks.append(net(x_seg))
        del x_seg
        i += sz
    return torch.cat(chunks, dim=0)


# -----------------------------------------------------------------------------
def find_density_npz(user_arg):
    """Search common locations for m1_dvsg_NA.npz."""
    if user_arg and os.path.exists(user_arg):
        return user_arg
    candidates = [
        'm1_dvsg_NA.npz',
        'AEROS/m1_dvsg_NA.npz',
        '/data/yhr/AEROS/m1_dvsg_NA.npz',
        os.path.expanduser('~/AEROS/m1_dvsg_NA.npz'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def make_input(T, b, C, H, device, seed, density_npz):
    """
    Build input that mimics real event-camera structure if density_npz exists,
    otherwise uniform Bernoulli at rate 0.3.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rates = None
    if density_npz:
        try:
            d = np.load(density_npz, allow_pickle=True)
            bc = d['bin_counts']  # (n_samples, T_train)
            density_per_t = bc.mean(axis=0)
            if T <= len(density_per_t):
                rates = density_per_t[:T]
            else:
                rates = np.concatenate([density_per_t,
                                        np.full(T - len(density_per_t), density_per_t[-1])])
            rates = 0.1 + 0.4 * (rates - rates.min()) / (rates.max() - rates.min() + 1e-9)
            print(f'  Using real DVS Gesture density profile from {density_npz}')
            print(f'    rates ∈ [{rates.min():.3f}, {rates.max():.3f}]')
        except Exception as e:
            print(f'  WARN: failed to load density_npz ({e}); using uniform Bernoulli p=0.3')
            rates = None
    if rates is None:
        print(f'  Using uniform Bernoulli input (p=0.3); '
              f'no density file found (looked for m1_dvsg_NA.npz in cwd, AEROS/, '
              f'/data/yhr/AEROS/)')
        rates = np.full(T, 0.3)

    rates_t = torch.tensor(rates, device=device, dtype=torch.float32).view(T, 1, 1, 1, 1)
    u = torch.rand(T, b, C, H, H, generator=g, device=device)
    x = (u < rates_t).float()
    return x


def load_or_build_net(checkpoint, device, num_classes=11):
    print('Building spiking_resnet18...')
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True,
                           num_classes=num_classes).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')

    has_pretrained = False
    if checkpoint and os.path.exists(checkpoint):
        try:
            sd = torch.load(checkpoint, map_location=device)
            if isinstance(sd, dict):
                if 'net' in sd: sd = sd['net']
                elif 'state_dict' in sd: sd = sd['state_dict']
                elif 'model' in sd: sd = sd['model']
            net.load_state_dict(sd, strict=False)
            print(f'  Loaded pretrained: {checkpoint}')
            has_pretrained = True
        except Exception as e:
            print(f'  WARN: failed to load checkpoint ({e}); using random weights')
    else:
        print('  Using random weights (bit-exact validation does not require trained weights)')
    return net, has_pretrained


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--b', type=int, default=16)
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--num_classes', type=int, default=11)
    ap.add_argument('--density_npz', type=str, default=None,
                    help='path to m1_dvsg_NA.npz (auto-located if not specified)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)

    net, has_pretrained = load_or_build_net(args.checkpoint, device, args.num_classes)

    print(f'\n=== P5-1: bit-exactness on heterogeneous input '
          f'(T={args.T}, b={args.b}, H={args.H}) ===')
    density_path = find_density_npz(args.density_npz)
    x = make_input(args.T, args.b, 3, args.H, device, args.seed, density_path)

    # --- Reference: unchunked baseline ---
    print('\n--- Reference: baseline (unchunked) ---')

    # Warmup (cuDNN autotune, etc.)
    _ = baseline_forward(net, x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    y_base = baseline_forward(net, x)
    torch.cuda.synchronize()
    base_wall = (time.perf_counter() - t0) * 1000
    print(f'  baseline: shape={tuple(y_base.shape)}, wall={base_wall:.1f}ms')

    pred_base = None
    if has_pretrained:
        pred_base = y_base.sum(dim=0).argmax(dim=1).cpu().numpy()
        print(f'  baseline pred (first 10): {pred_base[:10]}')

    # --- Methods to test ---
    methods = [
        ('AEROS κ=4',                  lambda: chunked_forward(net, x, 4)),
        ('AEROS κ=8',                  lambda: chunked_forward(net, x, 8)),
        ('AEROS κ=10 (non-divisor)',   lambda: chunked_forward(net, x, 10)),
        ('AEROS streaming κ=8',        lambda: streaming_forward(net, x, 8)),
    ]

    # Build header (no conditional inside f-string spec)
    print(f'\n--- Methods (bit-exact vs baseline) ---')
    if has_pretrained:
        header = f'{"Method":<30}  {"max_err":>12}  {"shape":>20}  {"wall_ms":>8}  {"pred_match":>10}'
    else:
        header = f'{"Method":<30}  {"max_err":>12}  {"shape":>20}  {"wall_ms":>8}'
    print(header)
    print('-' * len(header))

    rows = []
    for name, fn in methods:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000

        max_err = (y - y_base).abs().max().item()
        shape_str = str(tuple(y.shape))

        pred_match_pct = None
        if has_pretrained:
            pred = y.sum(dim=0).argmax(dim=1).cpu().numpy()
            pred_match_pct = float((pred == pred_base).mean()) * 100
            line = f'{name:<30}  {max_err:>12.2e}  {shape_str:>20}  {wall_ms:>8.1f}  {pred_match_pct:>9.2f}%'
        else:
            line = f'{name:<30}  {max_err:>12.2e}  {shape_str:>20}  {wall_ms:>8.1f}'
        print(line)

        rows.append({
            'method': name, 'max_err': max_err,
            'shape_match': tuple(y.shape) == tuple(y_base.shape),
            'wall_ms': wall_ms,
            'pred_match_pct': pred_match_pct,
        })

    # --- Save artifacts ---
    np.savez('p5_1_results.npz',
             rows=rows, T=args.T, b=args.b, H=args.H,
             has_pretrained=has_pretrained,
             checkpoint=str(args.checkpoint),
             baseline_wall_ms=base_wall)

    print('\n=== Verdict ===')
    n_zero = sum(1 for r in rows if r['max_err'] == 0.0)
    print(f'  Methods with max_err == 0.0 (bit-exact): {n_zero}/{len(rows)}')
    if has_pretrained:
        n_acc = sum(1 for r in rows
                    if r['pred_match_pct'] is not None and r['pred_match_pct'] >= 100.0 - 1e-9)
        print(f'  Methods with prediction-identical to baseline: {n_acc}/{len(rows)}')
    print('  Saved to p5_1_results.npz')


if __name__ == '__main__':
    main()