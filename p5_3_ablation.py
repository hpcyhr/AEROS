"""
P5-3: Component ablation — what does each AEROS mechanism contribute?

Four ablations, each isolating one mechanism:
  (a) No M3 (selector):     use heuristic κ=8 always vs selector
  (b) No M4 (planner):      force κ|T (no non-uniform residual) vs non-uniform
  (c) No streaming:         full-input only vs streaming at long T
  (d) No state-carry:       reset_net before each chunk (NEGATIVE control —
                            shows what would break if state-carry removed)

For each ablation, we measure the relevant outcome (compliance, peak memory,
max feasible T, or output divergence) on a small targeted config set.

Usage:
    python p5_3_ablation.py
    python p5_3_ablation.py --skip d   # skip the slow (d) divergence test
"""
import argparse, os, time, sys
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


# -----------------------------------------------------------------------------
def build_net(device):
    net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True).to(device)
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    return net


def kA_selector(M, eps, M_0, a_in, a_K, T):
    headroom = M * (1.0 - eps) - M_0 - a_in * T
    if headroom <= 0:
        return None
    return max(1, min(int(headroom / a_K), T))


@torch.no_grad()
def run_chunked(net, x, K, do_state_carry=True):
    """Standard chunked. If do_state_carry=False, reset between chunks."""
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        if (not do_state_carry) and i > 0:
            functional.reset_net(net)
        sz = min(K, T - i)
        chunks.append(net(x[i:i + sz]))
        i += sz
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def run_baseline(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def run_streaming(net, x, K):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    i = 0
    while i < T:
        sz = min(K, T - i)
        seg = x[i:i + sz].clone()
        chunks.append(net(seg))
        del seg
        i += sz
    return torch.cat(chunks, dim=0)


def peak_mem_GB():
    return torch.cuda.max_memory_allocated() / 1e9


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# -----------------------------------------------------------------------------
def ablation_a_no_selector(net, device, fits, budget=5.0):
    """Selector (κ_A*) vs heuristic (κ=8 always)."""
    print('\n=== Ablation (a): No M3 (selector) — heuristic κ=8 vs AEROS selector ===')
    print(f'{"T":>4} {"b":>3} {"H":>4}  {"kA*":>5} {"sel mem":>8} {"sel ok":>7}  '
          f'{"k=8 mem":>8} {"k=8 ok":>7}')
    rows = []
    Ts = [64, 128, 256]
    bs = [16, 32, 64]
    Hs = [64, 128, 224]
    eps = 0.05
    cnt_sel_ok, cnt_h8_ok = 0, 0
    n_total = 0
    for T in Ts:
        for b in bs:
            for H in Hs:
                if (b, H) not in fits:
                    continue
                M_0, a_in, a_K, _ = fits[(b, H)]
                k_sel = kA_selector(budget, eps, M_0, a_in, a_K, T)
                if k_sel is None:
                    continue
                # Snap to nearest power-of-2
                k_sel = max([g for g in [1,2,4,8,16,32,64,128,256] if g <= min(k_sel, T)], default=1)
                k_h8 = min(8, T)

                # selector
                reset_mem()
                try:
                    x = torch.rand(T, b, 3, H, H, device=device)
                    _ = run_chunked(net, x, k_sel)
                    torch.cuda.synchronize()
                    sel_mem = peak_mem_GB()
                    del x
                except torch.cuda.OutOfMemoryError:
                    sel_mem = None
                    torch.cuda.empty_cache()

                # heuristic κ=8
                reset_mem()
                try:
                    x = torch.rand(T, b, 3, H, H, device=device)
                    _ = run_chunked(net, x, k_h8)
                    torch.cuda.synchronize()
                    h8_mem = peak_mem_GB()
                    del x
                except torch.cuda.OutOfMemoryError:
                    h8_mem = None
                    torch.cuda.empty_cache()

                sel_ok = sel_mem is not None and sel_mem <= budget
                h8_ok  = h8_mem  is not None and h8_mem  <= budget
                cnt_sel_ok += sel_ok
                cnt_h8_ok  += h8_ok
                n_total += 1
                sel_str = f'{sel_mem:.2f}' if sel_mem else 'OOM'
                h8_str  = f'{h8_mem:.2f}'  if h8_mem  else 'OOM'
                print(f'{T:>4} {b:>3} {H:>4}  {k_sel:>5} {sel_str:>8} {"Y" if sel_ok else "N":>7}  '
                      f'{h8_str:>8} {"Y" if h8_ok else "N":>7}')
                rows.append({'T':T,'b':b,'H':H,'k_sel':k_sel,'sel_mem':sel_mem,
                             'h8_mem':h8_mem,'sel_ok':sel_ok,'h8_ok':h8_ok})
    print(f'\n  Selector compliance: {cnt_sel_ok}/{n_total}')
    print(f'  Heuristic κ=8 compliance: {cnt_h8_ok}/{n_total}')
    return rows


def ablation_b_no_planner(net, device):
    """Non-uniform planner [κ,κ,...,residual] vs uniform-only (must round κ down)."""
    print('\n=== Ablation (b): No M4 (planner) — non-uniform residual vs uniform-only ===')
    print(f'{"T":>5} {"target κ":>9}  {"uniform κ":>10} {"# segs (uniform)":>17}  '
          f'{"# segs (nonuniform)":>20}')
    rows = []
    # Target T values that are NOT power-of-2 (so non-uniform planner matters)
    # T_target, b, H selected to fit comfortably
    test_cases = [(33, 16, 64), (50, 16, 64), (100, 16, 128),
                  (257, 16, 64), (513, 16, 64), (1000, 8, 64)]
    target_k = 8
    for T, b, H in test_cases:
        # Non-uniform: [8, 8, ..., 8, residual]
        n_full = T // target_k
        residual = T % target_k
        n_seg_nu = n_full + (1 if residual else 0)

        # Uniform-only: must use largest κ' ≤ target_k that divides T
        k_div = target_k
        while T % k_div != 0 and k_div > 1:
            k_div -= 1
        n_seg_u = T // k_div if k_div > 0 else T

        # Run AEROS non-uniform
        reset_mem()
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            t0 = time.perf_counter()
            y_nu = run_chunked(net, x, target_k)
            torch.cuda.synchronize()
            wall_nu = (time.perf_counter()-t0)*1000
            mem_nu = peak_mem_GB()
            del x, y_nu
        except torch.cuda.OutOfMemoryError:
            wall_nu = None; mem_nu = None
            torch.cuda.empty_cache()

        # Run uniform-only with smaller κ
        reset_mem()
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            t0 = time.perf_counter()
            _ = run_chunked(net, x, k_div)
            torch.cuda.synchronize()
            wall_u = (time.perf_counter()-t0)*1000
            mem_u = peak_mem_GB()
            del x
        except torch.cuda.OutOfMemoryError:
            wall_u = None; mem_u = None
            torch.cuda.empty_cache()

        print(f'{T:>5} {target_k:>9}  {k_div:>10} {n_seg_u:>17}  {n_seg_nu:>20}')
        if mem_nu is not None and wall_nu is not None:
            print(f'        nonuniform: mem={mem_nu:.2f}GB wall={wall_nu:.0f}ms')
        if mem_u is not None and wall_u is not None:
            print(f'        uniform:    mem={mem_u:.2f}GB wall={wall_u:.0f}ms  '
                  f'(slowdown vs nonuniform: '
                  f'{wall_u/wall_nu if wall_nu else 0:.2f}×)')
        rows.append({'T':T,'b':b,'H':H,'target_k':target_k,'k_uniform':k_div,
                     'n_seg_uniform':n_seg_u,'n_seg_nonuniform':n_seg_nu,
                     'mem_nu':mem_nu,'mem_u':mem_u,'wall_nu':wall_nu,'wall_u':wall_u})
    return rows


def ablation_c_no_streaming(net, device):
    """Full-input AEROS vs streaming at progressively larger T."""
    print('\n=== Ablation (c): No streaming — full-input AEROS vs streaming ===')
    print(f'{"T":>5} {"full-input mem":>16} {"full status":>12}  '
          f'{"streaming mem":>14} {"stream status":>14}')
    rows = []
    # T progressively larger to stress full-input first
    Ts = [256, 512, 1024, 2048, 4096, 8192]
    b, H, K = 8, 128, 8

    for T in Ts:
        # Full-input
        reset_mem()
        full_mem = None; full_ok = False
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            _ = run_chunked(net, x, K)
            torch.cuda.synchronize()
            full_mem = peak_mem_GB()
            full_ok = True
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # Streaming
        reset_mem()
        stream_mem = None; stream_ok = False
        try:
            x = torch.rand(T, b, 3, H, H, device=device)
            _ = run_streaming(net, x, K)
            torch.cuda.synchronize()
            stream_mem = peak_mem_GB()
            stream_ok = True
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        full_str = f'{full_mem:.2f}GB' if full_mem else '—'
        stream_str = f'{stream_mem:.2f}GB' if stream_mem else '—'
        print(f'{T:>5} {full_str:>16} {"OK" if full_ok else "OOM":>12}  '
              f'{stream_str:>14} {"OK" if stream_ok else "OOM":>14}')
        rows.append({'T':T,'b':b,'H':H,'K':K,
                     'full_mem':full_mem,'full_ok':full_ok,
                     'stream_mem':stream_mem,'stream_ok':stream_ok})

    max_full = max([r['T'] for r in rows if r['full_ok']], default=0)
    max_stream = max([r['T'] for r in rows if r['stream_ok']], default=0)
    print(f'\n  Max feasible T (full-input):  {max_full}')
    print(f'  Max feasible T (streaming):   {max_stream}')
    return rows


@torch.no_grad()
def ablation_d_no_state_carry(net, device):
    """State-carry vs reset-between-chunks (negative ablation: should diverge)."""
    print('\n=== Ablation (d): No state-carry — reset_net before each chunk ===')
    print(f'{"T":>4} {"K":>3} {"max_err (state-carry)":>22}  '
          f'{"max_err (no state-carry)":>26}  {"diverged":>10}')
    rows = []
    test_cases = [(32, 4), (32, 8), (64, 8), (64, 16)]
    b, H = 8, 64
    for T, K in test_cases:
        x = torch.rand(T, b, 3, H, H, device=device)
        y_base = run_baseline(net, x)
        y_carry = run_chunked(net, x, K, do_state_carry=True)
        y_nocarry = run_chunked(net, x, K, do_state_carry=False)
        err_carry = (y_carry - y_base).abs().max().item()
        err_nocarry = (y_nocarry - y_base).abs().max().item()
        diverged = err_nocarry > 1e-3
        print(f'{T:>4} {K:>3} {err_carry:>22.2e}  {err_nocarry:>26.2e}  '
              f'{"Y" if diverged else "N":>10}')
        rows.append({'T':T,'K':K,'b':b,'H':H,
                     'err_carry':err_carry,'err_nocarry':err_nocarry,
                     'diverged':diverged})
        del x, y_base, y_carry, y_nocarry
    return rows


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', nargs='+', default=[], choices=['a','b','c','d'])
    ap.add_argument('--cache', type=str, default='phase1_1d_sweep.npz')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    print('Building spiking_resnet18...')
    net = build_net(device)

    # Load cached fits if available (for ablation a)
    fits = {}
    if os.path.exists(args.cache):
        d = np.load(args.cache, allow_pickle=True)
        results = d['results']
        # Per-(b, H) fit
        bh_pts = {}
        for r in results:
            key = (int(r['B']), int(r['H']))
            bh_pts.setdefault(key, [])
            for kr in r['K_results']:
                if kr.get('peak_mem_GB') is not None:
                    bh_pts[key].append((int(r['T']), int(kr['K']), float(kr['peak_mem_GB'])))
        for key, pts in bh_pts.items():
            if len(pts) < 4: continue
            A = np.array([[1, T, k] for T, k, _ in pts])
            y = np.array([m for _, _, m in pts])
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coeffs
            R2 = 1 - ((y-pred)**2).sum()/max(((y-y.mean())**2).sum(), 1e-9)
            fits[key] = tuple(coeffs.tolist()) + (R2,)
        print(f'  Loaded {len(fits)} per-(b,H) fits')

    out = {}
    if 'a' not in args.skip:
        out['a'] = ablation_a_no_selector(net, device, fits)
    if 'b' not in args.skip:
        out['b'] = ablation_b_no_planner(net, device)
    if 'c' not in args.skip:
        out['c'] = ablation_c_no_streaming(net, device)
    if 'd' not in args.skip:
        out['d'] = ablation_d_no_state_carry(net, device)

    np.savez('p5_3_results.npz', **{f'ablation_{k}': v for k, v in out.items()})
    print('\n  Saved to p5_3_results.npz')


if __name__ == '__main__':
    main()