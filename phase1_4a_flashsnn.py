"""
Phase 1.4a — flash-snn V100 + Triton 2.1.0 compatibility sanity.

Goal:
  1. Verify flash-snn LIF kernel runs on V100 (sm_70)
  2. Compare flash-snn LIF vs SJ JIT inference path (single layer)
  3. Decide whether AEROS can integrate flash-snn ops as kernel layer

Risk: high. flash-snn targets newer Triton, may fail on V100.
"""
import sys
import time, gc
import torch

# Add flash-snn to path (cloned earlier to /tmp/flash-snn)
sys.path.insert(0, '/tmp/flash-snn')

print('=== Step 1: import flash-snn ===')
try:
    import flashsnn
    print(f'  flashsnn imported, dir: {[x for x in dir(flashsnn) if not x.startswith("_")][:5]}')
except Exception as e:
    print(f'  IMPORT FAIL: {str(e)[:200]}')
    sys.exit(1)

print('\n=== Step 2: test flash-snn multistep LIF kernel ===')
try:
    from flashsnn.ops.lif import multistep_lif_forward
    print(f'  multistep_lif_forward imported')
except Exception as e:
    try:
        # Try alternate API
        from flashsnn.ops import lif as lif_module
        print(f'  flashsnn.ops.lif module: {[x for x in dir(lif_module) if not x.startswith("_")]}')
    except Exception as e2:
        print(f'  IMPORT FAIL: {e}')
        print(f'  ALT FAIL: {e2}')
        sys.exit(1)

print('\n=== Step 3: try forward kernel on V100 ===')
device = torch.device('cuda:0')
T, B, C = 32, 16, 64
x = torch.randn(T, B, C, device=device)
v0 = torch.zeros(B, C, device=device)
print(f'  Input shape: T={T} B={B} C={C}')

try:
    # Try 1: keyword args
    spike, v_final = multistep_lif_forward(
        x_seq=x,
        v_init=v0,
        v_th=1.0,
        v_reset=0.0,
        tau=2.0,
    )
    print(f'  forward OK, spike shape: {spike.shape}, v_final shape: {v_final.shape}')
    print(f'  spike dtype: {spike.dtype}, sum: {spike.sum().item():.0f}')
except Exception as e:
    print(f'  forward FAIL (kw args): {str(e)[:300]}')
    try:
        # Try 2: positional args
        spike, v_final = multistep_lif_forward(x, v0, 1.0, 0.0, 2.0)
        print(f'  forward OK (positional), spike shape: {spike.shape}')
    except Exception as e2:
        print(f'  forward FAIL (positional): {str(e2)[:300]}')
        sys.exit(1)

print('\n=== Step 4: benchmark flash-snn vs SJ JIT (single layer) ===')
from spikingjelly.activation_based import neuron, functional


@torch.no_grad()
def sj_forward(x):
    n = neuron.LIFNode(detach_reset=True).to(x.device)
    n.eval()
    functional.set_step_mode(n, step_mode='m')
    return n(x)


def measure(fn, *args, n_warmup=3, n_iters=10):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000

# Larger test (matches Phase C config)
T, B, C, H, W = 32, 16, 64, 56, 56
x_5d = torch.randn(T, B, C, H, W, device=device)

print(f'  Test config: T={T} B={B} C={C}*{H}*{W}')

# SJ inference path
sj_ms = measure(sj_forward, x_5d)
print(f'  SJ JIT (eval mode): {sj_ms:.2f} ms')

# flash-snn (need to flatten to [T, batch_dim])
x_flat = x_5d.reshape(T, B * C * H * W)
v0_flat = torch.zeros(B * C * H * W, device=device)


def flashsnn_forward(x):
    return multistep_lif_forward(x, v0_flat, 1.0, 0.0, 2.0)


try:
    fs_ms = measure(flashsnn_forward, x_flat)
    print(f'  flash-snn Triton:    {fs_ms:.2f} ms')
    print(f'  speedup: {sj_ms / fs_ms:.2f}x')
except Exception as e:
    print(f'  flash-snn benchmark FAIL: {str(e)[:200]}')

print('\n=== Verdict ===')
print('  If speedup > 1.5x: flash-snn worth integrating')
print('  If speedup < 1.0x: V100 + Triton 2.1 hurts flash-snn, skip integration')
print('  If FAIL: flash-snn V100 incompatible, paper cites it but does not integrate')
