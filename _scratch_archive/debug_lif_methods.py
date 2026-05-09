"""Quick debug: find which method SpikingJelly LIFNode calls in multi-step mode."""
import inspect
from spikingjelly.activation_based import neuron, functional
import torch

print("=" * 80)
print("LIFNode method inspection")
print("=" * 80)

lif = neuron.LIFNode(tau=2.0, v_threshold=1.0)
functional.set_step_mode(lif, 'm')

print(f"\nstep_mode: {lif.step_mode}")
print()

print("Methods on LIFNode:")
for n in ['forward', 'single_step_forward', 'multi_step_forward',
          'neuronal_charge', 'neuronal_fire', 'neuronal_reset']:
    m = getattr(lif, n, None)
    if m is not None:
        try:
            cls = m.__qualname__.split('.')[0] if hasattr(m, '__qualname__') else 'unknown'
            print(f"  {n}: defined in {cls}")
        except Exception:
            print(f"  {n}: present")

print()
print("--- forward signature & first lines ---")
try:
    print(inspect.getsource(lif.__class__.forward).split('\n', 30)[:30].__str__()[:1500])
except Exception as e:
    print(f"  err: {e}")

print()
print("--- multi_step_forward source ---")
try:
    src = inspect.getsource(lif.multi_step_forward)
    print(src[:2000])
except Exception as e:
    print(f"  err: {e}")

print()
print("--- neuronal_charge source ---")
try:
    src = inspect.getsource(lif.neuronal_charge)
    print(src)
except Exception as e:
    print(f"  err: {e}")

print()
print("--- neuronal_fire source ---")
try:
    src = inspect.getsource(lif.neuronal_fire)
    print(src)
except Exception as e:
    print(f"  err: {e}")

# Test which method actually gets called
print()
print("=" * 80)
print("Run-time hook test")
print("=" * 80)

calls = []
orig_single = lif.__class__.single_step_forward
orig_multi  = lif.__class__.multi_step_forward
orig_charge = lif.__class__.neuronal_charge
orig_fire   = lif.__class__.neuronal_fire

def trace(name, orig):
    def wrapped(self, *a, **kw):
        calls.append(name)
        return orig(self, *a, **kw)
    return wrapped

lif.__class__.single_step_forward = trace('single_step_forward', orig_single)
lif.__class__.multi_step_forward  = trace('multi_step_forward',  orig_multi)
lif.__class__.neuronal_charge     = trace('neuronal_charge',     orig_charge)
lif.__class__.neuronal_fire       = trace('neuronal_fire',       orig_fire)

# T=4, B=2, C=8, H=W=4
x = torch.randn(4, 2, 8, 4, 4) * 0.5
out = lif(x)
functional.reset_net(lif)

print(f"\nMethod call sequence (first 20):")
for c in calls[:20]:
    print(f"  - {c}")
print(f"\nTotal calls: {len(calls)}")
print(f"Unique methods called: {set(calls)}")
print(f"\nOutput shape: {out.shape}")