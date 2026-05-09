"""Inspect SpikingJelly LIFNode in eval mode + check what multi_step_forward does."""
import inspect
from spikingjelly.activation_based import neuron, functional
import torch

print("=" * 80)
print("Full multi_step_forward source")
print("=" * 80)
src = inspect.getsource(neuron.LIFNode.multi_step_forward)
print(src)
print()

print("=" * 80)
print("Eval-mode trace test")
print("=" * 80)
lif = neuron.LIFNode(tau=2.0, v_threshold=1.0)
functional.set_step_mode(lif, 'm')
lif.eval()  # <- KEY: eval mode

print(f"step_mode: {lif.step_mode}")
print(f"backend:   {lif.backend}")
print(f"training:  {lif.training}")
print()

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

x = torch.randn(4, 2, 8, 4, 4) * 0.5
with torch.no_grad():
    out = lif(x)
functional.reset_net(lif)

print(f"Method calls: {calls}")
print(f"Unique: {set(calls)}")
print()

print("=" * 80)
print("MemoryModule.multi_step_forward (super())")
print("=" * 80)
from spikingjelly.activation_based.base import MemoryModule
print(inspect.getsource(MemoryModule.multi_step_forward))