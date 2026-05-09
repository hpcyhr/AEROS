"""
Verify class-level patch actually changes the bound method.
Run on the actual SEW-RN18 net to see what's happening.
"""
import sys, os
sys.path.insert(0, '/data/yhr/CATFuse/training')

import torch
from phaseC1_train_sew_cifar10 import build_sew_model, encode
from spikingjelly.activation_based import functional, neuron

device = torch.device('cuda:0')
net = build_sew_model(num_classes=10, v_threshold=1.0, tau=2.0,
                      cifar10_stem=True).to(device).eval()

# Find first LIF
first_lif = None
first_lif_name = None
for name, mod in net.named_modules():
    if isinstance(mod, neuron.LIFNode):
        first_lif = mod
        first_lif_name = name
        break

print(f"First LIF: {first_lif_name}")
print(f"  type: {type(first_lif)}")
print(f"  step_mode: {first_lif.step_mode}")
print(f"  backend: {first_lif.backend}")
print(f"  training: {first_lif.training}")
print()

# Check the multi_step_forward bound to it
print(f"  multi_step_forward bound method: {first_lif.multi_step_forward}")
print(f"  multi_step_forward.__qualname__: {first_lif.multi_step_forward.__qualname__}")
print(f"  is from class: {first_lif.multi_step_forward.__func__ is type(first_lif).multi_step_forward}")
print()

# Apply class-level patch
print("Applying class patch...")
calls_log = []
orig_multi = neuron.LIFNode.multi_step_forward
orig_charge = neuron.LIFNode.neuronal_charge
orig_fire = neuron.BaseNode.neuronal_fire

def traced_multi(self, x_seq):
    calls_log.append(('multi_step_forward', id(self), x_seq.shape))
    return orig_multi(self, x_seq)

def traced_charge(self, x):
    calls_log.append(('neuronal_charge', id(self)))
    return orig_charge(self, x)

def traced_fire(self):
    calls_log.append(('neuronal_fire', id(self)))
    return orig_fire(self)

neuron.LIFNode.multi_step_forward = traced_multi
neuron.LIFNode.neuronal_charge = traced_charge
neuron.BaseNode.neuronal_fire = traced_fire

print(f"  After patch, first_lif.multi_step_forward: {first_lif.multi_step_forward}")
print(f"  __qualname__: {first_lif.multi_step_forward.__qualname__}")
print()

# Run forward
print("Running forward on real data (B=4, just 1 batch for trace)...")
dummy = torch.randn(4, 3, 32, 32, device=device)
with torch.no_grad():
    functional.reset_net(net)
    _ = net(encode(dummy, 4))

# Restore
neuron.LIFNode.multi_step_forward = orig_multi
neuron.LIFNode.neuronal_charge = orig_charge
neuron.BaseNode.neuronal_fire = orig_fire

print(f"\nTotal traced calls: {len(calls_log)}")
print(f"Method types called: {set(c[0] for c in calls_log)}")
print()

# Group by method
from collections import Counter
method_counts = Counter(c[0] for c in calls_log)
print(f"Method counts: {dict(method_counts)}")
print()

# Check: is first_lif's multi_step_forward called?
first_lif_id = id(first_lif)
first_lif_calls = [c for c in calls_log if len(c) > 1 and c[1] == first_lif_id]
print(f"Calls to first_lif (id={first_lif_id}): {len(first_lif_calls)}")
for c in first_lif_calls[:10]:
    print(f"  {c}")

print()
print("First 30 calls overall:")
for c in calls_log[:30]:
    print(f"  {c}")