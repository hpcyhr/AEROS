# AEROS

Long-horizon GPU runtime for event-camera SNN inference. Closed-form
memory–wallclock selectors and streaming-input execution on commodity
GPUs.

## Status
Pre-submission. Paper draft in `aeros.tex`. Experiment scripts:
- `p7_3c_extended_multinet.py` — 17-network memory model + selector + max-T sweep
- `p7_1_train_cifar10.py` — CIFAR-10 trainer (any of 17 architectures)
- `p7_1_train_dvsg.py` — DVS128 Gesture trainer (any of 17 architectures)
- `p7_1_eval_trained.py` — bit-exact AEROS preservation eval

## Hardware
V100-SXM2-32GB (sm_70) and A100-SXM4-40GB (sm_80).

## Dependencies
PyTorch 2.1.0 + cu118, SpikingJelly 0.0.0.0.14, CUDA 11.8.
Network wrappers borrowed from `/data/yhr/CATFuse/` (sibling project).

## Reproduction
See `run_overnight.sh` (V100) / `run_a100_overnight.sh` (A100).
