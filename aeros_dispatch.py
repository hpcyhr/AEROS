"""
AEROS shared dispatch utility for ablation/calibration tools.

Provides a unified API for building, resetting, and configuring per-arch
input shapes across both the SNN suite (spikingjelly) and the extended
suite (aeros_models_extended). Tools that previously hardcoded SNN-only
behavior should import this module.

Usage:
    from aeros_dispatch import (
        build_any_net, reset_any_net, get_arch_config,
        load_unified_bundle,
    )

    bundle = load_unified_bundle("p9_1a_full16.json")
    for net_name in bundle["coeffs"]:
        cfg = get_arch_config(bundle, net_name)
        # cfg = {"b":32, "C":3, "H":128 or 32, "num_classes":10, "provenance": "snn"|"extended"}
        net = build_any_net(net_name, cfg["provenance"],
                            num_classes=cfg["num_classes"],
                            H=cfg["H"], C=cfg["C"])
        net = net.to(device)
        reset_any_net(net, cfg["provenance"])
        # ... profile/measure ...
"""

from __future__ import annotations

import json
from typing import Dict, Optional


# ============================================================================
# Bundle loading + provenance lookup
# ============================================================================

def load_unified_bundle(path: str) -> Dict:
    """Load a coefficient bundle. Supports three layouts:
      (a) 16-arch unified: {config_snn, config_extended, coeffs, raw,
                             arch_provenance}
      (b) 10-arch SNN-only: {config, coeffs, raw}
      (c) Legacy bare dict (no top-level): treat all as SNN
    """
    with open(path) as f:
        data = json.load(f)
    if "arch_provenance" in data:
        return data  # unified
    if "coeffs" in data and "config" in data:
        # SNN-only legacy: synthesize provenance
        provenance = {name: "snn" for name in data["coeffs"]}
        return {
            "config_snn": data["config"],
            "coeffs": data["coeffs"],
            "raw": data.get("raw", {}),
            "arch_provenance": provenance,
        }
    # Bare dict
    coeffs = data
    provenance = {name: "snn" for name in coeffs}
    return {
        "config_snn": {"b": 32, "C": 3, "H": 128, "num_classes": 10},
        "coeffs": coeffs,
        "raw": {},
        "arch_provenance": provenance,
    }


def get_arch_config(bundle: Dict, net_name: str) -> Dict:
    """Return per-arch (b, C, H, num_classes, provenance) tuple as dict.

    SNN archs default to b=32 C=3 H=128 (p9_1a SNN config).
    Extended archs default to b=32 C=3 H=32 (p9_1a extended config).
    Falls back to SNN config if provenance is unknown.
    """
    prov = bundle.get("arch_provenance", {}).get(net_name, "snn")
    if prov == "extended":
        cfg = bundle.get("config_extended", {})
        b = cfg.get("b", 32)
        C = cfg.get("C", 3)
        H = cfg.get("H", 32)
        nc = cfg.get("num_classes", 10)
    else:
        cfg = bundle.get("config_snn", bundle.get("config", {}))
        b = cfg.get("b", 32)
        C = cfg.get("C", 3)
        H = cfg.get("H", 128)
        nc = cfg.get("num_classes", 10)
    return {"b": b, "C": C, "H": H, "num_classes": nc, "provenance": prov}


# ============================================================================
# Build dispatch
# ============================================================================

def build_any_net(net_name: str, provenance: str,
                  num_classes: int = 10, H: int = 128, C: int = 3):
    """Build a net. Dispatches by provenance.

    SNN: spikingjelly.activation_based.model.{spiking_resnet, sew_resnet,
                                              spiking_vgg}
    Extended: aeros_models_extended.build_extended_net
    """
    if provenance == "extended":
        from aeros_models_extended import build_extended_net
        return build_extended_net(net_name, num_classes=num_classes,
                                   H=H, in_ch=C)

    # SNN suite (default)
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model.spiking_resnet import (
        spiking_resnet18, spiking_resnet34, spiking_resnet50)
    from spikingjelly.activation_based.model.sew_resnet import (
        sew_resnet18, sew_resnet50, sew_resnet101)
    from spikingjelly.activation_based.model.spiking_vgg import (
        spiking_vgg11_bn, spiking_vgg13_bn, spiking_vgg16_bn,
        spiking_vgg19_bn)
    common = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=num_classes,
    )
    table = {
        "SR-18":   lambda: spiking_resnet18(**common),
        "SR-34":   lambda: spiking_resnet34(**common),
        "SR-50":   lambda: spiking_resnet50(**common),
        "SEW-18":  lambda: sew_resnet18(cnf="ADD", **common),
        "SEW-50":  lambda: sew_resnet50(cnf="ADD", **common),
        "SEW-101": lambda: sew_resnet101(cnf="ADD", **common),
        "VGG-11-BN": lambda: spiking_vgg11_bn(**common),
        "VGG-13-BN": lambda: spiking_vgg13_bn(**common),
        "VGG-16-BN": lambda: spiking_vgg16_bn(**common),
        "VGG-19-BN": lambda: spiking_vgg19_bn(**common),
    }
    if net_name not in table:
        raise ValueError(f"unknown SNN net: {net_name}")
    net = table[net_name]()
    net.eval()
    functional.set_step_mode(net, "m")
    return net


def reset_any_net(net, provenance: str) -> None:
    """Reset hidden state. Dispatches by provenance."""
    if provenance == "extended":
        from aeros_models_extended import reset_state_extended
        reset_state_extended(net)
        return
    # SNN suite
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(net)
    except Exception:
        pass