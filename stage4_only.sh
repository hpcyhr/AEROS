#!/bin/bash
# stage4_only.sh — standalone Stage 4 (17 networks × CIFAR-10 training).
#
# This script is a self-contained replacement for run_p7_1_all.sh that
# inlines the network list and skip-existing-checkpoint logic, so it
# does not depend on any other shell script being present.
#
# Designed to be relaunched after an overnight master driver hit rc=127
# on Stage 4. Stages 1-3 outputs (p7_3c_*_results.npz, sanity ckpt) are
# untouched.
#
# Usage:
#     cd /data/yhr/AEROS/
#     conda activate snn118
#     nohup bash stage4_only.sh \
#         > logs/stage4_relaunch_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#     tail -f logs/stage4_relaunch_*.log
#
# Time: ~30-50 hours on V100 (17 nets × 50 epochs × ~1-3 min/epoch)

set -u

EPOCHS=${EPOCHS:-50}
T=${T:-4}
BATCH=${BATCH:-128}
LR=${LR:-0.1}
DATA_PATH=${DATA_PATH:-/data/yhr/datasets/cifar10}
CKPT_DIR=${CKPT_DIR:-checkpoints/}
LOG_DIR=${LOG_DIR:-logs/}

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Sanity: p7_1_train_cifar10.py must exist
if [[ ! -f "p7_1_train_cifar10.py" ]]; then
    echo "[FATAL] p7_1_train_cifar10.py not found in $(pwd)"
    echo "[FATAL] aborting"
    exit 1
fi

# Per-arch config: "ARCH:LR:EPOCHS:V_THRESHOLD"
# Networks ordered: lightweight first, heavy last.
#
# v6 hparams (after overnight log diagnosis 2026-05-04):
#
# ROOT CAUSE for failures: deep sequential SNNs (>10 sequential LIF
# layers) with v_th=1.0 hit "spike die-off" — each layer ~30% spike
# rate, so depth=19 ⇒ signal dies at 0.3^19 ≈ 10^-10. Lower v_th=0.5
# restores propagation. ResNet's skip connections rescue SR-18 (residual
# path bypasses LIF), but SR-34/50 have too many residual blocks for
# rescue alone.
#
# v_th=0.5 (deep sequential / no-skip):
#   AlexNet, ZFNet (5+ stages, no skip)        — verified 84.74% / 85.81%
#   MobileNet-V1   (depthwise sep, 13 layers)
#   VGG-13/16/19-BN (deep sequential)
#   SR-34, SR-50   (many residual blocks, skip insufficient)
#   SEW-50, SEW-101 (similar to SR-50)
#
# v_th=1.0 (shallow or strong skip):
#   SR-18, SEW-18                                — verified 86.11% / 89.53%
#   VGG-11-BN                                    — verified 65.18% (acceptable)
#   Spike Transformers
#
# LR rationale unchanged from v5:
#   lr=0.1 for SR-18/SEW-18/VGG-11-BN (verified)
#   lr=0.05 for deep nets (lr=0.1 unstable)
#   lr=0.01 for AlexNet/ZFNet (large param)
#   lr=0.005 for transformers (SGD limit)
#
# SEW-101 BATCH FIX:
#   v5 SEW-101 crashed at init (rc=1, 29s) — likely OOM at b=128 with
#   T=4 AMP. SEW-101 has 44.55M params + heavy activation maps.
#   v6 reduces SEW-101 to per-arch BATCH=64 (handled in trainer launch
#   below).
ARCHS_CFG=(
    "AlexNet:0.01:50:0.5"
    "ZFNet:0.01:50:0.5"
    "MobileNet-V1:0.05:50:0.5"
    "VGG-11-BN:0.05:50:1.0"
    "SR-18:0.1:50:1.0"
    "SEW-18:0.1:50:1.0"
    "Spikformer-T:0.005:50:1.0"
    "QKFormer-T:0.005:50:1.0"
    "SDTv1-T:0.005:50:1.0"
    "VGG-13-BN:0.05:50:0.5"
    "SR-34:0.05:50:0.5"
    "VGG-16-BN:0.05:50:0.5"
    "Spikformer-S:0.005:50:1.0"
    "VGG-19-BN:0.05:50:0.5"
    "SR-50:0.05:50:0.5"
    "SEW-50:0.05:50:0.5"
    "SEW-101:0.05:50:0.5"
)

# Per-arch batch override (helps OOM-prone nets)
declare -A BATCH_OVERRIDE
BATCH_OVERRIDE["SEW-101"]=64
BATCH_OVERRIDE["SEW-50"]=64
BATCH_OVERRIDE["SR-50"]=64

echo "=================================================="
echo "=== Stage 4 standalone: 17-network CIFAR-10 train"
echo "=================================================="
echo "  Networks:  ${#ARCHS_CFG[@]}"
echo "  Per-arch lr: see config table below"
echo "  T:         $T"
echo "  Batch:     $BATCH"
echo "  Data:      $DATA_PATH"
echo "  Ckpt dir:  $CKPT_DIR"
echo "  Log dir:   $LOG_DIR"
echo "  Start:     $(date)"
echo ""

DONE=0
SKIPPED=0
FAILED=0

for CFG in "${ARCHS_CFG[@]}"; do
    # Parse ARCH:LR:EPOCHS:V_THRESHOLD
    ARCH=$(echo "$CFG" | cut -d: -f1)
    ARCH_LR=$(echo "$CFG" | cut -d: -f2)
    ARCH_EPOCHS=$(echo "$CFG" | cut -d: -f3)
    ARCH_VTH=$(echo "$CFG" | cut -d: -f4)

    ARCH_SAFE="${ARCH//-/_}"
    LOG_FILE="${LOG_DIR}/p7_1_${ARCH_SAFE}_$(date +%Y%m%d_%H%M%S).log"
    CKPT_FILE="${CKPT_DIR}/${ARCH_SAFE}_cifar10_best.pth"

    # Determine batch size: override if specified, else default
    ARCH_BATCH=${BATCH_OVERRIDE[$ARCH]:-$BATCH}

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $ARCH  (lr=$ARCH_LR, epochs=$ARCH_EPOCHS, v_th=$ARCH_VTH, b=$ARCH_BATCH)"

    if [[ -f "$CKPT_FILE" ]]; then
        echo "    [SKIP] $CKPT_FILE already exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "    Log: $LOG_FILE"
    set +e
    python p7_1_train_cifar10.py \
        --arch "$ARCH" \
        --data-path "$DATA_PATH" \
        --output-dir "$CKPT_DIR" \
        --epochs "$ARCH_EPOCHS" \
        -b "$ARCH_BATCH" \
        --T "$T" \
        --lr "$ARCH_LR" \
        --v-threshold "$ARCH_VTH" \
        --amp \
        --eval-every 2 \
        > "$LOG_FILE" 2>&1
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        echo "    [FAIL rc=$rc] $ARCH — see $LOG_FILE"
        FAILED=$((FAILED + 1))
    else
        BEST=$(grep "Best test acc" "$LOG_FILE" | tail -1 | awk -F: '{print $NF}' | tr -d ' ')
        echo "    [OK] $ARCH:  best=$BEST"
        DONE=$((DONE + 1))
    fi
done

echo ""
echo "=================================================="
echo "=== Stage 4 done"
echo "=================================================="
echo "  End:       $(date)"
echo "  Done:      $DONE / ${#ARCHS_CFG[@]}"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo ""
echo "  Run eval next:"
echo "    python p7_1_eval_trained.py --ckpt-dir $CKPT_DIR --T $T --b 64"