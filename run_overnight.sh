#!/bin/bash
# run_overnight.sh — master driver: 4 stages run sequentially.
#
# Stage 1: Track A CNN sweep (H=128, b=32, skip transformers)  ~2-3h
# Stage 2: Track A Transformer sweep (H=32, b=128)              ~30-60m
# Stage 3: Track B sanity training (SR-18, 30 epochs)            ~1h
# Stage 4: Track B full training (17 networks, 50 epochs each)   ~30-50h
#
# Total Stage 1-3:  ~4-5 hours (you can stop here for v8.3 §5.2 + §5.6)
# Total Stage 1-4:  ~35-55 hours (1-2 nights, full submission ready)
#
# Each stage runs only if previous succeeded. Logs go to logs/.
#
# Usage:
#     cd /data/yhr/AEROS/
#     conda activate snn118
#     nohup bash run_overnight.sh > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#     tail -f logs/master_*.log
#
# Stop after Stage 3 (skip full training):
#     STAGE_4_SKIP=1 nohup bash run_overnight.sh ...

set -u  # don't set -e; we want each stage's failure logged but continue

LOG_DIR=${LOG_DIR:-logs/}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "=== AEROS overnight master driver ==="
echo "=========================================="
echo "  Start: $(date)"
echo "  Logs:  $LOG_DIR/"
echo ""

# ============================================================
# STAGE 1: Track A CNN sweep (13 nets, H=128, b=32)
# ============================================================
STAGE1_LOG="${LOG_DIR}/stage1_track_a_cnn_${TS}.log"
echo ""
echo ">>> [$(date +%H:%M:%S)] STAGE 1: Track A CNN sweep"
echo "    Output: $STAGE1_LOG"
echo "    Expected: ~2-3 hours, 13 networks"

python p7_3c_extended_multinet.py \
    --H 128 --b 32 --skip-transformers \
    --out p7_3c_cnn_results.npz \
    > "$STAGE1_LOG" 2>&1
RC1=$?

if [[ $RC1 -eq 0 ]]; then
    echo "    [Stage 1 OK]  $(date +%H:%M:%S)"
else
    echo "    [Stage 1 FAIL rc=$RC1]  $(date +%H:%M:%S) — see $STAGE1_LOG"
    echo "    Continuing anyway..."
fi

# ============================================================
# STAGE 2: Track A Transformer sweep (4 nets, H=32, b=128)
# ============================================================
STAGE2_LOG="${LOG_DIR}/stage2_track_a_transformer_${TS}.log"
echo ""
echo ">>> [$(date +%H:%M:%S)] STAGE 2: Track A Transformer sweep"
echo "    Output: $STAGE2_LOG"
echo "    Expected: ~30-60 minutes, 4 transformer networks"

python p7_3c_extended_multinet.py \
    --H 32 --b 128 \
    --archs Spikformer-T,Spikformer-S,QKFormer-T,SDTv1-T \
    --out p7_3c_transformer_results.npz \
    > "$STAGE2_LOG" 2>&1
RC2=$?

if [[ $RC2 -eq 0 ]]; then
    echo "    [Stage 2 OK]  $(date +%H:%M:%S)"
else
    echo "    [Stage 2 FAIL rc=$RC2]  $(date +%H:%M:%S) — see $STAGE2_LOG"
fi

# ============================================================
# STAGE 3: Track B sanity (SR-18, 30 epochs)
# ============================================================
STAGE3_LOG="${LOG_DIR}/stage3_sanity_sr18_${TS}.log"
echo ""
echo ">>> [$(date +%H:%M:%S)] STAGE 3: Track B sanity training (SR-18, 30ep)"
echo "    Output: $STAGE3_LOG"
echo "    Expected: ~1 hour. Goal: confirm trainer reaches 85-90% top-1"

python p7_1_train_cifar10.py \
    --arch SR-18 --epochs 30 -b 128 --T 4 --lr 0.1 --amp \
    --output-dir checkpoints_sanity/ \
    > "$STAGE3_LOG" 2>&1
RC3=$?

# Extract sanity result
if [[ $RC3 -eq 0 ]]; then
    BEST=$(grep "Best test acc" "$STAGE3_LOG" | tail -1)
    echo "    [Stage 3 OK]  $(date +%H:%M:%S) — $BEST"

    # Pull the best acc number from log
    BEST_NUM=$(grep "Best test acc" "$STAGE3_LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [[ -n "$BEST_NUM" ]]; then
        # Use awk for portable float comparison (bash can't do float)
        UNDER_80=$(awk "BEGIN {print ($BEST_NUM < 80)}")
        if [[ "$UNDER_80" == "1" ]]; then
            echo ""
            echo "    [WARN] Sanity acc $BEST_NUM% < 80% — trainer config may be wrong"
            echo "    [WARN] Skipping Stage 4 to avoid wasting V100 time"
            echo "    [WARN] Investigate $STAGE3_LOG before retrying"
            STAGE_4_SKIP=1
        fi
    fi
else
    echo "    [Stage 3 FAIL rc=$RC3]  $(date +%H:%M:%S) — see $STAGE3_LOG"
    echo "    [WARN] Skipping Stage 4 since sanity failed"
    STAGE_4_SKIP=1
fi

# ============================================================
# STAGE 4: Track B full training (17 networks, 50 epochs)
# ============================================================
echo ""
if [[ "${STAGE_4_SKIP:-0}" == "1" ]]; then
    echo ">>> [$(date +%H:%M:%S)] STAGE 4: SKIPPED"
else
    STAGE4_LOG="${LOG_DIR}/stage4_full_train_${TS}.log"
    echo ">>> [$(date +%H:%M:%S)] STAGE 4: Track B full training (17 nets × 50ep)"
    echo "    Output: $STAGE4_LOG"
    echo "    Expected: ~30-50 hours (1-2 nights)"

    EPOCHS=50 BATCH=128 T=4 LR=0.1 \
    bash run_p7_1_all.sh > "$STAGE4_LOG" 2>&1
    RC4=$?

    if [[ $RC4 -eq 0 ]]; then
        echo "    [Stage 4 OK]  $(date +%H:%M:%S)"
    else
        echo "    [Stage 4 FAIL rc=$RC4]  $(date +%H:%M:%S) — see $STAGE4_LOG"
    fi

    # Stage 5 (eval) — only if Stage 4 produced ckpts
    if ls checkpoints/*_cifar10_best.pth >/dev/null 2>&1; then
        STAGE5_LOG="${LOG_DIR}/stage5_eval_${TS}.log"
        echo ""
        echo ">>> [$(date +%H:%M:%S)] STAGE 5: Eval trained checkpoints"
        echo "    Output: $STAGE5_LOG"

        python p7_1_eval_trained.py \
            --ckpt-dir checkpoints/ --T 4 --b 64 \
            > "$STAGE5_LOG" 2>&1
        RC5=$?
        if [[ $RC5 -eq 0 ]]; then
            echo "    [Stage 5 OK]  $(date +%H:%M:%S)"
        else
            echo "    [Stage 5 FAIL rc=$RC5]  see $STAGE5_LOG"
        fi
    fi
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "=== Master driver complete ==="
echo "=========================================="
echo "  End: $(date)"
echo ""
echo "  Stage 1 (CNN sweep H=128):       $([[ $RC1 -eq 0 ]] && echo OK || echo "FAIL ($RC1)")"
echo "  Stage 2 (Transformer sweep):     $([[ $RC2 -eq 0 ]] && echo OK || echo "FAIL ($RC2)")"
echo "  Stage 3 (sanity SR-18 30ep):     $([[ $RC3 -eq 0 ]] && echo OK || echo "FAIL ($RC3)")"
echo "  Stage 4 (full train 17 nets):    $([[ ${STAGE_4_SKIP:-0} -eq 1 ]] && echo SKIPPED || echo "${RC4:-?}")"
echo ""
echo "  Logs saved to $LOG_DIR/*${TS}.log"
echo "  Result npz files in current directory: p7_3c_*_results.npz"
echo "  Trained checkpoints in checkpoints/ (if Stage 4 ran)"