#!/usr/bin/env bash
# AEROS Phase 1 post-training pipeline:
#   Stage B: T1-B Bit-exact wrapper invariance + boundary stress for 6
#            trained extended archs (extends §5.1 to 23 trained archs)
#   Stage C: T1-C Cross-family flat-T full 16-arch grid (upgrades §5.4-bis
#            from 5-arch smoke to 16-arch full)
#   Stage D: T2-A Cross-architecture held-out E2 (3 random splits, 3
#            held-out archs each)
#
# Prerequisites:
#   - 6 extended archs already trained (training_summary.json exists)
#   - p9_1a_full16.json (16-arch unified coeff bundle)
#   - p9_bitexact_extended.py
#   - p9_iostream_flat_t_v3.py
#   - p9_6e_cross_arch_holdout.py
#
# Usage:
#   bash run_phase1_post_training.sh

set -e
set -u

WORK_DIR="/data/yhr/AEROS"
LOG_DIR="${WORK_DIR}/logs"
TS=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="${LOG_DIR}/phase1_post_${TS}.log"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

echo "================================================================" | tee "$SESSION_LOG"
echo "AEROS Phase 1 post-training pipeline"                                | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                    | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

# Pre-flight
echo "Pre-flight check..."                                                 | tee -a "$SESSION_LOG"
required=(
    "p9_1a_full16.json"
    "p9_bitexact_extended.py"
    "p9_iostream_flat_t_v3.py"
    "p9_6e_cross_arch_holdout.py"
    "aeros_dispatch.py"
    "aeros_models_extended.py"
    "train_extended.py"
    "checkpoints_extended/training_summary.json"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  [FATAL] missing: $f"                                       | tee -a "$SESSION_LOG"
        exit 1
    fi
    echo "  ok: $f"                                                         | tee -a "$SESSION_LOG"
done

# Verify training completed for 6 archs
echo "" | tee -a "$SESSION_LOG"
echo "Trained extended checkpoints:"                                       | tee -a "$SESSION_LOG"
for arch in ConvLSTM-2L ConvGRU-2L LSTM-4L GRU-4L CausalTCN-8L MinimalSSM-2L; do
    if [[ -f "checkpoints_extended/${arch}_best.pth" ]]; then
        sz=$(du -h "checkpoints_extended/${arch}_best.pth" | cut -f1)
        echo "  ok: ${arch}_best.pth ($sz)"                                 | tee -a "$SESSION_LOG"
    else
        echo "  [FATAL] missing: ${arch}_best.pth"                           | tee -a "$SESSION_LOG"
        exit 1
    fi
done

START_TS=$(date +%s)

# -----------------------------------------------------------------------------
# Stage B: T1-B Bit-exact for trained extended archs
# -----------------------------------------------------------------------------
SB_LOG="${LOG_DIR}/p9_bitexact_ext_${TS}.log"
SB_OUT="p9_bitexact_extended"

echo "" | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage B: T1-B Bit-exact extended (6 trained archs)"                  | tee -a "$SESSION_LOG"
echo "Estimated: ~10-30 min on V100"                                       | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                    | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

SB_START=$(date +%s)
python p9_bitexact_extended.py --output "$SB_OUT" 2>&1 | tee "$SB_LOG"
SB_END=$(date +%s)

if [[ ! -f "${SB_OUT}.json" ]]; then
    echo "[WARN] Stage B did not produce ${SB_OUT}.json; continuing"       | tee -a "$SESSION_LOG"
fi
echo "Stage B done in $(((SB_END-SB_START)/60))m $(((SB_END-SB_START)%60))s" | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Stage C: T1-C Cross-family flat-T full 16-arch grid
# -----------------------------------------------------------------------------
SC_LOG="${LOG_DIR}/p9_iostream_full_${TS}.log"
SC_OUT="p9_iostream_flat_t_full"

echo "" | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage C: T1-C Cross-family flat-T full 16-arch grid"                 | tee -a "$SESSION_LOG"
echo "Estimated: ~60-120 min on V100 (CausalTCN T=16384 may be skipped)"   | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                    | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

SC_START=$(date +%s)
python p9_iostream_flat_t_v3.py \
    --coeffs p9_1a_full16.json \
    --Ts 128 256 512 1024 2048 4096 8192 16384 \
    --kappa 8 \
    --output "$SC_OUT" \
    2>&1 | tee "$SC_LOG"
SC_END=$(date +%s)

if [[ ! -f "${SC_OUT}.json" ]]; then
    echo "[WARN] Stage C did not produce ${SC_OUT}.json; continuing"       | tee -a "$SESSION_LOG"
fi
echo "Stage C done in $(((SC_END-SC_START)/60))m $(((SC_END-SC_START)%60))s" | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Stage D: T2-A Cross-architecture held-out E2
# -----------------------------------------------------------------------------
SD_LOG="${LOG_DIR}/p9_6e_${TS}.log"
SD_OUT="p9_6e_cross_arch_holdout"

echo "" | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage D: T2-A Cross-architecture held-out E2 (3 splits, 3 held-out)" | tee -a "$SESSION_LOG"
echo "Estimated: ~30-90 min on V100"                                       | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                    | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

SD_START=$(date +%s)
python p9_6e_cross_arch_holdout.py \
    --coeffs p9_1a_full16.json \
    --Ts 128 1024 \
    --Ms 4.0 8.0 16.0 24.0 \
    --modes 2 4 \
    --n_splits 3 \
    --n_holdout 3 \
    --output "$SD_OUT" \
    2>&1 | tee "$SD_LOG"
SD_END=$(date +%s)

echo "Stage D done in $(((SD_END-SD_START)/60))m $(((SD_END-SD_START)%60))s" | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
END_TS=$(date +%s)
TOTAL=$((END_TS - START_TS))

echo "" | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"
echo "Phase 1 post-training pipeline complete"                             | tee -a "$SESSION_LOG"
echo "Total: $((TOTAL/60))m $((TOTAL%60))s"                                | tee -a "$SESSION_LOG"
echo ""                                                                    | tee -a "$SESSION_LOG"
echo "Output files:"                                                       | tee -a "$SESSION_LOG"
for f in "${SB_OUT}.json" "${SC_OUT}.json" "${SD_OUT}.json"; do
    if [[ -f "$f" ]]; then
        sz=$(du -h "$f" | cut -f1); echo "  $f ($sz)" | tee -a "$SESSION_LOG"
    else
        echo "  $f (MISSING)" | tee -a "$SESSION_LOG"
    fi
done
echo ""                                                                    | tee -a "$SESSION_LOG"
echo "Per-stage logs:"                                                     | tee -a "$SESSION_LOG"
echo "  Stage B: $SB_LOG"                                                  | tee -a "$SESSION_LOG"
echo "  Stage C: $SC_LOG"                                                  | tee -a "$SESSION_LOG"
echo "  Stage D: $SD_LOG"                                                  | tee -a "$SESSION_LOG"
echo "  Session: $SESSION_LOG"                                             | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

echo ""                                                                    | tee -a "$SESSION_LOG"
echo "=== Quick summary preview ==="                                       | tee -a "$SESSION_LOG"
echo ""                                                                    | tee -a "$SESSION_LOG"
echo "Stage B summary tail:"                                               | tee -a "$SESSION_LOG"
tail -25 "$SB_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true
echo ""                                                                    | tee -a "$SESSION_LOG"
echo "Stage C summary tail:"                                               | tee -a "$SESSION_LOG"
tail -25 "$SC_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true
echo ""                                                                    | tee -a "$SESSION_LOG"
echo "Stage D summary tail:"                                               | tee -a "$SESSION_LOG"
tail -25 "$SD_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true