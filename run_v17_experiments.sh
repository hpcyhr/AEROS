#!/usr/bin/env bash
# AEROS v16 → v17 phase: 16-arch downstream experiments
#
# Sequential execution of:
#   Stage 1: Ablation 2 (no-envelope solver) on 16 archs
#   Stage 2: Ablation 3 (manual fixed-kappa) on 16 archs   [depends on Stage 1 output]
#   Stage 3: §5.4 IO-streaming flat-T smoke (5 archs cross-family)
#
# Prerequisites in /data/yhr/AEROS/:
#   - p9_1a_full16.json (already merged)
#   - p9_6b_no_envelope_solver_v3.py
#   - p9_6c_manual_fixed_kappa_v3.py
#   - p9_iostream_flat_t.py
#   - aeros_dispatch.py
#   - aeros_models_extended.py
#
# Usage:
#   bash run_v17_experiments.sh

set -e  # halt on any error
set -u  # halt on undefined variable

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
WORK_DIR="/data/yhr/AEROS"
LOG_DIR="${WORK_DIR}/logs"
TS=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="${LOG_DIR}/v17_session_${TS}.log"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

echo "================================================================" | tee "$SESSION_LOG"
echo "AEROS v17 16-arch experiment pipeline"                              | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                   | tee -a "$SESSION_LOG"
echo "Work dir: $WORK_DIR"                                                | tee -a "$SESSION_LOG"
echo "Session log: $SESSION_LOG"                                          | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"
echo ""                                                                  | tee -a "$SESSION_LOG"

START_TS=$(date +%s)

# Pre-flight: verify required files
echo "Pre-flight check..."                                                | tee -a "$SESSION_LOG"
for f in p9_1a_full16.json \
         p9_6b_no_envelope_solver_v3.py \
         p9_6c_manual_fixed_kappa_v3.py \
         p9_iostream_flat_t.py \
         aeros_dispatch.py \
         aeros_models_extended.py; do
    if [[ ! -f "$f" ]]; then
        echo "  [FATAL] missing: $f"                                      | tee -a "$SESSION_LOG"
        exit 1
    fi
    echo "  ok: $f"                                                        | tee -a "$SESSION_LOG"
done
echo ""                                                                   | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Stage 1: Ablation 2 — no-envelope solver on 16 archs
# -----------------------------------------------------------------------------
STAGE1_LOG="${LOG_DIR}/p9_6b_full16_${TS}.log"
STAGE1_OUT="p9_6b_no_envelope_full16"

echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage 1: Ablation 2 — no-envelope solver (16 archs)"                | tee -a "$SESSION_LOG"
echo "Estimated: ~30-45 min on V100"                                      | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                   | tee -a "$SESSION_LOG"
echo "Output:  ${STAGE1_OUT}.json"                                        | tee -a "$SESSION_LOG"
echo "Log:     ${STAGE1_LOG}"                                             | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

S1_START=$(date +%s)
python p9_6b_no_envelope_solver_v3.py \
    --coeffs p9_1a_full16.json \
    --Ts 128 1024 \
    --Ms 4.0 8.0 16.0 \
    --modes 2 4 \
    --output "$STAGE1_OUT" \
    2>&1 | tee "$STAGE1_LOG"
S1_END=$(date +%s)
S1_DURATION=$((S1_END - S1_START))

if [[ ! -f "${STAGE1_OUT}.json" ]]; then
    echo ""                                                               | tee -a "$SESSION_LOG"
    echo "[FATAL] Stage 1 did not produce ${STAGE1_OUT}.json"              | tee -a "$SESSION_LOG"
    echo "Halting before Stage 2 (which depends on Stage 1 output)."      | tee -a "$SESSION_LOG"
    exit 1
fi

echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 1 complete in $((S1_DURATION / 60))m $((S1_DURATION % 60))s"   | tee -a "$SESSION_LOG"
echo ""                                                                   | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Stage 2: Ablation 3 — manual fixed-kappa baseline
# -----------------------------------------------------------------------------
STAGE2_LOG="${LOG_DIR}/p9_6c_full16_${TS}.log"
STAGE2_OUT="p9_6c_manual_kappa_full16"

echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage 2: Ablation 3 — manual fixed-kappa (16 archs)"                | tee -a "$SESSION_LOG"
echo "Estimated: ~40-60 min on V100"                                      | tee -a "$SESSION_LOG"
echo "Reads: ${STAGE1_OUT}.json"                                          | tee -a "$SESSION_LOG"
echo "Output: ${STAGE2_OUT}.json"                                         | tee -a "$SESSION_LOG"
echo "Log:    ${STAGE2_LOG}"                                              | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                   | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

S2_START=$(date +%s)
python p9_6c_manual_fixed_kappa_v3.py \
    --coeffs p9_1a_full16.json \
    --ablation2_json "${STAGE1_OUT}.json" \
    --Ts 128 1024 \
    --Ms 4.0 8.0 16.0 \
    --modes 2 4 \
    --output "$STAGE2_OUT" \
    2>&1 | tee "$STAGE2_LOG"
S2_END=$(date +%s)
S2_DURATION=$((S2_END - S2_START))

if [[ ! -f "${STAGE2_OUT}.json" ]]; then
    echo ""                                                               | tee -a "$SESSION_LOG"
    echo "[WARN] Stage 2 did not produce ${STAGE2_OUT}.json"               | tee -a "$SESSION_LOG"
    echo "Continuing to Stage 3 (independent)..."                          | tee -a "$SESSION_LOG"
fi

echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 2 complete in $((S2_DURATION / 60))m $((S2_DURATION % 60))s"   | tee -a "$SESSION_LOG"
echo ""                                                                   | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Stage 3: §5.4 IO-streaming flat-T cross-family smoke
# -----------------------------------------------------------------------------
STAGE3_LOG="${LOG_DIR}/p9_iostream_smoke_${TS}.log"
STAGE3_OUT="p9_iostream_flat_t_smoke"

echo "================================================================" | tee -a "$SESSION_LOG"
echo "Stage 3: §5.4 IO-streaming flat-T smoke (5 archs cross-family)"     | tee -a "$SESSION_LOG"
echo "Estimated: ~10-20 min on V100"                                      | tee -a "$SESSION_LOG"
echo "Output: ${STAGE3_OUT}.json"                                         | tee -a "$SESSION_LOG"
echo "Log:    ${STAGE3_LOG}"                                              | tee -a "$SESSION_LOG"
echo "Started: $(date)"                                                   | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

S3_START=$(date +%s)
python p9_iostream_flat_t.py \
    --coeffs p9_1a_full16.json \
    --nets SR-18 VGG-19-BN ConvLSTM-2L LSTM-4L MinimalSSM-2L \
    --Ts 128 1024 4096 16384 \
    --kappa 8 \
    --output "$STAGE3_OUT" \
    2>&1 | tee "$STAGE3_LOG"
S3_END=$(date +%s)
S3_DURATION=$((S3_END - S3_START))

echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 3 complete in $((S3_DURATION / 60))m $((S3_DURATION % 60))s"   | tee -a "$SESSION_LOG"
echo ""                                                                   | tee -a "$SESSION_LOG"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
END_TS=$(date +%s)
TOTAL=$((END_TS - START_TS))

echo "================================================================" | tee -a "$SESSION_LOG"
echo "All stages complete"                                                | tee -a "$SESSION_LOG"
echo "Total: $((TOTAL / 60))m $((TOTAL % 60))s"                           | tee -a "$SESSION_LOG"
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Output files:"                                                      | tee -a "$SESSION_LOG"
for f in "${STAGE1_OUT}.json" "${STAGE2_OUT}.json" "${STAGE3_OUT}.json"; do
    if [[ -f "$f" ]]; then
        sz=$(du -h "$f" | cut -f1)
        echo "  $f  ($sz)"                                                | tee -a "$SESSION_LOG"
    else
        echo "  $f  (MISSING)"                                            | tee -a "$SESSION_LOG"
    fi
done
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Per-stage logs:"                                                    | tee -a "$SESSION_LOG"
echo "  Stage 1: $STAGE1_LOG"                                             | tee -a "$SESSION_LOG"
echo "  Stage 2: $STAGE2_LOG"                                             | tee -a "$SESSION_LOG"
echo "  Stage 3: $STAGE3_LOG"                                             | tee -a "$SESSION_LOG"
echo "  Session: $SESSION_LOG"                                            | tee -a "$SESSION_LOG"
echo "================================================================" | tee -a "$SESSION_LOG"

# Print one-line summary lines from each stage's last summary block
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "=== Quick summary preview (grep summary lines) ==="                 | tee -a "$SESSION_LOG"
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 1 (Ablation 2) summary tail:"                                 | tee -a "$SESSION_LOG"
tail -20 "$STAGE1_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 2 (Ablation 3) summary tail:"                                 | tee -a "$SESSION_LOG"
tail -20 "$STAGE2_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true
echo ""                                                                   | tee -a "$SESSION_LOG"
echo "Stage 3 (flat-T smoke) summary tail:"                               | tee -a "$SESSION_LOG"
tail -20 "$STAGE3_LOG" 2>/dev/null | tee -a "$SESSION_LOG" || true