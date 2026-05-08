#!/usr/bin/env bash
# AEROS — train remaining 4 extended-suite archs sequentially.
# ConvLSTM-2L and ConvGRU-2L are already trained.
# This runs LSTM-4L, GRU-4L, CausalTCN-8L, MinimalSSM-2L in order.
#
# Usage: bash train_remaining.sh

set -e
set -u

cd /data/yhr/AEROS/
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

START=$(date +%s)

for arch in LSTM-4L GRU-4L CausalTCN-8L MinimalSSM-2L; do
    if [[ -f "checkpoints_extended/${arch}_best.pth" ]]; then
        echo "[skip] ${arch} already trained"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "Training ${arch} — $(date)"
    echo "================================================================"

    LOG="logs/train_${arch}_${TS}.log"
    A_START=$(date +%s)

    python train_extended.py --arch "${arch}" --epochs 10 2>&1 | tee "${LOG}"

    A_END=$(date +%s)
    echo "[ok] ${arch} done in $(((A_END-A_START)/60))m $(((A_END-A_START)%60))s"
done

END=$(date +%s)
TOTAL=$((END - START))

echo ""
echo "================================================================"
echo "All remaining archs trained"
echo "Total wall: $((TOTAL/60))m $((TOTAL%60))s"
echo "================================================================"
echo ""
echo "Final summary:"
cat checkpoints_extended/training_summary.json | python -m json.tool