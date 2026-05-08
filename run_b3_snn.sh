#!/usr/bin/env bash
# AEROS B3 re-run — disjoint-fold variance, restricted to 10 SNN archs.
#
# The previous B3 run crashed on ConvLSTM-2L because p9_6d_disjoint_calibration
# uses its own build_net which only knows the 10 SNN archs. Original §5.6 is
# scoped to those 10 archs anyway, so this re-run is the historically correct
# scope.
#
# Wall: ~2-3 hr V100.
#
# Usage:
#   bash run_b3_snn.sh
# Or background:
#   nohup bash run_b3_snn.sh > /dev/null 2>&1 &
#   echo $!
#   tail -f logs/b3_variance_snn_*.log

set -e
set -u

cd /data/yhr/AEROS/
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

LOG="logs/b3_variance_snn_${TS}.log"

echo "================================================================"
echo "AEROS B3 re-run (disjoint-fold variance, 10 SNN archs)"
echo "Started: $(date)"
echo "Log: ${LOG}"
echo "================================================================"

# Pre-flight
for f in p9_1a_full16.json p9_6d_disjoint_calibration.py p9_6d_run_variance.py; do
    if [[ ! -f "${f}" ]]; then
        echo "  [FATAL] missing: ${f}"
        exit 1
    fi
done

START=$(date +%s)

python -u p9_6d_run_variance.py \
    --coeffs p9_1a_full16.json \
    --seeds 42 123 7 \
    --Ts 128 1024 4096 \
    --Ms 4 8 16 24 \
    --modes 2 4 \
    --nets SR-18 SR-34 SR-50 SEW-18 SEW-50 SEW-101 \
           VGG-11-BN VGG-13-BN VGG-16-BN VGG-19-BN \
    --batch_size 32 \
    --H 128 --C 3 \
    --delta 0.05 \
    --epsilon_safety 0.02 \
    --output_prefix p9_6d_variance_snn \
    2>&1 | tee "${LOG}"

END=$(date +%s)
DT=$((END - START))

echo ""
echo "================================================================"
echo "B3 re-run done in $((DT / 60))m $((DT % 60))s"
echo ""
echo "Output:"
LATEST=$(ls -t p9_6d_variance_snn_aggregate_*.json 2>/dev/null | head -1)
echo "  ${LATEST:-MISSING}"
echo ""

if [[ -n "${LATEST:-}" ]]; then
    echo "=== B3 aggregate summary ==="
    python -c "
import json
d = json.load(open('${LATEST}'))
for tag in ['in_sample', 'held_out']:
    if tag in d['aggregate']:
        a = d['aggregate'][tag]
        print(f'  {tag:<11s}: compliance = {a[\"compliance_pct_mean\"]:.2f}% ± {a[\"compliance_pct_std\"]:.2f}')
        print(f'              OOM        = {a[\"oom_pct_mean\"]:.2f}% ± {a[\"oom_pct_std\"]:.2f}')
        print(f'              violation  = {a[\"violation_pct_mean\"]:.2f}% ± {a[\"violation_pct_std\"]:.2f}')
        print(f'              per-seed compliance: {[f\"{x:.2f}%\" for x in a[\"compliance_per_seed\"]]}')
"
fi
echo "================================================================"