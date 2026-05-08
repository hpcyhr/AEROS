#!/usr/bin/env bash
# AEROS Phase 1 variance bands runner (B1 + B3)
#
# B1: §5.7 A4 cross-arch held-out — n_splits=10 (was 3)
# B3: §5.6 disjoint-fold compliance — 3 fold-split seeds {42, 123, 7}
#
# Total wall: ~4-6 hr V100. Foreground; tee'd to per-stage logs.
#
# Usage:
#   bash run_phase1_variance_bands.sh
# Or background:
#   nohup bash run_phase1_variance_bands.sh > /dev/null 2>&1 &
#   echo $!  # PID
#   tail -f logs/variance_session_*.log

set -e
set -u

cd /data/yhr/AEROS/
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

SESSION_LOG="logs/variance_session_${TS}.log"
exec > >(tee -a "${SESSION_LOG}") 2>&1

echo "================================================================"
echo "AEROS Phase 1 variance bands (B1 + B3)"
echo "Started: $(date)"
echo "Session log: ${SESSION_LOG}"
echo "================================================================"

echo ""
echo "Pre-flight check..."
for f in p9_1a_full16.json \
         p9_6d_disjoint_calibration.py \
         p9_6e_cross_arch_holdout_v2.py \
         p9_6d_run_variance.py \
         aeros_dispatch.py \
         aeros_models_extended.py; do
    if [[ -f "${f}" ]]; then
        echo "  ok: ${f}"
    else
        echo "  [FATAL] missing: ${f}"
        exit 1
    fi
done

OVERALL_START=$(date +%s)

# ============================================================================
# B1: §5.7 A4 cross-arch held-out — n_splits=10
# ============================================================================
echo ""
echo "================================================================"
echo "B1: cross-arch held-out, n_splits=10"
echo "Estimated: ~2-3 hr V100"
echo "Started: $(date)"
echo "================================================================"

B1_LOG="logs/b1_n10_${TS}.log"
B1_START=$(date +%s)

python -u p9_6e_cross_arch_holdout_v2.py \
    --coeffs p9_1a_full16.json \
    --Ts 128 1024 \
    --Ms 4 8 16 24 \
    --modes 2 4 \
    --n_splits 10 \
    --n_holdout 3 \
    --output p9_6e_cross_arch_holdout_n10 \
    2>&1 | tee "${B1_LOG}"

B1_END=$(date +%s)
B1_DT=$((B1_END - B1_START))
echo ""
echo "B1 done in $((B1_DT / 60))m $((B1_DT % 60))s"

# ============================================================================
# B3: §5.6 disjoint-fold compliance — 3-seed variance
# ============================================================================
echo ""
echo "================================================================"
echo "B3: disjoint-fold compliance, 3 seeds {42, 123, 7}"
echo "Estimated: ~2-3 hr V100"
echo "Started: $(date)"
echo "================================================================"

B3_LOG="logs/b3_variance_${TS}.log"
B3_START=$(date +%s)

python -u p9_6d_run_variance.py \
    --coeffs p9_1a_full16.json \
    --seeds 42 123 7 \
    --Ts 128 1024 4096 \
    --Ms 4 8 16 24 \
    --modes 2 4 \
    --batch_size 32 \
    --H 128 --C 3 \
    --delta 0.05 \
    --epsilon_safety 0.02 \
    --output_prefix p9_6d_variance \
    2>&1 | tee "${B3_LOG}"

B3_END=$(date +%s)
B3_DT=$((B3_END - B3_START))
echo ""
echo "B3 done in $((B3_DT / 60))m $((B3_DT % 60))s"

# ============================================================================
# Summary
# ============================================================================
OVERALL_END=$(date +%s)
TOTAL=$((OVERALL_END - OVERALL_START))

echo ""
echo "================================================================"
echo "Phase 1 variance bands complete"
echo "Total: $((TOTAL / 60))m $((TOTAL % 60))s"
echo ""
echo "Output files:"
echo "  B1: $(ls -la p9_6e_cross_arch_holdout_n10.json 2>/dev/null || echo MISSING)"
echo "  B3: $(ls -la p9_6d_variance_aggregate_*.json 2>/dev/null | tail -1 || echo MISSING)"
echo ""
echo "Per-stage logs:"
echo "  B1: ${B1_LOG}"
echo "  B3: ${B3_LOG}"
echo "  Session: ${SESSION_LOG}"
echo "================================================================"

echo ""
echo "=== Quick summary preview ==="

if [[ -f p9_6e_cross_arch_holdout_n10.json ]]; then
    echo ""
    echo "B1 (n_splits=10) aggregate:"
    python -c "
import json
d = json.load(open('p9_6e_cross_arch_holdout_n10.json'))
agg = d['aggregate']
print(f'  Pooled cross-arch held-out compliance:')
print(f'    mean={agg[\"pooled_compliance_mean\"]:.2f}%  std={agg[\"pooled_compliance_std\"]:.2f}')
print(f'    per-split: {agg[\"pooled_compliance_per_split\"]}')
print(f'  Per-arch (E1-style) compliance:')
print(f'    mean={agg[\"per_arch_compliance_mean\"]:.2f}%  std={agg[\"per_arch_compliance_std\"]:.2f}')
print(f'    per-split: {agg[\"per_arch_compliance_per_split\"]}')
"
fi

LATEST_B3=$(ls -t p9_6d_variance_aggregate_*.json 2>/dev/null | head -1)
if [[ -n "${LATEST_B3}" ]]; then
    echo ""
    echo "B3 (3-seed disjoint-fold) aggregate:"
    python -c "
import json
d = json.load(open('${LATEST_B3}'))
for tag in ['in_sample', 'held_out']:
    if tag in d['aggregate']:
        a = d['aggregate'][tag]
        print(f'  {tag}: compliance = {a[\"compliance_pct_mean\"]:.2f}% ± {a[\"compliance_pct_std\"]:.2f}')
        print(f'           OOM = {a[\"oom_pct_mean\"]:.2f}% ± {a[\"oom_pct_std\"]:.2f}')
        print(f'           per-seed: {a[\"compliance_per_seed\"]}')
"
fi