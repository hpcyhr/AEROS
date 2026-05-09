#!/usr/bin/env bash
# AEROS Head-to-head against AutoChunk — final data collection
#
# Sweeps ConvLSTM-2L / SR-18 / VGG-19-BN at multiple T to determine
# at what T AutoChunk's symbolic_trace fails (RecursionError /
# segfault / OOM-kill) vs succeeds. Each cell runs in a subshell with:
#   - ulimit -s 65536  (raise C stack from 8MB to 64MB; bypass segfault)
#   - 5min timeout    (avoid hangs on OOM-kill recovery)
#
# Output: 1 JSON per cell + summary table.

set -u

cd /data/yhr/AEROS/
mkdir -p autochunk_results
TS=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="autochunk_results/session_${TS}.log"
exec > >(tee -a "${SESSION_LOG}") 2>&1

echo "================================================================"
echo "AEROS — Head-to-head against AutoChunk (final round)"
echo "================================================================"
echo "Started: $(date)"
echo "Session log: ${SESSION_LOG}"
echo ""

# ---------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------
# Format: arch, T, B, H_snn (used only for SNN), max_memory_mb
declare -a CELLS=(
    # ConvLSTM-2L sweep
    "ConvLSTM-2L  2  2  -    4096"
    "ConvLSTM-2L  4  2  -    4096"
    "ConvLSTM-2L  8  2  -    4096"
    "ConvLSTM-2L  16 2  -    4096"
    "ConvLSTM-2L  32 2  -    4096"
    # SR-18 (small T because SNN graph already large)
    "SR-18        2  1  64   8192"
    "SR-18        4  1  64   8192"
    "SR-18        8  1  64   8192"
    # VGG-19-BN
    "VGG-19-BN    2  1  64   8192"
    "VGG-19-BN    4  1  64   8192"
)

# ---------------------------------------------------------------------
# Run each cell with ulimit + timeout
# ---------------------------------------------------------------------
for cell in "${CELLS[@]}"; do
    read -r arch T B Hsnn budget <<< "$cell"
    out_json="autochunk_results/cell_${arch}_T${T}_B${B}.json"
    out_log="autochunk_results/cell_${arch}_T${T}_B${B}.log"

    echo ""
    echo "================================================================"
    echo "Cell: arch=${arch} T=${T} B=${B} budget=${budget}MB"
    echo "Started: $(date)"
    echo "================================================================"

    H_snn_arg=""
    if [[ "${Hsnn}" != "-" ]]; then
        H_snn_arg="--H_snn ${Hsnn}"
    fi

    # Subshell so ulimit doesn't affect parent
    (
        ulimit -s 65536
        timeout 300 python apply_autochunk_to_aeros.py \
            --archs "${arch}" \
            --T "${T}" \
            --B "${B}" \
            ${H_snn_arg} \
            --max_memory_mb "${budget}" \
            --output "${out_json}" \
            2>&1 | tee "${out_log}"
        ec=${PIPESTATUS[0]}
        echo ""
        echo "  [cell exit code: ${ec}]"
        if [[ ${ec} -eq 124 ]]; then
            echo "  [TIMEOUT after 5 min]"
            # Write a stub JSON so summary can pick it up
            echo "{\"config\": {\"archs\":[\"${arch}\"],\"T\":${T},\"B\":${B}}, \"results\": [{\"arch\": \"${arch}\", \"status\": \"timeout_5min\", \"n_chunks\": 0}]}" > "${out_json}"
        elif [[ ${ec} -eq 137 ]]; then
            echo "  [OOM-KILLED (SIGKILL)]"
            echo "{\"config\": {\"archs\":[\"${arch}\"],\"T\":${T},\"B\":${B}}, \"results\": [{\"arch\": \"${arch}\", \"status\": \"oom_killed\", \"n_chunks\": 0}]}" > "${out_json}"
        elif [[ ${ec} -eq 139 ]]; then
            echo "  [SEGFAULT]"
            echo "{\"config\": {\"archs\":[\"${arch}\"],\"T\":${T},\"B\":${B}}, \"results\": [{\"arch\": \"${arch}\", \"status\": \"segfault\", \"n_chunks\": 0}]}" > "${out_json}"
        fi
    )

    sleep 2  # let GPU mem release
done

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "FINAL SUMMARY"
echo "================================================================"

python << 'PYEOF'
import json
import glob
import os

cells = sorted(glob.glob("autochunk_results/cell_*.json"))
print(f"\n  Found {len(cells)} cell results\n")

print(f"{'Arch':<14} {'T':<4} {'B':<3} {'Status':<25} {'GraphNodes':<12} {'Chunks':<8} {'Verdict'}")
print("-" * 105)

for f in cells:
    try:
        with open(f) as fh:
            d = json.load(fh)
        cfg = d.get("config", {})
        T = cfg.get("T", "?")
        B = cfg.get("B", "?")
        for r in d.get("results", []):
            arch = r.get("arch", "?")
            status = r.get("status", "?")
            n_chunks = r.get("n_chunks", 0)
            n_nodes = r.get("n_graph_nodes", -1)
            n_nodes_str = str(n_nodes) if n_nodes >= 0 else "--"
            if status == "ok" and n_chunks > 0:
                verdict = f"ACCEPTED ({n_chunks} plans)"
            elif status == "no_chunk_found":
                verdict = "REJECTED (no legal chunk)"
            elif status == "trace_exploded":
                verdict = "TRACE EXPLODED (RecursionError)"
            elif status == "metainfo_failed":
                verdict = f"META-INFO FAILED ({r.get('exception_type','?')})"
            elif status == "autochunk_failed":
                verdict = f"AUTOCHUNK CRASHED ({r.get('exception_type','?')})"
            elif status == "oom_killed":
                verdict = "OOM-KILLED (host RAM)"
            elif status == "segfault":
                verdict = "C-STACK SEGFAULT"
            elif status == "timeout_5min":
                verdict = "TIMEOUT (>5min)"
            else:
                verdict = status
            print(f"{arch:<14} {str(T):<4} {str(B):<3} {status:<25} {n_nodes_str:<12} {str(n_chunks):<8} {verdict}")
    except Exception as e:
        print(f"  [error reading {f}: {e}]")
PYEOF

echo ""
echo "================================================================"
echo "Done: $(date)"
echo "All cell JSONs in autochunk_results/"
echo "Session log: ${SESSION_LOG}"
echo "================================================================"