#!/usr/bin/env bash
# Sequential runner για Sprint 1+2 architectural experiments.
# Όλα clone του winner (CIR+B+RE, FT+Lov 50/50) — αλλάζει μόνο fusion/ASVF.
#
# Usage:
#   nohup bash scripts/run_sprint_experiments.sh > nohup_sprint.log 2>&1 & disown
#
# Reference baseline (squeeze_excite, no ASVF): mIoU=0.8430, F1=0.9118

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a EXPERIMENTS=(
    "coord_attn:Channels_BestCombo_CoordAttn"
    "gated:Fusion_Gated"
    "asvf_ndvi:ASVF_NDVI"
)

mkdir -p logs
OVERALL_START=$(date +%s)
echo "============================================================"
echo "Sprint 1+2 Experiments Runner"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]} (coord_attn, gated, asvf_ndvi)"
echo "Baseline ref: mIoU=0.8430, F1=0.9118"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_LIST=()

for entry in "${EXPERIMENTS[@]}"; do
    NAME="${entry%%:*}"
    YAML_BASE="${entry##*:}"
    YAML="params/WeedsGalore/${YAML_BASE}.yaml"

    if [[ ! -f "$YAML" ]]; then
        echo "⚠️  SKIP $NAME: δεν βρέθηκε $YAML"
        FAILED=$((FAILED + 1)); FAILED_LIST+=("$NAME (no yaml)"); continue
    fi

    TS=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="sprint_${NAME}_${TS}"
    LOG="logs/${RUN_NAME}.log"

    echo ""
    echo "============================================================"
    echo "▶  [$((PASSED + FAILED + 1))/${#EXPERIMENTS[@]}] $NAME"
    echo "   YAML: $YAML | Log: $LOG | Start: $(date)"
    echo "============================================================"
    START=$(date +%s)

    docker run --rm --gpus all \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -v "$(pwd)":/workspace \
        -v "$HOME/datasets/weedsgalore":/workspace/datasets/weedsgalore \
        --name "$RUN_NAME" \
        weedmap:latest \
        python wd.py experiment -f "$YAML" >"$LOG" 2>&1

    EC=$?
    DUR_MIN=$(( ($(date +%s) - START) / 60 ))
    if [[ $EC -eq 0 ]]; then
        echo "✅ $NAME OK σε ${DUR_MIN} λεπτά"
        PASSED=$((PASSED + 1))
    else
        echo "❌ $NAME FAILED (exit=$EC) μετά ${DUR_MIN} λεπτά — δες $LOG"
        FAILED=$((FAILED + 1)); FAILED_LIST+=("$NAME (exit=$EC)")
    fi
done

OVERALL_MIN=$(( ($(date +%s) - OVERALL_START) / 60 ))
echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "  ✅ Passed: $PASSED / ${#EXPERIMENTS[@]}"
echo "  ❌ Failed: $FAILED / ${#EXPERIMENTS[@]}"
for f in "${FAILED_LIST[@]:-}"; do [[ -n "$f" ]] && echo "    - $f"; done
echo "  Total: $((OVERALL_MIN / 60))h $((OVERALL_MIN % 60))m"
echo "============================================================"
