#!/usr/bin/env bash
# Sequential runner για OSAVI/MSAVI vegetation-index channel ablations.
# 4 runs:
#   1. RGB + OSAVI
#   2. RGB + MSAVI
#   3. Best Combo (CIR+B+RE) + OSAVI
#   4. Best Combo (CIR+B+RE) + MSAVI
# All με winner loss config (FT+Lov 50/50, α=0.4/β=0.6).
#
# Usage:
#   nohup bash scripts/run_index_ablations.sh > nohup_indices.log 2>&1 & disown

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a COMBOS=(
    "rgb_osavi:Channels_RGB_OSAVI"
    "rgb_msavi:Channels_RGB_MSAVI"
    "bestcombo_osavi:Channels_BestCombo_OSAVI"
    "bestcombo_msavi:Channels_BestCombo_MSAVI"
)

mkdir -p logs

OVERALL_START=$(date +%s)
echo "============================================================"
echo "Vegetation-Index Channel Ablations"
echo "Started: $(date)"
echo "Runs:    ${#COMBOS[@]} (RGB+OSAVI/MSAVI + BestCombo+OSAVI/MSAVI)"
echo "Loss:    FT+Lovász 50/50 (α=0.4, β=0.6)"
echo "Reference Best Combo (no index): mIoU=0.8430, F1=0.9118"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_VARIANTS=()

for entry in "${COMBOS[@]}"; do
    NAME="${entry%%:*}"
    YAML_BASE="${entry##*:}"
    YAML="params/WeedsGalore/${YAML_BASE}.yaml"

    if [[ ! -f "$YAML" ]]; then
        echo "⚠️  SKIP $NAME: δεν βρέθηκε $YAML"
        FAILED=$((FAILED + 1))
        FAILED_VARIANTS+=("$NAME (no yaml)")
        continue
    fi

    TS=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="index_${NAME}_${TS}"
    LOG="logs/${RUN_NAME}.log"

    echo ""
    echo "============================================================"
    echo "▶  [$((PASSED + FAILED + 1))/${#COMBOS[@]}] $NAME"
    echo "   YAML:  $YAML"
    echo "   Log:   $LOG"
    echo "   Start: $(date)"
    echo "============================================================"
    START=$(date +%s)

    docker run --rm --gpus all \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -v "$(pwd)":/workspace \
        -v "$HOME/datasets/weedsgalore":/workspace/datasets/weedsgalore \
        --name "$RUN_NAME" \
        weedmap:latest \
        python wd.py experiment -f "$YAML" >"$LOG" 2>&1

    EXIT_CODE=$?
    DUR=$(( $(date +%s) - START ))
    DUR_MIN=$((DUR / 60))

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "✅ $NAME finished OK σε ${DUR_MIN} λεπτά"
        PASSED=$((PASSED + 1))
    else
        echo "❌ $NAME FAILED (exit=$EXIT_CODE) μετά ${DUR_MIN} λεπτά"
        FAILED=$((FAILED + 1))
        FAILED_VARIANTS+=("$NAME (exit=$EXIT_CODE)")
    fi
done

OVERALL_DUR=$(( $(date +%s) - OVERALL_START ))
OVERALL_HRS=$(( OVERALL_DUR / 3600 ))
OVERALL_MIN=$(( (OVERALL_DUR % 3600) / 60 ))

echo ""
echo "============================================================"
echo "INDEX ABLATIONS DONE — $(date)"
echo "  ✅ Passed: $PASSED / ${#COMBOS[@]}"
echo "  ❌ Failed: $FAILED / ${#COMBOS[@]}"
if [[ $FAILED -gt 0 ]]; then
    for f in "${FAILED_VARIANTS[@]}"; do
        echo "    - $f"
    done
fi
echo "  Total: ${OVERALL_HRS}h ${OVERALL_MIN}m"
echo "============================================================"
