#!/usr/bin/env bash
# Sequential runner για multi-seed verification του ASVF NDVI winner.
# Trains 2 extra seeds (43, 44) — έχουμε ήδη seed=42 → mIoU=0.8442.
# Output: mean ± std across 3 seeds για statistical reporting στο thesis.
#
# Usage:
#   nohup bash scripts/run_multiseed.sh > nohup_multiseed.log 2>&1 & disown

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a SEEDS=(
    "seed_43:ASVF_NDVI_seed43"
    "seed_44:ASVF_NDVI_seed44"
)

mkdir -p logs

OVERALL_START=$(date +%s)
echo "============================================================"
echo "Multi-Seed Winner Verification"
echo "Started: $(date)"
echo "Runs:    ${#SEEDS[@]} (seeds 43, 44 — seed 42 already done)"
echo "Reference (seed=42, ASVF NDVI winner): mIoU=0.8442, F1=0.9118"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_RUNS=()

for entry in "${SEEDS[@]}"; do
    NAME="${entry%%:*}"
    YAML_BASE="${entry##*:}"
    YAML="params/WeedsGalore/${YAML_BASE}.yaml"

    if [[ ! -f "$YAML" ]]; then
        echo "⚠️  SKIP $NAME: δεν βρέθηκε $YAML"
        FAILED=$((FAILED + 1))
        FAILED_RUNS+=("$NAME (no yaml)")
        continue
    fi

    TS=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="multiseed_${NAME}_${TS}"
    LOG="logs/${RUN_NAME}.log"

    echo ""
    echo "============================================================"
    echo "▶  [$((PASSED + FAILED + 1))/${#SEEDS[@]}] $NAME"
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
        FAILED_RUNS+=("$NAME (exit=$EXIT_CODE)")
    fi
done

OVERALL_DUR=$(( $(date +%s) - OVERALL_START ))
OVERALL_HRS=$(( OVERALL_DUR / 3600 ))
OVERALL_MIN=$(( (OVERALL_DUR % 3600) / 60 ))

echo ""
echo "============================================================"
echo "MULTI-SEED DONE — $(date)"
echo "  ✅ Passed: $PASSED / ${#SEEDS[@]}"
echo "  ❌ Failed: $FAILED / ${#SEEDS[@]}"
if [[ $FAILED -gt 0 ]]; then
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - $f"
    done
fi
echo "  Total: ${OVERALL_HRS}h ${OVERALL_MIN}m"
echo "============================================================"
