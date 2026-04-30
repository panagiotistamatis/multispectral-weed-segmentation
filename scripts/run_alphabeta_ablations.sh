#!/usr/bin/env bash
# Sequential runner για τα 3 α/β ablations του Focal Tversky.
# Base config: SplitLawin B0 + RGB+NIR+RE (winner channels).
# Variants: (0.5,0.5), (0.4,0.6), (0.6,0.4).
#
# Usage:
#   nohup bash scripts/run_alphabeta_ablations.sh > nohup_ab.log 2>&1 & disown
#
# Reference: τρέχον winner = α=0.3, β=0.7 → F1=0.905, mIoU=0.833

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a COMBOS=(
    "ab_05_05:AlphaBeta_05_05"
    "ab_04_06:AlphaBeta_04_06"
    "ab_06_04:AlphaBeta_06_04"
)

mkdir -p logs

OVERALL_START=$(date +%s)
echo "============================================================"
echo "α/β Ablation Sequential Runner"
echo "Started: $(date)"
echo "Variants: ${#COMBOS[@]} (a=0.5/b=0.5, a=0.4/b=0.6, a=0.6/b=0.4)"
echo "Base: SplitLawin B0 + RGB+NIR+RE + Focal Tversky"
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
    RUN_NAME="ab_full_${NAME}_${TS}"
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
        echo "❌ $NAME FAILED (exit=$EXIT_CODE) μετά από ${DUR_MIN} λεπτά"
        FAILED=$((FAILED + 1))
        FAILED_VARIANTS+=("$NAME (exit=$EXIT_CODE)")
    fi
done

OVERALL_DUR=$(( $(date +%s) - OVERALL_START ))
OVERALL_HRS=$(( OVERALL_DUR / 3600 ))
OVERALL_MIN=$(( (OVERALL_DUR % 3600) / 60 ))

echo ""
echo "============================================================"
echo "α/β ABLATION DONE — $(date)"
echo "  ✅ Passed: $PASSED / ${#COMBOS[@]}"
echo "  ❌ Failed: $FAILED / ${#COMBOS[@]}"
if [[ $FAILED -gt 0 ]]; then
    for f in "${FAILED_VARIANTS[@]}"; do
        echo "    - $f"
    done
fi
echo "  Total: ${OVERALL_HRS}h ${OVERALL_MIN}m"
echo "============================================================"
