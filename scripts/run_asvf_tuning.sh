#!/usr/bin/env bash
# Sequential runner για ASVF tuning experiments (Sprint 2 follow-up).
# Όλα clone του ASVF winner (CIR+B+RE, FT+Lov 50/50, ASVF) με μία διαφοροποίηση.
#
# Usage:
#   nohup bash scripts/run_asvf_tuning.sh > nohup_asvf_tuning.log 2>&1 & disown
#
# Reference: ASVF NDVI-only baseline mIoU=0.8442, F1=0.9118

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a EXPERIMENTS=(
    "dual_index:ASVF_DualIndex"
    "combo_coord:ASVF_Combo_CoordAttn"
    "alpha_03:ASVF_AlphaInit_03"
)

mkdir -p logs
OVERALL_START=$(date +%s)
echo "============================================================"
echo "ASVF Tuning Experiments Runner"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]} (dual_index, combo_coord, alpha_03)"
echo "Reference: ASVF baseline mIoU=0.8442"
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
    RUN_NAME="asvf_tune_${NAME}_${TS}"
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
