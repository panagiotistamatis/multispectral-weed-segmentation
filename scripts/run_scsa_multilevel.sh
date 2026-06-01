#!/usr/bin/env bash
# Sequential runner για multi-level SCSA experiments.
# Reference (F4 only): mIoU=0.8469
# Tests if SCSA at multiple feature levels gives orthogonal gain.
#
# Usage:
#   nohup bash scripts/run_scsa_multilevel.sh > nohup_scsa_ml.log 2>&1 & disown

set -u

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set."
    exit 1
fi

declare -a EXPERIMENTS=(
    "f3_f4:ASVF_SCSA_F3F4"
    "f2_f3_f4:ASVF_SCSA_F2F3F4"
)

mkdir -p logs
OVERALL_START=$(date +%s)
echo "============================================================"
echo "Multi-Level SCSA Runner"
echo "Started: $(date)"
echo "Runs: ${#EXPERIMENTS[@]} (F3+F4, F2+F3+F4)"
echo "Reference (F4 only): mIoU=0.8469"
echo "============================================================"

PASSED=0; FAILED=0; FAILED_LIST=()

for entry in "${EXPERIMENTS[@]}"; do
    NAME="${entry%%:*}"
    YAML="params/WeedsGalore/${entry##*:}.yaml"
    [[ ! -f "$YAML" ]] && { echo "⚠️  SKIP $NAME (no $YAML)"; FAILED=$((FAILED+1)); continue; }

    TS=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="scsa_ml_${NAME}_${TS}"
    LOG="logs/${RUN_NAME}.log"

    echo ""
    echo "▶  [$((PASSED+FAILED+1))/${#EXPERIMENTS[@]}] $NAME — Start: $(date)"
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
        PASSED=$((PASSED+1))
    else
        echo "❌ $NAME FAILED (exit=$EC) μετά ${DUR_MIN} λεπτά"
        FAILED=$((FAILED+1)); FAILED_LIST+=("$NAME (exit=$EC)")
    fi
done

OVERALL_MIN=$(( ($(date +%s) - OVERALL_START) / 60 ))
echo ""
echo "ALL DONE — $(date) | Passed: $PASSED/${#EXPERIMENTS[@]} | Total: $((OVERALL_MIN/60))h $((OVERALL_MIN%60))m"
for f in "${FAILED_LIST[@]:-}"; do [[ -n "$f" ]] && echo "  ❌ $f"; done
