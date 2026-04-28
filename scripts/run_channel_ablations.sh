#!/usr/bin/env bash
# Sequential runner για channel ablations (combos 2-7).
# Trains each combo με Focal Tversky loss, 500 epochs, batch=4, mixed_precision=True.
#
# Usage:
#   bash scripts/run_channel_ablations.sh           # full runs (default)
#   bash scripts/run_channel_ablations.sh smoke     # smoke runs (1 epoch each)
#
# Recommended overnight invocation:
#   nohup bash scripts/run_channel_ablations.sh > nohup_ablations.log 2>&1 & disown
#
# Notes:
#   - Συνεχίζει στο επόμενο combo ακόμα κι αν κάποιο σπάσει.
#   - Logs: logs/ablation_<mode>_<combo>_<timestamp>.log (ένα ανά combo).
#   - Combo #1 (RGB+NIR+RE) έχει ήδη γίνει — Focal Tversky F1=0.905, mIoU=0.833.

set -u  # error on unset vars (αλλά ΟΧΙ set -e — να συνεχίζει αν σπάσει run)

MODE="${1:-full}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "Usage: $0 [smoke|full]"
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set. Έλεγξε ~/.bashrc."
    exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
    SUFFIX="_smoke"
    LOG_PREFIX="smoke"
    echo "🔥 SMOKE mode: 1 epoch ανά combo (~10 λεπτά συνολικά)"
else
    SUFFIX=""
    LOG_PREFIX="full"
    echo "🚀 FULL mode: 500 epochs ανά combo, early_stopping patience=30 (~25-30h συνολικά)"
fi

# Mapping: combo_id : YAML_basename
declare -a COMBOS=(
    "c02_rgb_ndvi:Channels_2_RGB_NDVI"
    "c03_rgb_cir:Channels_3_RGB_CIR"
    "c04_cir_alone:Channels_4_CIR"
    "c05_rgb_nir:Channels_5_RGB_NIR"
    "c06_rgb_ndvi_nir_re:Channels_6_RGB_NDVI_NIR_RE"
    "c07_full:Channels_7_RGB_NDVI_CIR_RE_NIR"
)

mkdir -p logs

OVERALL_START=$(date +%s)
echo "============================================================"
echo "Channel Ablations Sequential Runner"
echo "Started: $(date)"
echo "Combos:  ${#COMBOS[@]}"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_COMBOS=()

for entry in "${COMBOS[@]}"; do
    NAME="${entry%%:*}"
    YAML_BASE="${entry##*:}"
    YAML="params/WeedsGalore/${YAML_BASE}${SUFFIX}.yaml"

    if [[ ! -f "$YAML" ]]; then
        echo "⚠️  SKIP $NAME: δεν βρέθηκε $YAML"
        FAILED=$((FAILED + 1))
        FAILED_COMBOS+=("$NAME (no yaml)")
        continue
    fi

    TS=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${LOG_PREFIX}_${NAME}_${TS}"
    LOG="logs/ablation_${RUN_NAME}.log"

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
        echo "❌ $NAME FAILED (exit=$EXIT_CODE) μετά από ${DUR_MIN} λεπτά — δες $LOG"
        FAILED=$((FAILED + 1))
        FAILED_COMBOS+=("$NAME (exit=$EXIT_CODE)")
    fi
done

OVERALL_DUR=$(( $(date +%s) - OVERALL_START ))
OVERALL_HRS=$(( OVERALL_DUR / 3600 ))
OVERALL_MIN=$(( (OVERALL_DUR % 3600) / 60 ))

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "  ✅ Passed: $PASSED / ${#COMBOS[@]}"
echo "  ❌ Failed: $FAILED / ${#COMBOS[@]}"
if [[ $FAILED -gt 0 ]]; then
    echo "  Failed combos:"
    for f in "${FAILED_COMBOS[@]}"; do
        echo "    - $f"
    done
fi
echo "  Total duration: ${OVERALL_HRS}h ${OVERALL_MIN}m"
echo "============================================================"
