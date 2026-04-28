#!/usr/bin/env bash
# Sequential runner για όλα τα channel ablations.
#
# Usage (στον orion):
#   bash scripts/run_channel_ablations.sh smoke   # τρέχει 5 smokes (~5 λεπτά)
#   bash scripts/run_channel_ablations.sh full    # τρέχει 5 full runs (~20-25 ώρες)
#
# Συνιστάται:
#   1. Πρώτα 'smoke' για να επιβεβαιώσεις ότι όλα τα YAMLs δουλεύουν
#   2. Μετά 'full' με nohup για overnight execution:
#        nohup bash scripts/run_channel_ablations.sh full > nohup_ablations.log 2>&1 &
#        disown

set -e

MODE="${1:-smoke}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "Usage: $0 [smoke|full]"
    exit 1
fi

# Channel combos (όνομα = filename suffix, όχι extension)
COMBOS=(
    "RGB_NDVI"
    "RGB_CIR"
    "RGB_NDVI_CIR"
    "RGB_NDVI_CIR_RE"
    "CIR_NDVI"
)

if [[ "$MODE" == "smoke" ]]; then
    SUFFIX="_smoke"
    GROUP="smoke"
else
    SUFFIX=""
    GROUP="full"
fi

TOTAL=${#COMBOS[@]}
START_TIME=$(date +%s)

echo "===================================="
echo "Channel Ablations Runner"
echo "  Mode:      $MODE"
echo "  Combos:    $TOTAL"
echo "  Started:   $(date)"
echo "===================================="

for i in "${!COMBOS[@]}"; do
    COMBO="${COMBOS[$i]}"
    YAML="params/WeedsGalore/Channels_${COMBO}${SUFFIX}.yaml"
    NAME="ch_${COMBO,,}_${GROUP}"
    IDX=$((i + 1))

    echo ""
    echo "[$IDX/$TOTAL] Starting: $COMBO ($MODE)"
    echo "  YAML: $YAML"
    echo "  Name: $NAME"
    echo "  Time: $(date)"
    echo "------------------------------------"

    if [[ ! -f "$YAML" ]]; then
        echo "ERROR: missing $YAML"
        exit 1
    fi

    # Set +e ώστε να συνεχίσουμε αν ένα run αποτύχει (continue_with_errors)
    set +e
    bash scripts/run_orion.sh "$YAML" "$NAME"
    RC=$?
    set -e

    if [[ $RC -ne 0 ]]; then
        echo "WARNING: $COMBO exited με κωδικό $RC, συνεχίζουμε στο επόμενο..."
    else
        echo "[$IDX/$TOTAL] DONE: $COMBO"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "===================================="
echo "All ablations finished"
echo "  Mode:    $MODE"
echo "  Elapsed: ${ELAPSED} λεπτά"
echo "  Ended:   $(date)"
echo "===================================="
