#!/usr/bin/env bash
# Wrapper για να τρέχουμε experiments στον orion μέσω Docker.
# Usage:
#   bash scripts/run_orion.sh <yaml_path> [run_name]
#
# Παράδειγμα:
#   bash scripts/run_orion.sh params/WeedsGalore/Loss_FocalCE_orion_smoke.yaml smoke_focal_ce
#   bash scripts/run_orion.sh params/WeedsGalore/Loss_FocalCE.yaml focal_ce_full

set -e

YAML="${1:?Πρέπει να δώσεις YAML path. π.χ. bash scripts/run_orion.sh params/.../Loss_X.yaml}"
NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"
LOG="logs/${NAME}_$(date +%Y%m%d_%H%M).log"

mkdir -p logs

if [[ -z "$WANDB_API_KEY" ]]; then
    echo "ERROR: \$WANDB_API_KEY δεν είναι set. Έλεγξε ~/.bashrc."
    exit 1
fi

if [[ ! -f "$YAML" ]]; then
    echo "ERROR: δεν βρέθηκε YAML: $YAML"
    exit 1
fi

echo "=== Starting run ==="
echo "  YAML:    $YAML"
echo "  Name:    $NAME"
echo "  Log:     $LOG"
echo "  Started: $(date)"
echo "===================="

docker run --rm --gpus all \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -v "$(pwd)":/workspace \
    -v "$HOME/datasets/weedsgalore":/workspace/datasets/weedsgalore \
    --name "$NAME" \
    weedmap:latest \
    python wd.py experiment -f "$YAML" 2>&1 | tee "$LOG"
