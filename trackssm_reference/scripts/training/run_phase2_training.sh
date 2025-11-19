#!/bin/bash
#
# Launch Phase 2 fine-tuning: Full model training with differential LR
#
# Prerequisites:
#   - Dataset interpolato generato
#   - Phase 1 checkpoint completato
#
# Usage:
#   bash scripts/run_phase2_training.sh

set -e

echo "========================================"
echo "PHASE 2: Full Fine-tuning"
echo "========================================"

# Configurazione
DATAROOT=${NUSC_INTERP_ROOT:-"./data/nuscenes_mot_6cams_interpolated"}
CONFIG="./configs/nuscenes_phase2.yaml"
OUTPUT_DIR="./weights/phase2"
PHASE1_CHECKPOINT="./weights/phase1/phase1_decoder_best.pth"

echo "Data root: $DATAROOT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Phase 1 checkpoint: $PHASE1_CHECKPOINT"
echo ""

# Verifica Phase 1 checkpoint
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CHECKPOINT"
    echo "Please complete Phase 1 training first:"
    echo "  bash scripts/run_phase1_training.sh"
    exit 1
fi

# Verifica dataset
if [ ! -d "$DATAROOT/train" ] || [ ! -d "$DATAROOT/val" ]; then
    echo "ERROR: Dataset not found at $DATAROOT"
    exit 1
fi

# Crea output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo "Starting Phase 2 training..."
echo "========================================"

python scripts/training/train_phase2_full.py \
    --config "$CONFIG" \
    --data_root "$DATAROOT" \
    --output_dir "$OUTPUT_DIR" \
    --phase1_checkpoint "$PHASE1_CHECKPOINT"

echo ""
echo "========================================"
echo "Phase 2 training complete!"
echo "Best checkpoint saved to: $OUTPUT_DIR/phase2_full_best.pth"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set: python evaluate_final.py --checkpoint $OUTPUT_DIR/phase2_full_best.pth"
echo "  2. Compare with baselines: python compare_metrics.py"
echo "========================================"
