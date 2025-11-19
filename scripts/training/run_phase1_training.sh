#!/bin/bash
#
# Launch Phase 1 fine-tuning: Decoder-only training
#
# Prerequisites:
#   - Dataset interpolato generato in data/nuscenes_mot_6cams_interpolated/
#   - Pretrained MOT17 checkpoint (optional)
#
# Usage:
#   bash scripts/run_phase1_training.sh

set -e

echo "========================================"
echo "PHASE 1: Decoder-only Fine-tuning"
echo "========================================"

# Configurazione
DATAROOT=${NUSC_INTERP_ROOT:-"./data/nuscenes_mot_6cams_interpolated"}
CONFIG="./configs/nuscenes_phase1.yaml"
OUTPUT_DIR="./weights/phase1"
PRETRAINED=${MOT17_CHECKPOINT:-""}

echo "Data root: $DATAROOT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Pretrained: ${PRETRAINED:-'None (training from scratch)'}"
echo ""

# Verifica dataset
if [ ! -d "$DATAROOT/train" ]; then
    echo "ERROR: Training data not found at $DATAROOT/train"
    echo "Please run: bash scripts/generate_splits.sh first"
    exit 1
fi

if [ ! -d "$DATAROOT/val" ]; then
    echo "ERROR: Validation data not found at $DATAROOT/val"
    echo "Please run: bash scripts/generate_splits.sh first"
    exit 1
fi

# Crea output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo "Starting Phase 1 training..."
echo "========================================"

if [ -n "$PRETRAINED" ]; then
    python scripts/training/train_phase1_decoder.py \
        --config "$CONFIG" \
        --data_root "$DATAROOT" \
        --output_dir "$OUTPUT_DIR" \
        --pretrained "$PRETRAINED"
else
    python scripts/training/train_phase1_decoder.py \
        --config "$CONFIG" \
        --data_root "$DATAROOT" \
        --output_dir "$OUTPUT_DIR"
fi

echo ""
echo "========================================"
echo "Phase 1 training complete!"
echo "Best checkpoint saved to: $OUTPUT_DIR/phase1_decoder_best.pth"
echo ""
echo "Next steps:"
echo "  1. Evaluate Phase 1: python evaluate_phase1.py --checkpoint $OUTPUT_DIR/phase1_decoder_best.pth"
echo "  2. Start Phase 2: bash scripts/training/run_phase2_training.sh"
echo "========================================"
