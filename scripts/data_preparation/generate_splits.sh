#!/bin/bash
#
# Generate train/val/test splits for NuScenes interpolated dataset
#
# This script processes NuScenes scenes across all 6 cameras with interpolation,
# creating separate directories for train (600 scene-cams), val (170), test (80).
#
# Prerequisites:
#   - NuScenes dataset installed at $NUSC_ROOT
#   - nuscenes-devkit installed: pip install nuscenes-devkit
#   - prepare_nuscenes_interpolated.py in scripts/
#
# Usage:
#   export NUSC_ROOT=/mnt/datasets/Nuscense
#   export OUT_ROOT=./data/nuscenes_mot_6cams_interpolated
#   bash scripts/data_preparation/generate_splits.sh
#

set -e  # Exit on error

# === Configuration ===
NUSC_ROOT=${NUSC_ROOT:-"/mnt/datasets/Nuscense"}
OUT_ROOT=${OUT_ROOT:-"./data/nuscenes_mot_6cams_interpolated"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP_SCRIPT="$SCRIPT_DIR/../data_preparation/prepare_nuscenes_interpolated.py"

echo "========================================"
echo "NuScenes Interpolated Split Generation"
echo "========================================"
echo "NuScenes root: $NUSC_ROOT"
echo "Output root: $OUT_ROOT"
echo "Preparation script: $PREP_SCRIPT"
echo ""

# Check if preparation script exists
if [ ! -f "$PREP_SCRIPT" ]; then
    echo "ERROR: Preparation script not found: $PREP_SCRIPT"
    exit 1
fi

# Check if NuScenes dataset exists
if [ ! -d "$NUSC_ROOT" ]; then
    echo "ERROR: NuScenes dataset not found: $NUSC_ROOT"
    echo "Please set NUSC_ROOT environment variable"
    exit 1
fi

# === All 6 cameras ===
CAMERAS="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT"

# === Generate TRAIN split ===
echo "========================================" 
echo "Processing TRAIN split..."
echo "========================================" 
python "$PREP_SCRIPT" \
    --dataroot "$NUSC_ROOT" \
    --version v1.0-trainval \
    --split train \
    --output_dir "$OUT_ROOT" \
    --cameras $CAMERAS \
    --target_fps 12 \
    --min_visibility 1

echo ""
echo "✓ TRAIN split completed"
echo ""

# === Generate VAL split ===
echo "========================================" 
echo "Processing VAL split..."
echo "========================================" 
python "$PREP_SCRIPT" \
    --dataroot "$NUSC_ROOT" \
    --version v1.0-trainval \
    --split val \
    --output_dir "$OUT_ROOT" \
    --cameras $CAMERAS \
    --target_fps 12 \
    --min_visibility 1

echo ""
echo "✓ VAL split completed"
echo ""

# === Generate TEST split ===
echo "========================================" 
echo "Processing TEST split..."
echo "========================================" 
python "$PREP_SCRIPT" \
    --dataroot "$NUSC_ROOT" \
    --version v1.0-trainval \
    --split test \
    --output_dir "$OUT_ROOT" \
    --cameras $CAMERAS \
    --target_fps 12 \
    --min_visibility 1

echo ""
echo "✓ TEST split completed"
echo ""

# === Summary Statistics ===
echo "========================================"
echo "SPLIT GENERATION COMPLETE"
echo "========================================"
echo ""

echo "Directory structure:"
echo "  $OUT_ROOT/"
echo "    ├── train/     (600 scene-cameras)"
echo "    ├── val/       (170 scene-cameras)"
echo "    └── test/      (80 scene-cameras)"
echo ""

# Count scene-cameras per split
if [ -d "$OUT_ROOT/train" ]; then
    TRAIN_COUNT=$(find "$OUT_ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Train scene-cameras: $TRAIN_COUNT"
fi

if [ -d "$OUT_ROOT/val" ]; then
    VAL_COUNT=$(find "$OUT_ROOT/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Val scene-cameras: $VAL_COUNT"
fi

if [ -d "$OUT_ROOT/test" ]; then
    TEST_COUNT=$(find "$OUT_ROOT/test" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Test scene-cameras: $TEST_COUNT"
fi

echo ""
echo "Statistics files:"
echo "  - $OUT_ROOT/train/dataset_stats.json"
echo "  - $OUT_ROOT/val/dataset_stats.json"
echo "  - $OUT_ROOT/test/dataset_stats.json"
echo ""
echo "Next steps:"
echo "  1. Verify splits with: python scripts/verify_splits.py"
echo "  2. Start Phase 1 training: bash scripts/train_phase1_decoder.sh"
echo "========================================"
