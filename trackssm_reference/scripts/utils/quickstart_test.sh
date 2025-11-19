#!/bin/bash
# Quick Start Guide per Fine-tuning Pipeline TrackSSM
# Test rapido con subset ridotto per verificare funzionamento

set -e

echo "============================================================================"
echo "TrackSSM Fine-tuning Pipeline - QUICK START TEST"
echo "============================================================================"
echo ""
echo "Questo script esegue un test rapido della pipeline con dataset ridotto:"
echo "  - Solo 10 scene dal validation split (invece di 700 train)"
echo "  - Solo CAM_FRONT (invece di 6 camere)"
echo "  - Training ridotto: 5 epochs Phase 1, 10 epochs Phase 2"
echo ""
echo "ATTENZIONE: Questo è solo per TEST. Per training reale, usa pipeline completa."
echo ""
read -p "Premere ENTER per continuare o CTRL+C per annullare..."
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================
NUSC_ROOT="${NUSC_ROOT:-/mnt/datasets/Nuscense}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$PROJECT_ROOT/data/nuscenes_quicktest"
WEIGHTS_ROOT="$PROJECT_ROOT/weights/quicktest"

echo "Configuration:"
echo "  NuScenes root: $NUSC_ROOT"
echo "  Project root: $PROJECT_ROOT"
echo "  Data output: $DATA_ROOT"
echo "  Weights output: $WEIGHTS_ROOT"
echo ""

# ============================================================================
# STEP 1: Generate Small Test Dataset
# ============================================================================
echo "============================================================================"
echo "STEP 1: Generating test dataset (val split, CAM_FRONT only, interpolated)"
echo "============================================================================"
echo ""

cd "$PROJECT_ROOT"

python prepare_nuscenes_interpolated.py \
    --nusc_root "$NUSC_ROOT" \
    --version v1.0-trainval \
    --out_root "$DATA_ROOT" \
    --split val \
    --cameras CAM_FRONT \
    --interpolate \
    --min_vis 2

echo ""
echo "✓ Dataset generated"
echo ""

# Verifica
TRAIN_COUNT=$(ls -1 "$DATA_ROOT/val/" 2>/dev/null | wc -l)
echo "Generated sequences: $TRAIN_COUNT"
echo ""

if [ "$TRAIN_COUNT" -lt 50 ]; then
    echo "ERROR: Not enough sequences generated (expected ~150 from val split)"
    exit 1
fi

# ============================================================================
# STEP 2: Dataset Loader Check
# ============================================================================
echo "============================================================================"
echo "STEP 2: Dataset Loader Implementation Status"
echo "============================================================================"
echo ""

if [ -f "$PROJECT_ROOT/dataset/nuscenes_interpolated_dataset.py" ]; then
    echo "✓ Dataset loader found: dataset/nuscenes_interpolated_dataset.py"
    echo ""
else
    echo "⚠ Dataset loader NOT implemented yet"
    echo ""
    echo "NEXT STEPS:"
    echo "1. Implement dataset/nuscenes_interpolated_dataset.py"
    echo "2. Test with: python test_dataset_loader.py"
    echo "3. Re-run this script to continue"
    echo ""
    echo "Dataset generation completed successfully. Training skipped."
    exit 0
fi

# ============================================================================
# STEP 3: Phase 1 Training (Quick Test)
# ============================================================================
echo "============================================================================"
echo "STEP 3: Phase 1 Training - Decoder Only (5 epochs test)"
echo "============================================================================"
echo ""

# Create quick test config
cat > "$PROJECT_ROOT/configs/quicktest_phase1.yaml" <<EOF
# Quick test config for Phase 1
encoder_dim: 256
n_layer: 2
vocab_size: 6400
interval: 5
use_diffmot: false

batch_size: 16  # Reduced for quick test
lr: 0.0001
weight_decay: 0.0001
max_epochs: 5  # Reduced for quick test
warmup_epochs: 1

early_stop_patience: 3
early_stop_metric: "val_loss"

loss_bbox_weight: 1.0
loss_iou_weight: 2.0
loss_temporal_weight: 0.5

data_root: "$DATA_ROOT"
train_split: "val"  # Using val as train for quick test
val_split: "val"    # Same for validation

num_classes: 7
frame_rate: 12
image_width: 1600
image_height: 900

device: "cuda"
num_workers: 2
pin_memory: true
EOF

echo "Config created: configs/quicktest_phase1.yaml"
echo ""

# Check if pretrained checkpoint exists
PRETRAINED_PATH="$PROJECT_ROOT/weights/mot17_pretrained.pth"
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠ MOT17 pretrained checkpoint not found: $PRETRAINED_PATH"
    echo "Training will start from random initialization (not recommended for real training)"
    echo ""
    PRETRAINED_ARG=""
else
    echo "✓ Using pretrained checkpoint: $PRETRAINED_PATH"
    echo ""
    PRETRAINED_ARG="--pretrained $PRETRAINED_PATH"
fi

# Run Phase 1 training
python train_phase1_decoder.py \
    --config configs/quicktest_phase1.yaml \
    --data_root "$DATA_ROOT" \
    $PRETRAINED_ARG \
    --output_dir "$WEIGHTS_ROOT/phase1"

echo ""
echo "✓ Phase 1 training completed (or skipped if dataset loader missing)"
echo ""

# ============================================================================
# STEP 4: Phase 2 Training (Quick Test)
# ============================================================================
echo "============================================================================"
echo "STEP 4: Phase 2 Training - Full Model (10 epochs test)"
echo "============================================================================"
echo ""

# Check Phase 1 checkpoint
PHASE1_CHECKPOINT="$WEIGHTS_ROOT/phase1/phase1_decoder_best.pth"
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "⚠ Phase 1 checkpoint not found: $PHASE1_CHECKPOINT"
    echo "Skipping Phase 2 training"
    echo ""
    exit 0
fi

# Create quick test config
cat > "$PROJECT_ROOT/configs/quicktest_phase2.yaml" <<EOF
# Quick test config for Phase 2
encoder_dim: 256
n_layer: 2
vocab_size: 6400
interval: 5
use_diffmot: false

batch_size: 16
lr_encoder: 0.00001
lr_decoder: 0.00005
weight_decay: 0.00005
max_epochs: 10  # Reduced for quick test
warmup_epochs: 2

early_stop_patience: 4
early_stop_metric: "val_loss"

loss_bbox_weight: 1.0
loss_iou_weight: 2.0
loss_temporal_weight: 0.5

data_root: "$DATA_ROOT"
train_split: "val"
val_split: "val"

num_classes: 7
frame_rate: 12
image_width: 1600
image_height: 900

device: "cuda"
num_workers: 2
pin_memory: true
EOF

echo "Config created: configs/quicktest_phase2.yaml"
echo ""

# Run Phase 2 training
python train_phase2_full.py \
    --config configs/quicktest_phase2.yaml \
    --data_root "$DATA_ROOT" \
    --phase1_checkpoint "$PHASE1_CHECKPOINT" \
    --output_dir "$WEIGHTS_ROOT/phase2"

echo ""
echo "✓ Phase 2 training completed"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================================"
echo "QUICK START TEST COMPLETED"
echo "============================================================================"
echo ""
echo "Generated files:"
echo "  Dataset: $DATA_ROOT/val/ ($TRAIN_COUNT sequences)"
echo "  Phase 1 config: configs/quicktest_phase1.yaml"
echo "  Phase 2 config: configs/quicktest_phase2.yaml"
echo ""
if [ -f "$PHASE1_CHECKPOINT" ]; then
    echo "  Phase 1 checkpoint: $PHASE1_CHECKPOINT"
fi
if [ -f "$WEIGHTS_ROOT/phase2/phase2_full_best.pth" ]; then
    echo "  Phase 2 checkpoint: $WEIGHTS_ROOT/phase2/phase2_full_best.pth"
fi
echo ""
echo "Next steps for REAL training:"
echo "  1. Verify dataset loader implementation works correctly"
echo "  2. Generate full dataset with all cameras: bash scripts/generate_finetuning_splits.sh all"
echo "  3. Run full Phase 1 training: python train_phase1_decoder.py --config configs/nuscenes_finetuning_phase1.yaml"
echo "  4. Run full Phase 2 training: python train_phase2_full.py --config configs/nuscenes_finetuning_phase2.yaml"
echo "  5. Evaluate on test set"
echo ""
echo "Tensorboard logs: tensorboard --logdir runs/"
echo ""
echo "============================================================================"
