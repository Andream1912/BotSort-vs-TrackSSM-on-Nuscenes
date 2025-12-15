#!/bin/bash
# Launch YOLOX training with validation enabled
# This will train the detector and compute validation metrics at every epoch

set -e

PROJECT_DIR="/user/amarino/tesi_project_amarino"
cd "$PROJECT_DIR"

echo "=========================================="
echo "YOLOX TRAINING WITH VALIDATION"
echo "=========================================="
echo ""

# Configuration
CONFIG="yolox_finetuning/configs/yolox_l_nuscenes_stable.py"
BATCH_SIZE=32
DEVICES=1
OUTPUT_DIR="external/YOLOX/YOLOX_outputs/yolox_x_nuscenes_7class_with_val"
LOG_FILE="yolox_finetuning/logs/training_with_validation.log"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

# Check if data exists
if [ ! -d "data/nuscenes_yolox_detector" ]; then
    echo "❌ Training data not found: data/nuscenes_yolox_detector"
    exit 1
fi

echo "📋 Configuration:"
echo "   Config: $CONFIG"
echo "   Batch size: $BATCH_SIZE"
echo "   Devices: $DEVICES"
echo "   Output: $OUTPUT_DIR"
echo "   Log: $LOG_FILE"
echo ""

# Show key config parameters
echo "🔍 Config parameters:"
echo "   max_epoch: 30"
echo "   eval_interval: 1  ← VALIDATION EVERY EPOCH ✓"
echo "   batch_size: 32"
echo "   learning_rate: 0.0006/64 per img"
echo "   warmup_epochs: 3"
echo "   no_aug_epochs: 8"
echo ""

# Create output and log directories
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""
echo "⏰ Training will take approximately 24-36 hours"
echo "📊 Validation metrics will be computed every epoch"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check validation results:"
echo "  grep -A10 'Average forward time\\|mAP' $LOG_FILE"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 1
fi

# Change to YOLOX directory
cd external/YOLOX

# Launch training with nohup
nohup python3 -m yolox.tools.train \
    -f "../../$CONFIG" \
    -d $DEVICES \
    -b $BATCH_SIZE \
    --fp16 \
    -o \
    --logger tensorboard \
    > "../../$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training started!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check progress:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "Kill if needed:"
echo "   kill $TRAIN_PID"
