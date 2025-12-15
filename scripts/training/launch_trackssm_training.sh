#!/bin/bash
# Launch TrackSSM training to regenerate all logs and plots
# Same configuration as original training

set -e

PROJECT_DIR="/user/amarino/tesi_project_amarino"
cd "$PROJECT_DIR"

echo "=========================================="
echo "TRACKSSM TRAINING - Regenerate Logs/Plots"
echo "=========================================="
echo ""

# Configuration
TRACKER_PATH="external/TrackSSM"
CONFIG="external/TrackSSM/configs/nuscenes_trackssm.yaml"
OUTPUT_DIR="weights/trackssm_training_$(date +%Y%m%d)"
LOG_FILE="logs/trackssm_training_$(date +%Y%m%d_%H%M).log"

# Check if tracker exists
if [ ! -d "$TRACKER_PATH" ]; then
    echo "❌ TrackSSM not found: $TRACKER_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

echo "📋 Configuration:"
echo "   Tracker: TrackSSM"
echo "   Config: $CONFIG"
echo "   Output: $OUTPUT_DIR"
echo "   Log: $LOG_FILE"
echo ""

# Create directories
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "STARTING TRACKSSM TRAINING"
echo "=========================================="
echo ""
echo "📝 Log file: $LOG_FILE"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_FILE"
echo ""

# Launch training
cd "$TRACKER_PATH"

nohup python3 train.py \
    --config "$PROJECT_DIR/$CONFIG" \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    > "$PROJECT_DIR/$LOG_FILE" 2>&1 &

TRAIN_PID=$!

cd "$PROJECT_DIR"

echo "✅ TrackSSM training started!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check progress:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
