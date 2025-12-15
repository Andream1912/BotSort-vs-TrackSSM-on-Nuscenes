#!/bin/bash
# Monitor YOLOX training progress and validation

LOG_FILE="/user/amarino/tesi_project_amarino/yolox_finetuning/logs/training_with_validation_FINAL.log"

echo "=========================================="
echo "YOLOX TRAINING MONITOR"
echo "=========================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Training process NOT running!"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit 1
else
    echo "✅ Training process RUNNING (PID: $PID)"
fi

echo ""
echo "📊 Current Progress:"
echo "===================="

# Get current epoch and iteration
CURRENT=$(tail -100 "$LOG_FILE" | grep "epoch.*iter" | tail -1)
if [ ! -z "$CURRENT" ]; then
    echo "$CURRENT"
else
    echo "No training iterations logged yet..."
fi

echo ""
echo "🔍 Validation Status:"
echo "===================="

# Check if validation has run
VAL_COUNT=$(grep -c "Average forward time" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$VAL_COUNT" -gt "0" ]; then
    echo "✅ Validation has run $VAL_COUNT time(s)"
    echo ""
    echo "Latest validation results:"
    grep -A10 "Average forward time" "$LOG_FILE" | tail -15
else
    echo "⏳ Waiting for first validation (after epoch 1)..."
fi

echo ""
echo "📈 Latest Training Loss:"
echo "===================="
tail -100 "$LOG_FILE" | grep "epoch.*iter.*total_loss" | tail -5

echo ""
echo "=========================================="
echo "Commands:"
echo "  Watch live:    tail -f $LOG_FILE"
echo "  Check process: ps aux | grep train.py"
echo "  This script:   bash $0"
echo "=========================================="
