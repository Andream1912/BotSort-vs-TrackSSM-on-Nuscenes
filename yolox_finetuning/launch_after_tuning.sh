#!/bin/bash
# Script to launch smooth training AFTER tuning experiments complete

echo "=============================================="
echo "Waiting for tuning experiments to complete..."
echo "=============================================="

# Wait for all 4 tuning processes to finish
while true; do
    COUNT=$(ps aux | grep "[t]rack.py" | grep "GRID_SEARCH" | wc -l)
    if [ $COUNT -eq 0 ]; then
        echo "✓ All tuning experiments completed!"
        break
    else
        echo "  Still running: $COUNT / 4 processes"
        sleep 60  # Check every minute
    fi
done

echo ""
echo "=============================================="
echo "Launching YOLOX Smooth Training (Batch 32)"
echo "=============================================="

# Activate environment
source /user/amarino/miniconda3/bin/activate trackssm

cd /user/amarino/tesi_project_amarino

# Launch smooth training
echo "Starting training with:"
echo "  - Batch size: 32 (4x larger for stability)"
echo "  - Cosine annealing scheduler"
echo "  - 10 epochs"
echo ""

nohup python yolox_finetuning/scripts/train_smooth.py > yolox_finetuning/training_smooth.log 2>&1 &
PID=$!

echo "Training PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f yolox_finetuning/training_smooth.log"
echo ""
echo "Check progress:"
echo "  ps aux | grep train_smooth"
