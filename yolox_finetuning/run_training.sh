#!/bin/bash

# Activate conda environment
source /user/amarino/miniconda3/bin/activate trackssm

cd /user/amarino/tesi_project_amarino

# Launch training 1
echo "Launching Training 1 (LR=0.002)..."
nohup python yolox_finetuning/scripts/train_clean.py > yolox_finetuning/training_clean.log 2>&1 &
PID1=$!
echo "Training 1 PID: $PID1"

# Wait a bit
sleep 5

# Launch training 2
echo "Launching Training 2 (LR=0.001)..."
nohup python yolox_finetuning/scripts/train_clean_v2.py > yolox_finetuning/training_clean_v2.log 2>&1 &
PID2=$!
echo "Training 2 PID: $PID2"

echo ""
echo "=== Training Summary ==="
echo "Training 1: LR=0.002/64, Output=yolox_l_nuscenes_clean/"
echo "Training 2: LR=0.001/64, Output=yolox_l_nuscenes_clean_v2/"
echo ""
echo "Monitor with:"
echo "  tail -f yolox_finetuning/training_clean.log"
echo "  tail -f yolox_finetuning/training_clean_v2.log"
