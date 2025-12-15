#!/bin/bash

# Pre-warm Triton cache before grid search
# This runs 1 scene to compile all Mamba/Triton kernels

echo "=========================================="
echo "PRE-WARMING TRITON CACHE"
echo "=========================================="
echo "This will run 1 scene to compile Mamba kernels"
echo "After this, all grid search experiments will start fast"
echo ""

cd /user/amarino/tesi_project_amarino

export TRITON_CACHE_DIR="/user/amarino/.triton_cache"

# Run on 1 scene only
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
    --conf-thresh 0.3 \
    --nms-thresh 0.65 \
    --match-thresh 0.85 \
    --track-thresh 0.6 \
    --output results/WARMUP_CACHE \
    --gt-data data/nuscenes_mot_front/val \
    --scenes scene-0003_CAM_FRONT

echo ""
echo "✓ Cache warmed up! Triton kernels compiled."
echo "  Grid search experiments will now start immediately."
