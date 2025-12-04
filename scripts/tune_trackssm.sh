#!/bin/bash
# Quick tuning tests for TrackSSM to beat BotSort IDSW

source /user/amarino/miniconda3/bin/activate trackssm
cd /user/amarino/tesi_project_amarino

echo "=========================================="
echo "TrackSSM Tuning Tests"
echo "Target: IDSW < 2754 (beat BotSort)"
echo "Current: IDSW = 2857 (need -103)"
echo "=========================================="

# Test 1: Higher match threshold (more conservative ID assignment)
echo ""
echo "[Test 1/4] Match threshold 0.85 (more conservative)..."
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_10.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_TUNE_MATCH085 \
    --match-thresh 0.85 \
    --max-age 30 \
    --track-thresh 0.6 \
    --conf-thresh 0.3 \
    --evaluate

# Test 2: Lower confidence threshold (keep more detections)
echo ""
echo "[Test 2/4] Confidence threshold 0.25 (keep more detections)..."
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_10.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_TUNE_CONF025 \
    --match-thresh 0.8 \
    --max-age 30 \
    --track-thresh 0.5 \
    --conf-thresh 0.25 \
    --evaluate

# Test 3: Higher max_age (more patient before losing track)
echo ""
echo "[Test 3/4] Max age 40 (more patient)..."
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_10.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_TUNE_AGE40 \
    --match-thresh 0.8 \
    --max-age 40 \
    --track-thresh 0.6 \
    --conf-thresh 0.3 \
    --evaluate

# Test 4: Combined best settings
echo ""
echo "[Test 4/4] Combined: match 0.85, conf 0.25, age 35..."
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_10.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_TUNE_COMBINED \
    --match-thresh 0.85 \
    --max-age 35 \
    --track-thresh 0.5 \
    --conf-thresh 0.25 \
    --evaluate

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Compare IDSW results:"
echo "=========================================="

for dir in results/TRACKSSM_TUNE_*/; do
    if [ -f "${dir}metrics_motmetrics.txt" ]; then
        echo ""
        echo "$(basename $dir):"
        grep "IDSW:" "${dir}metrics_motmetrics.txt" || echo "  (no results yet)"
    fi
done

echo ""
echo "Target: IDSW < 2754"
echo "Baseline TrackSSM: IDSW = 2857"
