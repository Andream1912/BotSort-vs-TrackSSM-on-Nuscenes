#!/bin/bash
# QUICK LAUNCH - Per lancio veloce senza conferme

cd /user/amarino/tesi_project_amarino/trackssm_reference

nohup python scripts/training/train_phase1_decoder.py \
    --config configs/nuscenes_phase1_gpu.yaml \
    --data_root ./data/nuscenes_mot_6cams_interpolated \
    --output_dir weights/phase1 \
    > logs/phase1_training.log 2>&1 &

TRAIN_PID=$!

echo "🚀 Training Phase 1 avviato in background!"
echo ""
echo "   PID: $TRAIN_PID"
echo "   Config: GPU (batch=64, workers=12)"
echo "   Dataset: 4.5M train tracks"
echo "   Log: logs/phase1_training.log"
echo ""
echo "📝 Monitor:"
echo "   tail -f logs/phase1_training.log"
echo "   bash scripts/utils/monitor_training.sh"
echo ""
echo "⏱️  Tempo stimato: 12-16 ore"
