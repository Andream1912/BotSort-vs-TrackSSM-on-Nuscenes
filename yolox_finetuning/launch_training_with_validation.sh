#!/bin/bash
# Script per lanciare training YOLOX con VALIDATION
# Data: 14 dicembre 2025

set -e

echo "════════════════════════════════════════════════════════════════"
echo "         YOLOX-L Training with VALIDATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check current directory
if [ ! -d "yolox_finetuning" ]; then
    echo "❌ Errore: Esegui da /user/amarino/tesi_project_amarino/"
    exit 1
fi

# Check config exists
if [ ! -f "yolox_finetuning/configs/yolox_l_nuscenes_stable.py" ]; then
    echo "❌ Errore: Config stable non trovato"
    exit 1
fi

# Check checkpoint exists
if [ ! -f "weights/detectors/yolox_l.pth" ]; then
    echo "❌ Errore: COCO checkpoint non trovato"
    exit 1
fi

# Check validation annotations
if [ ! -f "data/nuscenes_yolox_6cams/annotations/val.json" ]; then
    echo "❌ Errore: Validation annotations non trovate"
    exit 1
fi

echo "✅ Directory: $(pwd)"
echo "✅ Config: yolox_finetuning/configs/yolox_l_nuscenes_stable.py"
echo "✅ Checkpoint: weights/detectors/yolox_l.pth"
echo "✅ Val annotations: data/nuscenes_yolox_6cams/annotations/val.json"
echo ""

# Show configuration
echo "📊 CONFIGURAZIONE TRAINING:"
echo "─────────────────────────────────────────────────────────────"
grep -E "max_epoch|batch_size|basic_lr_per_img|warmup_epochs|no_aug_epochs|mosaic_prob|mixup_prob" \
    yolox_finetuning/configs/yolox_l_nuscenes_stable.py | \
    grep -v "^#" | sed 's/self./  /g' | sed 's/=/ = /g'
echo "─────────────────────────────────────────────────────────────"
echo ""

echo "📈 NUOVE FEATURES:"
echo "  ✨ Validation ogni 5 epoche"
echo "  ✨ Calcolo mAP durante training"
echo "  ✨ Salvataggio best model automatico"
echo "  ✨ History validation in JSON"
echo "  ✨ Validation extra alle epoche: 1, 5, 10, 15, 20, 25, 30"
echo ""

# Estimate time
echo "⏱️  STIMA TEMPI:"
echo "  Tempo per epoca training: ~26 minuti"
echo "  Tempo per validation: ~5 minuti"
echo "  Totale 30 epoche: ~14.5 ore (con 6 validation)"
echo "  Checkpoint: epochs 1, 5, 10, 15, 20, 25, 30 + best_model.pth"
echo ""

echo "📁 OUTPUT:"
echo "  Checkpoints: yolox_finetuning/yolox_l_nuscenes_stable/"
echo "  Best model: yolox_finetuning/yolox_l_nuscenes_stable/best_model.pth"
echo "  Val history: yolox_finetuning/yolox_l_nuscenes_stable/validation_history.json"
echo "  Training log: yolox_finetuning/training_with_validation.log"
echo ""

# Ask confirmation
read -p "🚀 Vuoi lanciare il training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training annullato"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🚀 LANCIO TRAINING CON VALIDATION..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Launch training
nohup python yolox_finetuning/scripts/train_with_validation.py > yolox_finetuning/training_with_validation.log 2>&1 &
PID=$!

echo "✅ Training avviato con validation!"
echo ""
echo "📋 INFORMAZIONI:"
echo "  PID: $PID"
echo "  Log: yolox_finetuning/training_with_validation.log"
echo "  Output dir: yolox_finetuning/yolox_l_nuscenes_stable/"
echo ""
echo "📊 MONITORAGGIO:"
echo ""
echo "  # Segui il training in tempo reale"
echo "  tail -f yolox_finetuning/training_with_validation.log"
echo ""
echo "  # Verifica validation results"
echo "  cat yolox_finetuning/yolox_l_nuscenes_stable/validation_history.json"
echo ""
echo "  # Verifica checkpoints"
echo "  ls -lth yolox_finetuning/yolox_l_nuscenes_stable/*.pth"
echo ""
echo "  # Estrai solo validation metrics dal log"
echo "  grep 'mAP@' yolox_finetuning/training_with_validation.log"
echo ""
echo "  # Controlla se running"
echo "  ps aux | grep $PID"
echo ""
echo "  # Best model finora"
echo "  grep 'New best model' yolox_finetuning/training_with_validation.log"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✨ Training in corso con validation! Tempo stimato: ~14.5 ore"
echo "════════════════════════════════════════════════════════════════"
