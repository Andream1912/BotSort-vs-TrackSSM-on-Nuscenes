#!/bin/bash
# Script per lanciare training ULTRA STABLE V3
# Data: 5 dicembre 2025

set -e

echo "════════════════════════════════════════════════════════════════"
echo "         YOLOX-L Training ULTRA STABLE V3"
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

echo "✅ Directory: $(pwd)"
echo "✅ Config: yolox_finetuning/configs/yolox_l_nuscenes_stable.py"
echo "✅ Checkpoint: weights/detectors/yolox_l.pth"
echo ""

# Show configuration
echo "📊 CONFIGURAZIONE TRAINING:"
echo "─────────────────────────────────────────────────────────────"
grep -E "max_epoch|batch_size|basic_lr_per_img|warmup_epochs|no_aug_epochs|mosaic_prob|mixup_prob|min_lr_ratio" \
    yolox_finetuning/configs/yolox_l_nuscenes_stable.py | \
    grep -v "^#" | sed 's/self./  /g' | sed 's/=/ = /g'
echo "─────────────────────────────────────────────────────────────"
echo ""

# Calculate effective LR
echo "📈 PARAMETRI EFFETTIVI:"
echo "  LR effettivo = 0.0006/64 × 32 = 0.000300 (-40% vs Training 2)"
echo "  Warmup: 3 epoche (stabilizzazione iniziale)"
echo "  Fine-tune: 8 epoche senza aug (epoch 22-30)"
echo "  Mosaic: 0.2 (ridotto 60%)"
echo "  Mixup: 0.15 (ridotto 70%)"
echo ""

# Estimate time
echo "⏱️  STIMA TEMPI:"
echo "  Tempo per epoca: ~26 minuti"
echo "  Totale 30 epoche: ~13 ore"
echo "  Checkpoint: epochs 1, 5, 10, 15, 20, 25, 30"
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
echo "🚀 LANCIO TRAINING..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Launch training
nohup python yolox_finetuning/scripts/train_stable.py > yolox_finetuning/training_stable.log 2>&1 &
PID=$!

echo "✅ Training avviato!"
echo ""
echo "📋 INFORMAZIONI:"
echo "  PID: $PID"
echo "  Log: yolox_finetuning/training_stable.log"
echo "  Output dir: yolox_finetuning/yolox_l_nuscenes_stable/"
echo ""
echo "📊 MONITORAGGIO:"
echo ""
echo "  # Segui il training in tempo reale"
echo "  tail -f yolox_finetuning/training_stable.log"
echo ""
echo "  # Verifica checkpoints"
echo "  ls -lth yolox_finetuning/yolox_l_nuscenes_stable/*.pth"
echo ""
echo "  # Estrai loss recenti"
echo "  grep 'iter.*loss' yolox_finetuning/training_stable.log | tail -20"
echo ""
echo "  # Controlla se running"
echo "  ps aux | grep $PID"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✨ Training in corso! Tempo stimato: ~13 ore"
echo "════════════════════════════════════════════════════════════════"
