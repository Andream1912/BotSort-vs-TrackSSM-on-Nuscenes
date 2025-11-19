#!/bin/bash

# Script per avviare training automaticamente dopo generazione dataset

echo "========================================="
echo "Auto-Train Watcher"
echo "========================================="
echo "Attende completamento generazione dataset..."
echo "Poi avvia training Phase 1 automaticamente"
echo ""

DATASET_PID=$1
DATASET_PATH="/mnt/datasets/Nuscense/nuscenes_mot_6cams_interpolated"
TRAINING_SCRIPT="/user/amarino/tesi_project_amarino/trackssm_reference/scripts/training/train_phase1_decoder.py"
CONFIG="/user/amarino/tesi_project_amarino/trackssm_reference/configs/nuscenes_phase1.yaml"

# Aspetta che generazione dataset finisca
echo "Monitoraggio PID: $DATASET_PID"
while ps -p $DATASET_PID > /dev/null 2>&1; do
    sleep 30
done

echo ""
echo "✅ Generazione dataset completata!"
echo ""

# Verifica che dataset esista
if [ ! -d "$DATASET_PATH/train" ]; then
    echo "❌ ERRORE: Dataset non trovato in $DATASET_PATH/train"
    echo "   Controlla logs/dataset_generation.log per errori"
    exit 1
fi

echo "✅ Dataset verificato: $DATASET_PATH"
echo ""

# Conta sequenze
TRAIN_COUNT=$(ls -1 $DATASET_PATH/train 2>/dev/null | wc -l)
VAL_COUNT=$(ls -1 $DATASET_PATH/val 2>/dev/null | wc -l)
TEST_COUNT=$(ls -1 $DATASET_PATH/test 2>/dev/null | wc -l)

echo "📊 Dataset statistiche:"
echo "   Train: $TRAIN_COUNT sequenze"
echo "   Val: $VAL_COUNT sequenze"
echo "   Test: $TEST_COUNT sequenze"
echo ""

# Test completo del pipeline
echo "🧪 Eseguo test completo del pipeline..."
echo "   (Verifica: dataset loader, modelli, forward/backward pass)"
echo ""

cd /user/amarino/tesi_project_amarino/trackssm_reference
python scripts/utils/test_training_pipeline.py > logs/pipeline_test.log 2>&1

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ ERRORE: Pipeline test fallito!"
    echo "   Controlla: logs/pipeline_test.log"
    echo ""
    echo "   Test falliti - TRAINING NON AVVIATO"
    echo "   Risolvi gli errori e rilancia manualmente"
    exit 1
fi

echo "✅ Pipeline test PASSATO!"
echo "   Tutti i componenti funzionano correttamente"
echo ""

# Avvia training
echo "🚀 Avvio Training Phase 1 su CPU..."
echo "   Config: $CONFIG"
echo "   Batch size: 8 (CPU mode)"
echo "   Tempo stimato: 3-5 giorni"
echo ""

cd /user/amarino/tesi_project_amarino/trackssm_reference

nohup python $TRAINING_SCRIPT \
    --config $CONFIG \
    --data_root $DATASET_PATH \
    --output_dir weights/phase1 \
    > logs/phase1_training_cpu.log 2>&1 &

TRAIN_PID=$!

echo "✅ Training avviato in background!"
echo ""
echo "   PID: $TRAIN_PID"
echo "   Log: logs/phase1_training_cpu.log"
echo ""
echo "📝 Monitoraggio:"
echo "   tail -f logs/phase1_training_cpu.log"
echo "   ps -p $TRAIN_PID"
echo ""
echo "========================================="
echo "Auto-Train completato!"
echo "Training in esecuzione: PID $TRAIN_PID"
echo "========================================="
