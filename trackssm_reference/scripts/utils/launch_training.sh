#!/bin/bash

################################################################################
# SCRIPT DI LANCIO TRAINING - PRONTO PER DOMANI MATTINA
################################################################################

echo "========================================="
echo "🚀 TrackSSM Phase 1 Training Launcher"
echo "========================================="
echo "Data: $(date)"
echo ""

# Percorsi
PROJECT_ROOT="/user/amarino/tesi_project_amarino/trackssm_reference"
DATA_ROOT="$PROJECT_ROOT/data/nuscenes_mot_6cams_interpolated"
CONFIG="$PROJECT_ROOT/configs/nuscenes_phase1.yaml"
OUTPUT_DIR="$PROJECT_ROOT/weights/phase1"
LOG_FILE="$PROJECT_ROOT/logs/phase1_training.log"

cd $PROJECT_ROOT

# 1. Verifica CUDA
echo "1️⃣  Verifica CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -ne 0 ]; then
    echo "❌ ERRORE: CUDA non disponibile!"
    echo "   Attendi che la GPU torni disponibile"
    exit 1
fi

echo "✅ CUDA OK!"
echo ""

# 2. Verifica dataset
echo "2️⃣  Verifica dataset..."
if [ ! -d "$DATA_ROOT/train" ]; then
    echo "❌ ERRORE: Dataset non trovato in $DATA_ROOT/train"
    exit 1
fi

TRAIN_COUNT=$(ls -1 "$DATA_ROOT/train" | wc -l)
VAL_COUNT=$(ls -1 "$DATA_ROOT/val" | wc -l)
TEST_COUNT=$(ls -1 "$DATA_ROOT/test" | wc -l)

echo "✅ Dataset OK!"
echo "   Train: $TRAIN_COUNT sequenze"
echo "   Val: $VAL_COUNT sequenze"
echo "   Test: $TEST_COUNT sequenze"
echo ""

# 3. Verifica config
echo "3️⃣  Verifica config..."
if [ ! -f "$CONFIG" ]; then
    echo "❌ ERRORE: Config non trovata: $CONFIG"
    exit 1
fi

echo "✅ Config OK: $CONFIG"
cat $CONFIG | grep -E "batch_size|device|num_workers|lr"
echo ""

# 4. Verifica script training
echo "4️⃣  Verifica script training..."
python -c "
import sys
sys.path.insert(0, '.')
compile(open('scripts/training/train_phase1_decoder.py').read(), 'train_phase1_decoder.py', 'exec')
print('✅ Script sintatticamente corretto!')
"

if [ $? -ne 0 ]; then
    echo "❌ ERRORE: Script training ha errori di sintassi"
    exit 1
fi
echo ""

# 5. Crea directory output
echo "5️⃣  Preparazione directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $(dirname $LOG_FILE)
mkdir -p runs
echo "✅ Directories pronte"
echo ""

# 6. Test veloce (1 batch)
echo "6️⃣  Test rapido (1 batch)..."
timeout 60 python scripts/training/train_phase1_decoder.py \
    --config $CONFIG \
    --data_root $DATA_ROOT \
    --output_dir ${OUTPUT_DIR}_test \
    > /tmp/quick_test.log 2>&1

if grep -q "RuntimeError\|Error\|Exception" /tmp/quick_test.log; then
    echo "❌ ERRORE nel test rapido:"
    tail -30 /tmp/quick_test.log
    exit 1
fi

echo "✅ Test rapido OK!"
echo ""

# 7. Lancio training VERO
echo "========================================="
echo "🚀 LANCIO TRAINING PHASE 1"
echo "========================================="
echo ""
echo "Configurazione:"
echo "  Config: $CONFIG"
echo "  Dataset: $DATA_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo ""
echo "Training specs:"
echo "  - Batch size: 64 (GPU)"
echo "  - Workers: 12"
echo "  - Epochs: 40"
echo "  - LR: 1e-4"
echo "  - Device: CUDA"
echo ""
echo "Dataset:"
echo "  - Train samples: ~4.5M tracks"
echo "  - Val samples: ~1M tracks"
echo "  - Format: (B, 5, 8) - track-level"
echo ""
echo "⏱️  Tempo stimato: 12-16 ore su GPU"
echo ""

read -p "Vuoi procedere con il training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training annullato."
    exit 0
fi

echo ""
echo "🚀 Avvio training in background..."

PRETRAINED="/user/amarino/tesi_project_amarino/weights/MOT17_epoch160.pt"

nohup python scripts/training/train_phase1_decoder.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --pretrained "$PRETRAINED" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training avviato!"
echo ""
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "📝 Monitoraggio:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 60 'tail -30 $LOG_FILE'"
echo "   bash scripts/utils/monitor_training.sh"
echo ""
echo "🔄 Per fermare:"
echo "   kill $TRAIN_PID"
echo ""
echo "📊 Report progress:"
echo "   bash scripts/utils/night_report.sh"
echo ""
echo "========================================="
echo "Training in esecuzione!"
echo "========================================="

# Aspetta 30 secondi e mostra inizio
sleep 30
echo ""
echo "📊 Prime righe del training:"
tail -50 $LOG_FILE

echo ""
echo "✅ Setup completato!"
echo "   Il training è in esecuzione in background."
echo "   Controlla il log per il progresso."
