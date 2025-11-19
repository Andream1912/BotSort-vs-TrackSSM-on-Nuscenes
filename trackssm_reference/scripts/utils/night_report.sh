#!/bin/bash

# Script completo per verificare risultati del training notturno

echo "========================================="
echo "📊 TRAINING NIGHT REPORT"
echo "========================================="
echo "Data: $(date)"
echo ""

# 1. Check processi attivi
echo "1️⃣  PROCESSI ATTIVI"
echo "----------------------------------------"

TRAIN_PID=$(pgrep -f "train_phase1_decoder.py")
if [ ! -z "$TRAIN_PID" ]; then
    RUNTIME=$(ps -p $TRAIN_PID -o etime= | xargs)
    CPU=$(ps -p $TRAIN_PID -o %cpu= | xargs)
    MEM=$(ps -p $TRAIN_PID -o %mem= | xargs)
    echo "✅ Training ATTIVO"
    echo "   PID: $TRAIN_PID"
    echo "   Runtime: $RUNTIME"
    echo "   CPU: $CPU%"
    echo "   Memory: $MEM%"
else
    echo "⚠️  Training NON attivo"
    echo "   (Potrebbe essere completato o terminato con errore)"
fi
echo ""

# 2. Training Progress
echo "2️⃣  TRAINING PROGRESS"
echo "----------------------------------------"

TRAIN_LOG="/user/amarino/tesi_project_amarino/trackssm_reference/logs/phase1_training_cpu.log"

if [ -f "$TRAIN_LOG" ]; then
    # Epoche completate
    COMPLETED_EPOCHS=$(grep -c "Epoch [0-9]*/40" "$TRAIN_LOG")
    echo "Epoche completate: $COMPLETED_EPOCHS / 40"
    
    # Ultima epoca
    LAST_EPOCH=$(grep "Epoch" "$TRAIN_LOG" | tail -1)
    if [ ! -z "$LAST_EPOCH" ]; then
        echo "Ultima epoca: $LAST_EPOCH"
    fi
    
    # Loss trend (ultime 5 epoche)
    echo ""
    echo "📉 Loss Trend (ultime 5 epoche):"
    grep "Train Loss:" "$TRAIN_LOG" | tail -5
    
    # Migliore val loss
    BEST_VAL=$(grep "Best model saved" "$TRAIN_LOG" | tail -1)
    if [ ! -z "$BEST_VAL" ]; then
        echo ""
        echo "🏆 Best checkpoint: $BEST_VAL"
    fi
else
    echo "⚠️  Log non trovato: $TRAIN_LOG"
fi
echo ""

# 3. Checkpoint salvati
echo "3️⃣  CHECKPOINT SALVATI"
echo "----------------------------------------"

WEIGHTS_DIR="/user/amarino/tesi_project_amarino/trackssm_reference/weights/phase1"

if [ -d "$WEIGHTS_DIR" ]; then
    echo "Directory: $WEIGHTS_DIR"
    echo ""
    
    # Lista checkpoint
    CHECKPOINTS=$(ls -lh "$WEIGHTS_DIR"/*.pth 2>/dev/null)
    if [ ! -z "$CHECKPOINTS" ]; then
        echo "$CHECKPOINTS"
        echo ""
        CHECKPOINT_COUNT=$(ls -1 "$WEIGHTS_DIR"/*.pth 2>/dev/null | wc -l)
        echo "Totale checkpoint: $CHECKPOINT_COUNT"
    else
        echo "⚠️  Nessun checkpoint trovato"
    fi
else
    echo "⚠️  Directory weights non trovata"
fi
echo ""

# 4. Errori nel log
echo "4️⃣  ERRORI RILEVATI"
echo "----------------------------------------"

if [ -f "$TRAIN_LOG" ]; then
    ERROR_COUNT=$(grep -i "error\|exception\|traceback" "$TRAIN_LOG" | wc -l)
    
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "❌ Trovati $ERROR_COUNT errori nel log"
        echo ""
        echo "Ultimi errori:"
        grep -i "error\|exception" "$TRAIN_LOG" | tail -5
    else
        echo "✅ Nessun errore rilevato"
    fi
else
    echo "⚠️  Log non disponibile"
fi
echo ""

# 5. Spazio disco
echo "5️⃣  SPAZIO DISCO"
echo "----------------------------------------"

df -h /user/amarino/tesi_project_amarino | tail -1
echo ""

# 6. Prossimi step
echo "6️⃣  PROSSIMI STEP"
echo "----------------------------------------"

if [ ! -z "$TRAIN_PID" ]; then
    echo "Training in corso - continua a monitorare:"
    echo "  tail -f logs/phase1_training_cpu.log"
    echo "  bash scripts/utils/monitor_training.sh"
elif [ -f "$WEIGHTS_DIR/phase1_decoder_best.pth" ]; then
    echo "✅ Training completato! Prossimi step:"
    echo "  1. Valuta risultati: check checkpoint migliore"
    echo "  2. Se GPU disponibile: rilancia per velocizzare"
    echo "  3. Oppure procedi con Phase 2"
else
    echo "⚠️  Training non completato - verifica errori:"
    echo "  cat logs/phase1_training_cpu.log | grep -i error"
    echo "  bash scripts/utils/test_training_pipeline.py"
fi

echo ""
echo "========================================="
echo "Report completato: $(date)"
echo "========================================="
