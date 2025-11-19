#!/bin/bash

# Script per lanciare training quando GPU diventa disponibile

echo "========================================="
echo "GPU WATCHER - Auto-launch Training"
echo "========================================="
echo "Attende che GPU diventi disponibile..."
echo "Poi lancia automaticamente training Phase 1"
echo ""

# Funzione per check GPU
check_gpu() {
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
    return $?
}

# Loop di attesa
WAIT_COUNT=0
while true; do
    if check_gpu; then
        echo ""
        echo "✅ GPU DISPONIBILE!"
        echo ""
        
        # Mostra info GPU
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
        echo ""
        
        # Lancia training
        echo "🚀 Lancio Training Phase 1..."
        echo ""
        
        cd /user/amarino/tesi_project_amarino/trackssm_reference
        
        nohup python scripts/training/train_phase1_decoder.py \
            --config configs/nuscenes_phase1_gpu.yaml \
            --data_root ./data/nuscenes_mot_6cams_interpolated \
            --output_dir weights/phase1 \
            > logs/phase1_training_gpu.log 2>&1 &
        
        TRAIN_PID=$!
        
        echo "✅ Training avviato in background!"
        echo ""
        echo "   PID: $TRAIN_PID"
        echo "   Config: configs/nuscenes_phase1_gpu.yaml"
        echo "   Batch size: 64"
        echo "   Workers: 12"
        echo "   Device: CUDA"
        echo ""
        echo "📝 Monitoraggio:"
        echo "   tail -f logs/phase1_training_gpu.log"
        echo "   bash scripts/utils/monitor_training.sh"
        echo ""
        echo "⏱️  Tempo stimato: ~12-16 ore"
        echo ""
        
        # Attendi che training si avvii
        sleep 30
        
        # Verifica che sia ancora attivo
        if ps -p $TRAIN_PID > /dev/null; then
            echo "✅ Training in esecuzione!"
            echo ""
            echo "Prime righe log:"
            tail -30 logs/phase1_training_gpu.log
        else
            echo "❌ Training terminato subito - controlla errori:"
            tail -50 logs/phase1_training_gpu.log
        fi
        
        break
    fi
    
    # Mostra progress ogni 60 secondi
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Attesa GPU... (check ogni 1 sec)"
    fi
    
    WAIT_COUNT=$((WAIT_COUNT + 1))
    sleep 1
done

echo ""
echo "========================================="
echo "GPU Watcher completato!"
echo "========================================="
