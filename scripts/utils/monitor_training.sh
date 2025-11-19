#!/bin/bash

# Script per monitorare lo stato del training durante la notte

echo "========================================="
echo "Training Monitor Dashboard"
echo "========================================="
echo ""

# Check dataset generation
DATASET_GEN_PID=$(pgrep -f "generate_splits.sh")
if [ ! -z "$DATASET_GEN_PID" ]; then
    echo "📦 Dataset Generation: ✅ IN CORSO (PID: $DATASET_GEN_PID)"
    DATASET_LOG="/user/amarino/tesi_project_amarino/trackssm_reference/logs/dataset_generation.log"
    if [ -f "$DATASET_LOG" ]; then
        LAST_LINE=$(tail -1 "$DATASET_LOG")
        echo "   └─ $LAST_LINE"
    fi
else
    echo "📦 Dataset Generation: ✅ COMPLETATO"
fi

echo ""

# Check watcher
WATCHER_PID=$(pgrep -f "auto_train_after_dataset.sh")
if [ ! -z "$WATCHER_PID" ]; then
    echo "🤖 Auto-Train Watcher: ✅ ATTIVO (PID: $WATCHER_PID)"
else
    echo "🤖 Auto-Train Watcher: ⚠️  NON ATTIVO"
fi

echo ""

# Check training
TRAIN_PID=$(pgrep -f "train_phase1_decoder.py")
if [ ! -z "$TRAIN_PID" ]; then
    echo "🚀 Training Phase 1: ✅ IN CORSO (PID: $TRAIN_PID)"
    
    # Get runtime
    RUNTIME=$(ps -p $TRAIN_PID -o etime= | xargs)
    echo "   ├─ Runtime: $RUNTIME"
    
    # Check log
    TRAIN_LOG="/user/amarino/tesi_project_amarino/trackssm_reference/logs/phase1_training_cpu.log"
    if [ -f "$TRAIN_LOG" ]; then
        # Get current epoch
        EPOCH=$(grep -oP "Epoch \K\d+(?=/)" "$TRAIN_LOG" | tail -1)
        if [ ! -z "$EPOCH" ]; then
            echo "   ├─ Current Epoch: $EPOCH / 40"
        fi
        
        # Get latest loss
        TRAIN_LOSS=$(grep "Train Loss:" "$TRAIN_LOG" | tail -1 | grep -oP "Train Loss: \K[0-9.]+")
        VAL_LOSS=$(grep "Val Loss:" "$TRAIN_LOG" | tail -1 | grep -oP "Val Loss: \K[0-9.]+")
        
        if [ ! -z "$TRAIN_LOSS" ]; then
            echo "   ├─ Latest Train Loss: $TRAIN_LOSS"
        fi
        
        if [ ! -z "$VAL_LOSS" ]; then
            echo "   └─ Latest Val Loss: $VAL_LOSS"
        fi
    fi
else
    echo "🚀 Training Phase 1: ⏳ IN ATTESA"
    
    # Check if test passed
    TEST_LOG="/user/amarino/tesi_project_amarino/trackssm_reference/logs/pipeline_test.log"
    if [ -f "$TEST_LOG" ]; then
        if grep -q "TUTTI I TEST PASSATI" "$TEST_LOG"; then
            echo "   └─ Pipeline test: ✅ PASSATO"
        elif grep -q "TEST FALLITI" "$TEST_LOG"; then
            echo "   └─ Pipeline test: ❌ FALLITO - controlla logs/pipeline_test.log"
        fi
    fi
fi

echo ""
echo "========================================="
echo "Log Files:"
echo "========================================="
echo "Dataset Gen:  logs/dataset_generation.log"
echo "Pipeline Test: logs/pipeline_test.log"
echo "Training:     logs/phase1_training_cpu.log"
echo "Watcher:      logs/auto_train_watcher.log"
echo ""
echo "Comandi utili:"
echo "  tail -f logs/phase1_training_cpu.log    # Segui training live"
echo "  watch -n 60 bash scripts/utils/monitor_training.sh  # Auto-refresh"
echo "========================================="
