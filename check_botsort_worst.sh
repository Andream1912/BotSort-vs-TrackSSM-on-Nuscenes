#!/bin/bash
echo "========================================"
echo "BoT-SORT WORST - Status Monitor"
echo "========================================"
echo ""

# Check process
if ps aux | grep "python3 track.py.*BOTSORT_WORST" | grep -v grep > /dev/null; then
    echo "✓ Processo ATTIVO (PID: $(pgrep -f 'python3 track.py.*BOTSORT_WORST'))"
else
    echo "✗ Processo NON ATTIVO"
fi
echo ""

# Check progress from log
echo "--- Progresso ---"
tail -3 logs/botsort_worst_experiment.log | grep "Overall Progress" | tail -1
echo ""

# Check output files
if [ -d "results/TEST_BOTSORT_WORST/data" ]; then
    num_files=$(ls results/TEST_BOTSORT_WORST/data/*.txt 2>/dev/null | wc -l)
    echo "✓ File generati: $num_files/150 scene"
    
    if [ $num_files -gt 0 ]; then
        percent=$((num_files * 100 / 150))
        echo "  Completamento: $percent%"
        
        # Stima tempo rimanente
        if [ $num_files -gt 5 ]; then
            # Assume ~4.5s per scene
            remaining=$((150 - num_files))
            eta_sec=$((remaining * 5))
            eta_min=$((eta_sec / 60))
            echo "  ETA: ~$eta_min minuti"
        fi
    fi
else
    echo "✗ Cartella output non ancora creata"
fi
echo ""
echo "Per vedere il log completo:"
echo "  tail -f logs/botsort_worst_experiment.log"
echo ""
