#!/bin/bash
# Monitor progress of 4 parallel TrackSSM tuning tests

RESULTS_DIR="results/GRID_SEARCH_AFTER_DETECTOR_FINETUNING"

echo "=========================================="
echo "TrackSSM Tuning - Progress Monitor"
echo "Target: IDSW < 2754 (beat BotSort)"
echo "Baseline: IDSW = 2857"
echo "=========================================="
echo ""

# Check each test
for test in TEST1_MATCH0.85 TEST2_CONF0.25 TEST3_AGE40 TEST4_COMBINED; do
    echo "[$test]"
    
    # Check if running
    if ps aux | grep -q "[t]rack.py.*$test"; then
        echo "  Status: 🔄 Running"
        
        # Check progress from log
        if [ -f "$RESULTS_DIR/${test}.log" ]; then
            # Get last progress line
            PROGRESS=$(tail -50 "$RESULTS_DIR/${test}.log" | grep "Overall Progress" | tail -1)
            if [ ! -z "$PROGRESS" ]; then
                echo "  Progress: $PROGRESS"
            fi
            
            # Check for errors
            if grep -q "Error\|Traceback" "$RESULTS_DIR/${test}.log"; then
                echo "  ⚠️  Errors detected in log!"
            fi
        fi
    else
        echo "  Status: ⏸️  Not running (completed or not started)"
        
        # Check if results exist
        if [ -f "$RESULTS_DIR/$test/metrics_motmetrics.txt" ]; then
            echo "  Status: ✅ Completed"
            
            # Extract key metrics
            IDSW=$(grep "IDSW:" "$RESULTS_DIR/$test/metrics_motmetrics.txt" | awk '{print $2}')
            MOTA=$(grep "MOTA:" "$RESULTS_DIR/$test/metrics_motmetrics.txt" | awk '{print $2}')
            IDF1=$(grep "IDF1:" "$RESULTS_DIR/$test/metrics_motmetrics.txt" | awk '{print $2}')
            
            echo "  Results:"
            echo "    IDSW: $IDSW"
            echo "    MOTA: $MOTA"
            echo "    IDF1: $IDF1"
            
            if [ ! -z "$IDSW" ] && [ "$IDSW" -lt 2754 ]; then
                echo "    🎉 BEATS BOTSORT!"
            fi
        fi
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="

# Count completed
COMPLETED=0
for test in TEST1_MATCH0.85 TEST2_CONF0.25 TEST3_AGE40 TEST4_COMBINED; do
    if [ -f "$RESULTS_DIR/$test/metrics_motmetrics.txt" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done

echo "Completed: $COMPLETED / 4"
echo ""

if [ $COMPLETED -eq 4 ]; then
    echo "All tests completed! Finding best result..."
    echo ""
    
    BEST_IDSW=9999
    BEST_TEST=""
    
    for test in TEST1_MATCH0.85 TEST2_CONF0.25 TEST3_AGE40 TEST4_COMBINED; do
        if [ -f "$RESULTS_DIR/$test/metrics_motmetrics.txt" ]; then
            IDSW=$(grep "IDSW:" "$RESULTS_DIR/$test/metrics_motmetrics.txt" | awk '{print $2}')
            
            if [ ! -z "$IDSW" ] && [ "$IDSW" -lt "$BEST_IDSW" ]; then
                BEST_IDSW=$IDSW
                BEST_TEST=$test
            fi
        fi
    done
    
    echo "🏆 BEST RESULT:"
    echo "  Test: $BEST_TEST"
    echo "  IDSW: $BEST_IDSW"
    echo ""
    echo "Comparison:"
    echo "  BotSort: 2754 IDSW"
    echo "  Best TrackSSM: $BEST_IDSW IDSW"
    
    if [ "$BEST_IDSW" -lt 2754 ]; then
        DIFF=$((2754 - BEST_IDSW))
        echo "  🎉 TrackSSM WINS by $DIFF switches!"
    else
        DIFF=$((BEST_IDSW - 2754))
        echo "  ⚠️  Still behind by $DIFF switches"
        echo "  Next step: Fine-tune TrackSSM model on NuScenes"
    fi
fi

echo "=========================================="
