#!/bin/bash
# Quick status check for grid search

cd /user/amarino/tesi_project_amarino

echo "========================================"
echo "GRID SEARCH STATUS - $(date '+%H:%M:%S')"
echo "========================================"

# Main process
MAIN_PID=$(ps aux | grep "grid_search_simple.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$MAIN_PID" ]; then
    echo "❌ Main process NOT running"
else
    echo "✅ Main process running (PID: $MAIN_PID)"
fi

# Workers
WORKERS=$(ps aux | grep "amarino.*python.*track.py" | grep -v grep | wc -l)
echo "🔧 Active workers: $WORKERS"

# Experiments
TOTAL=$(ls -1d results/GRID_SEARCH/exp_* 2>/dev/null | wc -l)
COMPLETED=$(find results/GRID_SEARCH/exp_* -name "metrics.json" 2>/dev/null | wc -l)
echo "📊 Experiments: $COMPLETED completed / $TOTAL launched / 320 total"

if [ $COMPLETED -gt 0 ]; then
    PCT=$((COMPLETED * 100 / 320))
    REMAINING=$((320 - COMPLETED))
    echo "📈 Progress: $PCT% ($REMAINING remaining)"
fi

# Current experiments
echo ""
echo "🏃 Currently running:"
ps aux | grep "amarino.*track.py" | grep -v grep | while read line; do
    EXP=$(echo $line | grep -o "exp_[0-9]*")
    if [ -n "$EXP" ]; then
        LOG="results/GRID_SEARCH/$EXP/tracking.log"
        if [ -f "$LOG" ]; then
            PROG=$(tail -5 "$LOG" 2>/dev/null | grep "Progress:" | tail -1 | grep -oP '\d+%' | head -1)
            SCENES=$(ls results/GRID_SEARCH/$EXP/data/*.txt 2>/dev/null | wc -l)
            echo "  $EXP: ${PROG:-0%} ($SCENES/151 scenes)"
        fi
    fi
done

echo ""
echo "========================================"
