#!/bin/bash

# Monitor Grid Search Progress

cd /user/amarino/tesi_project_amarino

echo "========================================"
echo "GRID SEARCH MONITORING"
echo "========================================"
echo ""

# Check main process
MAIN_PID=$(ps aux | grep "grid_search_simple.py" | grep -v grep | awk '{print $2}')
if [ -z "$MAIN_PID" ]; then
    echo "❌ Main process NOT running"
else
    echo "✅ Main process running (PID: $MAIN_PID)"
fi

# Count active workers
WORKERS=$(ps aux | grep "amarino.*python.*track.py" | grep -v grep | wc -l)
echo "🔧 Active workers: $WORKERS"

echo ""
echo "========================================"
echo "EXPERIMENT STATUS"
echo "========================================"

# Count total experiments
TOTAL_EXPS=$(ls -1d results/GRID_SEARCH/exp_* 2>/dev/null | wc -l)
echo "📊 Total experiments launched: $TOTAL_EXPS / 320"

# Count completed (with metrics.json)
COMPLETED=$(find results/GRID_SEARCH/exp_*/metrics.json 2>/dev/null | wc -l)
echo "✅ Completed: $COMPLETED"
echo "🔄 In progress: $((TOTAL_EXPS - COMPLETED))"

if [ $COMPLETED -gt 0 ]; then
    PCT=$((COMPLETED * 100 / 320))
    echo "📈 Progress: $PCT%"
    
    # Estimate time remaining
    if [ $COMPLETED -gt 3 ]; then
        # Rough estimate: 11 min per exp, 4 workers
        REMAINING=$((320 - COMPLETED))
        TIME_MIN=$((REMAINING * 11 / 4))
        TIME_HOURS=$((TIME_MIN / 60))
        echo "⏱️  Estimated time remaining: ~${TIME_HOURS}h $((TIME_MIN % 60))min"
    fi
fi

echo ""
echo "========================================"
echo "CURRENT BEST CONFIGURATION"
echo "========================================"

if [ -f "results/GRID_SEARCH/best_config.json" ]; then
    echo "🏆 Best config found:"
    python3 -c "
import json
with open('results/GRID_SEARCH/best_config.json') as f:
    data = json.load(f)
    print(f\"  Experiment: {data['experiment_id']:04d}\")
    print(f\"  Score: {data['score']:.2f}\")
    print(f\"  MOTA: {data['metrics'].get('MOTA', 0):.2f}%\")
    print(f\"  IDF1: {data['metrics'].get('IDF1', 0):.2f}%\")
    print(f\"  HOTA: {data['metrics'].get('HOTA', 0):.2f}%\")
    print(f\"  IDSW: {data['metrics'].get('IDSW', 0)}\")
    print(f\"  Config: conf={data['config']['conf_thresh']}, match={data['config']['match_thresh']}, track={data['config']['track_thresh']}, nms={data['config']['nms_thresh']}\")
" 2>/dev/null
else
    echo "⏳ No completed experiments yet"
fi

echo ""
echo "========================================"
echo "ACTIVE EXPERIMENTS"
echo "========================================"

# Show active experiments with progress
for exp_dir in results/GRID_SEARCH/exp_*; do
    if [ -d "$exp_dir" ] && [ ! -f "$exp_dir/metrics.json" ]; then
        exp_name=$(basename $exp_dir)
        log_file="$exp_dir/tracking.log"
        if [ -f "$log_file" ]; then
            # Get last progress line
            progress=$(tail -5 "$log_file" 2>/dev/null | grep "Progress:" | tail -1)
            if [ -n "$progress" ]; then
                # Extract percentage
                pct=$(echo "$progress" | grep -oP '\d+%' | head -1)
                echo "  $exp_name: $pct"
            fi
        fi
    fi
done | head -10

echo ""
echo "========================================"
echo "Run this script again to update status"
echo "  watch -n 30 ./scripts/grid_search/monitor.sh"
echo "========================================"
