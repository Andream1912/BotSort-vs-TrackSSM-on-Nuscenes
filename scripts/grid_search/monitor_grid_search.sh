#!/bin/bash

# Monitor grid search progress in real-time
# Shows: progress, active experiments, current best config

BEST_CONFIG="results/GRID_SEARCH/best_config.json"
PROGRESS="results/GRID_SEARCH/progress.json"

echo "=========================================="
echo "GRID SEARCH MONITOR"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "GRID SEARCH PROGRESS MONITOR"
    echo "Last update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Progress
    if [ -f "$PROGRESS" ]; then
        echo "📊 PROGRESS:"
        python3 -c "
import json
with open('$PROGRESS', 'r') as f:
    data = json.load(f)
    completed = data.get('completed', 0)
    total = data.get('total', 0)
    pct = data.get('progress_pct', 0)
    active = data.get('active_experiments', [])
    
    print(f'  Completed: {completed}/{total} ({pct:.1f}%)')
    print(f'  Active experiments: {active}')
" 2>/dev/null || echo "  No progress data yet"
        echo ""
    else
        echo "📊 PROGRESS: Not started yet"
        echo ""
    fi
    
    # Best config
    if [ -f "$BEST_CONFIG" ]; then
        echo "🏆 CURRENT BEST CONFIGURATION:"
        python3 -c "
import json
with open('$BEST_CONFIG', 'r') as f:
    data = json.load(f)
    exp_id = data.get('experiment_id', 0)
    score = data.get('score', 0)
    config = data.get('config', {})
    metrics = data.get('metrics', {})
    
    print(f'  Experiment ID: {exp_id:04d}')
    print(f'  Score: {score:.2f}')
    print(f'')
    print(f'  Configuration:')
    print(f'    conf_thresh:  {config.get(\"conf_thresh\", 0)}')
    print(f'    match_thresh: {config.get(\"match_thresh\", 0)}')
    print(f'    track_thresh: {config.get(\"track_thresh\", 0)}')
    print(f'    nms_thresh:   {config.get(\"nms_thresh\", 0)}')
    print(f'')
    print(f'  Metrics:')
    print(f'    MOTA:  {metrics.get(\"MOTA\", 0):.2f}%')
    print(f'    IDF1:  {metrics.get(\"IDF1\", 0):.2f}%')
    print(f'    HOTA:  {metrics.get(\"HOTA\", 0):.2f}%')
    print(f'    IDSW:  {metrics.get(\"IDSW\", 0)}')
    print(f'    Recall: {metrics.get(\"RECALL\", 0):.2f}%')
    print(f'    Prec:  {metrics.get(\"precision\", 0):.2f}%')
" 2>/dev/null || echo "  No best config yet"
    else
        echo "🏆 CURRENT BEST: No experiments completed yet"
    fi
    
    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to exit monitor"
    echo "Refreshing every 30 seconds..."
    echo "=========================================="
    
    sleep 30
done
