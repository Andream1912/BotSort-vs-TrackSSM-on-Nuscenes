#!/bin/bash
# Batch evaluate all experiments without metrics.json using motmetrics

cd /user/amarino/tesi_project_amarino

echo "=== BATCH EVALUATION WITH MOTMETRICS ==="
echo ""

# Count experiments to evaluate
total=0
for exp_dir in results/GRID_SEARCH/exp_*/; do
    if [ -d "$exp_dir/data" ] && [ ! -f "$exp_dir/metrics.json" ]; then
        total=$((total + 1))
    fi
done

echo "Found $total experiments to evaluate"
echo ""

# Evaluate each one
current=0
for exp_dir in results/GRID_SEARCH/exp_*/; do
    exp_name=$(basename "$exp_dir")
    
    # Check if needs evaluation
    if [ -d "$exp_dir/data" ] && [ ! -f "$exp_dir/metrics.json" ]; then
        current=$((current + 1))
        echo "[$current/$total] Evaluating $exp_name..."
        
        python3 scripts/evaluation/evaluate_motmetrics.py \
            --gt-folder data/nuscenes_mot_front/val \
            --pred-folder "$exp_dir/data" \
            --output "$exp_dir/metrics.json" \
            --iou-threshold 0.5 \
            > "$exp_dir/evaluation.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Success"
        else
            echo "  ✗ Failed (check $exp_dir/evaluation.log)"
        fi
    fi
done

echo ""
echo "=== EVALUATION COMPLETE ==="
echo "Evaluated: $current experiments"
