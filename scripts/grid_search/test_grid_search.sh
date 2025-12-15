#!/bin/bash

# Test grid search with 2 configurations (quick test)
# This will verify: tracking works, evaluation works, best config updates

cd /user/amarino/tesi_project_amarino

echo "=========================================="
echo "GRID SEARCH - QUICK TEST (8 configs)"
echo "=========================================="
echo ""

# Create test directory
mkdir -p results/GRID_SEARCH_TEST

# Run grid search with max 8 experiments
python3 scripts/grid_search/grid_search_parallel.py \
    --detector yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
    --workers 2 \
    --max-experiments 8 \
    2>&1 | tee results/GRID_SEARCH_TEST/test_run.log

echo ""
echo "=========================================="
echo "TEST COMPLETE - Checking results..."
echo "=========================================="

# Check if best config was created
if [ -f "results/GRID_SEARCH/best_config.json" ]; then
    echo "✅ Best config file created"
    echo ""
    echo "Best configuration:"
    cat results/GRID_SEARCH/best_config.json | python3 -m json.tool | head -30
else
    echo "❌ Best config file NOT created"
fi

echo ""
echo "Experiments completed:"
ls -1 results/GRID_SEARCH/exp_* 2>/dev/null | wc -l

echo ""
echo "Check full results:"
echo "  cat results/GRID_SEARCH/all_results.json | python3 -m json.tool"
