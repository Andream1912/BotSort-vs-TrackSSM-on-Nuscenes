#!/bin/bash

# Start parallel grid search for TrackSSM hyperparameters
# Usage: ./start_grid_search.sh [detector_epoch] [num_workers]

DETECTOR_EPOCH=${1:-30}
NUM_WORKERS=${2:-4}

DETECTOR_WEIGHT="yolox_finetuning/yolox_l_nuscenes_stable/epoch_${DETECTOR_EPOCH}.pth"

echo "=========================================="
echo "STARTING PARALLEL GRID SEARCH"
echo "=========================================="
echo "Detector: epoch_${DETECTOR_EPOCH}.pth"
echo "Parallel workers: ${NUM_WORKERS}"
echo "Output: results/GRID_SEARCH/"
echo "=========================================="
echo ""

# Check detector exists
if [ ! -f "$DETECTOR_WEIGHT" ]; then
    echo "❌ ERROR: Detector not found: $DETECTOR_WEIGHT"
    echo "Available detectors:"
    ls -lh yolox_finetuning/yolox_l_nuscenes_stable/*.pth
    exit 1
fi

# Create output directory
mkdir -p results/GRID_SEARCH

# Make scripts executable
chmod +x grid_search_parallel.py
chmod +x monitor_grid_search.sh

# Run grid search in background
nohup python3 scripts/grid_search/grid_search_parallel.py \
    --detector "$DETECTOR_WEIGHT" \
    --workers $NUM_WORKERS \
    > results/GRID_SEARCH/grid_search.log 2>&1 &

GRID_PID=$!

echo "✓ Grid search started (PID: $GRID_PID)"
echo ""
echo "📊 Monitor progress with:"
echo "   ./monitor_grid_search.sh"
echo ""
echo "📄 View logs with:"
echo "   tail -f results/GRID_SEARCH/grid_search.log"
echo ""
echo "🏆 Check best config anytime:"
echo "   cat results/GRID_SEARCH/best_config.json | python3 -m json.tool"
echo ""
echo "🛑 Stop grid search with:"
echo "   kill $GRID_PID"
echo ""
echo "Grid search running in background..."
echo "Best config updates automatically in: results/GRID_SEARCH_PARALLEL/best_config.json"
