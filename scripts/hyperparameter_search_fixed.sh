#!/bin/bash
#
# Hyperparameter Grid Search for TrackSSM (Fixed Parameter Mapping)
# 
# After discovering parameter mapping bug, we re-run grid search with:
# - match_thresh: {0.6, 0.7, 0.8}
# - max_age: {30, 45, 60}
# - track_thresh: {0.6, 0.7}
#
# Total: 3 × 3 × 2 = 18 experiments
# Estimated time: ~18 hours on CPU (1h per experiment)
#

set -e

BASE_DIR="/user/amarino/tesi_project_amarino"
OUTPUT_BASE="${BASE_DIR}/results/HYPERPARAM_SEARCH_FIXED"
LOG_DIR="${BASE_DIR}/logs/hyperparam_search"

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

echo "=================================================="
echo "HYPERPARAMETER GRID SEARCH (FIXED PARAMETERS)"
echo "=================================================="
echo "Output: $OUTPUT_BASE"
echo "Logs: $LOG_DIR"
echo ""

# Grid parameters
MATCH_THRESHS=(0.6 0.7 0.8)
MAX_AGES=(30 45 60)
TRACK_THRESHS=(0.6 0.7)

TOTAL=$((${#MATCH_THRESHS[@]} * ${#MAX_AGES[@]} * ${#TRACK_THRESHS[@]}))
COUNTER=0

echo "Total experiments: $TOTAL"
echo "Starting at: $(date)"
echo ""

for match_thresh in "${MATCH_THRESHS[@]}"; do
    for max_age in "${MAX_AGES[@]}"; do
        for track_thresh in "${TRACK_THRESHS[@]}"; do
            COUNTER=$((COUNTER + 1))
            
            EXP_NAME="match${match_thresh}_age${max_age}_track${track_thresh}"
            OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
            LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
            METRICS_FILE="${OUTPUT_DIR}/metrics.json"
            
            echo "[$COUNTER/$TOTAL] ${EXP_NAME}"
            echo "  match_thresh=$match_thresh, max_age=$max_age, track_thresh=$track_thresh"
            
            # Skip if already completed
            if [ -f "$METRICS_FILE" ]; then
                IDSW=$(grep -o '"IDSW": [0-9]*' "$METRICS_FILE" | awk '{print $2}')
                echo "  ✓ Already completed (IDSW: $IDSW)"
                continue
            fi
            
            # Run experiment
            python track.py \
                --tracker trackssm \
                --data data/nuscenes_mot_front/val \
                --output "$OUTPUT_DIR" \
                --match-thresh "$match_thresh" \
                --max-age "$max_age" \
                --track-thresh "$track_thresh" \
                --conf-thresh 0.7 \
                --nms-thresh 0.65 \
                > "$LOG_FILE" 2>&1
            
            # Evaluate
            python evaluate_motmetrics.py \
                --gt-folder data/nuscenes_mot_front/val \
                --pred-folder "${OUTPUT_DIR}/data" \
                --output "$METRICS_FILE" \
                >> "$LOG_FILE" 2>&1
            
            # Extract and display IDSW
            if [ -f "$METRICS_FILE" ]; then
                IDSW=$(grep -o '"IDSW": [0-9]*' "$METRICS_FILE" | awk '{print $2}')
                MOTA=$(grep -o '"MOTA": [0-9.]*' "$METRICS_FILE" | awk '{print $2}')
                echo "  ✓ Completed - IDSW: $IDSW, MOTA: $MOTA%"
            else
                echo "  ✗ Failed - check $LOG_FILE"
            fi
            
            echo ""
        done
    done
done

echo "=================================================="
echo "GRID SEARCH COMPLETE"
echo "Finished at: $(date)"
echo "=================================================="

# Generate summary
echo ""
echo "📊 SUMMARY (sorted by IDSW):"
echo "---------------------------------------------------"
echo "Config                                    IDSW    MOTA"
echo "---------------------------------------------------"

for result in "${OUTPUT_BASE}"/*/metrics.json; do
    if [ -f "$result" ]; then
        config=$(basename $(dirname "$result"))
        idsw=$(grep -o '"IDSW": [0-9]*' "$result" | awk '{print $2}')
        mota=$(grep -o '"MOTA": [0-9.]*' "$result" | awk '{print $2}')
        printf "%-40s %6s  %6s%%\n" "$config" "$idsw" "$mota"
    fi
done | sort -k2 -n

echo "---------------------------------------------------"
echo ""
echo "✓ Results saved to: $OUTPUT_BASE"
echo "✓ Best config: $(ls -t ${OUTPUT_BASE}/*/metrics.json | head -1 | xargs -I{} bash -c 'grep -o '\"IDSW\": [0-9]*' {} | awk '\''{print $2}'\'' | xargs -I[] bash -c '\''dirname $(find ${OUTPUT_BASE} -name metrics.json -exec grep -l '\"IDSW\": []' {} \\;)'\'' | xargs basename')"
