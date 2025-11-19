#!/bin/bash
# Show the organized plot structure

echo ""
echo "================================================================================"
echo "                    FINAL EVALUATION PLOTS STRUCTURE"
echo "================================================================================"
echo ""
echo "📊 Location: results/final_evaluation/plots/"
echo ""
echo "📁 Total: 17 high-resolution plots (200 DPI) organized in 6 categories"
echo ""

# Function to list files in a directory
show_category() {
    local cat_num=$1
    local cat_name=$2
    local cat_dir=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $cat_num $cat_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -d "results/final_evaluation/plots/$cat_dir" ]; then
        for file in results/final_evaluation/plots/$cat_dir/*.png; do
            if [ -f "$file" ]; then
                local size=$(du -h "$file" | cut -f1)
                local filename=$(basename "$file")
                printf "    ✓ %-40s %s\n" "$filename" "($size)"
            fi
        done
        local total_size=$(du -sh "results/final_evaluation/plots/$cat_dir" | cut -f1)
        echo "    └─ Subtotal: $total_size"
    else
        echo "    ⚠ Directory not found!"
    fi
    echo ""
}

# Show each category
show_category "1️⃣" "TRACKING ACCURACY" "01_tracking_accuracy"
show_category "2️⃣" "IDENTITY METRICS" "02_identity_metrics"
show_category "3️⃣" "DETECTION QUALITY" "03_detection_quality"
show_category "4️⃣" "ERROR ANALYSIS" "04_error_analysis"
show_category "5️⃣" "HOTA ANALYSIS (BotSort)" "05_hota_analysis"
show_category "6️⃣" "SUMMARY VIEWS" "06_summary_views"

echo "================================================================================"

# Total size
if [ -d "results/final_evaluation/plots" ]; then
    total=$(du -sh results/final_evaluation/plots | cut -f1)
    echo "📊 TOTAL SIZE: $total"
else
    echo "⚠ Plots directory not found!"
fi

echo ""
echo "📖 For detailed documentation, see: results/final_evaluation/plots/README.md"
echo ""
echo "🎯 To regenerate plots: python scripts/generate_final_plots.py"
echo ""
echo "================================================================================"
echo ""
