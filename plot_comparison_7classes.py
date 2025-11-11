#!/usr/bin/env python3
"""
Generate comparison plots between TrackSSM and BotSort for all 7 NuScenes classes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Class names in order
CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']
CLASS_LABELS = ['Car', 'Truck', 'Bus', 'Trailer', 'Pedestrian', 'Motorcycle', 'Bicycle']

# Colors
TRACKSSM_COLOR = '#2E86AB'  # Blue
BOTSORT_COLOR = '#A23B72'   # Purple/Pink

def load_metrics(trackssm_file, botsort_file):
    """Load metrics from JSON files."""
    with open(trackssm_file, 'r') as f:
        trackssm_data = json.load(f)
    
    with open(botsort_file, 'r') as f:
        botsort_data = json.load(f)
    
    return trackssm_data, botsort_data

def extract_metric_arrays(trackssm_data, botsort_data, metric_name):
    """Extract metric values for all classes."""
    trackssm_vals = []
    botsort_vals = []
    
    for cls_name in CLASS_NAMES:
        # TrackSSM
        if cls_name in trackssm_data['per_class_metrics']:
            trackssm_vals.append(trackssm_data['per_class_metrics'][cls_name].get(metric_name, 0))
        else:
            trackssm_vals.append(0)
        
        # BotSort
        if cls_name in botsort_data:
            botsort_vals.append(botsort_data[cls_name].get(metric_name, 0))
        else:
            botsort_vals.append(0)
    
    return np.array(trackssm_vals), np.array(botsort_vals)

def plot_grouped_bars(trackssm_vals, botsort_vals, metric_name, ylabel, title, output_path, 
                      percentage=True, higher_is_better=True):
    """Create grouped bar chart comparing two methods."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(CLASS_LABELS))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, trackssm_vals, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, botsort_vals, width, label='BotSort (GT Detections)', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_xlabel('Object Class', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars, vals):
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if percentage:
                label = f'{val:.1f}%'
            elif val >= 1000:
                label = f'{int(val):,}'
            else:
                label = f'{int(val)}'
            
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(bars1, trackssm_vals)
    autolabel(bars2, botsort_vals)
    
    # Set y-axis limits with some padding
    if percentage:
        ax.set_ylim(0, max(max(trackssm_vals), max(botsort_vals)) * 1.15)
    else:
        ax.set_ylim(0, max(max(trackssm_vals), max(botsort_vals)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_object_counts(trackssm_data, botsort_data, output_path):
    """Plot object counts per class."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    trackssm_counts = []
    botsort_counts = []
    
    for cls_name in CLASS_NAMES:
        if cls_name in trackssm_data['per_class_metrics']:
            trackssm_counts.append(trackssm_data['per_class_metrics'][cls_name].get('num_objects', 0))
        else:
            trackssm_counts.append(0)
        
        if cls_name in botsort_data:
            botsort_counts.append(botsort_data[cls_name].get('GT_IDs', 0))
        else:
            botsort_counts.append(0)
    
    x = np.arange(len(CLASS_LABELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, trackssm_counts, width, label='TrackSSM Dataset', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, botsort_counts, width, label='BotSort Dataset', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Object Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Ground Truth Tracks', fontsize=13, fontweight='bold')
    ax.set_title('Dataset Size Comparison: Ground Truth Tracks per Class', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars, counts in [(bars1, trackssm_counts), (bars2, botsort_counts)]:
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{int(count):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_combined_metrics(trackssm_data, botsort_data, output_path):
    """Create a 2x2 subplot with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TrackSSM vs BotSort: Comprehensive Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    metrics = [
        ('MOTA', 'MOTA (%)', True, True),
        ('IDF1', 'IDF1 (%)', True, True),
        ('IDSW', 'ID Switches', False, False),
        ('Precision', 'Precision (%)', True, True)
    ]
    
    for idx, (metric, ylabel, percentage, higher_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        trackssm_vals, botsort_vals = extract_metric_arrays(trackssm_data, botsort_data, metric)
        
        x = np.arange(len(CLASS_LABELS))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trackssm_vals, width, label='TrackSSM', 
                       color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, botsort_vals, width, label='BotSort', 
                       color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, fontsize=10, rotation=0)
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars, vals in [(bars1, trackssm_vals), (bars2, botsort_vals)]:
            for bar, val in zip(bars, vals):
                height = bar.get_height()
                if percentage:
                    label = f'{val:.1f}'
                else:
                    label = f'{int(val)}'
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_summary_table(trackssm_data, botsort_data, output_path):
    """Create a summary table image."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = [['Class', 'Metric', 'TrackSSM', 'BotSort', 'Δ (BotSort - TrackSSM)']]
    
    metrics_to_show = ['MOTA', 'IDF1', 'IDSW', 'Precision', 'Recall']
    
    for cls_name, cls_label in zip(CLASS_NAMES, CLASS_LABELS):
        # Add class header
        table_data.append([cls_label, '', '', '', ''])
        
        for metric in metrics_to_show:
            trackssm_val = trackssm_data['per_class_metrics'].get(cls_name, {}).get(metric, 0)
            botsort_val = botsort_data.get(cls_name, {}).get(metric, 0)
            
            if metric == 'IDSW':
                # Lower is better
                delta = botsort_val - trackssm_val
                delta_str = f'{int(delta):+,}'
            elif metric in ['MOTA', 'IDF1', 'Precision', 'Recall']:
                # Percentage metrics
                delta = botsort_val - trackssm_val
                delta_str = f'{delta:+.1f}%'
                trackssm_val = f'{trackssm_val:.1f}%'
                botsort_val = f'{botsort_val:.1f}%'
            else:
                delta = botsort_val - trackssm_val
                delta_str = f'{delta:+.1f}'
                trackssm_val = f'{trackssm_val:.1f}'
                botsort_val = f'{botsort_val:.1f}'
            
            table_data.append(['', f'  {metric}', str(trackssm_val), str(botsort_val), delta_str])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.15, 0.15, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4A4A4A')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Style class headers
    row_idx = 1
    for cls_name in CLASS_NAMES:
        cell = table[(row_idx, 0)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold', fontsize=10)
        row_idx += 6  # 5 metrics + 1 class header
    
    plt.title('TrackSSM vs BotSort: Detailed Metrics Comparison (All Classes)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    # Paths
    trackssm_file = './results/final_evaluation/trackssm_7classes_per_class_metrics.json'
    botsort_file = '../results/results_botsort/baseline_gt_7classes/raw_with_classes/metrics_per_class.json'
    output_dir = Path('./results/final_evaluation/comparison_plots_7classes')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING TRACKSSM VS BOTSORT COMPARISON PLOTS (7 CLASSES)")
    print("="*80 + "\n")
    
    # Load data
    print("📂 Loading metrics...")
    trackssm_data, botsort_data = load_metrics(trackssm_file, botsort_file)
    print(f"   ✓ TrackSSM: {len(trackssm_data['per_class_metrics'])} classes")
    print(f"   ✓ BotSort: {len(botsort_data)} classes\n")
    
    # Generate plots
    print("📊 Generating comparison plots...\n")
    
    # 1. MOTA comparison
    trackssm_mota, botsort_mota = extract_metric_arrays(trackssm_data, botsort_data, 'MOTA')
    plot_grouped_bars(trackssm_mota, botsort_mota, 'MOTA', 'MOTA (%)', 
                     'MOTA Comparison by Class: TrackSSM vs BotSort',
                     output_dir / 'mota_comparison.png', percentage=True, higher_is_better=True)
    
    # 2. IDF1 comparison
    trackssm_idf1, botsort_idf1 = extract_metric_arrays(trackssm_data, botsort_data, 'IDF1')
    plot_grouped_bars(trackssm_idf1, botsort_idf1, 'IDF1', 'IDF1 (%)', 
                     'IDF1 Comparison by Class: TrackSSM vs BotSort',
                     output_dir / 'idf1_comparison.png', percentage=True, higher_is_better=True)
    
    # 3. ID Switches comparison
    trackssm_idsw, botsort_idsw = extract_metric_arrays(trackssm_data, botsort_data, 'IDSW')
    plot_grouped_bars(trackssm_idsw, botsort_idsw, 'IDSW', 'Number of ID Switches', 
                     'ID Switches by Class: TrackSSM vs BotSort (Lower is Better)',
                     output_dir / 'idsw_comparison.png', percentage=False, higher_is_better=False)
    
    # 4. Precision comparison
    trackssm_prec, botsort_prec = extract_metric_arrays(trackssm_data, botsort_data, 'Precision')
    # BotSort uses DetPr instead of Precision
    botsort_prec = []
    for cls_name in CLASS_NAMES:
        if cls_name in botsort_data:
            botsort_prec.append(botsort_data[cls_name].get('DetPr', 0))
        else:
            botsort_prec.append(0)
    botsort_prec = np.array(botsort_prec)
    
    plot_grouped_bars(trackssm_prec, botsort_prec, 'Precision', 'Precision (%)', 
                     'Precision Comparison by Class: TrackSSM vs BotSort',
                     output_dir / 'precision_comparison.png', percentage=True, higher_is_better=True)
    
    # 5. Recall comparison
    trackssm_recall, botsort_recall = extract_metric_arrays(trackssm_data, botsort_data, 'Recall')
    # BotSort uses DetRe instead of Recall
    botsort_recall = []
    for cls_name in CLASS_NAMES:
        if cls_name in botsort_data:
            botsort_recall.append(botsort_data[cls_name].get('DetRe', 0))
        else:
            botsort_recall.append(0)
    botsort_recall = np.array(botsort_recall)
    
    plot_grouped_bars(trackssm_recall, botsort_recall, 'Recall', 'Recall (%)', 
                     'Recall Comparison by Class: TrackSSM vs BotSort',
                     output_dir / 'recall_comparison.png', percentage=True, higher_is_better=True)
    
    # 6. Object counts
    print()
    plot_object_counts(trackssm_data, botsort_data, output_dir / 'object_counts.png')
    
    # 7. Combined metrics (2x2 grid)
    print()
    plot_combined_metrics(trackssm_data, botsort_data, output_dir / 'combined_metrics.png')
    
    # 8. Summary table
    print()
    create_summary_table(trackssm_data, botsort_data, output_dir / 'summary_table.png')
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("="*80 + "\n")
    
    # Print summary statistics
    print("📈 SUMMARY STATISTICS:\n")
    print(f"{'Class':<12} {'TrackSSM MOTA':<15} {'BotSort MOTA':<15} {'Δ MOTA':<10}")
    print("-" * 60)
    for cls_name, cls_label in zip(CLASS_NAMES, CLASS_LABELS):
        ts_mota = trackssm_data['per_class_metrics'][cls_name]['MOTA']
        bs_mota = botsort_data[cls_name]['MOTA']
        delta = bs_mota - ts_mota
        print(f"{cls_label:<12} {ts_mota:>6.2f}%        {bs_mota:>6.2f}%        {delta:>+6.2f}%")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
