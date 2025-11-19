#!/usr/bin/env python3
"""
Generate comprehensive comparison plots between TrackSSM and BotSort.
Creates per-class comparisons and overall averages for all metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Colors
TRACKSSM_COLOR = '#FF6B6B'  # Red
BOTSORT_COLOR = '#4ECDC4'   # Teal

# Class order (BotSort doesn't have trailer)
CLASS_ORDER_TRACKSSM = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']
CLASS_ORDER_BOTSORT = ['car', 'truck', 'bus', 'pedestrian', 'motorcycle', 'bicycle']
CLASS_LABELS = ['Car', 'Truck', 'Bus', 'Trailer', 'Pedestrian', 'Motorcycle', 'Bicycle']
CLASS_LABELS_NO_TRAILER = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Motorcycle', 'Bicycle']

def load_data():
    """Load all evaluation data."""
    
    # Load comparison data
    with open('results/final_evaluation/comparison_trackssm_vs_botsort.json', 'r') as f:
        comparison = json.load(f)
    
    # Load TrackSSM per-class metrics
    with open('results/final_evaluation/trackssm_7classes_per_class_metrics.json', 'r') as f:
        trackssm_per_class = json.load(f)
    
    return comparison, trackssm_per_class

def create_per_class_comparison(metric_name, trackssm_values, botsort_values, 
                                ylabel, title, filename, percentage=True, use_trailer=False):
    """Create a grouped bar chart comparing TrackSSM and BotSort per class."""
    
    labels = CLASS_LABELS if use_trailer else CLASS_LABELS_NO_TRAILER
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, trackssm_values, width, label='TrackSSM', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, botsort_values, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if percentage and height > 0:
                label = f'{height:.1f}%'
            elif height > 0:
                label = f'{int(height)}'
            else:
                label = '0'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Object Class', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0 for metrics that can be negative
    if min(min(trackssm_values), min(botsort_values)) < 0:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'results/final_evaluation/plots/{filename}', bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Created {filename}")

def create_overall_comparison(metric_names, trackssm_values, botsort_values,
                             ylabel, title, filename, percentage=True):
    """Create a grouped bar chart comparing overall averages."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, trackssm_values, width, label='TrackSSM', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, botsort_values, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if percentage and height > 0:
                label = f'{height:.1f}%'
            elif height > 0:
                label = f'{int(height)}'
            else:
                label = '0'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0 for metrics that can be negative
    if min(min(trackssm_values), min(botsort_values)) < 0:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'results/final_evaluation/plots/{filename}', bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Created {filename}")

def main():
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS: TrackSSM vs BotSort")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    comparison, trackssm_per_class = load_data()
    
    botsort_per_class = comparison['methods']['BotSort']['per_class_metrics']
    trackssm_overall = comparison['methods']['TrackSSM']['metrics']
    botsort_overall = comparison['methods']['BotSort']['metrics']
    
    # Create output directory
    Path('results/final_evaluation/plots').mkdir(parents=True, exist_ok=True)
    
    print("\n📈 Generating per-class comparison plots...")
    
    # ========== HOTA Metrics ==========
    print("\n1. HOTA Metrics")
    
    # HOTA (BotSort doesn't have trailer)
    botsort_hota = [botsort_per_class[cls]['HOTA'] for cls in CLASS_ORDER_BOTSORT]
    trackssm_hota = [trackssm_overall['HOTA']] * len(CLASS_ORDER_BOTSORT)  # Repeat overall value
    
    create_per_class_comparison(
        'HOTA', trackssm_hota, botsort_hota,
        'HOTA Score (%)', 
        'HOTA: Higher Order Tracking Accuracy\n(TrackSSM: overall value repeated, no per-class data)',
        '01_hota_per_class.png'
    )
    
    # DetA
    botsort_deta = [botsort_per_class[cls]['DetA'] for cls in CLASS_ORDER_BOTSORT]
    trackssm_deta = [trackssm_overall['DetA']] * len(CLASS_ORDER_BOTSORT)
    
    create_per_class_comparison(
        'DetA', trackssm_deta, botsort_deta,
        'Detection Accuracy (%)', 
        'DetA: Detection Accuracy Component of HOTA\n(TrackSSM: overall value repeated, no per-class data)',
        '02_deta_per_class.png'
    )
    
    # AssA
    botsort_assa = [botsort_per_class[cls]['AssA'] for cls in CLASS_ORDER_BOTSORT]
    trackssm_assa = [trackssm_overall['AssA']] * len(CLASS_ORDER_BOTSORT)
    
    create_per_class_comparison(
        'AssA', trackssm_assa, botsort_assa,
        'Association Accuracy (%)', 
        'AssA: Association Accuracy Component of HOTA\n(TrackSSM: overall value repeated, no per-class data)',
        '03_assa_per_class.png'
    )
    
    # ========== Traditional Metrics ==========
    print("\n2. Traditional Tracking Metrics")
    
    # MOTA
    trackssm_mota = [trackssm_per_class['per_class_metrics'][cls]['MOTA'] for cls in CLASS_ORDER_BOTSORT]
    botsort_mota = [botsort_per_class[cls]['MOTA'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'MOTA', trackssm_mota, botsort_mota,
        'MOTA (%)', 
        'MOTA: Multiple Object Tracking Accuracy',
        '04_mota_per_class.png'
    )
    
    # IDF1
    trackssm_idf1 = [trackssm_per_class['per_class_metrics'][cls]['IDF1'] for cls in CLASS_ORDER_BOTSORT]
    botsort_idf1 = [botsort_per_class[cls]['IDF1'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'IDF1', trackssm_idf1, botsort_idf1,
        'IDF1 Score (%)', 
        'IDF1: Identity F1 Score',
        '05_idf1_per_class.png'
    )
    
    # ========== Detection Quality ==========
    print("\n3. Detection Quality Metrics")
    
    # Precision
    trackssm_prec = [trackssm_per_class['per_class_metrics'][cls]['Precision'] for cls in CLASS_ORDER_BOTSORT]
    botsort_prec = [botsort_per_class[cls]['DetPr'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'Precision', trackssm_prec, botsort_prec,
        'Precision (%)', 
        'Detection Precision',
        '06_precision_per_class.png'
    )
    
    # Recall
    trackssm_recall = [trackssm_per_class['per_class_metrics'][cls]['Recall'] for cls in CLASS_ORDER_BOTSORT]
    botsort_recall = [botsort_per_class[cls]['DetRe'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'Recall', trackssm_recall, botsort_recall,
        'Recall (%)', 
        'Detection Recall',
        '07_recall_per_class.png'
    )
    
    # ========== Identity Metrics ==========
    print("\n4. Identity Switch Metrics")
    
    # ID Switches (log scale for better visualization)
    trackssm_idsw = [trackssm_per_class['per_class_metrics'][cls]['IDSW'] for cls in CLASS_ORDER_BOTSORT]
    botsort_idsw = [botsort_per_class[cls]['IDSW'] for cls in CLASS_ORDER_BOTSORT]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASS_LABELS_NO_TRAILER))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, trackssm_idsw, width, label='TrackSSM', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, botsort_idsw, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Object Class', fontweight='bold')
    ax.set_ylabel('ID Switches (count)', fontweight='bold')
    ax.set_title('ID Switches: Number of Identity Changes\n(Lower is Better)', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS_NO_TRAILER, rotation=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/final_evaluation/plots/08_idsw_per_class.png', bbox_inches='tight')
    plt.close()
    print(f"   ✓ Created 08_idsw_per_class.png")
    
    # ========== Track Quality ==========
    print("\n5. Track Quality Metrics")
    
    # Mostly Tracked (MT)
    trackssm_mt = [trackssm_per_class['per_class_metrics'][cls]['MT'] for cls in CLASS_ORDER_BOTSORT]
    botsort_mt = [botsort_per_class[cls]['MT'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'MT', trackssm_mt, botsort_mt,
        'Mostly Tracked (count)', 
        'MT: Objects Tracked for >80% of their Lifetime',
        '09_mt_per_class.png',
        percentage=False
    )
    
    # Mostly Lost (ML)
    trackssm_ml = [trackssm_per_class['per_class_metrics'][cls]['ML'] for cls in CLASS_ORDER_BOTSORT]
    botsort_ml = [botsort_per_class[cls]['ML'] for cls in CLASS_ORDER_BOTSORT]
    
    create_per_class_comparison(
        'ML', trackssm_ml, botsort_ml,
        'Mostly Lost (count)', 
        'ML: Objects Tracked for <20% of their Lifetime\n(Lower is Better)',
        '10_ml_per_class.png',
        percentage=False
    )
    
    # ========== OVERALL COMPARISONS ==========
    print("\n📊 Generating overall average comparison plots...")
    
    # HOTA Components
    create_overall_comparison(
        ['HOTA', 'DetA', 'AssA'],
        [trackssm_overall['HOTA'], trackssm_overall['DetA'], trackssm_overall['AssA']],
        [botsort_overall['HOTA'], botsort_overall['DetA'], botsort_overall['AssA']],
        'Score (%)',
        'Overall HOTA Metrics Comparison',
        '11_overall_hota_metrics.png'
    )
    
    # Traditional Tracking Metrics
    create_overall_comparison(
        ['MOTA', 'IDF1', 'MOTP'],
        [trackssm_overall['MOTA'], trackssm_overall['IDF1'], trackssm_overall['MOTP']*100],
        [botsort_overall['MOTA'], botsort_overall['IDF1'], 0],  # BotSort doesn't report MOTP
        'Score (%)',
        'Overall Traditional Tracking Metrics',
        '12_overall_tracking_metrics.png'
    )
    
    # Detection Quality
    create_overall_comparison(
        ['Precision', 'Recall'],
        [trackssm_overall['Precision'], trackssm_overall['Recall']],
        [botsort_overall['DetPr'], botsort_overall['DetRe']],
        'Score (%)',
        'Overall Detection Quality',
        '13_overall_detection_quality.png'
    )
    
    # Identity Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['ID Switches', 'MT', 'ML']
    trackssm_vals = [trackssm_overall['IDSW'], trackssm_overall['MT'], trackssm_overall['ML']]
    botsort_vals = [botsort_overall['IDSW'], botsort_overall['MT'], botsort_overall['ML']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, trackssm_vals, width, label='TrackSSM', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, botsort_vals, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Overall Identity Performance\n(MT/IDSW/ML)', fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/final_evaluation/plots/14_overall_identity_performance.png', bbox_inches='tight')
    plt.close()
    print(f"   ✓ Created 14_overall_identity_performance.png")
    
    # ========== SUMMARY RADAR CHART ==========
    print("\n6. Summary Visualization")
    
    # Normalize metrics to 0-100 scale for radar chart
    metrics = ['HOTA', 'MOTA', 'IDF1', 'Precision', 'Recall']
    trackssm_norm = [
        trackssm_overall['HOTA'],
        trackssm_overall['MOTA'],
        trackssm_overall['IDF1'],
        trackssm_overall['Precision'],
        trackssm_overall['Recall']
    ]
    botsort_norm = [
        botsort_overall['HOTA'],
        botsort_overall['MOTA'],
        botsort_overall['IDF1'],
        botsort_overall['DetPr'],
        botsort_overall['DetRe']
    ]
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    trackssm_norm += trackssm_norm[:1]
    botsort_norm += botsort_norm[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, trackssm_norm, 'o-', linewidth=2, label='TrackSSM', 
            color=TRACKSSM_COLOR, markersize=8)
    ax.fill(angles, trackssm_norm, alpha=0.25, color=TRACKSSM_COLOR)
    ax.plot(angles, botsort_norm, 'o-', linewidth=2, label='BotSort', 
            color=BOTSORT_COLOR, markersize=8)
    ax.fill(angles, botsort_norm, alpha=0.25, color=BOTSORT_COLOR)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Overall Performance Comparison\n(Radar Chart)', 
                 fontweight='bold', fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig('results/final_evaluation/plots/15_overall_radar_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"   ✓ Created 15_overall_radar_comparison.png")
    
    # ========== SUMMARY STATISTICS ==========
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n📊 Overall Comparison:")
    print(f"\n{'Metric':<20} {'TrackSSM':>12} {'BotSort':>12} {'Difference':>15}")
    print("-" * 60)
    print(f"{'HOTA':<20} {trackssm_overall['HOTA']:>11.2f}% {botsort_overall['HOTA']:>11.2f}% {trackssm_overall['HOTA']-botsort_overall['HOTA']:>14.2f}%")
    print(f"{'DetA':<20} {trackssm_overall['DetA']:>11.2f}% {botsort_overall['DetA']:>11.2f}% {trackssm_overall['DetA']-botsort_overall['DetA']:>14.2f}%")
    print(f"{'AssA':<20} {trackssm_overall['AssA']:>11.2f}% {botsort_overall['AssA']:>11.2f}% {trackssm_overall['AssA']-botsort_overall['AssA']:>14.2f}%")
    print(f"{'MOTA':<20} {trackssm_overall['MOTA']:>11.2f}% {botsort_overall['MOTA']:>11.2f}% {trackssm_overall['MOTA']-botsort_overall['MOTA']:>14.2f}%")
    print(f"{'IDF1':<20} {trackssm_overall['IDF1']:>11.2f}% {botsort_overall['IDF1']:>11.2f}% {trackssm_overall['IDF1']-botsort_overall['IDF1']:>14.2f}%")
    print(f"{'Precision':<20} {trackssm_overall['Precision']:>11.2f}% {botsort_overall['DetPr']:>11.2f}% {trackssm_overall['Precision']-botsort_overall['DetPr']:>14.2f}%")
    print(f"{'Recall':<20} {trackssm_overall['Recall']:>11.2f}% {botsort_overall['DetRe']:>11.2f}% {trackssm_overall['Recall']-botsort_overall['DetRe']:>14.2f}%")
    print(f"{'ID Switches':<20} {trackssm_overall['IDSW']:>12} {botsort_overall['IDSW']:>12} {trackssm_overall['IDSW']-botsort_overall['IDSW']:>15}")
    
    print("\n✅ All plots generated successfully!")
    print(f"   📁 Output directory: results/final_evaluation/plots/")
    print(f"   📊 Total plots: 15")
    print()

if __name__ == '__main__':
    main()
