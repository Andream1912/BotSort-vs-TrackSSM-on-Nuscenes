#!/usr/bin/env python3
"""
Generate systematic and organized comparison plots between TrackSSM and BotSort
Structure:
  - Individual plots for each metric (single focus, clear comparison)
  - HOTA analysis with detailed breakdown
  - Professional styling and high readability
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Color scheme
TRACKSSM_COLOR = '#1f77b4'  # Blue
BOTSORT_COLOR = '#ff7f0e'   # Orange
POSITIVE_COLOR = '#2ca02c'  # Green
NEGATIVE_COLOR = '#d62728'  # Red
COLORS_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def load_data():
    """Load metrics from JSON files"""
    results_dir = Path('results/final_evaluation')
    
    with open(results_dir / 'trackssm_7classes_per_class_metrics.json', 'r') as f:
        trackssm_data = json.load(f)
    
    with open(results_dir / 'comparison_trackssm_vs_botsort.json', 'r') as f:
        comparison_data = json.load(f)
    
    return trackssm_data, comparison_data

def get_class_order():
    """Define class order for consistent plotting"""
    return ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']

def format_class_name(class_name):
    """Format class name for display"""
    return class_name.capitalize()

def create_output_structure():
    """Create organized output directory structure"""
    base_dir = Path('results/final_evaluation')
    
    dirs = {
        'main': base_dir / 'plots',
        'tracking': base_dir / 'plots/01_tracking_accuracy',
        'identity': base_dir / 'plots/02_identity_metrics',
        'detection': base_dir / 'plots/03_detection_quality',
        'errors': base_dir / 'plots/04_error_analysis',
        'hota': base_dir / 'plots/05_hota_analysis',
        'summary': base_dir / 'plots/06_summary_views'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

# ============================================================================
# 01. TRACKING ACCURACY
# ============================================================================

def plot_mota_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for MOTA comparison"""
    classes = get_class_order()
    
    trackssm_mota = [trackssm_data['per_class_metrics'][cls]['MOTA'] for cls in classes]
    botsort_mota = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('MOTA', 0) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_mota, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_mota, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotations
    for i, (t_val, b_val) in enumerate(zip(trackssm_mota, botsort_mota)):
        if b_val != 0:
            diff = t_val - b_val
            color = POSITIVE_COLOR if diff > 0 else NEGATIVE_COLOR
            ax.text(i, max(t_val, b_val) + 4, f'{diff:+.1f}%', 
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    ax.set_ylabel('MOTA (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Multiple Object Tracking Accuracy (MOTA) - Per Class Comparison\nHigher is Better', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
    ax.set_ylim(min(trackssm_mota + botsort_mota) - 10, max(trackssm_mota + botsort_mota) + 10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mota_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ MOTA comparison")

def plot_motp_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for MOTP comparison"""
    classes = get_class_order()
    
    trackssm_motp = [trackssm_data['per_class_metrics'][cls]['MOTP'] * 100 for cls in classes]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = ax.bar(x, trackssm_motp, width, label='TrackSSM', 
                  color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('MOTP (Precision %)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Multiple Object Tracking Precision (MOTP) - TrackSSM\nLower is Better (Average localization error)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motp_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ MOTP comparison")

# ============================================================================
# 02. IDENTITY METRICS
# ============================================================================

def plot_idf1_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for IDF1 comparison"""
    classes = get_class_order()
    
    trackssm_idf1 = [trackssm_data['per_class_metrics'][cls]['IDF1'] for cls in classes]
    botsort_idf1 = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDF1', 0) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_idf1, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_idf1, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotations
    for i, (t_val, b_val) in enumerate(zip(trackssm_idf1, botsort_idf1)):
        if b_val != 0:
            diff = t_val - b_val
            color = POSITIVE_COLOR if diff > 0 else NEGATIVE_COLOR
            ax.text(i, max(t_val, b_val) + 4, f'{diff:+.1f}%', 
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    ax.set_ylabel('IDF1 Score (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('ID F1 Score (IDF1) - Per Class Comparison\nHigher is Better (Identity Consistency)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.set_ylim(0, max(trackssm_idf1 + botsort_idf1) + 12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'idf1_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ IDF1 comparison")

def plot_idsw_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for ID Switches comparison"""
    classes = get_class_order()
    
    trackssm_idsw = [trackssm_data['per_class_metrics'][cls]['IDSW'] for cls in classes]
    botsort_idsw = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDSW', 1) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_idsw, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_idsw, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add ratio annotations
    for i, (t_val, b_val) in enumerate(zip(trackssm_idsw, botsort_idsw)):
        if b_val > 0:
            ratio = t_val / b_val
            ax.text(i, max(t_val, b_val) * 1.4, f'{ratio:.1f}×', 
                   ha='center', va='bottom', fontsize=10, color=NEGATIVE_COLOR, fontweight='bold')
    
    ax.set_ylabel('ID Switches (count)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Identity Switches - Per Class Comparison (Log Scale)\nLower is Better', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--', which='both')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'idsw_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ ID Switches comparison")

def plot_fragmentation(trackssm_data, comparison_data, output_dir):
    """Individual plot for Fragmentation"""
    classes = get_class_order()
    
    trackssm_frag = [trackssm_data['per_class_metrics'][cls]['Frag'] for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = ax.bar(x, trackssm_frag, width, label='TrackSSM', 
                  color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
               f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Fragmentations (count)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Track Fragmentations - TrackSSM\nLower is Better (Track interruptions)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fragmentation.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Fragmentation analysis")

# ============================================================================
# 03. DETECTION QUALITY
# ============================================================================

def plot_precision_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for Precision comparison"""
    classes = get_class_order()
    
    trackssm_precision = [trackssm_data['per_class_metrics'][cls]['Precision'] for cls in classes]
    botsort_precision = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetPr', 0) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_precision, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_precision, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotations
    for i, (t_val, b_val) in enumerate(zip(trackssm_precision, botsort_precision)):
        if b_val != 0:
            diff = t_val - b_val
            color = POSITIVE_COLOR if diff > 0 else NEGATIVE_COLOR
            ax.text(i, max(t_val, b_val) + 3, f'{diff:+.1f}%', 
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    ax.set_ylabel('Precision (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Detection Precision - Per Class Comparison\nHigher is Better (Few False Positives)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.set_ylim(0, 108)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Precision comparison")

def plot_recall_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for Recall comparison"""
    classes = get_class_order()
    
    trackssm_recall = [trackssm_data['per_class_metrics'][cls]['Recall'] for cls in classes]
    botsort_recall = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetRe', 0) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_recall, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_recall, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotations
    for i, (t_val, b_val) in enumerate(zip(trackssm_recall, botsort_recall)):
        if b_val != 0:
            diff = t_val - b_val
            color = POSITIVE_COLOR if diff > 0 else NEGATIVE_COLOR
            ax.text(i, max(t_val, b_val) + 3, f'{diff:+.1f}%', 
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    ax.set_ylabel('Recall (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Detection Recall - Per Class Comparison\nHigher is Better (Few False Negatives)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.set_ylim(0, 108)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Recall comparison")

# ============================================================================
# 04. ERROR ANALYSIS
# ============================================================================

def plot_false_positives(trackssm_data, comparison_data, output_dir):
    """Individual plot for False Positives"""
    classes = get_class_order()
    
    trackssm_fp = [trackssm_data['per_class_metrics'][cls]['FP'] for cls in classes]
    botsort_fp = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDFP', 1) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_fp, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_fp, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('False Positives (count)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('False Positives - Per Class Comparison (Log Scale)\nLower is Better', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--', which='both')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'false_positives.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ False Positives analysis")

def plot_false_negatives(trackssm_data, comparison_data, output_dir):
    """Individual plot for False Negatives"""
    classes = get_class_order()
    
    trackssm_fn = [trackssm_data['per_class_metrics'][cls]['FN'] for cls in classes]
    botsort_fn = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDFN', 1) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_fn, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_fn, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('False Negatives (count)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('False Negatives - Per Class Comparison (Log Scale)\nLower is Better', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--', which='both')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'false_negatives.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ False Negatives analysis")

def plot_mt_ml_comparison(trackssm_data, comparison_data, output_dir):
    """Individual plot for MT/ML comparison"""
    classes = get_class_order()
    
    trackssm_mt = [trackssm_data['per_class_metrics'][cls]['MT'] for cls in classes]
    trackssm_ml = [trackssm_data['per_class_metrics'][cls]['ML'] for cls in classes]
    botsort_mt = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('MT', 0) for cls in classes]
    botsort_ml = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('ML', 0) for cls in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    # MT (Mostly Tracked)
    bars1 = ax1.bar(x - width/2, trackssm_mt, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, botsort_mt, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Mostly Tracked (count)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Object Class', fontweight='bold', fontsize=13)
    ax1.set_title('Mostly Tracked (MT) - Higher is Better', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=11)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=11, edgecolor='black')
    ax1.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    # ML (Mostly Lost)
    bars3 = ax2.bar(x - width/2, trackssm_ml, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, botsort_ml, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Mostly Lost (count)', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Object Class', fontweight='bold', fontsize=13)
    ax2.set_title('Mostly Lost (ML) - Lower is Better', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=11)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=11, edgecolor='black')
    ax2.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mt_ml_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ MT/ML comparison")

# ============================================================================
# 05. HOTA ANALYSIS
# ============================================================================

def plot_hota_single(trackssm_data, comparison_data, output_dir):
    """Individual plot for HOTA (BotSort only)"""
    classes = get_class_order()
    
    botsort_hota = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('HOTA', 0) for cls in classes]
    overall_hota = comparison_data['methods']['BotSort']['metrics']['HOTA']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = ax.bar(x, botsort_hota, width, color=BOTSORT_COLOR, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, label='BotSort')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add overall HOTA line
    ax.axhline(y=overall_hota, color=NEGATIVE_COLOR, linestyle='--', linewidth=2.5, 
               label=f'Overall HOTA: {overall_hota:.2f}%', alpha=0.8)
    
    ax.set_ylabel('HOTA Score (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('Higher Order Tracking Accuracy (HOTA) - BotSort\nBalances Detection and Association Performance', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.set_ylim(0, max(botsort_hota) * 1.15)
    
    # Add note
    ax.text(0.02, 0.98, 'Note: TrackSSM HOTA not available\n(requires different evaluation protocol)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hota_botsort.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ HOTA analysis")

def plot_hota_components(trackssm_data, comparison_data, output_dir):
    """Plot HOTA components: DetA and AssA"""
    classes = get_class_order()
    
    deta = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetA', 0) for cls in classes]
    assa = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('AssA', 0) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, deta, width, label='DetA (Detection Accuracy)', 
                   color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, assa, width, label='AssA (Association Accuracy)', 
                   color='#9467bd', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title('HOTA Components - BotSort\nDetection Accuracy (DetA) vs Association Accuracy (AssA)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    # Add explanation
    ax.text(0.02, 0.02, 'HOTA = √(DetA × AssA)\nDetA: How well detections match ground truth\nAssA: How well associations are maintained over time', 
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hota_components.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ HOTA components (DetA vs AssA)")

def plot_hota_precision_recall(trackssm_data, comparison_data, output_dir):
    """Plot HOTA precision and recall components"""
    classes = get_class_order()
    
    det_pr = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetPr', 0) for cls in classes]
    det_re = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetRe', 0) for cls in classes]
    ass_pr = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('AssPr', 0) for cls in classes]
    ass_re = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('AssRe', 0) for cls in classes]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    x = np.arange(len(classes))
    width = 0.6
    
    # Detection Precision
    bars1 = ax1.bar(x, det_pr, width, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.5)
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Detection Precision (DetPr)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Precision (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_class_name(c)[:3].upper() for c in classes], fontsize=10)
    ax1.grid(True, alpha=0.4, axis='y')
    ax1.set_ylim(0, 100)
    
    # Detection Recall
    bars2 = ax2.bar(x, det_re, width, color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.5)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_title('Detection Recall (DetRe)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Recall (%)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_class_name(c)[:3].upper() for c in classes], fontsize=10)
    ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_ylim(0, 100)
    
    # Association Precision
    bars3 = ax3.bar(x, ass_pr, width, color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1.5)
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_title('Association Precision (AssPr)', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Precision (%)', fontweight='bold')
    ax3.set_xlabel('Object Class', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([format_class_name(c)[:3].upper() for c in classes], fontsize=10)
    ax3.grid(True, alpha=0.4, axis='y')
    ax3.set_ylim(0, 100)
    
    # Association Recall
    bars4 = ax4.bar(x, ass_re, width, color='#d62728', alpha=0.85, edgecolor='black', linewidth=1.5)
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_title('Association Recall (AssRe)', fontweight='bold', fontsize=13)
    ax4.set_ylabel('Recall (%)', fontweight='bold')
    ax4.set_xlabel('Object Class', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([format_class_name(c)[:3].upper() for c in classes], fontsize=10)
    ax4.grid(True, alpha=0.4, axis='y')
    ax4.set_ylim(0, 100)
    
    fig.suptitle('HOTA Precision & Recall Components - BotSort', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hota_precision_recall.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ HOTA precision/recall components")

# ============================================================================
# 06. SUMMARY VIEWS
# ============================================================================

def plot_overall_comparison(trackssm_data, comparison_data, output_dir):
    """Overall metrics comparison"""
    trackssm_overall = comparison_data['methods']['TrackSSM']['metrics']
    botsort_overall = comparison_data['methods']['BotSort']['metrics']
    
    metrics = ['MOTA', 'IDF1', 'Precision', 'Recall']
    trackssm_values = [
        trackssm_overall['MOTA'],
        trackssm_overall['IDF1'],
        trackssm_overall['Precision'],
        trackssm_overall['Recall']
    ]
    botsort_values = [
        botsort_overall['MOTA'],
        botsort_overall['IDF1'],
        botsort_overall['DetPr'],
        botsort_overall['DetRe']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metrics))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, trackssm_values, width, label='TrackSSM (Zero-Shot)', 
                   color=TRACKSSM_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, botsort_values, width, label='BotSort (Optimized)', 
                   color=BOTSORT_COLOR, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=14)
    ax.set_title('Overall Performance Metrics Comparison\nAveraged Across All Classes', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, edgecolor='black')
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Overall comparison")

def plot_class_distribution(trackssm_data, comparison_data, output_dir):
    """Plot object class distribution"""
    classes = get_class_order()
    
    counts = [trackssm_data['per_class_metrics'][cls]['num_objects'] for cls in classes]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = ax.bar(x, counts, width, color=COLORS_PALETTE[:len(classes)], 
                  alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels and percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 200,
               f'{int(count)}\n({percentage:.1f}%)', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Objects', fontweight='bold', fontsize=14)
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=14)
    ax.set_title(f'Object Class Distribution in NuScenes Dataset\nTotal: {total:,} objects across 7 classes', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right', fontsize=12)
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Class distribution")

def create_summary_table(trackssm_data, comparison_data, output_dir):
    """Create comprehensive summary table"""
    classes = get_class_order()
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Class', 'MOTA\nT / B', 'IDF1\nT / B', 'IDSW\nT / B', 'Precision\nT / B', 'Recall\nT / B', 'FP\nT / B', 'FN\nT / B', 'Objects']
    
    table_data = []
    for cls in classes:
        t_metrics = trackssm_data['per_class_metrics'][cls]
        b_metrics = comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {})
        
        row = [
            format_class_name(cls),
            f"{t_metrics['MOTA']:.1f}\n{b_metrics.get('MOTA', 0):.1f}",
            f"{t_metrics['IDF1']:.1f}\n{b_metrics.get('IDF1', 0):.1f}",
            f"{t_metrics['IDSW']}\n{b_metrics.get('IDSW', 0)}",
            f"{t_metrics['Precision']:.1f}\n{b_metrics.get('DetPr', 0):.1f}",
            f"{t_metrics['Recall']:.1f}\n{b_metrics.get('DetRe', 0):.1f}",
            f"{t_metrics['FP']}\n{b_metrics.get('IDFP', 0)}",
            f"{t_metrics['FN']}\n{b_metrics.get('IDFN', 0)}",
            f"{t_metrics['num_objects']}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.09, 0.09, 0.09, 0.11, 0.10, 0.09, 0.09, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
    
    plt.title('Comprehensive Metrics Comparison - TrackSSM (T) vs BotSort (B)\nPer-Class Performance Summary', 
              fontsize=18, fontweight='bold', pad=30)
    
    # Add legend
    legend_text = 'T = TrackSSM (Zero-Shot, MOT17 weights)   |   B = BotSort (Optimized for NuScenes)'
    plt.text(0.5, 0.02, legend_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), transform=fig.transFigure)
    
    plt.savefig(output_dir / 'summary_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Summary table")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" "*20 + "SYSTEMATIC PLOT GENERATION")
    print("="*80 + "\n")
    
    # Load data
    print("📊 Loading data...")
    trackssm_data, comparison_data = load_data()
    print("   ✓ Data loaded successfully\n")
    
    # Create output structure
    print("📁 Creating output directory structure...")
    dirs = create_output_structure()
    print("   ✓ Directory structure created\n")
    
    # Generate plots by category
    print("🎨 Generating plots...\n")
    
    print("📈 [1/6] Tracking Accuracy Metrics")
    plot_mota_comparison(trackssm_data, comparison_data, dirs['tracking'])
    plot_motp_comparison(trackssm_data, comparison_data, dirs['tracking'])
    
    print("\n🔑 [2/6] Identity Metrics")
    plot_idf1_comparison(trackssm_data, comparison_data, dirs['identity'])
    plot_idsw_comparison(trackssm_data, comparison_data, dirs['identity'])
    plot_fragmentation(trackssm_data, comparison_data, dirs['identity'])
    
    print("\n🎯 [3/6] Detection Quality")
    plot_precision_comparison(trackssm_data, comparison_data, dirs['detection'])
    plot_recall_comparison(trackssm_data, comparison_data, dirs['detection'])
    
    print("\n⚠️  [4/6] Error Analysis")
    plot_false_positives(trackssm_data, comparison_data, dirs['errors'])
    plot_false_negatives(trackssm_data, comparison_data, dirs['errors'])
    plot_mt_ml_comparison(trackssm_data, comparison_data, dirs['errors'])
    
    print("\n🏆 [5/6] HOTA Analysis")
    plot_hota_single(trackssm_data, comparison_data, dirs['hota'])
    plot_hota_components(trackssm_data, comparison_data, dirs['hota'])
    plot_hota_precision_recall(trackssm_data, comparison_data, dirs['hota'])
    
    print("\n📊 [6/6] Summary Views")
    plot_overall_comparison(trackssm_data, comparison_data, dirs['summary'])
    plot_class_distribution(trackssm_data, comparison_data, dirs['summary'])
    create_summary_table(trackssm_data, comparison_data, dirs['summary'])
    
    # Print summary
    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    print("📂 OUTPUT STRUCTURE:")
    print(f"   {dirs['main']}/")
    print(f"   ├── 01_tracking_accuracy/")
    print(f"   │   ├── mota_comparison.png")
    print(f"   │   └── motp_comparison.png")
    print(f"   ├── 02_identity_metrics/")
    print(f"   │   ├── idf1_comparison.png")
    print(f"   │   ├── idsw_comparison.png")
    print(f"   │   └── fragmentation.png")
    print(f"   ├── 03_detection_quality/")
    print(f"   │   ├── precision_comparison.png")
    print(f"   │   └── recall_comparison.png")
    print(f"   ├── 04_error_analysis/")
    print(f"   │   ├── false_positives.png")
    print(f"   │   ├── false_negatives.png")
    print(f"   │   └── mt_ml_comparison.png")
    print(f"   ├── 05_hota_analysis/")
    print(f"   │   ├── hota_botsort.png")
    print(f"   │   ├── hota_components.png")
    print(f"   │   └── hota_precision_recall.png")
    print(f"   └── 06_summary_views/")
    print(f"       ├── overall_comparison.png")
    print(f"       ├── class_distribution.png")
    print(f"       └── summary_table.png")
    
    print("\n📝 TOTAL: 17 high-resolution plots organized in 6 categories")
    print("🎯 Ready for presentation and publication!\n")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
