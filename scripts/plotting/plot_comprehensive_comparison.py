#!/usr/bin/env python3
"""
Generate comprehensive comparison plots between TrackSSM and BotSort
with all important metrics including HOTA, FP, FN, and better readability
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for better readability
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define colors
TRACKSSM_COLOR = '#2E86AB'  # Blue
BOTSORT_COLOR = '#A23B72'   # Purple/Pink
COLORS_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B7E74']

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

def create_hota_comparison(trackssm_data, comparison_data, output_dir):
    """Create HOTA comparison plot (only BotSort has HOTA)"""
    classes = get_class_order()
    
    # Get BotSort HOTA values
    botsort_hota = []
    for cls in classes:
        if cls in comparison_data['methods']['BotSort']['per_class_metrics']:
            botsort_hota.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['HOTA'])
        else:
            botsort_hota.append(0)
    
    # BotSort overall HOTA
    overall_hota = comparison_data['methods']['BotSort']['metrics']['HOTA']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = ax.bar(x, botsort_hota, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add overall HOTA line
    ax.axhline(y=overall_hota, color='red', linestyle='--', linewidth=2, 
               label=f'BotSort Overall: {overall_hota:.1f}%', alpha=0.7)
    
    ax.set_ylabel('HOTA (%)', fontweight='bold')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_title('Higher Order Tracking Accuracy (HOTA) - BotSort Only\n(TrackSSM metrics not available in HOTA format)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(botsort_hota) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hota_botsort.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created HOTA comparison plot")

def create_mota_idf1_grouped(trackssm_data, comparison_data, output_dir):
    """Create grouped bar chart for MOTA and IDF1"""
    classes = get_class_order()
    
    # Extract data
    trackssm_mota = [trackssm_data['per_class_metrics'][cls]['MOTA'] for cls in classes]
    trackssm_idf1 = [trackssm_data['per_class_metrics'][cls]['IDF1'] for cls in classes]
    
    botsort_mota = []
    botsort_idf1 = []
    for cls in classes:
        if cls in comparison_data['methods']['BotSort']['per_class_metrics']:
            botsort_mota.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['MOTA'])
            botsort_idf1.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['IDF1'])
        else:
            botsort_mota.append(0)
            botsort_idf1.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # MOTA subplot
    bars1 = ax1.bar(x - width/2, trackssm_mota, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, botsort_mota, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels for MOTA
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('MOTA (%)', fontweight='bold')
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_title('Multiple Object Tracking Accuracy (MOTA)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # IDF1 subplot
    bars3 = ax2.bar(x - width/2, trackssm_idf1, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, botsort_idf1, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels for IDF1
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('IDF1 (%)', fontweight='bold')
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_title('ID F1 Score (IDF1)', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mota_idf1_grouped.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created MOTA & IDF1 grouped comparison")

def create_precision_recall_grouped(trackssm_data, comparison_data, output_dir):
    """Create grouped bar chart for Precision and Recall"""
    classes = get_class_order()
    
    # Extract data
    trackssm_precision = [trackssm_data['per_class_metrics'][cls]['Precision'] for cls in classes]
    trackssm_recall = [trackssm_data['per_class_metrics'][cls]['Recall'] for cls in classes]
    
    botsort_precision = []
    botsort_recall = []
    for cls in classes:
        if cls in comparison_data['methods']['BotSort']['per_class_metrics']:
            botsort_precision.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['DetPr'])
            botsort_recall.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['DetRe'])
        else:
            botsort_precision.append(0)
            botsort_recall.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Precision subplot
    bars1 = ax1.bar(x - width/2, trackssm_precision, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, botsort_precision, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Precision (%)', fontweight='bold')
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_title('Detection Precision', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)
    
    # Recall subplot
    bars3 = ax2.bar(x - width/2, trackssm_recall, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, botsort_recall, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Recall (%)', fontweight='bold')
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_title('Detection Recall', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_grouped.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created Precision & Recall grouped comparison")

def create_fp_fn_comparison(trackssm_data, comparison_data, output_dir):
    """Create comparison plots for False Positives and False Negatives"""
    classes = get_class_order()
    
    # Extract data
    trackssm_fp = [trackssm_data['per_class_metrics'][cls]['FP'] for cls in classes]
    trackssm_fn = [trackssm_data['per_class_metrics'][cls]['FN'] for cls in classes]
    
    botsort_fp = []
    botsort_fn = []
    for cls in classes:
        if cls in comparison_data['methods']['BotSort']['per_class_metrics']:
            botsort_fp.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls].get('IDFP', 0))
            botsort_fn.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls].get('IDFN', 0))
        else:
            botsort_fp.append(0)
            botsort_fn.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # False Positives subplot
    bars1 = ax1.bar(x - width/2, trackssm_fp, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, botsort_fp, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels for FP
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('False Positives (count)', fontweight='bold')
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_title('False Positives (Lower is Better)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # False Negatives subplot
    bars3 = ax2.bar(x - width/2, trackssm_fn, width, label='TrackSSM', 
                    color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, botsort_fn, width, label='BotSort', 
                    color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels for FN
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('False Negatives (count)', fontweight='bold')
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_title('False Negatives (Lower is Better)', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_fn_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created FP & FN comparison")

def create_idsw_comparison(trackssm_data, comparison_data, output_dir):
    """Create ID switches comparison with log scale"""
    classes = get_class_order()
    
    trackssm_idsw = [trackssm_data['per_class_metrics'][cls]['IDSW'] for cls in classes]
    
    botsort_idsw = []
    for cls in classes:
        if cls in comparison_data['methods']['BotSort']['per_class_metrics']:
            botsort_idsw.append(comparison_data['methods']['BotSort']['per_class_metrics'][cls]['IDSW'])
        else:
            botsort_idsw.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, trackssm_idsw, width, label='TrackSSM', 
                   color=TRACKSSM_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, botsort_idsw, width, label='BotSort', 
                   color=BOTSORT_COLOR, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('ID Switches (count, log scale)', fontweight='bold')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_title('ID Switches Comparison (Lower is Better)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([format_class_name(c) for c in classes], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'idsw_comparison_log.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created ID Switches comparison (log scale)")

def create_radar_chart(trackssm_data, comparison_data, output_dir):
    """Create radar chart for overall performance comparison"""
    
    # Overall metrics
    trackssm_metrics = comparison_data['methods']['TrackSSM']['metrics']
    botsort_metrics = comparison_data['methods']['BotSort']['metrics']
    
    # Normalize metrics to 0-100 scale for radar chart
    categories = ['MOTA', 'IDF1', 'Precision', 'Recall', 'ID Stability']
    
    # TrackSSM values
    trackssm_values = [
        trackssm_metrics['MOTA'],
        trackssm_metrics['IDF1'],
        trackssm_metrics['Precision'],
        trackssm_metrics['Recall'],
        100 - (trackssm_metrics['IDSW'] / trackssm_metrics['MT'] * 10)  # Inverse of IDSW/MT ratio
    ]
    
    # BotSort values
    botsort_values = [
        botsort_metrics['MOTA'],
        botsort_metrics['IDF1'],
        botsort_metrics['DetPr'],
        botsort_metrics['DetRe'],
        100 - (botsort_metrics['IDSW'] / botsort_metrics['MT'] * 10)
    ]
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    trackssm_values += trackssm_values[:1]
    botsort_values += botsort_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, trackssm_values, 'o-', linewidth=2, label='TrackSSM', 
            color=TRACKSSM_COLOR, markersize=8)
    ax.fill(angles, trackssm_values, alpha=0.25, color=TRACKSSM_COLOR)
    
    ax.plot(angles, botsort_values, 'o-', linewidth=2, label='BotSort', 
            color=BOTSORT_COLOR, markersize=8)
    ax.fill(angles, botsort_values, alpha=0.25, color=BOTSORT_COLOR)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)
    
    # Add title
    plt.title('Overall Performance Comparison\n(Radar Chart)', 
              size=16, fontweight='bold', pad=30)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created radar chart comparison")

def create_summary_dashboard(trackssm_data, comparison_data, output_dir):
    """Create comprehensive summary dashboard"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    classes = get_class_order()
    x = np.arange(len(classes))
    width = 0.35
    
    # 1. MOTA comparison
    ax1 = fig.add_subplot(gs[0, 0])
    trackssm_mota = [trackssm_data['per_class_metrics'][cls]['MOTA'] for cls in classes]
    botsort_mota = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('MOTA', 0) for cls in classes]
    ax1.bar(x - width/2, trackssm_mota, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax1.bar(x + width/2, botsort_mota, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax1.set_title('MOTA (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. IDF1 comparison
    ax2 = fig.add_subplot(gs[0, 1])
    trackssm_idf1 = [trackssm_data['per_class_metrics'][cls]['IDF1'] for cls in classes]
    botsort_idf1 = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDF1', 0) for cls in classes]
    ax2.bar(x - width/2, trackssm_idf1, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax2.bar(x + width/2, botsort_idf1, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax2.set_title('IDF1 (%)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. HOTA (BotSort only)
    ax3 = fig.add_subplot(gs[0, 2])
    botsort_hota = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('HOTA', 0) for cls in classes]
    ax3.bar(x, botsort_hota, width*2, color=BOTSORT_COLOR, alpha=0.8)
    ax3.set_title('HOTA (%) - BotSort', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Precision comparison
    ax4 = fig.add_subplot(gs[1, 0])
    trackssm_prec = [trackssm_data['per_class_metrics'][cls]['Precision'] for cls in classes]
    botsort_prec = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetPr', 0) for cls in classes]
    ax4.bar(x - width/2, trackssm_prec, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax4.bar(x + width/2, botsort_prec, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax4.set_title('Precision (%)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 105)
    
    # 5. Recall comparison
    ax5 = fig.add_subplot(gs[1, 1])
    trackssm_rec = [trackssm_data['per_class_metrics'][cls]['Recall'] for cls in classes]
    botsort_rec = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('DetRe', 0) for cls in classes]
    ax5.bar(x - width/2, trackssm_rec, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax5.bar(x + width/2, botsort_rec, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax5.set_title('Recall (%)', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 105)
    
    # 6. ID Switches (log scale)
    ax6 = fig.add_subplot(gs[1, 2])
    trackssm_idsw = [trackssm_data['per_class_metrics'][cls]['IDSW'] for cls in classes]
    botsort_idsw = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDSW', 1) for cls in classes]
    ax6.bar(x - width/2, trackssm_idsw, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax6.bar(x + width/2, botsort_idsw, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax6.set_title('ID Switches (log)', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # 7. False Positives (log scale)
    ax7 = fig.add_subplot(gs[2, 0])
    trackssm_fp = [trackssm_data['per_class_metrics'][cls]['FP'] for cls in classes]
    botsort_fp = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDFP', 1) for cls in classes]
    ax7.bar(x - width/2, trackssm_fp, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax7.bar(x + width/2, botsort_fp, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax7.set_title('False Positives (log)', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_yscale('log')
    
    # 8. False Negatives (log scale)
    ax8 = fig.add_subplot(gs[2, 1])
    trackssm_fn = [trackssm_data['per_class_metrics'][cls]['FN'] for cls in classes]
    botsort_fn = [comparison_data['methods']['BotSort']['per_class_metrics'].get(cls, {}).get('IDFN', 1) for cls in classes]
    ax8.bar(x - width/2, trackssm_fn, width, label='TrackSSM', color=TRACKSSM_COLOR, alpha=0.8)
    ax8.bar(x + width/2, botsort_fn, width, label='BotSort', color=BOTSORT_COLOR, alpha=0.8)
    ax8.set_title('False Negatives (log)', fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_yscale('log')
    
    # 9. Object distribution
    ax9 = fig.add_subplot(gs[2, 2])
    trackssm_objs = [trackssm_data['per_class_metrics'][cls]['num_objects'] for cls in classes]
    ax9.bar(x, trackssm_objs, width*2, color='#6A994E', alpha=0.8)
    ax9.set_title('Dataset Size (objects)', fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([c[:3].upper() for c in classes], fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('Comprehensive Tracking Metrics Comparison - TrackSSM vs BotSort', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created comprehensive dashboard")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("Creating Comprehensive Comparison Plots")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path('results/final_evaluation/comprehensive_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    trackssm_data, comparison_data = load_data()
    print("✓ Data loaded\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    create_hota_comparison(trackssm_data, comparison_data, output_dir)
    create_mota_idf1_grouped(trackssm_data, comparison_data, output_dir)
    create_precision_recall_grouped(trackssm_data, comparison_data, output_dir)
    create_fp_fn_comparison(trackssm_data, comparison_data, output_dir)
    create_idsw_comparison(trackssm_data, comparison_data, output_dir)
    create_radar_chart(trackssm_data, comparison_data, output_dir)
    create_summary_dashboard(trackssm_data, comparison_data, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ All plots generated successfully!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\nGenerated plots:")
    print("  1. hota_botsort.png - HOTA per class (BotSort only)")
    print("  2. mota_idf1_grouped.png - MOTA & IDF1 side-by-side")
    print("  3. precision_recall_grouped.png - Precision & Recall side-by-side")
    print("  4. fp_fn_comparison.png - False Positives & False Negatives")
    print("  5. idsw_comparison_log.png - ID Switches (log scale)")
    print("  6. radar_comparison.png - Overall performance radar chart")
    print("  7. comprehensive_dashboard.png - All metrics in one view (3x3 grid)")
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
