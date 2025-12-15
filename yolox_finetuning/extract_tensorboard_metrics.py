#!/usr/bin/env python3
"""
Extract all training and validation metrics from TensorBoard logs
for YOLOX fine-tuning documentation in thesis.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not installed. Installing...")
    os.system("pip install tensorboard")
    from tensorboard.backend.event_processing import event_accumulator


def extract_tensorboard_data(logdir):
    """Extract all scalars from TensorBoard logs."""
    
    print(f"📂 Loading TensorBoard data from: {logdir}")
    
    # Initialize event accumulator
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()
    
    # Get all available tags
    tags = ea.Tags()['scalars']
    print(f"📊 Found {len(tags)} metric tags:")
    for tag in tags:
        print(f"   - {tag}")
    
    # Extract data for each tag
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {
            'steps': steps,
            'values': values,
            'wall_time': [e.wall_time for e in events]
        }
        print(f"   ✓ {tag}: {len(steps)} data points")
    
    return data


def categorize_metrics(data):
    """Organize metrics by category."""
    
    categories = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'detection_metrics': [],  # mAP, AP50, etc.
        'other': []
    }
    
    for tag in data.keys():
        tag_lower = tag.lower()
        
        if 'train' in tag_lower and 'loss' in tag_lower:
            categories['train_loss'].append(tag)
        elif 'val' in tag_lower and 'loss' in tag_lower:
            categories['val_loss'].append(tag)
        elif 'lr' in tag_lower or 'learning' in tag_lower:
            categories['learning_rate'].append(tag)
        elif any(metric in tag_lower for metric in ['map', 'ap', 'precision', 'recall']):
            categories['detection_metrics'].append(tag)
        else:
            categories['other'].append(tag)
    
    return categories


def plot_all_metrics(data, categories, output_path):
    """Create comprehensive plots for thesis."""
    
    # Determine number of non-empty categories
    non_empty = {k: v for k, v in categories.items() if v}
    n_plots = len(non_empty)
    
    if n_plots == 0:
        print("⚠️  No metrics found to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot each category
    for category, tags in non_empty.items():
        ax = axes[plot_idx]
        
        for tag in tags:
            steps = data[tag]['steps']
            values = data[tag]['values']
            ax.plot(steps, values, label=tag, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{category.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {output_path}")
    
    return fig


def save_metrics_json(data, categories, output_path):
    """Save all metrics to JSON for reference."""
    
    # Convert numpy types to native Python types for JSON serialization
    json_data = {}
    for tag, metrics in data.items():
        json_data[tag] = {
            'steps': [int(s) for s in metrics['steps']],
            'values': [float(v) for v in metrics['values']],
            'summary': {
                'min': float(np.min(metrics['values'])),
                'max': float(np.max(metrics['values'])),
                'mean': float(np.mean(metrics['values'])),
                'final': float(metrics['values'][-1]) if metrics['values'] else None
            }
        }
    
    output_data = {
        'metrics': json_data,
        'categories': categories,
        'summary': {
            'total_tags': len(data),
            'categories': {k: len(v) for k, v in categories.items()}
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Metrics JSON saved to: {output_path}")


def main():
    # Paths
    tensorboard_dir = Path(__file__).parent.parent / "external/YOLOX/YOLOX_outputs/yolox_x_nuscenes_7class/tensorboard"
    output_plot = Path(__file__).parent / "complete_training_metrics.png"
    output_json = Path(__file__).parent / "tensorboard_metrics.json"
    
    if not tensorboard_dir.exists():
        print(f"❌ TensorBoard directory not found: {tensorboard_dir}")
        sys.exit(1)
    
    print("="*80)
    print("YOLOX TRAINING METRICS EXTRACTION")
    print("="*80)
    
    # Extract data
    data = extract_tensorboard_data(str(tensorboard_dir))
    
    if not data:
        print("❌ No metrics found in TensorBoard logs!")
        sys.exit(1)
    
    # Categorize metrics
    print("\n" + "="*80)
    print("CATEGORIZING METRICS")
    print("="*80)
    categories = categorize_metrics(data)
    
    for category, tags in categories.items():
        if tags:
            print(f"\n{category.upper()}:")
            for tag in tags:
                print(f"  - {tag}")
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    plot_all_metrics(data, categories, output_plot)
    
    # Save JSON
    print("\n" + "="*80)
    print("SAVING JSON SUMMARY")
    print("="*80)
    save_metrics_json(data, categories, output_json)
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"📊 Plot: {output_plot}")
    print(f"📄 JSON: {output_json}")


if __name__ == "__main__":
    main()
