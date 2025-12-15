#!/usr/bin/env python3
"""
Parse YOLOX training log to extract all loss metrics.
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Extract all metrics from training log."""
    
    print(f"📂 Parsing log file: {log_path}")
    
    # Pattern to match log lines with metrics
    pattern = r'epoch: (\d+)/\d+, iter: (\d+)/\d+.*?total_loss: ([\d.]+), iou_loss: ([\d.]+), l1_loss: ([\d.]+), conf_loss: ([\d.]+), cls_loss: ([\d.]+), lr: ([\de.-]+)'
    
    data = defaultdict(list)
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                total_loss = float(match.group(3))
                iou_loss = float(match.group(4))
                l1_loss = float(match.group(5))
                conf_loss = float(match.group(6))
                cls_loss = float(match.group(7))
                lr = float(match.group(8))
                
                data['epoch'].append(epoch)
                data['iteration'].append(iteration)
                data['total_loss'].append(total_loss)
                data['iou_loss'].append(iou_loss)
                data['l1_loss'].append(l1_loss)
                data['conf_loss'].append(conf_loss)
                data['cls_loss'].append(cls_loss)
                data['lr'].append(lr)
    
    print(f"✅ Extracted {len(data['iteration'])} data points")
    print(f"   Epochs: {min(data['epoch'])} to {max(data['epoch'])}")
    print(f"   Iterations: {min(data['iteration'])} to {max(data['iteration'])}")
    
    return dict(data)


def compute_epoch_statistics(data):
    """Compute average loss per epoch."""
    
    epochs = {}
    current_epoch = data['epoch'][0]
    epoch_data = defaultdict(list)
    
    for i in range(len(data['iteration'])):
        if data['epoch'][i] != current_epoch:
            # Save previous epoch
            epochs[current_epoch] = {
                key: np.mean(values) 
                for key, values in epoch_data.items()
            }
            # Reset for new epoch
            current_epoch = data['epoch'][i]
            epoch_data = defaultdict(list)
        
        # Accumulate data for current epoch
        epoch_data['total_loss'].append(data['total_loss'][i])
        epoch_data['iou_loss'].append(data['iou_loss'][i])
        epoch_data['conf_loss'].append(data['conf_loss'][i])
        epoch_data['cls_loss'].append(data['cls_loss'][i])
        epoch_data['lr'].append(data['lr'][i])
    
    # Save last epoch
    if epoch_data:
        epochs[current_epoch] = {
            key: np.mean(values) 
            for key, values in epoch_data.items()
        }
    
    return epochs


def plot_training_metrics(data, epoch_stats, output_path):
    """Create comprehensive training plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Total Loss over iterations (with smoothing)
    ax = axes[0, 0]
    iterations = data['iteration']
    total_loss = data['total_loss']
    
    # Plot raw data (transparent)
    ax.plot(iterations, total_loss, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
    
    # Plot smoothed (moving average)
    window = 50
    if len(total_loss) >= window:
        smoothed = np.convolve(total_loss, np.ones(window)/window, mode='valid')
        smooth_iter = iterations[window-1:]
        ax.plot(smooth_iter, smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Loss (Total)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss components by epoch
    ax = axes[0, 1]
    epoch_nums = sorted(epoch_stats.keys())
    iou_losses = [epoch_stats[e]['iou_loss'] for e in epoch_nums]
    conf_losses = [epoch_stats[e]['conf_loss'] for e in epoch_nums]
    cls_losses = [epoch_stats[e]['cls_loss'] for e in epoch_nums]
    
    ax.plot(epoch_nums, iou_losses, marker='o', linewidth=2, label='IoU Loss')
    ax.plot(epoch_nums, conf_losses, marker='s', linewidth=2, label='Confidence Loss')
    ax.plot(epoch_nums, cls_losses, marker='^', linewidth=2, label='Classification Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Components by Epoch', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax = axes[1, 0]
    ax.plot(iterations, data['lr'], color='green', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Total Loss by Epoch
    ax = axes[1, 1]
    total_losses_epoch = [epoch_stats[e]['total_loss'] for e in epoch_nums]
    ax.plot(epoch_nums, total_losses_epoch, marker='o', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Total Loss', fontsize=12)
    ax.set_title('Training Loss by Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final loss value as text
    final_loss = total_losses_epoch[-1]
    ax.text(0.05, 0.95, f'Final Loss: {final_loss:.3f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to: {output_path}")
    
    return fig


def save_metrics_json(data, epoch_stats, output_path):
    """Save metrics to JSON."""
    
    output_data = {
        'raw_data': {
            key: [float(v) for v in values]  # Convert to native Python types
            for key, values in data.items()
        },
        'epoch_statistics': {
            int(epoch): {k: float(v) for k, v in stats.items()}
            for epoch, stats in epoch_stats.items()
        },
        'summary': {
            'total_iterations': len(data['iteration']),
            'total_epochs': max(data['epoch']),
            'final_epoch_loss': float(epoch_stats[max(epoch_stats.keys())]['total_loss']),
            'min_loss': float(min(data['total_loss'])),
            'max_loss': float(max(data['total_loss']))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Metrics JSON saved to: {output_path}")


def main():
    # Paths
    log_path = Path(__file__).parent.parent / "external/YOLOX/YOLOX_outputs/yolox_x_nuscenes_7class/train_log.txt"
    output_plot = Path(__file__).parent / "training_loss_curves.png"
    output_json = Path(__file__).parent / "training_metrics.json"
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1
    
    print("="*80)
    print("YOLOX TRAINING LOG ANALYSIS")
    print("="*80)
    
    # Parse log
    data = parse_log_file(log_path)
    
    if not data or not data['iteration']:
        print("❌ No training data found in log!")
        return 1
    
    # Compute epoch statistics
    print("\n" + "="*80)
    print("COMPUTING EPOCH STATISTICS")
    print("="*80)
    epoch_stats = compute_epoch_statistics(data)
    print(f"✅ Computed statistics for {len(epoch_stats)} epochs")
    
    # Print epoch summary
    print("\nEpoch Summary:")
    for epoch in sorted(epoch_stats.keys())[-5:]:  # Last 5 epochs
        stats = epoch_stats[epoch]
        print(f"  Epoch {epoch}: Total Loss = {stats['total_loss']:.3f}, "
              f"IoU = {stats['iou_loss']:.3f}, "
              f"Conf = {stats['conf_loss']:.3f}, "
              f"Cls = {stats['cls_loss']:.3f}")
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    plot_training_metrics(data, epoch_stats, output_plot)
    
    # Save JSON
    print("\n" + "="*80)
    print("SAVING JSON")
    print("="*80)
    save_metrics_json(data, epoch_stats, output_json)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Plots: {output_plot}")
    print(f"📄 JSON: {output_json}")
    
    return 0


if __name__ == "__main__":
    exit(main())
