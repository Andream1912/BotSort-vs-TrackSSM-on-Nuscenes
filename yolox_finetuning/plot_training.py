#!/usr/bin/env python3
"""
Plot training curves from YOLOX fine-tuning log for presentation
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Parse log file
log_file = Path(__file__).parent / "training_clean_v2.log"
print(f"Parsing log: {log_file}")

# Store metrics per epoch
epochs = []
total_losses = []
iou_losses = []
conf_losses = []
cls_losses = []
learning_rates = []

# Regex pattern for loss lines
pattern = r'iter (\d+)/2667 - loss: total: ([\d.]+), iou: ([\d.]+), l1: ([\d.]+), conf: ([\d.]+), cls: ([\d.]+), lr: ([\d.]+)'

current_epoch = 0
iter_per_epoch = 2667

with open(log_file, 'r') as f:
    for line in f:
        # Track epoch changes
        if "Epoch " in line and "/10" in line:
            match = re.search(r'Epoch (\d+)/10', line)
            if match:
                current_epoch = int(match.group(1))
        
        # Parse loss values
        match = re.search(pattern, line)
        if match:
            iteration = int(match.group(1))
            total_loss = float(match.group(2))
            iou_loss = float(match.group(3))
            conf_loss = float(match.group(5))
            cls_loss = float(match.group(6))
            lr = float(match.group(7))
            
            # Calculate global iteration
            global_iter = (current_epoch - 1) * iter_per_epoch + iteration
            
            epochs.append(global_iter / iter_per_epoch)
            total_losses.append(total_loss)
            iou_losses.append(iou_loss)
            conf_losses.append(conf_loss)
            cls_losses.append(cls_loss)
            learning_rates.append(lr)

print(f"Parsed {len(epochs)} data points across {current_epoch} epochs")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('YOLOX Fine-Tuning on NuScenes - Training 2 (LR 0.001)', 
             fontsize=16, fontweight='bold')

# Plot 1: Total Loss
ax1 = axes[0, 0]
ax1.plot(epochs, total_losses, linewidth=2, color='#2E86AB', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Total Loss', fontsize=12)
ax1.set_title('Total Loss', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 10)

# Add text annotation for loss reduction
initial_loss = np.mean(total_losses[:10])
final_loss = np.mean(total_losses[-10:])
ax1.text(0.95, 0.95, f'Initial: {initial_loss:.2f}\nFinal: {final_loss:.2f}\nReduction: {((initial_loss-final_loss)/initial_loss*100):.1f}%',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: IoU Loss
ax2 = axes[0, 1]
ax2.plot(epochs, iou_losses, linewidth=2, color='#A23B72', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('IoU Loss', fontsize=12)
ax2.set_title('IoU Loss (Localization)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 10)

# Plot 3: Confidence Loss
ax3 = axes[1, 0]
ax3.plot(epochs, conf_losses, linewidth=2, color='#F18F01', alpha=0.7)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Confidence Loss', fontsize=12)
ax3.set_title('Confidence Loss (Objectness)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(1, 10)

# Plot 4: Classification Loss
ax4 = axes[1, 1]
ax4.plot(epochs, cls_losses, linewidth=2, color='#C73E1D', alpha=0.7)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Classification Loss', fontsize=12)
ax4.set_title('Classification Loss (7 Classes)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(1, 10)

# Add text annotation for class loss reduction
initial_cls = np.mean(cls_losses[:10])
final_cls = np.mean(cls_losses[-10:])
ax4.text(0.95, 0.95, f'Initial: {initial_cls:.2f}\nFinal: {final_cls:.2f}\nReduction: {((initial_cls-final_cls)/initial_cls*100):.1f}%',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = Path(__file__).parent / "training_curves.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: {output_file}")

# Create a second figure with combined losses
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(epochs, total_losses, linewidth=2.5, label='Total Loss', color='#2E86AB', alpha=0.8)
ax.plot(epochs, iou_losses, linewidth=2, label='IoU Loss', color='#A23B72', alpha=0.7)
ax.plot(epochs, conf_losses, linewidth=2, label='Confidence Loss', color='#F18F01', alpha=0.7)
ax.plot(epochs, cls_losses, linewidth=2, label='Classification Loss', color='#C73E1D', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax.set_title('YOLOX Fine-Tuning - All Losses (Training 2, LR 0.001)', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 10)

# Add configuration info
config_text = (
    "Configuration:\n"
    "• Resolution: 800×1440\n"
    "• Backbone: Frozen (86.1%)\n"
    "• Head: Trainable (13.9%)\n"
    "• Batch size: 8\n"
    "• Learning rate: 0.001\n"
    "• Classes: 7 (NuScenes)"
)
ax.text(0.02, 0.98, config_text, transform=ax.transAxes, 
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.tight_layout()

# Save combined figure
output_file2 = Path(__file__).parent / "training_curves_combined.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved combined plot to: {output_file2}")

# Print summary statistics
print("\n" + "="*50)
print("Training Summary:")
print("="*50)
print(f"Total Loss:   {initial_loss:.2f} → {final_loss:.2f} ({((initial_loss-final_loss)/initial_loss*100):.1f}% reduction)")
print(f"IoU Loss:     {np.mean(iou_losses[:10]):.2f} → {np.mean(iou_losses[-10:]):.2f}")
print(f"Conf Loss:    {np.mean(conf_losses[:10]):.2f} → {np.mean(conf_losses[-10:]):.2f}")
print(f"Class Loss:   {initial_cls:.2f} → {final_cls:.2f} ({((initial_cls-final_cls)/initial_cls*100):.1f}% reduction)")
print(f"Final LR:     {learning_rates[-1]:.6f}")
print("="*50)

plt.show()
