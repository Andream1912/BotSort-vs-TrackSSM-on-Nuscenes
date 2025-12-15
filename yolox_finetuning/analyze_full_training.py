import re
import matplotlib.pyplot as plt
import numpy as np

# Parse log file
with open('training_stable.log', 'r') as f:
    content = f.read()

# Extract all iteration data
pattern = r'Epoch (\d+)/30.*?iter (\d+)/666 - loss: total: ([\d.]+), iou: ([\d.]+), l1: ([\d.]+), conf: ([\d.]+), cls: ([\d.]+), lr: ([\d.]+)'
matches = re.findall(pattern, content, re.DOTALL)

data = []
for match in matches:
    epoch, iter_n, loss_total, loss_iou, loss_l1, loss_conf, loss_cls, lr = match
    data.append({
        'epoch': int(epoch),
        'iter': int(iter_n),
        'global_iter': (int(epoch) - 1) * 666 + int(iter_n),
        'loss_total': float(loss_total),
        'loss_iou': float(loss_iou),
        'loss_conf': float(loss_conf),
        'loss_cls': float(loss_cls),
        'lr': float(lr)
    })

print(f"Parsed {len(data)} data points across {max([d['epoch'] for d in data])} epochs\n")

# Create comprehensive plot
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Total Loss over time
ax1 = fig.add_subplot(gs[0, :])
x = [d['global_iter'] for d in data]
y_total = [d['loss_total'] for d in data]
ax1.plot(x, y_total, 'b-', linewidth=1.5, alpha=0.8, label='Total Loss')

# Mark phases
ax1.axvline(x=666*3, color='red', linestyle='--', linewidth=2, alpha=0.6, label='End Warmup (Epoch 3)')
ax1.axvline(x=666*22, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Start No-Aug (Epoch 23)')

# Epoch markers every 5 epochs
for ep in [1, 5, 10, 15, 20, 25, 30]:
    ax1.axvline(x=666*(ep-1), color='gray', linestyle=':', alpha=0.3, linewidth=1)
    if ep == 1 or ep % 10 == 0:
        ax1.text(666*(ep-1)+200, max(y_total)*0.95, f'E{ep}', fontsize=10, fontweight='bold', alpha=0.7)

ax1.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
ax1.set_title('YOLOX-L Stable V3: Complete Training (30 Epochs)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim([0, max(x)])

# 2. Learning Rate Schedule
ax2 = fig.add_subplot(gs[1, :])
y_lr = [d['lr'] * 1000 for d in data]  # Convert to millis
ax2.plot(x, y_lr, 'g-', linewidth=1.5, alpha=0.8, label='Learning Rate')
ax2.axvline(x=666*3, color='red', linestyle='--', linewidth=2, alpha=0.6, label='End Warmup')
ax2.axvline(x=666*22, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Start No-Aug')

for ep in [1, 5, 10, 15, 20, 25, 30]:
    ax2.axvline(x=666*(ep-1), color='gray', linestyle=':', alpha=0.3, linewidth=1)

ax2.set_ylabel('Learning Rate (×10⁻³)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Iterations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim([0, max(x)])

# 3. Loss Components
ax3 = fig.add_subplot(gs[2, 0])
# Average per epoch
epochs_list = list(range(1, 31))
avg_iou = []
avg_conf = []
avg_cls = []

for ep in epochs_list:
    ep_data = [d for d in data if d['epoch'] == ep]
    if ep_data:
        avg_iou.append(np.mean([d['loss_iou'] for d in ep_data]))
        avg_conf.append(np.mean([d['loss_conf'] for d in ep_data]))
        avg_cls.append(np.mean([d['loss_cls'] for d in ep_data]))

ax3.plot(epochs_list, avg_iou, 'r-', linewidth=2, label='IoU Loss', marker='o', markersize=4)
ax3.plot(epochs_list, avg_conf, 'b-', linewidth=2, label='Conf Loss', marker='s', markersize=4)
ax3.plot(epochs_list, avg_cls, 'g-', linewidth=2, label='Cls Loss', marker='^', markersize=4)
ax3.axvline(x=3, color='red', linestyle='--', alpha=0.4, label='End Warmup')
ax3.axvline(x=22, color='orange', linestyle='--', alpha=0.4, label='No-Aug Start')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss Components', fontsize=11, fontweight='bold')
ax3.set_title('Loss Decomposition', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# 4. Per-Epoch Statistics
ax4 = fig.add_subplot(gs[2, 1])
avg_total = []
std_total = []

for ep in epochs_list:
    ep_data = [d['loss_total'] for d in data if d['epoch'] == ep]
    if ep_data:
        avg_total.append(np.mean(ep_data))
        std_total.append(np.std(ep_data))

ax4.errorbar(epochs_list, avg_total, yerr=std_total, fmt='bo-', linewidth=2, 
             markersize=5, capsize=3, alpha=0.7, label='Avg ± Std')
ax4.axvline(x=3, color='red', linestyle='--', alpha=0.4)
ax4.axvline(x=22, color='orange', linestyle='--', alpha=0.4)
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Average Loss per Epoch', fontsize=11, fontweight='bold')
ax4.set_title('Epoch-wise Stability', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

plt.savefig('training_stable_complete_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved: training_stable_complete_analysis.png\n")

# Detailed Statistics
print("=" * 80)
print("  YOLOX-L STABLE V3: COMPLETE TRAINING ANALYSIS (30 EPOCHS)")
print("=" * 80)

# Overall
first_loss = data[0]['loss_total']
last_loss = data[-1]['loss_total']
print(f"\n📊 Overall Progress:")
print(f"  Initial Loss (Epoch 1, iter 1):    {first_loss:.2f}")
print(f"  Final Loss (Epoch 30, iter 650):   {last_loss:.2f}")
print(f"  Total Reduction:                    {first_loss - last_loss:.2f} ({(last_loss - first_loss) / first_loss * 100:+.1f}%)")

# Phase-wise analysis
print(f"\n📈 Phase-wise Analysis:")

# Warmup (Epoch 1-3)
warmup_data = [d['loss_total'] for d in data if d['epoch'] <= 3]
warmup_avg = np.mean(warmup_data)
warmup_start = warmup_data[0]
warmup_end = [d['loss_total'] for d in data if d['epoch'] == 3][-1]
print(f"\n  WARMUP (Epochs 1-3):")
print(f"    Start:  {warmup_start:.2f}")
print(f"    End:    {warmup_end:.2f}")
print(f"    Change: {warmup_end - warmup_start:.2f} ({(warmup_end - warmup_start) / warmup_start * 100:+.1f}%)")

# Training (Epoch 4-22)
training_data = [d['loss_total'] for d in data if 4 <= d['epoch'] <= 22]
training_avg = np.mean(training_data)
training_std = np.std(training_data)
training_start = [d['loss_total'] for d in data if d['epoch'] == 4][0]
training_end = [d['loss_total'] for d in data if d['epoch'] == 22][-1]
print(f"\n  TRAINING (Epochs 4-22):")
print(f"    Start:    {training_start:.2f}")
print(f"    End:      {training_end:.2f}")
print(f"    Avg:      {training_avg:.2f} ± {training_std:.2f}")
print(f"    Change:   {training_end - training_start:.2f} ({(training_end - training_start) / training_start * 100:+.1f}%)")

# Fine-tune (Epoch 23-30)
finetune_data = [d['loss_total'] for d in data if d['epoch'] >= 23]
finetune_avg = np.mean(finetune_data)
finetune_std = np.std(finetune_data)
finetune_start = [d['loss_total'] for d in data if d['epoch'] == 23][0]
finetune_end = [d['loss_total'] for d in data if d['epoch'] == 30][-1]
print(f"\n  FINE-TUNE (Epochs 23-30, No-Aug):")
print(f"    Start:    {finetune_start:.2f}")
print(f"    End:      {finetune_end:.2f}")
print(f"    Avg:      {finetune_avg:.2f} ± {finetune_std:.2f}")
print(f"    Change:   {finetune_end - finetune_start:.2f} ({(finetune_end - finetune_start) / finetune_start * 100:+.1f}%)")

# Best epochs
print(f"\n🏆 Best Checkpoints:")
epoch_avgs = {}
for ep in epochs_list:
    ep_data = [d['loss_total'] for d in data if d['epoch'] == ep]
    if ep_data:
        epoch_avgs[ep] = np.mean(ep_data)

sorted_epochs = sorted(epoch_avgs.items(), key=lambda x: x[1])
for i, (ep, avg_loss) in enumerate(sorted_epochs[:5]):
    print(f"  {i+1}. Epoch {ep:2d}: Avg Loss = {avg_loss:.2f}")

# Stability metrics
print(f"\n✅ Stability Metrics:")
all_losses = [d['loss_total'] for d in data]
overall_std = np.std(all_losses)
overall_cv = overall_std / np.mean(all_losses) * 100  # Coefficient of variation

post_warmup_losses = [d['loss_total'] for d in data if d['epoch'] > 3]
post_warmup_std = np.std(post_warmup_losses)
post_warmup_cv = post_warmup_std / np.mean(post_warmup_losses) * 100

print(f"  Overall Std Dev:        {overall_std:.3f}")
print(f"  Overall CV:             {overall_cv:.2f}%")
print(f"  Post-Warmup Std Dev:    {post_warmup_std:.3f}")
print(f"  Post-Warmup CV:         {post_warmup_cv:.2f}%")

if post_warmup_cv < 5:
    status = "✅ VERY STABLE"
elif post_warmup_cv < 10:
    status = "✓ STABLE"
else:
    status = "⚠️ VARIABLE"

print(f"\n  Training Status: {status}")

# Training time
import re
time_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
timestamps = re.findall(time_pattern, content)
if len(timestamps) >= 2:
    from datetime import datetime
    start_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S')
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600
    print(f"\n⏱️  Training Duration: {hours:.1f} hours ({duration.total_seconds()/60:.0f} minutes)")
    print(f"   Time per Epoch: {hours/30:.1f} hours (~{hours*60/30:.0f} minutes)")

print(f"\n{'=' * 80}\n")

