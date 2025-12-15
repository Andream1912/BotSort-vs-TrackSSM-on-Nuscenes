import re
import matplotlib.pyplot as plt
import numpy as np

# Parse log
with open('training_stable.log', 'r') as f:
    content = f.read()

# Extract iter data with epoch context
pattern = r'Epoch (\d+)/30.*?iter (\d+)/666 - loss: total: ([\d.]+), iou: ([\d.]+), l1: ([\d.]+), conf: ([\d.]+), cls: ([\d.]+), lr: ([\d.]+)'
matches = re.findall(pattern, content, re.DOTALL)

data = []
for match in matches:
    epoch, iter_n, loss_total, loss_iou, loss_l1, loss_conf, loss_cls, lr = match
    data.append({
        'epoch': int(epoch),
        'iter': int(iter_n),
        'loss': float(loss_total),
        'lr': float(lr) * 1000,  # to millis
        'conf': float(loss_conf),
        'cls': float(loss_cls)
    })

if not data:
    print("No data found!")
    exit(1)

# Global iterations
for i, d in enumerate(data):
    d['global_iter'] = (d['epoch'] - 1) * 666 + d['iter']

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Loss plot
ax1 = axes[0]
x = [d['global_iter'] for d in data]
y = [d['loss'] for d in data]
ax1.plot(x, y, 'b-', linewidth=2.5, alpha=0.8, label='Total Loss')

# Warmup end line
ax1.axvline(x=666*3, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Warmup End (Epoch 3)')

# Epoch separators
for ep in range(1, 6):
    ax1.axvline(x=666*(ep-1), color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax1.text(666*(ep-1)+100, 10.5, f'Epoch {ep}', fontsize=11, fontweight='bold', alpha=0.7)

ax1.set_ylabel('Total Loss', fontsize=13, fontweight='bold')
ax1.set_title('YOLOX-L Stable V3: Training Progress (Epoch 1-5)', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_ylim([4, 11])

# LR plot
ax2 = axes[1]
lr_vals = [d['lr'] for d in data]
ax2.plot(x, lr_vals, 'g-', linewidth=2.5, alpha=0.8, label='Learning Rate')

# Warmup end
ax2.axvline(x=666*3, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Warmup End')

# Epoch separators
for ep in range(1, 6):
    ax2.axvline(x=666*(ep-1), color='gray', linestyle=':', alpha=0.4, linewidth=1)

ax2.set_xlabel('Iterations', fontsize=13, fontweight='bold')
ax2.set_ylabel('Learning Rate (x10^-3)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('training_stable_epoch5.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved: training_stable_epoch5.png\n")

# Statistics
print("=" * 70)
print("  TRAINING STABLE V3 - ANALYSIS (EPOCH 1-5)")
print("=" * 70)

# Overall
first_loss = data[0]['loss']
last_loss = data[-1]['loss']
print(f"\n📊 Overall Progress:")
print(f"  Start Loss:  {first_loss:.2f} (iter 1, epoch 1)")
print(f"  Current Loss: {last_loss:.2f} (iter {data[-1]['iter']}, epoch {data[-1]['epoch']})")
print(f"  Reduction:    {first_loss - last_loss:.2f} ({(last_loss - first_loss) / first_loss * 100:+.1f}%)")

# Per epoch
print(f"\n📈 Per-Epoch Statistics:")
print(f"  {'Epoch':<8} {'Avg Loss':<10} {'Min Loss':<10} {'Max Loss':<10} {'StdDev':<10} {'Avg LR':<10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for ep in range(1, 6):
    ep_data = [d for d in data if d['epoch'] == ep]
    if not ep_data:
        continue
    
    losses = [d['loss'] for d in ep_data]
    lrs = [d['lr'] for d in ep_data]
    
    avg_loss = np.mean(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    std_loss = np.std(losses)
    avg_lr = np.mean(lrs)
    
    print(f"  {ep:<8} {avg_loss:<10.2f} {min_loss:<10.2f} {max_loss:<10.2f} {std_loss:<10.2f} {avg_lr:<10.3f}")

# Stability check (epoch 4 vs 5)
print(f"\n✅ Stability Analysis (Post-Warmup):")
ep4_data = [d['loss'] for d in data if d['epoch'] == 4]
ep5_data = [d['loss'] for d in data if d['epoch'] == 5]

if ep4_data and ep5_data:
    avg4 = np.mean(ep4_data)
    avg5 = np.mean(ep5_data)
    std4 = np.std(ep4_data)
    std5 = np.std(ep5_data)
    
    print(f"  Epoch 4: Avg={avg4:.2f}, StdDev={std4:.2f}")
    print(f"  Epoch 5: Avg={avg5:.2f}, StdDev={std5:.2f}")
    print(f"  Change:  {avg5 - avg4:+.2f} ({(avg5 - avg4) / avg4 * 100:+.1f}%)")
    
    if std5 < 0.5 and abs((avg5 - avg4) / avg4) < 0.05:
        status = "✅ MOLTO STABILE - Varianza bassa, convergenza graduale"
    elif std5 < 0.8:
        status = "✓ STABILE - Convergenza normale"
    else:
        status = "⚠️ VARIABILE - Monitorare"
    
    print(f"\n  Status: {status}")

print(f"\n{'=' * 70}\n")

