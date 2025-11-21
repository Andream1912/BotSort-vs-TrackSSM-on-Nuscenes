import argparse
import re
import matplotlib.pyplot as plt
import os

def parse_log(log_path):
    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    loss_pattern = re.compile(r"Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+)")
    
    current_epoch = 0
    
    for line in lines:
        # Check for epoch header (optional validation)
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            
        # Check for loss line
        loss_match = loss_pattern.search(line)
        if loss_match:
            train_loss = float(loss_match.group(1))
            val_loss = float(loss_match.group(2))
            
            # If we didn't find an explicit epoch line before, just increment
            if len(epochs) > 0:
                expected_epoch = epochs[-1] + 1
            else:
                expected_epoch = 1
                
            # Use found epoch if available, else increment
            epoch = current_epoch if current_epoch > 0 else expected_epoch
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Reset current epoch for next iteration
            current_epoch = 0
            
    return epochs, train_losses, val_losses

def plot_curves(epochs, train_losses, val_losses, output_path, title="Training Progress"):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Highlight min val loss
    if len(val_losses) > 0:
        min_val_loss = min(val_losses)
        min_epoch = epochs[val_losses.index(min_val_loss)]
        plt.annotate(f'Min Val: {min_val_loss:.4f}', 
                     xy=(min_epoch, min_val_loss), 
                     xytext=(min_epoch, min_val_loss + (max(val_losses)-min(val_losses))*0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log file")
    parser.add_argument('--log_file', required=True, help='Path to log file')
    parser.add_argument('--output', required=True, help='Path to output image')
    parser.add_argument('--title', default='Phase 1 Training: Decoder Fine-tuning', help='Plot title')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
        
    epochs, train_losses, val_losses = parse_log(args.log_file)
    
    if not epochs:
        print("No training data found in log file.")
        return
        
    print(f"Found {len(epochs)} epochs.")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    plot_curves(epochs, train_losses, val_losses, args.output, args.title)

if __name__ == "__main__":
    main()
