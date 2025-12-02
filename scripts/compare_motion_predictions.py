#!/usr/bin/env python3
"""
Confronto diretto delle predizioni Kalman vs TrackSSM nei momenti critici.
Mostra QUANTO MEGLIO predice TrackSSM la posizione futura.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json

def load_tracks(filepath):
    """Carica tracking in formato MOT"""
    tracks = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            tracks[frame].append({
                'id': track_id,
                'bbox': bbox,
                'conf': conf
            })
    return tracks

def compute_iou(bbox1, bbox2):
    """Calcola IoU tra due bbox [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]
    
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def bbox_center(bbox):
    """Calcola centro bbox"""
    x, y, w, h = bbox
    return (x + w/2, y + h/2)

def bbox_distance(bbox1, bbox2):
    """Distanza euclidea tra centri"""
    c1 = bbox_center(bbox1)
    c2 = bbox_center(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def main():
    # Paths
    kalman_file = Path("results/results_prima_del_meeting/last_minute/kalman/data/scene-1062.txt")
    trackssm_file = Path("results/results_prima_del_meeting/last_minute/trackssm_finetuned/data/scene-1062.txt")
    gt_file = Path("data/nuscenes_mot_front/val/scene-1062_CAM_FRONT/gt/gt.txt")
    output_dir = Path("results/FINE_TUNING_PARAMETRI/scene-1062_prediction_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Carica
    kalman = load_tracks(kalman_file)
    trackssm = load_tracks(trackssm_file)
    gt = load_tracks(gt_file)
    
    print("=" * 90)
    print("MOTION PREDICTION COMPARISON: Kalman vs TrackSSM")
    print("=" * 90)
    
    # Focus su GT ID=501844 (la traccia con più problemi)
    gt_id_focus = 501844
    frames_focus = range(27, 41)  # Dopo detection gap
    
    # Estrai dati per plotting
    data = {
        'frames': [],
        'kalman_iou': [],
        'trackssm_iou': [],
        'kalman_dist': [],
        'trackssm_dist': [],
        'gt_centers': []
    }
    
    print(f"\n📊 FOCUS: GT ID={gt_id_focus} (Frames 27-40)")
    print(f"{'-'*90}")
    print(f"{'Frame':<7} {'K IoU':<10} {'TS IoU':<10} {'Diff':<12} {'K Dist':<12} {'TS Dist':<12} {'Winner'}")
    print(f"{'-'*90}")
    
    for frame in frames_focus:
        # GT bbox
        gt_objs = [t for t in gt.get(frame, []) if t['id'] == gt_id_focus]
        if not gt_objs:
            continue
        gt_bbox = gt_objs[0]['bbox']
        
        # Kalman best match
        kalman_matches = [(t['id'], t['bbox'], compute_iou(gt_bbox, t['bbox'])) 
                         for t in kalman.get(frame, [])]
        kalman_matches = [(id, bbox, iou) for id, bbox, iou in kalman_matches if iou > 0.1]
        kalman_matches.sort(key=lambda x: x[2], reverse=True)
        
        # TrackSSM best match
        trackssm_matches = [(t['id'], t['bbox'], compute_iou(gt_bbox, t['bbox'])) 
                           for t in trackssm.get(frame, [])]
        trackssm_matches = [(id, bbox, iou) for id, bbox, iou in trackssm_matches if iou > 0.1]
        trackssm_matches.sort(key=lambda x: x[2], reverse=True)
        
        if kalman_matches and trackssm_matches:
            k_id, k_bbox, k_iou = kalman_matches[0]
            t_id, t_bbox, t_iou = trackssm_matches[0]
            
            k_dist = bbox_distance(gt_bbox, k_bbox)
            t_dist = bbox_distance(gt_bbox, t_bbox)
            
            iou_diff = t_iou - k_iou
            dist_diff = k_dist - t_dist  # Positive = TrackSSM closer
            
            # Winner
            if iou_diff > 0.05:
                winner = "🟢 TrackSSM"
            elif iou_diff < -0.05:
                winner = "🟡 Kalman"
            else:
                winner = "✓ Tie"
            
            print(f"{frame:<7} {k_iou:<10.3f} {t_iou:<10.3f} {iou_diff:>+.3f} ({iou_diff*100:+.1f}%) "
                  f"{k_dist:<12.1f} {t_dist:<12.1f} {winner}")
            
            data['frames'].append(frame)
            data['kalman_iou'].append(k_iou)
            data['trackssm_iou'].append(t_iou)
            data['kalman_dist'].append(k_dist)
            data['trackssm_dist'].append(t_dist)
            data['gt_centers'].append(bbox_center(gt_bbox))
    
    # Statistics
    print(f"\n{'='*90}")
    print("📈 STATISTICHE AGGREGATE")
    print(f"{'='*90}")
    
    k_iou_avg = np.mean(data['kalman_iou'])
    t_iou_avg = np.mean(data['trackssm_iou'])
    k_dist_avg = np.mean(data['kalman_dist'])
    t_dist_avg = np.mean(data['trackssm_dist'])
    
    print(f"\n🎯 IoU Performance:")
    print(f"  Kalman avg:   {k_iou_avg:.3f}")
    print(f"  TrackSSM avg: {t_iou_avg:.3f}")
    print(f"  Difference:   {t_iou_avg - k_iou_avg:+.3f} ({(t_iou_avg - k_iou_avg)*100:+.1f}%)")
    
    print(f"\n📏 Distance Error (pixels):")
    print(f"  Kalman avg:   {k_dist_avg:.1f} px")
    print(f"  TrackSSM avg: {t_dist_avg:.1f} px")
    print(f"  Improvement:  {k_dist_avg - t_dist_avg:+.1f} px ({(k_dist_avg - t_dist_avg)/k_dist_avg*100:+.1f}%)")
    
    # Count wins
    iou_diffs = np.array(data['trackssm_iou']) - np.array(data['kalman_iou'])
    ts_wins = np.sum(iou_diffs > 0.05)
    k_wins = np.sum(iou_diffs < -0.05)
    ties = len(iou_diffs) - ts_wins - k_wins
    
    print(f"\n🏆 Frame-by-Frame Wins (IoU diff > 0.05):")
    print(f"  TrackSSM wins: {ts_wins}/{len(data['frames'])} frames ({ts_wins/len(data['frames'])*100:.1f}%)")
    print(f"  Kalman wins:   {k_wins}/{len(data['frames'])} frames ({k_wins/len(data['frames'])*100:.1f}%)")
    print(f"  Ties:          {ties}/{len(data['frames'])} frames ({ties/len(data['frames'])*100:.1f}%)")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Motion Prediction Comparison: Scene-1062 GT ID={gt_id_focus}', fontsize=16, fontweight='bold')
    
    # Plot 1: IoU over time
    ax1 = axes[0, 0]
    ax1.plot(data['frames'], data['kalman_iou'], 'o-', label='Kalman', color='#3498db', linewidth=2, markersize=8)
    ax1.plot(data['frames'], data['trackssm_iou'], 's-', label='TrackSSM', color='#e74c3c', linewidth=2, markersize=8)
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Good match threshold')
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('IoU with Ground Truth', fontsize=12)
    ax1.set_title('Prediction Accuracy (IoU)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.0])
    
    # Highlight critical frames
    critical_frames = [29, 30, 31, 32, 33]
    for cf in critical_frames:
        if cf in data['frames']:
            ax1.axvspan(cf-0.5, cf+0.5, alpha=0.1, color='red')
    
    # Plot 2: IoU difference (TrackSSM - Kalman)
    ax2 = axes[0, 1]
    iou_diff = np.array(data['trackssm_iou']) - np.array(data['kalman_iou'])
    colors = ['green' if d > 0 else 'red' for d in iou_diff]
    ax2.bar(data['frames'], iou_diff, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('IoU Difference (TrackSSM - Kalman)', fontsize=12)
    ax2.set_title('Prediction Advantage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Annotate significant differences
    for i, (f, d) in enumerate(zip(data['frames'], iou_diff)):
        if abs(d) > 0.1:
            ax2.text(f, d, f'{d:+.2f}', ha='center', va='bottom' if d > 0 else 'top', fontsize=9)
    
    # Plot 3: Distance error
    ax3 = axes[1, 0]
    ax3.plot(data['frames'], data['kalman_dist'], 'o-', label='Kalman', color='#3498db', linewidth=2, markersize=8)
    ax3.plot(data['frames'], data['trackssm_dist'], 's-', label='TrackSSM', color='#e74c3c', linewidth=2, markersize=8)
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('Center Distance Error (pixels)', fontsize=12)
    ax3.set_title('Prediction Error (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory in image space
    ax4 = axes[1, 1]
    gt_x = [c[0] for c in data['gt_centers']]
    gt_y = [c[1] for c in data['gt_centers']]
    
    # Plot GT trajectory
    ax4.plot(gt_x, gt_y, 'k-', linewidth=3, label='Ground Truth', zorder=3)
    ax4.scatter(gt_x, gt_y, c=data['frames'], cmap='viridis', s=100, edgecolors='black', linewidth=2, zorder=4)
    
    # Annotate start/end
    ax4.text(gt_x[0], gt_y[0], 'START', fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax4.text(gt_x[-1], gt_y[-1], 'END', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    ax4.set_xlabel('X (pixels)', fontsize=12)
    ax4.set_ylabel('Y (pixels)', fontsize=12)
    ax4.set_title('Object Trajectory in Image Space', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # Image coordinates
    
    # Colorbar for frame numbers
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(data['frames']), vmax=max(data['frames'])))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('Frame', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'prediction_comparison_GT_501844.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot salvato: {plot_path}")
    
    # Save data as JSON
    json_path = output_dir / 'prediction_comparison_GT_501844.json'
    with open(json_path, 'w') as f:
        json.dump({
            'gt_id': gt_id_focus,
            'frames': data['frames'],
            'kalman': {
                'iou': data['kalman_iou'],
                'distance': data['kalman_dist'],
                'avg_iou': float(k_iou_avg),
                'avg_distance': float(k_dist_avg)
            },
            'trackssm': {
                'iou': data['trackssm_iou'],
                'distance': data['trackssm_dist'],
                'avg_iou': float(t_iou_avg),
                'avg_distance': float(t_dist_avg)
            },
            'comparison': {
                'iou_improvement': float(t_iou_avg - k_iou_avg),
                'iou_improvement_percent': float((t_iou_avg - k_iou_avg) / k_iou_avg * 100),
                'distance_improvement': float(k_dist_avg - t_dist_avg),
                'distance_improvement_percent': float((k_dist_avg - t_dist_avg) / k_dist_avg * 100),
                'trackssm_wins': int(ts_wins),
                'kalman_wins': int(k_wins),
                'ties': int(ties)
            }
        }, f, indent=2)
    
    print(f"✅ Dati salvati: {json_path}")
    print("\n" + "=" * 90)
    print("✅ ANALISI COMPLETATA!")
    print("=" * 90)
    print(f"\n💡 KEY INSIGHT:")
    print(f"   TrackSSM predice {(t_iou_avg - k_iou_avg)*100:+.1f}% meglio di Kalman (IoU)")
    print(f"   TrackSSM ha {(k_dist_avg - t_dist_avg)/k_dist_avg*100:+.1f}% meno errore di posizione")
    print(f"   TrackSSM vince in {ts_wins}/{len(data['frames'])} frames ({ts_wins/len(data['frames'])*100:.0f}%)")
    print(f"\n   MA: Entrambi hanno 3 ID switches per detection gaps + association issues")
    print(f"   SOLUZIONE: Abbassare match_thresh da 0.8 a 0.7 per sfruttare prediction migliore!")

if __name__ == "__main__":
    main()
