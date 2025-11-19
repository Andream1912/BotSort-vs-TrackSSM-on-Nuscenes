"""
Recompute ALL metrics correctly using motmetrics on the same data.
This will ensure consistency between overall and per-class metrics.
"""
import motmetrics as mm
import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def load_mot_file(file_path):
    """Load MOT format file."""
    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls_id = int(parts[7]) if len(parts) > 7 else -1
            
            data[frame_id].append({
                'id': obj_id,
                'bbox': [x, y, w, h],
                'conf': conf,
                'class': cls_id
            })
    return data


def compute_metrics_for_sequence(gt_file, pred_file, seq_name):
    """Compute metrics for a single sequence."""
    gt_data = load_mot_file(gt_file)
    pred_data = load_mot_file(pred_file)
    
    acc = mm.MOTAccumulator(auto_id=True)
    
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    for frame_id in all_frames:
        gt_objs = gt_data.get(frame_id, [])
        pred_objs = pred_data.get(frame_id, [])
        
        gt_ids = [obj['id'] for obj in gt_objs]
        pred_ids = [obj['id'] for obj in pred_objs]
        
        # Compute IoU distances
        if len(gt_objs) > 0 and len(pred_objs) > 0:
            distances = []
            for gt_obj in gt_objs:
                row = []
                for pred_obj in pred_objs:
                    iou = bbox_iou(gt_obj['bbox'], pred_obj['bbox'])
                    row.append(1 - iou)  # Distance = 1 - IoU
                distances.append(row)
            distances = np.array(distances)
        else:
            distances = np.empty((len(gt_objs), len(pred_objs)))
        
        acc.update(gt_ids, pred_ids, distances)
    
    return acc


def bbox_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_overall_metrics(gt_dir, pred_dir):
    """Compute overall metrics across all sequences."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    # Collect all sequences
    sequences = []
    for seq_dir in sorted(gt_dir.iterdir()):
        if seq_dir.is_dir():
            gt_file = seq_dir / 'gt' / 'gt.txt'
            pred_file = pred_dir / f"{seq_dir.name}.txt"
            if gt_file.exists() and pred_file.exists():
                sequences.append((seq_dir.name, gt_file, pred_file))
    
    print(f"Found {len(sequences)} sequences")
    
    # Compute accumulator for each sequence
    accumulators = []
    seq_names = []
    
    for seq_name, gt_file, pred_file in sequences:
        print(f"Processing {seq_name}...")
        acc = compute_metrics_for_sequence(gt_file, pred_file, seq_name)
        accumulators.append(acc)
        seq_names.append(seq_name)
    
    # Compute metrics
    mh = mm.metrics.create()
    
    summary = mh.compute_many(
        accumulators, 
        metrics=[
            'num_frames', 'mota', 'motp', 'idf1', 
            'num_switches', 'num_fragmentations',
            'num_false_positives', 'num_misses',
            'mostly_tracked', 'mostly_lost',
            'precision', 'recall', 'num_objects'
        ],
        names=seq_names
    )
    
    print("\n" + "="*70)
    print("SUMMARY PER SEQUENCE")
    print("="*70)
    print(summary.head(10))
    
    # Compute OVERALL (weighted by num_objects)
    weights = summary['num_objects'].values
    total_objects = weights.sum()
    
    mota_overall = (summary['mota'] * weights).sum() / total_objects
    motp_overall = (summary['motp'] * weights).sum() / total_objects
    idf1_overall = (summary['idf1'] * weights).sum() / total_objects
    precision_overall = (summary['precision'] * weights).sum() / total_objects
    recall_overall = (summary['recall'] * weights).sum() / total_objects
    
    # Sum counts
    num_switches = int(summary['num_switches'].sum())
    num_frag = int(summary['num_fragmentations'].sum())
    num_fp = int(summary['num_false_positives'].sum())
    num_fn = int(summary['num_misses'].sum())
    num_mt = int(summary['mostly_tracked'].sum())
    num_ml = int(summary['mostly_lost'].sum())
    num_frames = int(summary['num_frames'].sum())
    
    results = {
        'method': 'TrackSSM (Zero-Shot)',
        'model_weights': 'MOT17_epoch160.pt',
        'dataset': 'NuScenes val CAM_FRONT',
        'num_sequences': len(sequences),
        'num_frames': num_frames,
        'total_objects': int(total_objects),
        'metrics': {
            'MOTA': float(mota_overall * 100),
            'MOTP': float(motp_overall),
            'IDF1': float(idf1_overall * 100),
            'IDSW': num_switches,
            'Frag': num_frag,
            'FP': num_fp,
            'FN': num_fn,
            'MT': num_mt,
            'ML': num_ml,
            'Precision': float(precision_overall * 100),
            'Recall': float(recall_overall * 100)
        }
    }
    
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"MOTA:      {results['metrics']['MOTA']:.2f}%")
    print(f"IDF1:      {results['metrics']['IDF1']:.2f}%")
    print(f"MOTP:      {results['metrics']['MOTP']:.4f}")
    print(f"Precision: {results['metrics']['Precision']:.2f}%")
    print(f"Recall:    {results['metrics']['Recall']:.2f}%")
    print(f"IDSW:      {results['metrics']['IDSW']}")
    print(f"Frag:      {results['metrics']['Frag']}")
    print(f"FP:        {results['metrics']['FP']}")
    print(f"FN:        {results['metrics']['FN']}")
    print(f"MT:        {results['metrics']['MT']}")
    print(f"ML:        {results['metrics']['ML']}")
    print("="*70)
    
    # Verify consistency
    print("\nVERIFYING CONSISTENCY...")
    tp_from_precision = results['metrics']['Precision'] / 100 * num_fp / (1 - results['metrics']['Precision'] / 100)
    tp_from_recall = results['metrics']['Recall'] / 100 * num_fn / (1 - results['metrics']['Recall'] / 100)
    
    print(f"TP from Precision: {tp_from_precision:.0f}")
    print(f"TP from Recall:    {tp_from_recall:.0f}")
    print(f"Difference:        {abs(tp_from_precision - tp_from_recall):.0f}")
    
    if abs(tp_from_precision - tp_from_recall) < 100:
        print("✓ Metrics are CONSISTENT!")
    else:
        print("⚠️  WARNING: Metrics are INCONSISTENT!")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth directory')
    parser.add_argument('--pred_dir', type=str, required=True, help='Predictions directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    
    results = compute_overall_metrics(args.gt_dir, args.pred_dir)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Results saved to: {output_path}")
