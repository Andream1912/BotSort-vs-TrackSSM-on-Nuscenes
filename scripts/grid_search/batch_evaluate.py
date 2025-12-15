#!/usr/bin/env python3
"""
Batch evaluation for grid search experiments using TrackEval
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.mot_evaluator import NuScenesMultiClassEvaluator

def evaluate_experiment(exp_dir, gt_folder='data/nuscenes_mot_front/val'):
    """Evaluate a single experiment"""
    exp_path = Path(exp_dir)
    data_dir = exp_path / 'data'
    metrics_file = exp_path / 'metrics.json'
    
    # Skip if already evaluated
    if metrics_file.exists():
        return None, "already_evaluated"
    
    # Skip if no tracking data
    if not data_dir.exists() or len(list(data_dir.glob('*.txt'))) == 0:
        return None, "no_tracking_data"
    
    try:
        # Get scene list from data directory
        scene_files = list(data_dir.glob('*.txt'))
        scene_list = [f.stem for f in scene_files]
        
        # Create evaluator
        evaluator = NuScenesMultiClassEvaluator(
            gt_folder=gt_folder,
            pred_folder=str(data_dir)
        )
        
        # Run evaluation
        results = evaluator.evaluate_all(scene_list=scene_list)
        
        # Save metrics
        evaluator.save_results(results, metrics_file)
        
        return results, "success"
        
    except Exception as e:
        return None, f"error: {str(e)}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-start', type=int, default=161, help='Start experiment ID')
    parser.add_argument('--exp-end', type=int, default=260, help='End experiment ID')
    parser.add_argument('--gt-folder', type=str, default='data/nuscenes_mot_front/val')
    args = parser.parse_args()
    
    base_dir = Path('results/GRID_SEARCH')
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"Evaluating experiments {args.exp_start:04d} to {args.exp_end:04d}")
    print("=" * 80)
    
    for exp_id in range(args.exp_start, args.exp_end + 1):
        exp_dir = base_dir / f'exp_{exp_id:04d}'
        
        if not exp_dir.exists():
            continue
        
        print(f"\n[{exp_id:04d}] Evaluating...", end=' ')
        result, status = evaluate_experiment(exp_dir, args.gt_folder)
        
        if status == "success":
            mota = result.get('MOTA', 0)
            idf1 = result.get('IDF1', 0)
            idsw = result.get('IDSW', 0)
            print(f"✓ MOTA={mota:.2f}% IDF1={idf1:.2f}% IDSW={idsw}")
            success_count += 1
        elif status == "already_evaluated":
            print("⏭️  (already done)")
            skip_count += 1
        elif status == "no_tracking_data":
            print("⚠️  (no data)")
            skip_count += 1
        else:
            print(f"✗ {status}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print(f"✅ Evaluated: {success_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"✗ Errors: {error_count}")


if __name__ == '__main__':
    main()
