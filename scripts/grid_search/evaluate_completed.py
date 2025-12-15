#!/usr/bin/env python3
"""
Evaluate all completed tracking experiments
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.mot_evaluator import NuScenesMultiClassEvaluator

def main():
    # Find experiments to evaluate
    base_dir = Path('results/GRID_SEARCH')
    to_evaluate = []
    
    for exp_dir in sorted(base_dir.glob('exp_*')):
        metrics_file = exp_dir / 'metrics.json'
        data_dir = exp_dir / 'data'
        
        if not metrics_file.exists() and data_dir.exists():
            txt_files = list(data_dir.glob('*.txt'))
            if len(txt_files) >= 150:
                to_evaluate.append(exp_dir)
    
    print(f"📊 Found {len(to_evaluate)} experiments to evaluate")
    
    if not to_evaluate:
        print("✅ All experiments already evaluated!")
        return
    
    # Evaluate each
    evaluator = NuScenesMultiClassEvaluator(
        gt_folder='data/nuscenes_mot_front/val',
        pred_folder='results/GRID_SEARCH/exp_0001/data',  # Dummy, will change per-experiment
        iou_threshold=0.5
    )
    
    for i, exp_dir in enumerate(to_evaluate, 1):
        print(f"\n[{i}/{len(to_evaluate)}] Evaluating {exp_dir.name}...")
        
        try:
            # Update pred folder
            evaluator.pred_folder = str(exp_dir / 'data')
            
            # Get scene list
            scene_list = sorted([f.stem for f in (exp_dir / 'data').glob('*.txt')])
            
            # Evaluate
            results = evaluator.evaluate_all(scene_list=scene_list)
            
            # Save metrics
            metrics_file = exp_dir / 'metrics.json'
            evaluator.save_results(results, metrics_file)
            
            print(f"  ✓ {exp_dir.name}: MOTA={results.get('MOTA', 0):.2f}%, IDSW={results.get('IDSW', 0)}")
            
        except Exception as e:
            print(f"  ✗ {exp_dir.name}: {e}")
            continue
    
    print(f"\n✅ Evaluation completed!")

if __name__ == '__main__':
    main()
