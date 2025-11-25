#!/usr/bin/env python3
"""
Multi-Class MOT Evaluation Script

Supports NuScenes 7-class schema:
- 1: car, 2: truck, 3: bus, 4: trailer, 5: pedestrian, 6: motorcycle, 7: bicycle

Generates aggregate + per-class metrics (MOTA, Precision, Recall, IDSW, etc.)

Usage:
    python evaluate.py --gt data/nuscenes_mot_front_7classes/val --results results/trackssm/data --output metrics/trackssm.json
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation.mot_evaluator import NuScenesMultiClassEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class MOT evaluation')
    
    parser.add_argument('--gt', type=str, required=True,
                       help='Path to GT directory (e.g., data/nuscenes_mot_front_7classes/val)')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to tracking results directory (expects data/ subfolder)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for metrics')
    parser.add_argument('--seqmap', type=str, default=None,
                       help='Sequence map file (optional, will evaluate all if not provided)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment config JSON file (for metadata)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("MULTI-CLASS MOT EVALUATION")
    print("="*80)
    print(f"GT folder:      {args.gt}")
    print(f"Results folder: {args.results}")
    print(f"IoU threshold:  {args.iou_threshold}")
    print()
    
    # Ensure results path points to data/ subfolder
    results_path = Path(args.results)
    if not (results_path / 'data').exists() and results_path.name != 'data':
        results_path = results_path / 'data'
    
    print(f"Using predictions from: {results_path}")
    
    # Create evaluator
    evaluator = NuScenesMultiClassEvaluator(
        gt_folder=args.gt,
        pred_folder=str(results_path),
        iou_threshold=args.iou_threshold
    )
    
    # Load scene list if provided
    scene_list = None
    if args.seqmap and Path(args.seqmap).exists():
        with open(args.seqmap) as f:
            lines = f.readlines()[1:]  # Skip header
            scene_list = [line.strip() for line in lines if line.strip()]
        print(f"Evaluating {len(scene_list)} scenes from seqmap")
    
    # Run evaluation
    results = evaluator.evaluate_all(scene_list=scene_list)
    
    # Add experiment config if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            results['experiment_config'] = json.load(f)
    
    # Save results
    output_path = Path(args.output)
    evaluator.save_results(results, output_path)
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nMOTA:      {summary['mota']*100:>6.2f}%")
    print(f"Precision: {summary['precision']*100:>6.2f}%")
    print(f"Recall:    {summary['recall']*100:>6.2f}%")
    print(f"IDSW:      {summary['idsw']:>6d}")
    print(f"\nTP:  {summary['tp']:>6d}")
    print(f"FP:  {summary['fp']:>6d}")
    print(f"FN:  {summary['fn']:>6d}")
    print(f"\nGT detections:    {summary['gt_count']:>6d}")
    print(f"Pred detections:  {summary['pred_count']:>6d}")
    print(f"GT IDs:           {summary['gt_ids']:>6d}")
    print(f"Pred IDs:         {summary['pred_ids']:>6d}")
    
    # Per-class summary
    if 'class_metrics' in summary:
        print(f"\n{'='*80}")
        print("PER-CLASS METRICS")
        print(f"{'='*80}")
        print(f"{'Class':<15} {'GT':<8} {'Pred':<8} {'TP':<8} {'MOTA%':<10} {'Recall%':<10}")
        print(f"{'-'*80}")
        
        for cls_id, class_data in sorted(summary['class_metrics'].items()):
            class_name = class_data.get('class_name', f'class-{cls_id}')
            if class_data['gt'] > 0:  # Only show classes present in GT
                print(f"{class_name:<15} {class_data['gt']:<8} {class_data['pred']:<8} "
                      f"{class_data['tp']:<8} {class_data['mota']*100:<10.2f} {class_data['recall']*100:<10.2f}")
    
    print(f"\n✓ Full results saved to: {output_path}")
    
    # Create summary file
    summary_file = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_file, 'w') as f:
        # Extract key metrics for compatibility
        summary_compact = {
            'MOTA': summary['mota'],
            'Precision': summary['precision'],
            'Recall': summary['recall'],
            'IDSW': summary['idsw'],
            'FP': summary['fp'],
            'FN': summary['fn'],
            'Total_GT_IDs': summary['gt_ids'],
            'Total_Predicted_IDs': summary['pred_ids'],
            'Total_GT_Dets': summary['gt_count'],
            'Total_Predicted_Dets': summary['pred_count'],
        }
        json.dump(summary_compact, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()


def create_seqmap(gt_dir, output_file):
    """Create seqmap file from GT directory"""
    gt_path = Path(gt_dir)
    scenes = sorted([d.name for d in gt_path.iterdir() if d.is_dir() and (d / 'gt' / 'gt.txt').exists()])
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("name\n")
        for scene in scenes:
            f.write(f"{scene}\n")
    
    print(f"✓ Created seqmap with {len(scenes)} scenes: {output_file}")
    return scenes


def evaluate_tracking(gt_dir, results_dir, seqmap_file, output_file, quiet=False, config_file=None):
    """Evaluate tracking results using TrackEval"""
    
    # Load experiment config if provided
    experiment_config = None
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            experiment_config = json.load(f)
    
    # Ensure seqmap exists
    if not Path(seqmap_file).exists():
        print(f"Seqmap not found, creating: {seqmap_file}")
        create_seqmap(gt_dir, seqmap_file)
    
    # TrackEval configuration
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    
    # Dataset configuration (MOT format)
    dataset_config = {
        'GT_FOLDER': gt_dir,
        'TRACKERS_FOLDER': os.path.dirname(results_dir),
        'TRACKERS_TO_EVAL': [os.path.basename(results_dir)],
        'BENCHMARK': 'MOT17',  # Use MOT17 format
        'SPLIT_TO_EVAL': 'train',  # Doesn't matter, we use custom seqmap
        'SEQ_INFO': None,
        'SEQMAP_FILE': seqmap_file,
        'SKIP_SPLIT_FOL': True,
        # CRITICAL: Evaluate all NuScenes 7 classes with correct ID mapping
        # NuScenes: 1=car, 2=truck, 3=bus, 4=trailer, 5=pedestrian, 6=motorcycle, 7=bicycle
        'CLASSES_TO_EVAL': ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle'],
        'DISTRACTOR_CLASSES': [],  # Don't ignore any class
    }
    
    # Metrics to compute
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]
    
    # Create evaluator
    evaluator = trackeval.Evaluator(eval_config)
    
    # Load dataset
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    # Run evaluation
    print("\n" + "="*80)
    print("Running TrackEval...")
    print("="*80)
    
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    # Extract and save metrics
    tracker_name = os.path.basename(results_dir)
    
    if output_res and 'MotChallenge2DBox' in output_res:
        mot_results = output_res['MotChallenge2DBox']
        
        # Auto-detect tracker name if not found
        if tracker_name not in mot_results:
            available_trackers = list(mot_results.keys())
            if available_trackers:
                tracker_name = available_trackers[0]
        
        if tracker_name in mot_results:
            tracker_results = mot_results[tracker_name]
            
            # Aggregate metrics across all sequences
            aggregated = {}
            
            if 'COMBINED_SEQ' in tracker_results:
                combined = tracker_results['COMBINED_SEQ']
                
                # Extract key metrics for each class (usually just 'pedestrian')
                for class_name in combined:
                    class_metrics = combined[class_name]
                    for metric_class in ['HOTA', 'CLEAR', 'Identity', 'Count']:
                        if metric_class in class_metrics:
                            # Convert numpy arrays to lists/floats for JSON serialization
                            metrics_dict = {}
                            for key, value in class_metrics[metric_class].items():
                                if hasattr(value, 'tolist'):  # numpy array
                                    metrics_dict[key] = value.tolist() if value.ndim > 0 else float(value)
                                elif hasattr(value, 'item'):  # numpy scalar
                                    metrics_dict[key] = value.item()
                                else:
                                    metrics_dict[key] = value
                            aggregated[metric_class] = metrics_dict
            
            # Save to JSON if output_file is specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save full metrics
                with open(output_file, 'w') as f:
                    json.dump(aggregated, f, indent=2)
                
                # Save summary with main scalar values
                summary = {}
                
                # Main tracking metrics
                if 'HOTA' in aggregated and 'HOTA' in aggregated['HOTA']:
                    hota_val = aggregated['HOTA']['HOTA']
                    summary['HOTA'] = hota_val[0] if isinstance(hota_val, list) else hota_val
                
                if 'CLEAR' in aggregated:
                    # MOTA
                    if 'MOTA' in aggregated['CLEAR']:
                        summary['MOTA'] = aggregated['CLEAR']['MOTA']
                    # ID Switches - QUESTO È IL CAMPO CORRETTO!
                    if 'IDSW' in aggregated['CLEAR']:
                        summary['IDSW'] = int(aggregated['CLEAR']['IDSW'])
                    # False Positives
                    if 'CLR_FP' in aggregated['CLEAR']:
                        summary['FP'] = int(aggregated['CLEAR']['CLR_FP'])
                    # False Negatives
                    if 'CLR_FN' in aggregated['CLEAR']:
                        summary['FN'] = int(aggregated['CLEAR']['CLR_FN'])
                    # Precision
                    if 'CLR_Pr' in aggregated['CLEAR']:
                        summary['Precision'] = aggregated['CLEAR']['CLR_Pr']
                    # Recall
                    if 'CLR_Re' in aggregated['CLEAR']:
                        summary['Recall'] = aggregated['CLEAR']['CLR_Re']
                    # MOTP
                    if 'MOTP' in aggregated['CLEAR']:
                        summary['MOTP'] = aggregated['CLEAR']['MOTP']
                
                if 'Identity' in aggregated:
                    # IDF1
                    if 'IDF1' in aggregated['Identity']:
                        summary['IDF1'] = aggregated['Identity']['IDF1']
                    # ID Recall
                    if 'IDR' in aggregated['Identity']:
                        summary['IDR'] = aggregated['Identity']['IDR']
                    # ID Precision
                    if 'IDP' in aggregated['Identity']:
                        summary['IDP'] = aggregated['Identity']['IDP']
                
                if 'Count' in aggregated:
                    # Total GT Objects
                    if 'GT_IDs' in aggregated['Count']:
                        summary['Total_GT_IDs'] = int(aggregated['Count']['GT_IDs'])
                    # Total Predicted IDs
                    if 'IDs' in aggregated['Count']:
                        summary['Total_Predicted_IDs'] = int(aggregated['Count']['IDs'])
                    # Total GT Detections
                    if 'GT_Dets' in aggregated['Count']:
                        summary['Total_GT_Dets'] = int(aggregated['Count']['GT_Dets'])
                    # Total Predicted Detections
                    if 'Dets' in aggregated['Count']:
                        summary['Total_Predicted_Dets'] = int(aggregated['Count']['Dets'])
                
                # Add number of frames (can be calculated from GT_Dets / objects per frame)
                # This is an approximation - actual frame count would need sequence info
                summary['Note'] = "Metrics computed on all evaluated sequences"
                
                # Include experiment configuration if available
                if experiment_config:
                    summary['experiment_config'] = experiment_config
                
                # Save summary
                summary_file = output_path.parent / f"{output_path.stem}_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n✓ Metrics saved to: {output_file}")
                print(f"✓ Summary saved to: {summary_file}")
            
            # Print summary if not quiet mode
            if not quiet and aggregated:
                print("\n" + "="*80)
                print("SUMMARY METRICS")
                print("="*80)
                
                if 'HOTA' in aggregated:
                    print(f"\nHOTA Metrics:")
                    for key, value in aggregated['HOTA'].items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                
                if 'CLEAR' in aggregated:
                    print(f"\nCLEAR Metrics (MOTA, IDF1):")
                    for key in ['MOTA', 'MOTP', 'MT', 'ML', 'FP', 'FN', 'IDs']:
                        if key in aggregated['CLEAR']:
                            value = aggregated['CLEAR'][key]
                            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                
                if 'Identity' in aggregated:
                    print(f"\nIdentity Metrics:")
                    for key in ['IDF1', 'IDR', 'IDP']:
                        if key in aggregated['Identity']:
                            print(f"  {key}: {aggregated['Identity'][key]:.4f}")
            
            return aggregated
    
    print("⚠️  No results found in evaluation output")
    return None


def evaluate_per_class(gt_dir, results_dir, seqmap_file):
    """
    Evaluate tracking metrics per class
    
    Reads tracking results with class information and computes metrics
    for each class separately
    """
    print("\nComputing per-class metrics...")
    
    # Read GT and results to identify classes
    from collections import defaultdict
    
    gt_path = Path(gt_dir)
    results_path = Path(results_dir)
    
    # Find all unique classes in GT
    classes_in_gt = set()
    for scene_dir in gt_path.iterdir():
        if not scene_dir.is_dir():
            continue
        gt_file = scene_dir / 'gt' / 'gt.txt'
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        classes_in_gt.add(int(parts[7]))
    
    # NuScenes class mapping
    CLASS_NAMES = {
        1: 'pedestrian',
        2: 'car',
        3: 'truck',
        4: 'bus',
        5: 'motorcycle',
        6: 'bicycle',
        7: 'trailer'
    }
    
    print(f"  Found {len(classes_in_gt)} classes in GT: {sorted(classes_in_gt)}")
    
    # Compute metrics for each class
    per_class_metrics = {}
    
    for class_id in sorted(classes_in_gt):
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        print(f"\n  Evaluating class {class_id} ({class_name})...")
        
        # Create temporary GT and results directories with only this class
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_gt = Path(temp_dir) / 'gt'
            temp_results = Path(temp_dir) / 'results' / 'data'
            temp_gt.mkdir(parents=True)
            temp_results.mkdir(parents=True)
            
            # Filter GT for this class
            for scene_dir in gt_path.iterdir():
                if not scene_dir.is_dir():
                    continue
                gt_file = scene_dir / 'gt' / 'gt.txt'
                if not gt_file.exists():
                    continue
                
                # Create scene directory in temp GT
                temp_scene_gt = temp_gt / scene_dir.name / 'gt'
                temp_scene_gt.mkdir(parents=True)
                
                # Filter GT lines for this class
                with open(gt_file, 'r') as f_in:
                    with open(temp_scene_gt / 'gt.txt', 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split(',')
                            if len(parts) >= 8 and int(parts[7]) == class_id:
                                # Write with class_id=1 for TrackEval
                                parts[7] = '1'
                                f_out.write(','.join(parts) + '\n')
                
                # Filter results for this class
                result_file = results_path / f'{scene_dir.name}.txt'
                if result_file.exists():
                    with open(result_file, 'r') as f_in:
                        with open(temp_results / f'{scene_dir.name}.txt', 'w') as f_out:
                            for line in f_in:
                                parts = line.strip().split(',')
                                if len(parts) >= 8 and int(parts[7]) == class_id:
                                    # Write with class_id=1 for TrackEval
                                    parts[7] = '1'
                                    f_out.write(','.join(parts) + '\n')
            
            # Create temporary seqmap
            temp_seqmap = Path(temp_dir) / 'seqmap.txt'
            with open(seqmap_file, 'r') as f_in:
                with open(temp_seqmap, 'w') as f_out:
                    f_out.write(f_in.read())
            
            # Run evaluation for this class
            try:
                metrics = evaluate_tracking(
                    gt_dir=str(temp_gt),
                    results_dir=str(Path(temp_dir) / 'results'),
                    seqmap_file=str(temp_seqmap),
                    output_file=None,
                    quiet=True
                )
                
                if metrics:
                    per_class_metrics[class_name] = metrics
                    print(f"    ✓ {class_name}: HOTA={metrics.get('HOTA', {}).get('HOTA', 0):.4f}")
            except Exception as e:
                print(f"    ⚠️  Failed to evaluate {class_name}: {e}")
    
    return per_class_metrics


def main():
    args = parse_args()
    
    print("="*80)
    print("Unified Tracker Evaluation")
    print("="*80)
    print(f"GT Dir: {args.gt}")
    print(f"Results Dir: {args.results}")
    print(f"Output: {args.output}")
    
    # Run overall evaluation (unified class=1)
    print("\n[1/2] Computing overall metrics...")
    metrics_overall = evaluate_tracking(
        gt_dir=args.gt,
        results_dir=args.results,
        seqmap_file=args.seqmap,
        output_file=args.output,
        quiet=args.quiet,
        config_file=args.config
    )
    
    if not metrics_overall:
        print("\n⚠️  Overall evaluation failed")
        sys.exit(1)
    
    # Run per-class evaluation if requested
    if args.per_class:
        print("\n[2/2] Computing per-class metrics...")
        
        # Check if with_classes directory exists
        with_classes_dir = os.path.join(args.results, 'with_classes')
        if not os.path.exists(with_classes_dir):
            print(f"⚠️  Per-class results not found: {with_classes_dir}")
            print("    Skipping per-class evaluation")
        else:
            metrics_per_class = evaluate_per_class(
                gt_dir=args.gt,
                results_dir=with_classes_dir,
                seqmap_file=args.seqmap
            )
            
            # Save per-class metrics
            output_path = Path(args.output)
            per_class_output = output_path.parent / f"{output_path.stem}_per_class.json"
            
            with open(per_class_output, 'w') as f:
                json.dump(metrics_per_class, f, indent=2)
            
            print(f"\n✓ Per-class metrics saved to: {per_class_output}")
            
            # Print summary
            print("\n" + "="*80)
            print("PER-CLASS SUMMARY")
            print("="*80)
            for class_name, metrics in metrics_per_class.items():
                hota = metrics.get('HOTA', {}).get('HOTA', 0)
                mota = metrics.get('CLEAR', {}).get('MOTA', 0)
                idf1 = metrics.get('Identity', {}).get('IDF1', 0)
                print(f"\n{class_name}:")
                print(f"  HOTA: {hota:.4f}")
                print(f"  MOTA: {mota:.4f}")
                print(f"  IDF1: {idf1:.4f}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
