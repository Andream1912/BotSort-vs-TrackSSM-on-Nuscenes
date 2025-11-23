#!/usr/bin/env python3
"""
Unified Evaluation Script - Compute HOTA metrics for any tracker

Usage:
    python evaluate.py --gt data/nuscenes_mot_front/val --results results/trackssm --output metrics/trackssm.json
    python evaluate.py --gt data/nuscenes_mot_front/val --results results/botsort --output metrics/botsort.json
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add TrackEval to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'external', 'TrackEval'))

import trackeval


def parse_args():
    parser = argparse.ArgumentParser(description='Unified tracker evaluation')
    
    parser.add_argument('--gt', type=str, required=True,
                       help='Path to GT directory (e.g., data/nuscenes_mot_front/val)')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to tracking results directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for metrics')
    parser.add_argument('--seqmap', type=str, default='seqmaps/val.txt',
                       help='Sequence map file (default: seqmaps/val.txt)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--per-class', action='store_true',
                       help='Compute per-class metrics (requires with_classes/ directory)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment config JSON file')
    
    return parser.parse_args()


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
