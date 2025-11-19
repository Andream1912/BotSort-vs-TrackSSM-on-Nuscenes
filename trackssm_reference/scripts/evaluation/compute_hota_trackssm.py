#!/usr/bin/env python3
"""
Compute HOTA metrics for TrackSSM using TrackEval library.
This allows fair comparison with BotSort HOTA scores.
"""

import os
import sys
import json
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    import trackeval
except ImportError:
    print("⚠️  TrackEval not installed. Installing...")
    os.system("pip install git+https://github.com/JonathonLuiten/TrackEval.git")
    import trackeval

from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity
from trackeval import Evaluator

# NuScenes 7 classes mapping
CLASS_MAPPING = {
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'trailer',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle'
}

def convert_to_motchallenge_format(input_dir, output_dir, class_filter=None):
    """
    Convert NuScenes MOT format to MOTChallenge format for TrackEval.
    
    Args:
        input_dir: Directory with .txt files in MOT format
        output_dir: Output directory for MOTChallenge format
        class_filter: If specified, only include this class ID
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for txt_file in input_dir.glob('*.txt'):
        seq_name = txt_file.stem
        
        # Read and filter by class if needed
        lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    class_id = int(parts[7]) if len(parts) > 7 else -1
                    
                    if class_filter is None or class_id == class_filter:
                        # Keep line (MOTChallenge format is same as our format)
                        lines.append(line)
        
        # Write output
        if lines:
            out_file = output_dir / f"{seq_name}.txt"
            with open(out_file, 'w') as f:
                f.writelines(lines)

def prepare_trackeval_structure(gt_dir, pred_dir, output_dir, class_filter=None):
    """
    Prepare data structure for TrackEval.
    
    Structure:
    output_dir/
    ├── data/
    │   ├── gt/
    │   │   └── mot_challenge/
    │   │       └── nuscenes-train/
    │   │           └── {seq_name}/
    │   │               ├── gt/
    │   │               │   └── gt.txt
    │   │               └── seqinfo.ini
    │   └── trackers/
    │       └── mot_challenge/
    │           └── nuscenes-train/
    │               └── trackssm/
    │                   └── data/
    │                       └── {seq_name}.txt
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    gt_base = output_dir / 'data' / 'gt' / 'mot_challenge' / 'nuscenes-train'
    tracker_base = output_dir / 'data' / 'trackers' / 'mot_challenge' / 'nuscenes-train' / 'trackssm' / 'data'
    
    gt_base.mkdir(parents=True, exist_ok=True)
    tracker_base.mkdir(parents=True, exist_ok=True)
    
    # Get sequence list
    seqmap_file = output_dir / 'data' / 'gt' / 'mot_challenge' / 'seqmaps' / 'nuscenes-train.txt'
    seqmap_file.parent.mkdir(parents=True, exist_ok=True)
    
    sequences = []
    
    print(f"\n📁 Preparing TrackEval structure...")
    if class_filter:
        print(f"   Filtering for class: {CLASS_MAPPING.get(class_filter, 'unknown')}")
    
    # Process each sequence
    for gt_file in tqdm(sorted(gt_dir.glob('*.txt')), desc="   Processing sequences"):
        seq_name = gt_file.stem
        pred_file = pred_dir / f"{seq_name}.txt"
        
        if not pred_file.exists():
            continue
        
        # Read GT
        gt_lines = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    cls_id = int(parts[7])
                    if class_filter is None or cls_id == class_filter:
                        gt_lines.append(line)
        
        if not gt_lines:
            continue
        
        # Read predictions
        pred_lines = []
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    cls_id = int(parts[7])
                    if class_filter is None or cls_id == class_filter:
                        pred_lines.append(line)
        
        if not pred_lines:
            continue
        
        # Create sequence directory
        seq_gt_dir = gt_base / seq_name / 'gt'
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Write GT file
        with open(seq_gt_dir / 'gt.txt', 'w') as f:
            f.writelines(gt_lines)
        
        # Get sequence info
        frames = [int(line.split(',')[0]) for line in gt_lines]
        num_frames = max(frames) if frames else 0
        
        # Create seqinfo.ini
        seqinfo = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=2
seqLength={num_frames}
imWidth=1600
imHeight=900
imExt=.jpg
"""
        with open(gt_base / seq_name / 'seqinfo.ini', 'w') as f:
            f.write(seqinfo)
        
        # Write tracker file
        with open(tracker_base / f"{seq_name}.txt", 'w') as f:
            f.writelines(pred_lines)
        
        sequences.append(seq_name)
    
    # Write seqmap
    with open(seqmap_file, 'w') as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")
    
    print(f"   ✓ Prepared {len(sequences)} sequences")
    
    return output_dir, len(sequences)

def compute_hota_metrics(eval_dir, benchmark='nuscenes-train'):
    """Run TrackEval to compute HOTA metrics."""
    
    print("\n🔄 Running TrackEval HOTA computation...")
    
    # Configure evaluator
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': False,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': eval_dir / 'error.log',
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    
    # Configure dataset
    dataset_config = {
        'GT_FOLDER': str(eval_dir / 'data' / 'gt' / 'mot_challenge'),
        'TRACKERS_FOLDER': str(eval_dir / 'data' / 'trackers' / 'mot_challenge'),
        'OUTPUT_FOLDER': str(eval_dir / 'results'),
        'TRACKERS_TO_EVAL': ['trackssm'],
        'CLASSES_TO_EVAL': ['pedestrian'],  # Required by MOTChallenge
        'BENCHMARK': benchmark,
        'SPLIT_TO_EVAL': 'train',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': str(eval_dir / 'data' / 'gt' / 'mot_challenge' / 'seqmaps'),
        'SEQMAP_FILE': None,
        'SEQ_INFO': None,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': True,
    }
    
    try:
        # Initialize
        evaluator = Evaluator(eval_config)
        dataset_list = [MotChallenge2DBox(dataset_config)]
        metrics_list = [HOTA(), CLEAR(), Identity()]
        
        # Run evaluation
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        
        print("\n✅ TrackEval completed successfully")
        
        return output_res, output_msg
        
    except Exception as e:
        print(f"\n❌ Error running TrackEval: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def extract_hota_from_results(results):
    """Extract HOTA metrics from TrackEval results."""
    
    if not results:
        return None
    
    try:
        # Navigate TrackEval result structure
        tracker_results = results['MotChallenge2DBox']['trackssm']['COMBINED_SEQ']
        
        metrics = {
            'HOTA': tracker_results.get('HOTA', {}).get('HOTA', 0),
            'DetA': tracker_results.get('HOTA', {}).get('DetA', 0),
            'AssA': tracker_results.get('HOTA', {}).get('AssA', 0),
            'DetRe': tracker_results.get('HOTA', {}).get('DetRe', 0),
            'DetPr': tracker_results.get('HOTA', {}).get('DetPr', 0),
            'AssRe': tracker_results.get('HOTA', {}).get('AssRe', 0),
            'AssPr': tracker_results.get('HOTA', {}).get('AssPr', 0),
            'LocA': tracker_results.get('HOTA', {}).get('LocA', 0),
            'MOTA': tracker_results.get('CLEAR', {}).get('MOTA', 0),
            'MOTP': tracker_results.get('CLEAR', {}).get('MOTP', 0),
            'IDF1': tracker_results.get('Identity', {}).get('IDF1', 0),
            'IDR': tracker_results.get('Identity', {}).get('IDR', 0),
            'IDP': tracker_results.get('Identity', {}).get('IDP', 0),
        }
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Could not extract all metrics: {e}")
        return None

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute HOTA metrics for TrackSSM')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory with ground truth files')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with prediction files')
    parser.add_argument('--output', type=str, default='results/final_evaluation/trackssm_hota_metrics.json',
                       help='Output JSON file')
    parser.add_argument('--per_class', action='store_true',
                       help='Compute HOTA per class')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPUTING HOTA METRICS FOR TRACKSSM USING TRACKEVAL")
    print("="*80)
    
    all_results = {}
    
    if args.per_class:
        print("\n📊 MODE: Per-class HOTA computation")
        
        for cls_id, cls_name in CLASS_MAPPING.items():
            print(f"\n{'='*80}")
            print(f"CLASS: {cls_name.upper()} (ID: {cls_id})")
            print(f"{'='*80}")
            
            # Create evaluation directory for this class
            eval_dir = Path(f'results/trackeval_temp_{cls_name}')
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            try:
                eval_root, num_seqs = prepare_trackeval_structure(
                    args.gt_dir, args.pred_dir, eval_dir, class_filter=cls_id
                )
                
                if num_seqs == 0:
                    print(f"   ⚠️  No sequences found for {cls_name}, skipping...")
                    continue
                
                # Compute HOTA
                results, messages = compute_hota_metrics(eval_root, benchmark=f'nuscenes-{cls_name}')
                
                if results:
                    metrics = extract_hota_from_results(results)
                    if metrics:
                        all_results[cls_name] = metrics
                        print(f"\n   ✅ {cls_name}: HOTA = {metrics['HOTA']:.2f}%")
                    else:
                        print(f"   ⚠️  Could not extract metrics for {cls_name}")
                else:
                    print(f"   ❌ Failed to compute HOTA for {cls_name}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {cls_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Cleanup
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
    
    else:
        print("\n📊 MODE: Overall HOTA computation (all classes combined)")
        
        # Create evaluation directory
        eval_dir = Path('results/trackeval_temp_overall')
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare data
            eval_root, num_seqs = prepare_trackeval_structure(
                args.gt_dir, args.pred_dir, eval_dir, class_filter=None
            )
            
            if num_seqs == 0:
                print("   ❌ No sequences found!")
                return
            
            # Compute HOTA
            results, messages = compute_hota_metrics(eval_root)
            
            if results:
                metrics = extract_hota_from_results(results)
                if metrics:
                    all_results['overall'] = metrics
                    print(f"\n✅ Overall HOTA = {metrics['HOTA']:.2f}%")
                else:
                    print("   ⚠️  Could not extract metrics")
            else:
                print("   ❌ Failed to compute HOTA")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
    
    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ HOTA COMPUTATION COMPLETE!")
        print("="*80)
        print(f"\n📄 Results saved to: {output_path}")
        print(f"\n📊 Summary:")
        for key, metrics in all_results.items():
            if 'HOTA' in metrics:
                print(f"   {key:15s}: HOTA = {metrics['HOTA']:6.2f}%, "
                      f"DetA = {metrics['DetA']:6.2f}%, "
                      f"AssA = {metrics['AssA']:6.2f}%")
        print()
    else:
        print("\n❌ No results to save")

if __name__ == '__main__':
    main()
