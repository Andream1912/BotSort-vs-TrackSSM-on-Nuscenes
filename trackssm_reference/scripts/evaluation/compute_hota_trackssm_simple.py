#!/usr/bin/env python3
"""
Compute HOTA metrics for TrackSSM using TrackEval library.
Simplified version - data is already in MOTChallenge format.
"""

import os
import sys
import json
import shutil
from pathlib import Path

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

def prepare_tracker_data(pred_dir, output_dir):
    """
    Copy tracker results to TrackEval expected location.
    
    Structure needed:
    output_dir/trackers/mot_challenge/nuscenes-val/trackssm/data/*.txt
    """
    pred_dir = Path(pred_dir)
    tracker_data_dir = output_dir / 'trackers' / 'mot_challenge' / 'nuscenes-val' / 'trackssm' / 'data'
    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Preparing tracker data...")
    
    # Copy all prediction files
    count = 0
    for pred_file in pred_dir.glob('*.txt'):
        dest = tracker_data_dir / pred_file.name
        shutil.copy2(pred_file, dest)
        count += 1
    
    print(f"   ✓ Copied {count} tracker files")
    return count

def filter_by_class(input_dir, output_dir, class_id):
    """Filter sequences by class ID."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Filtering for class: {CLASS_MAPPING.get(class_id, 'unknown')}")
    
    count = 0
    for seq_dir in input_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        
        gt_file = seq_dir / 'gt' / 'gt.txt'
        if not gt_file.exists():
            continue
        
        # Read and filter GT
        filtered_lines = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    cls = int(parts[7])
                    if cls == class_id:
                        filtered_lines.append(line)
        
        if not filtered_lines:
            continue
        
        # Create output sequence directory
        out_seq_dir = output_dir / seq_dir.name
        out_gt_dir = out_seq_dir / 'gt'
        out_gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Write filtered GT
        with open(out_gt_dir / 'gt.txt', 'w') as f:
            f.writelines(filtered_lines)
        
        # Copy seqinfo.ini
        if (seq_dir / 'seqinfo.ini').exists():
            shutil.copy2(seq_dir / 'seqinfo.ini', out_seq_dir / 'seqinfo.ini')
        
        count += 1
    
    print(f"   ✓ Filtered {count} sequences")
    return count

def filter_tracker_by_class(tracker_dir, class_id):
    """Filter tracker files by class ID."""
    tracker_dir = Path(tracker_dir)
    
    print(f"📝 Filtering tracker files for class {CLASS_MAPPING.get(class_id, 'unknown')}...")
    
    count = 0
    for txt_file in tracker_dir.glob('*.txt'):
        # Read and filter
        filtered_lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    cls = int(parts[7])
                    if cls == class_id:
                        filtered_lines.append(line)
        
        # Overwrite with filtered data
        with open(txt_file, 'w') as f:
            f.writelines(filtered_lines)
        
        if filtered_lines:
            count += 1
    
    print(f"   ✓ Filtered {count} tracker files")
    return count

def compute_hota(gt_dir, tracker_dir, output_dir, benchmark='nuscenes-val'):
    """Run TrackEval to compute HOTA."""
    
    print("\n🔄 Running TrackEval HOTA computation...")
    
    # Get absolute paths
    gt_dir = Path(gt_dir).absolute()
    tracker_dir = Path(tracker_dir).absolute()
    output_dir = Path(output_dir).absolute()
    seqmap_dir = gt_dir.parent / 'seqmaps'
    
    print(f"   GT folder: {gt_dir}")
    print(f"   Tracker folder: {tracker_dir}")
    print(f"   Seqmap folder: {seqmap_dir}")
    print(f"   Seqmap file: {benchmark}.txt")
    
    # Configure evaluator
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': False,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': str(output_dir / 'error.log'),
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    
    # Configure dataset
    dataset_config = {
        'GT_FOLDER': str(gt_dir.parent),  # Go up one level for mot_challenge structure
        'TRACKERS_FOLDER': str(tracker_dir.parent),
        'OUTPUT_FOLDER': str(output_dir),
        'TRACKERS_TO_EVAL': ['trackssm'],
        'CLASSES_TO_EVAL': ['pedestrian'],  # MOTChallenge format requirement
        'BENCHMARK': benchmark,
        'SPLIT_TO_EVAL': gt_dir.name,  # Use 'val' as split
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': str(seqmap_dir),
        'SEQMAP_FILE': benchmark + '.txt',
        'SEQ_INFO': None,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': False,
    }
    
    try:
        # Initialize
        evaluator = Evaluator(eval_config)
        dataset_list = [MotChallenge2DBox(dataset_config)]
        metrics_list = [HOTA(), CLEAR(), Identity()]
        
        # Run evaluation
        print()
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        
        print("\n✅ TrackEval completed")
        return output_res, output_msg
        
    except Exception as e:
        print(f"\n❌ Error running TrackEval: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def extract_metrics(results):
    """Extract metrics from TrackEval results."""
    if not results:
        return None
    
    try:
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
            'IDSW': tracker_results.get('Identity', {}).get('IDSW', 0),
            'FP': tracker_results.get('CLEAR', {}).get('FP', 0),
            'FN': tracker_results.get('CLEAR', {}).get('FN', 0),
        }
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Could not extract metrics: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute HOTA for TrackSSM')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='GT directory in MOTChallenge format')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Predictions directory with .txt files')
    parser.add_argument('--output', type=str, default='results/final_evaluation/trackssm_hota.json',
                       help='Output JSON file')
    parser.add_argument('--per_class', action='store_true',
                       help='Compute HOTA per class')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRACKSSM HOTA COMPUTATION USING TRACKEVAL")
    print("="*80)
    
    all_results = {}
    
    if args.per_class:
        print("\n📊 MODE: Per-class HOTA")
        
        for cls_id, cls_name in CLASS_MAPPING.items():
            print(f"\n{'='*80}")
            print(f"CLASS: {cls_name.upper()} (ID: {cls_id})")
            print(f"{'='*80}")
            
            # Create temp directories
            temp_dir = Path(f'results/trackeval_temp_{cls_name}')
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_gt = temp_dir / 'gt' / 'mot_challenge' / 'nuscenes-val'
            temp_trackers = temp_dir / 'trackers' / 'mot_challenge' / 'nuscenes-val'
            temp_output = temp_dir / 'output'
            
            try:
                # Filter GT by class
                num_seqs = filter_by_class(args.gt_dir, temp_gt, cls_id)
                if num_seqs == 0:
                    print(f"   ⚠️  No sequences for {cls_name}, skipping...")
                    continue
                
                # Prepare tracker data
                num_files = prepare_tracker_data(args.pred_dir, temp_dir)
                if num_files == 0:
                    print(f"   ⚠️  No tracker files, skipping...")
                    continue
                
                # Filter tracker by class
                tracker_data_dir = temp_trackers / 'trackssm' / 'data'
                filter_tracker_by_class(tracker_data_dir, cls_id)
                
                # Compute HOTA
                results, messages = compute_hota(
                    temp_gt.parent,  # mot_challenge level
                    temp_trackers.parent,  # mot_challenge level  
                    temp_output,
                    benchmark='nuscenes-val'
                )
                
                if results:
                    metrics = extract_metrics(results)
                    if metrics:
                        all_results[cls_name] = metrics
                        print(f"\n   ✅ {cls_name}: HOTA = {metrics['HOTA']:.2f}%, DetA = {metrics['DetA']:.2f}%, AssA = {metrics['AssA']:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    else:
        print("\n📊 MODE: Overall HOTA (all classes)")
        
        temp_dir = Path('results/trackeval_temp_overall')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare tracker data
            num_files = prepare_tracker_data(args.pred_dir, temp_dir)
            if num_files == 0:
                print("   ❌ No tracker files found!")
                return
            
            # Setup paths for TrackEval
            gt_base = Path(args.gt_dir).absolute().parent  # Should be at mot_challenge level
            tracker_base = (temp_dir / 'trackers' / 'mot_challenge').absolute()
            output_dir = (temp_dir / 'output').absolute()
            
            # Compute HOTA
            results, messages = compute_hota(args.gt_dir, temp_dir / 'trackers' / 'mot_challenge', output_dir)
            
            if results:
                metrics = extract_metrics(results)
                if metrics:
                    all_results['overall'] = metrics
                    print(f"\n✅ Overall: HOTA = {metrics['HOTA']:.2f}%, DetA = {metrics['DetA']:.2f}%, AssA = {metrics['AssA']:.2f}%")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ HOTA COMPUTATION COMPLETE!")
        print("="*80)
        print(f"\n📄 Results saved to: {output_path}\n")
        print("📊 Summary:")
        for key, metrics in all_results.items():
            print(f"   {key:15s}: HOTA = {metrics['HOTA']:6.2f}%, "
                  f"DetA = {metrics['DetA']:6.2f}%, "
                  f"AssA = {metrics['AssA']:6.2f}%")
        print()
    else:
        print("\n❌ No results computed")

if __name__ == '__main__':
    main()
