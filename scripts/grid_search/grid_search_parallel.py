#!/usr/bin/env python3
"""
Parallel Grid Search for TrackSSM Hyperparameters
Runs 4 experiments in parallel, tracks best configuration in real-time
"""

import json
import subprocess
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import argparse

# Grid search space
SEARCH_SPACE = {
    'conf_thresh': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    'match_thresh': [0.80, 0.85, 0.90, 0.95],
    'track_thresh': [0.5, 0.6, 0.7, 0.8],
    'nms_thresh': [0.5, 0.6, 0.65, 0.7]
}

# Fixed parameters
DETECTOR_WEIGHT = "yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth"  # Will be set by argparse
GT_DATA = "data/nuscenes_mot_front/val"
BASE_OUTPUT = "results/GRID_SEARCH"
BEST_CONFIG_FILE = "results/GRID_SEARCH/best_config.json"
PROGRESS_FILE = "results/GRID_SEARCH/progress.json"

# Scoring function weights
WEIGHTS = {
    'MOTA': 0.35,
    'IDF1': 0.30,
    'HOTA': 0.25,
    'IDSW': -0.10  # Negative because lower is better
}

def compute_score(metrics):
    """Compute composite score for ranking configurations"""
    try:
        mota = metrics.get('MOTA', 0)
        idf1 = metrics.get('IDF1', 0)
        hota = metrics.get('HOTA', 0)
        idsw = metrics.get('IDSW', 10000)
        
        # Normalize IDSW (baseline ~3000, lower is better)
        idsw_norm = max(0, 100 - (idsw / 30))  # 3000 IDSW = 0 points, 0 IDSW = 100 points
        
        score = (
            WEIGHTS['MOTA'] * mota +
            WEIGHTS['IDF1'] * idf1 +
            WEIGHTS['HOTA'] * hota +
            WEIGHTS['IDSW'] * idsw_norm
        )
        return score
    except Exception as e:
        print(f"Error computing score: {e}")
        return -1000

def run_experiment(config, experiment_id, detector_weight):
    """Run a single tracking experiment with given hyperparameters"""
    conf_thresh = config['conf_thresh']
    match_thresh = config['match_thresh']
    track_thresh = config['track_thresh']
    nms_thresh = config['nms_thresh']
    
    # Create output directory
    output_dir = f"{BASE_OUTPUT}/exp_{experiment_id:04d}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = f"{output_dir}/config.json"
    with open(config_file, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'config': config,
            'detector': detector_weight,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Run tracking
    cmd = [
        'python', 'track.py',
        '--tracker', 'trackssm',
        '--detector-weights', detector_weight,
        '--conf-thresh', str(conf_thresh),
        '--nms-thresh', str(nms_thresh),
        '--match-thresh', str(match_thresh),
        '--track-thresh', str(track_thresh),
        '--output', output_dir,
        '--gt-data', GT_DATA
    ]
    
    log_file = f"{output_dir}/tracking.log"
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd='/user/amarino/tesi_project_amarino',
                env={**os.environ, 'TRITON_CACHE_DIR': '/user/amarino/.triton_cache'},
                timeout=1800  # 30 min timeout
            )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Exp {experiment_id} failed (exit code {result.returncode})")
            return None
        
        # Run evaluation
        metrics_file = f"{output_dir}/metrics.json"
        eval_cmd = [
            'python', 'scripts/evaluation/evaluate_motmetrics.py',
            '--pred-folder', f"{output_dir}/data",
            '--output', metrics_file
        ]
        
        subprocess.run(
            eval_cmd,
            cwd='/user/amarino/tesi_project_amarino',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        
        # Read metrics
        if not os.path.exists(metrics_file):
            print(f"⚠️  Exp {experiment_id} - metrics file not found")
            return None
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            metrics = metrics_data.get('summary', {})
        
        # Compute score
        score = compute_score(metrics)
        
        result_data = {
            'experiment_id': experiment_id,
            'config': config,
            'metrics': metrics,
            'score': score,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Exp {experiment_id:04d} | Score: {score:.2f} | MOTA: {metrics.get('MOTA', 0):.1f}% | "
              f"IDF1: {metrics.get('IDF1', 0):.1f}% | IDSW: {metrics.get('IDSW', 0)} | "
              f"Time: {duration/60:.1f}min")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Exp {experiment_id} timeout (30min)")
        return None
    except Exception as e:
        print(f"❌ Exp {experiment_id} error: {e}")
        return None

def update_best_config(result_data):
    """Update best configuration file if this result is better"""
    if result_data is None:
        return
    
    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # Read current best
    current_best = None
    if os.path.exists(BEST_CONFIG_FILE):
        try:
            with open(BEST_CONFIG_FILE, 'r') as f:
                current_best = json.load(f)
        except:
            pass
    
    # Compare scores
    new_score = result_data['score']
    is_new_best = False
    
    if current_best is None:
        is_new_best = True
    else:
        current_score = current_best.get('score', -1000)
        if new_score > current_score:
            is_new_best = True
    
    # Update if better
    if is_new_best:
        with open(BEST_CONFIG_FILE, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"🏆 NEW BEST CONFIGURATION! (Exp {result_data['experiment_id']:04d})")
        print(f"{'='*80}")
        print(f"Score: {new_score:.2f}")
        print(f"Config: {json.dumps(result_data['config'], indent=2)}")
        print(f"Metrics:")
        print(f"  MOTA:  {result_data['metrics'].get('MOTA', 0):.2f}%")
        print(f"  IDF1:  {result_data['metrics'].get('IDF1', 0):.2f}%")
        print(f"  HOTA:  {result_data['metrics'].get('HOTA', 0):.2f}%")
        print(f"  IDSW:  {result_data['metrics'].get('IDSW', 0)}")
        print(f"  Recall: {result_data['metrics'].get('RECALL', 0):.2f}%")
        print(f"  Prec:   {result_data['metrics'].get('precision', 0):.2f}%")
        print(f"{'='*80}\n")

def update_progress(completed, total, active_experiments):
    """Update progress file"""
    progress_data = {
        'completed': completed,
        'total': total,
        'progress_pct': (completed / total * 100) if total > 0 else 0,
        'active_experiments': active_experiments,
        'timestamp': datetime.now().isoformat()
    }
    
    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)

def generate_experiment_configs(search_space):
    """Generate all possible configurations from search space"""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    
    configs = []
    for i, combination in enumerate(itertools.product(*values), start=1):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs

def run_parallel_grid_search(detector_weight, num_workers=4, max_experiments=None):
    """Run grid search with parallel workers"""
    
    print(f"\n{'='*80}")
    print(f"PARALLEL GRID SEARCH - TrackSSM Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Detector: {detector_weight}")
    print(f"Workers: {num_workers} parallel experiments")
    print(f"Search Space:")
    for param, values in SEARCH_SPACE.items():
        print(f"  {param}: {values} ({len(values)} values)")
    
    # Generate all configurations
    all_configs = generate_experiment_configs(SEARCH_SPACE)
    
    if max_experiments:
        all_configs = all_configs[:max_experiments]
    
    total_experiments = len(all_configs)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Estimated time: {total_experiments * 3 / num_workers / 60:.1f} hours (3min/exp avg)")
    print(f"{'='*80}\n")
    
    # Initialize results
    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    
    completed = 0
    results = []
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch
        futures = {}
        for i, config in enumerate(all_configs[:num_workers], start=1):
            future = executor.submit(run_experiment, config, i, detector_weight)
            futures[future] = i
        
        # Process completed and submit new
        next_exp_id = num_workers + 1
        
        while futures:
            # Wait for any to complete
            done_iter = as_completed(futures)
            
            for future in done_iter:
                exp_id = futures[future]
                result = future.result()
                
                if result:
                    results.append(result)
                    update_best_config(result)
                
                completed += 1
                del futures[future]
                
                # Update progress
                active_ids = [futures[f] for f in futures]
                update_progress(completed, total_experiments, active_ids)
                
                print(f"\n[Progress: {completed}/{total_experiments} = {completed/total_experiments*100:.1f}%]")
                
                # Submit next experiment if available
                if next_exp_id <= total_experiments:
                    config = all_configs[next_exp_id - 1]
                    new_future = executor.submit(run_experiment, config, next_exp_id, detector_weight)
                    futures[new_future] = next_exp_id
                    print(f"🚀 Started Exp {next_exp_id:04d}")
                    next_exp_id += 1
                
                break  # Process one at a time
    
    # Save all results
    results_file = f"{BASE_OUTPUT}/all_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'total_experiments': total_experiments,
            'completed': completed,
            'results': sorted(results, key=lambda x: x['score'], reverse=True),
            'search_space': SEARCH_SPACE,
            'detector': detector_weight,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Results saved to: {results_file}")
    print(f"Best config saved to: {BEST_CONFIG_FILE}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel Grid Search for TrackSSM')
    parser.add_argument('--detector', type=str, required=True, help='Path to detector weights')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--max-experiments', type=int, default=None, help='Limit number of experiments')
    
    args = parser.parse_args()
    
    run_parallel_grid_search(
        detector_weight=args.detector,
        num_workers=args.workers,
        max_experiments=args.max_experiments
    )
