#!/usr/bin/env python3
"""
Simple Parallel Grid Search - Launches processes directly in background
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path
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

GT_DATA = "data/nuscenes_mot_front/val"
BASE_OUTPUT = "results/GRID_SEARCH"
BEST_CONFIG_FILE = f"{BASE_OUTPUT}/best_config.json"

def compute_score(metrics):
    """Compute composite score"""
    mota = metrics.get('MOTA', 0)
    idf1 = metrics.get('IDF1', 0)
    hota = metrics.get('HOTA', 0)
    idsw = metrics.get('IDSW', 10000)
    idsw_norm = max(0, 100 - (idsw / 30))
    return 0.35*mota + 0.30*idf1 + 0.25*hota + 0.10*idsw_norm

def generate_configs():
    """Generate all configurations"""
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    configs = []
    for combination in itertools.product(*values):
        configs.append(dict(zip(keys, combination)))
    return configs

def launch_experiment(config, exp_id, detector):
    """Launch experiment as background process"""
    output_dir = f"{BASE_OUTPUT}/exp_{exp_id:04d}"
    
    # Skip if already completed
    metrics_file = f"{output_dir}/metrics.json"
    if os.path.exists(metrics_file):
        print(f"⏭️  Skipping exp_{exp_id:04d} (already completed)")
        return None, output_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump({'experiment_id': exp_id, 'config': config}, f, indent=2)
    
    # Build command
    cmd = [
        sys.executable, 'track.py',
        '--tracker', 'trackssm',
        '--detector-weights', detector,
        '--conf-thresh', str(config['conf_thresh']),
        '--nms-thresh', str(config['nms_thresh']),
        '--match-thresh', str(config['match_thresh']),
        '--track-thresh', str(config['track_thresh']),
        '--output', output_dir,
        '--gt-data', GT_DATA
    ]
    
    # Launch in background with Triton optimizations
    log_file = f"{output_dir}/tracking.log"
    
    # Set environment variables to speed up Triton (no autotuning)
    env = os.environ.copy()
    env['TRITON_CACHE_DIR'] = os.path.expanduser('~/.triton/cache')
    env['TRITON_INTERPRET'] = '0'
    env['CUDA_LAUNCH_BLOCKING'] = '0'
    
    with open(log_file, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    
    return proc, output_dir

def check_metrics(output_dir):
    """Check if metrics.json exists and return it"""
    metrics_file = f"{output_dir}/metrics.json"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def update_best(config, metrics, exp_id):
    """Update best config if this one is better"""
    score = compute_score(metrics)
    
    best_data = {
        'experiment_id': exp_id,
        'score': score,
        'config': config,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if we should update
    should_update = True
    if os.path.exists(BEST_CONFIG_FILE):
        try:
            with open(BEST_CONFIG_FILE, 'r') as f:
                current_best = json.load(f)
                if current_best.get('score', -9999) >= score:
                    should_update = False
        except:
            pass
    
    if should_update:
        with open(BEST_CONFIG_FILE, 'w') as f:
            json.dump(best_data, f, indent=2)
        print(f"🏆 NEW BEST! Exp {exp_id:04d} - Score: {score:.2f}")
        print(f"   MOTA: {metrics.get('MOTA', 0):.2f}%, IDF1: {metrics.get('IDF1', 0):.2f}%, IDSW: {metrics.get('IDSW', 0)}")

def run_grid_search(detector, num_workers, max_experiments):
    """Run grid search with simple process management"""
    
    print(f"\n{'='*80}")
    print(f"SIMPLE PARALLEL GRID SEARCH")
    print(f"{'='*80}")
    print(f"Detector: {detector}")
    print(f"Workers: {num_workers}")
    
    # Generate configs
    all_configs = generate_configs()
    if max_experiments:
        all_configs = all_configs[:max_experiments]
    
    total = len(all_configs)
    print(f"Total experiments: {total}")
    print(f"Estimated time: {total * 11 / num_workers / 60:.1f} hours")
    print(f"{'='*80}\n")
    
    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # Track running processes
    running = {}  # {proc: (exp_id, output_dir, config)}
    completed = 0
    next_exp = 0
    
    # Launch initial batch - skip already completed experiments
    while len(running) < num_workers and next_exp < total:
        config = all_configs[next_exp]
        exp_id = next_exp + 1
        proc, output_dir = launch_experiment(config, exp_id, detector)
        if proc is not None:  # Only track if actually launched
            running[proc] = (exp_id, output_dir, config)
            print(f"🚀 Started Exp {exp_id:04d}: conf={config['conf_thresh']}, match={config['match_thresh']}, track={config['track_thresh']}, nms={config['nms_thresh']}")
        else:
            completed += 1  # Count skipped as completed
        next_exp += 1
        time.sleep(1)  # Stagger starts
    
    # Monitor and launch new ones as they complete
    while running or next_exp < total:
        time.sleep(10)  # Check every 10 seconds
        
        # Check which processes finished
        finished = []
        for proc, (exp_id, output_dir, config) in list(running.items()):
            ret = proc.poll()
            if ret is not None:  # Process finished
                finished.append((proc, exp_id, output_dir, config, ret))
        
        # Process finished experiments
        for proc, exp_id, output_dir, config, ret in finished:
            del running[proc]
            completed += 1
            
            if ret == 0:
                # Check metrics
                metrics = check_metrics(output_dir)
                if metrics:
                    update_best(config, metrics, exp_id)
                    print(f"✅ Completed Exp {exp_id:04d} [{completed}/{total}] - MOTA: {metrics.get('MOTA', 0):.2f}%")
                else:
                    print(f"⚠️  Exp {exp_id:04d} finished but no metrics found")
            else:
                print(f"❌ Exp {exp_id:04d} failed with code {ret}")
            
            # Launch next if available
            if next_exp < total:
                config = all_configs[next_exp]
                new_exp_id = next_exp + 1
                proc, output_dir = launch_experiment(config, new_exp_id, detector)
                if proc is not None:
                    running[proc] = (new_exp_id, output_dir, config)
                    print(f"🚀 Started Exp {new_exp_id:04d}: conf={config['conf_thresh']}, match={config['match_thresh']}")
                else:
                    completed += 1  # Count skipped as completed
                next_exp += 1
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{total}")
    if os.path.exists(BEST_CONFIG_FILE):
        with open(BEST_CONFIG_FILE, 'r') as f:
            best = json.load(f)
        print(f"Best: Exp {best['experiment_id']:04d} - Score {best['score']:.2f}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-experiments', type=int, default=None)
    args = parser.parse_args()
    
    run_grid_search(args.detector, args.workers, args.max_experiments)
