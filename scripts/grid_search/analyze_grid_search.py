#!/usr/bin/env python3
"""Analyze and visualize grid search results.

This project stores grid-search experiments under:

- results/GRID_SEARCH/exp_0001/ (config.json + metrics.json)

Older variants stored a consolidated JSON file. This script supports both.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


WEIGHTS = {
    'MOTA': 0.35,
    'IDF1': 0.30,
    'HOTA': 0.25,
    'IDSW': -0.10,  # Negative because lower is better
}


def compute_score(metrics: dict) -> float:
    mota = float(metrics.get('MOTA', 0) or 0)
    idf1 = float(metrics.get('IDF1', 0) or 0)
    hota = float(metrics.get('HOTA', 0) or 0)
    idsw = float(metrics.get('IDSW', 10000) or 10000)
    idsw_norm = max(0.0, 100.0 - (idsw / 30.0))
    return (
        WEIGHTS['MOTA'] * mota
        + WEIGHTS['IDF1'] * idf1
        + WEIGHTS['HOTA'] * hota
        + WEIGHTS['IDSW'] * idsw_norm
    )

def load_results(results_path='results/GRID_SEARCH'):
    """Load grid search results.

    Supports:
    - a directory containing exp_*/config.json + metrics.json (preferred)
    - a consolidated JSON file with {'results': [...]} (legacy)
    """
    p = Path(results_path)
    if not p.exists():
        print(f"❌ Results path not found: {results_path}")
        return None

    if p.is_file():
        with open(p, 'r') as f:
            return json.load(f)

    # Directory mode: scan exp_* folders
    results = []
    for exp_dir in sorted(p.glob('exp_*')):
        cfg_path = exp_dir / 'config.json'
        met_path = exp_dir / 'metrics.json'
        if not cfg_path.exists() or not met_path.exists():
            continue

        try:
            cfg = json.loads(cfg_path.read_text())
            metrics = json.loads(met_path.read_text())
            exp_id = int(cfg.get('experiment_id') or cfg.get('exp_id') or exp_dir.name.split('_')[-1])
            config = cfg.get('config', {})
            score = compute_score(metrics)

            results.append({
                'experiment_id': exp_id,
                'config': config,
                'metrics': metrics,
                'score': score,
            })
        except Exception as e:
            print(f"⚠️  Failed to load {exp_dir}: {e}")

    return {'results': results, 'source': str(p)}

def create_results_dataframe(data):
    """Convert results to pandas DataFrame"""
    results = data.get('results', [])
    
    rows = []
    for r in results:
        config = r.get('config', {})
        metrics = r.get('metrics', {})
        
        row = {
            'exp_id': r['experiment_id'],
            'score': r.get('score', compute_score(metrics)),
            'conf_thresh': config['conf_thresh'],
            'match_thresh': config['match_thresh'],
            'track_thresh': config['track_thresh'],
            'nms_thresh': config['nms_thresh'],
            'MOTA': metrics.get('MOTA', 0),
            'IDF1': metrics.get('IDF1', 0),
            'HOTA': metrics.get('HOTA', 0),
            'IDSW': metrics.get('IDSW', 0),
            'Recall': metrics.get('RECALL', 0),
            'Precision': metrics.get('precision', 0),
            'duration': r.get('duration', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(df)}")
    print(f"\nTop 10 Configurations (by score):")
    print("-"*80)
    
    top10 = df.nlargest(10, 'score')
    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        exp_id = int(row['exp_id']) if pd.notna(row.get('exp_id')) else -1
        print(f"\n#{rank} | Score: {row['score']:.2f} | Exp ID: {exp_id:04d}")
        print(f"  Config: conf={row['conf_thresh']:.2f}, match={row['match_thresh']:.2f}, "
              f"track={row['track_thresh']:.2f}, nms={row['nms_thresh']:.2f}")
        print(f"  Metrics: MOTA={row['MOTA']:.1f}%, IDF1={row['IDF1']:.1f}%, "
              f"HOTA={row['HOTA']:.1f}%, IDSW={int(row['IDSW'])}")
    
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    for param in ['conf_thresh', 'match_thresh', 'track_thresh', 'nms_thresh']:
        print(f"\n{param}:")
        grouped = df.groupby(param).agg({
            'score': 'mean',
            'MOTA': 'mean',
            'IDF1': 'mean',
            'IDSW': 'mean'
        }).round(2)
        print(grouped.to_string())

def create_visualizations(df, output_dir='results/GRID_SEARCH_PARALLEL'):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")

    best = df.nlargest(1, 'score').iloc[0] if len(df) else None
    
    # 1. Score distribution by parameter
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    params = ['conf_thresh', 'match_thresh', 'track_thresh', 'nms_thresh']
    param_labels = ['Confidence Threshold', 'Match Threshold', 'Track Threshold', 'NMS Threshold']
    
    for ax, param, label in zip(axes.flat, params, param_labels):
        grouped = df.groupby(param).agg({
            'score': ['mean', 'std'],
            'MOTA': 'mean',
            'IDF1': 'mean'
        }).reset_index()
        
        ax.errorbar(grouped[param], grouped[('score', 'mean')], 
                   yerr=grouped[('score', 'std')], 
                   marker='o', capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title(f'Score vs {label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if best is not None and param in best:
            ax.axvline(float(best[param]), color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(float(best[param]), ax.get_ylim()[1], ' best', color='red', va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_impact.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'parameter_impact.png'}")
    
    # 2. Heatmap: conf_thresh vs match_thresh
    plt.figure(figsize=(12, 8))
    pivot = df.pivot_table(values='score', 
                          index='conf_thresh', 
                          columns='match_thresh', 
                          aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=pivot.mean().mean())
    plt.title('Score Heatmap: Conf Threshold vs Match Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_conf_match.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'heatmap_conf_match.png'}")
    
    # 3. Metrics comparison for top 10
    plt.figure(figsize=(14, 8))
    top10 = df.nlargest(10, 'score')
    
    x = range(len(top10))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], top10['MOTA'], width, label='MOTA', alpha=0.8)
    plt.bar([i - 0.5*width for i in x], top10['IDF1'], width, label='IDF1', alpha=0.8)
    plt.bar([i + 0.5*width for i in x], top10['HOTA'], width, label='HOTA', alpha=0.8)
    plt.bar([i + 1.5*width for i in x], top10['IDSW']/50, width, label='IDSW/50', alpha=0.8)
    
    plt.xlabel('Configuration Rank', fontsize=12)
    plt.ylabel('Metric Value (%)', fontsize=12)
    plt.title('Top 10 Configurations - Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, [f"#{i+1}" for i in x])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'top10_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'top10_comparison.png'}")
    
    # 4. Trade-off: MOTA vs IDSW
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['IDSW'], df['MOTA'], c=df['score'], 
                         cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Score')
    
    # Annotate best points
    best = df.nlargest(5, 'score')
    for _, row in best.iterrows():
        exp_id = int(row['exp_id']) if pd.notna(row.get('exp_id')) else -1
        plt.annotate(f"#{exp_id:04d}", 
                    xy=(row['IDSW'], row['MOTA']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('ID Switches (IDSW)', fontsize=12)
    plt.ylabel('MOTA (%)', fontsize=12)
    plt.title('Trade-off: MOTA vs ID Switches', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tradeoff_mota_idsw.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tradeoff_mota_idsw.png'}")
    
    plt.close('all')

def export_top_configs(df, output_file='results/GRID_SEARCH_PARALLEL/top10_configs.json'):
    """Export top 10 configurations as JSON"""
    top10 = df.nlargest(10, 'score')
    
    configs = []
    for idx, row in top10.iterrows():
        config = {
            'rank': idx + 1,
            'experiment_id': int(row['exp_id']),
            'score': float(row['score']),
            'config': {
                'conf_thresh': float(row['conf_thresh']),
                'match_thresh': float(row['match_thresh']),
                'track_thresh': float(row['track_thresh']),
                'nms_thresh': float(row['nms_thresh'])
            },
            'metrics': {
                'MOTA': float(row['MOTA']),
                'IDF1': float(row['IDF1']),
                'HOTA': float(row['HOTA']),
                'IDSW': int(row['IDSW']),
                'Recall': float(row['Recall']),
                'Precision': float(row['Precision'])
            }
        }
        configs.append(config)
    
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"\n✓ Saved top 10 configs to: {output_file}")

if __name__ == '__main__':
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'results/GRID_SEARCH'
    
    print("Loading grid search results...")
    data = load_results(results_file)
    
    if data is None:
        sys.exit(1)
    
    print(f"Found {len(data.get('results', []))} completed experiments")
    
    df = create_results_dataframe(data)
    
    print_summary(df)
    # Save into results/GRID_SEARCH/analysis by default
    out_dir = 'results/GRID_SEARCH/analysis'
    create_visualizations(df, output_dir=out_dir)
    export_top_configs(df, output_file=str(Path(out_dir) / 'top10_configs.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
