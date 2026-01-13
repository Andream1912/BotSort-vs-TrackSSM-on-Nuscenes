#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


METRIC_KEYS = [
    'MOTA', 'IDF1', 'HOTA',
    'IDSW', 'num_switches',
    'num_fragmentations',
    'precision', 'recall',
    'TP', 'FP', 'FN',
    'Frames', 'GT Objects', 'Predictions',
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _pick(d: dict, keys) -> dict:
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _delta(a, b):
    # a - b
    af = _as_float(a)
    bf = _as_float(b)
    if af is None or bf is None:
        return None
    return af - bf


def _fmt_pct(x):
    if x is None:
        return '--'
    return f"{x:.2f}\\%"


def _fmt_num(x, nd=2):
    if x is None:
        return '--'
    if isinstance(x, bool):
        return str(x)
    try:
        if float(x).is_integer():
            return str(int(float(x)))
    except Exception:
        pass
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_signed(x, nd=2, is_pct=False):
    if x is None:
        return '--'
    s = f"{float(x):+.{nd}f}"
    return (s + "\\%") if is_pct else s


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    return _read_json(metrics_path)


def _load_bench(bench_dir: Path):
    if bench_dir is None:
        return None
    bench_path = bench_dir / 'benchmark.json'
    if not bench_path.exists():
        return None
    return _read_json(bench_path)


def _bench_summary(bench: dict | None) -> dict:
    if not bench:
        return {}

    def get_mean(section):
        v = bench.get(section)
        if isinstance(v, dict):
            return v.get('mean_ms')
        return None

    return {
        'fps_overall': bench.get('fps_overall'),
        'total_ms_mean': get_mean('total_ms'),
        'detect_ms_mean': get_mean('detect_ms'),
        'track_ms_mean': get_mean('track_ms'),
    }


def _make_latex_table(summary: dict) -> str:
    none = summary['none']
    flow = summary['flow']
    d = summary['delta_flow_minus_none']

    def row(metric_key, label, is_percent=True, nd=2):
        n = none.get(metric_key)
        f = flow.get(metric_key)
        dd = d.get(metric_key)
        if is_percent:
            return f"    {label} & {_fmt_pct(_as_float(n))} & {_fmt_pct(_as_float(f))} & {_fmt_signed(dd, nd=nd, is_pct=True)} \\\\"
        return f"    {label} & {_fmt_num(n, nd=nd)} & {_fmt_num(f, nd=nd)} & {_fmt_signed(dd, nd=nd, is_pct=False)} \\\\"

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{Confronto TrackSSM con CMC disattivato vs CMC (sparseOptFlow), YOLOX-L fine-tuned (epoch 30), test size 800\\times1440, batch prediction abilitata.}")
    lines.append("  \\label{tab:trackssm_cmc_ablation}")
    lines.append("  \\begin{tabular}{lrrr}")
    lines.append("    \\toprule")
    lines.append("    Metrica & CMC none & CMC sparseOptFlow & $\\Delta$(flow-none) \\\\"
    )
    lines.append("    \\midrule")

    lines.append(row('HOTA', 'HOTA', is_percent=True))
    lines.append(row('IDF1', 'IDF1', is_percent=True))
    lines.append(row('MOTA', 'MOTA', is_percent=True))
    lines.append(row('IDSW', 'IDSW', is_percent=False, nd=0))

    if 'fps_overall' in none and 'fps_overall' in flow:
        lines.append("    \\midrule")
        lines.append(row('fps_overall', 'FPS (benchmark)', is_percent=False, nd=2))
        lines.append(row('total_ms_mean', 'Latency totale [ms]', is_percent=False, nd=2))
        lines.append(row('detect_ms_mean', 'Detector [ms]', is_percent=False, nd=2))
        lines.append(row('track_ms_mean', 'Tracker [ms]', is_percent=False, nd=2))

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description='Compare TrackSSM CMC configurations (none vs sparseOptFlow).')
    ap.add_argument('--run-none', default='results/FULL_E30_800x1440_CMCnone_batch', help='Run dir for CMC none (must contain metrics.json)')
    ap.add_argument('--run-flow', default='results/FULL_E30_800x1440_CMCflow_batch', help='Run dir for CMC sparseOptFlow (must contain metrics.json)')
    ap.add_argument('--bench-none', default='results/BENCH_E30_C_800x1440_none_batch', help='Benchmark dir for CMC none (optional; benchmark.json)')
    ap.add_argument('--bench-flow', default='results/BENCH_E30_C_800x1440_flow_batch', help='Benchmark dir for CMC sparseOptFlow (optional; benchmark.json)')
    ap.add_argument('--out-dir', default='results/FINAL_COMPARISON', help='Output folder')
    args = ap.parse_args()

    run_none = Path(args.run_none)
    run_flow = Path(args.run_flow)
    bench_none = Path(args.bench_none) if args.bench_none else None
    bench_flow = Path(args.bench_flow) if args.bench_flow else None

    m_none = _load_metrics(run_none)
    m_flow = _load_metrics(run_flow)

    out_none = _pick(m_none, METRIC_KEYS)
    out_flow = _pick(m_flow, METRIC_KEYS)

    b_none = _bench_summary(_load_bench(bench_none))
    b_flow = _bench_summary(_load_bench(bench_flow))
    out_none.update(b_none)
    out_flow.update(b_flow)

    delta = {}
    for k in set(out_none.keys()) | set(out_flow.keys()):
        delta[k] = _delta(out_flow.get(k), out_none.get(k))

    summary = {
        'run_none': str(run_none),
        'run_flow': str(run_flow),
        'bench_none': str(bench_none) if bench_none else None,
        'bench_flow': str(bench_flow) if bench_flow else None,
        'none': out_none,
        'flow': out_flow,
        'delta_flow_minus_none': delta,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / 'cmc_none_vs_sparseOptFlow_summary.json'
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    out_tex = out_dir / 'cmc_none_vs_sparseOptFlow_table.tex'
    out_tex.write_text(_make_latex_table(summary), encoding='utf-8')

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_tex}")


if __name__ == '__main__':
    main()
