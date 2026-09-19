"""Aggregate CEM anytime-success dumps and plot stacked bars.

python analysis/make_mpc_table.py /path/to/npz-dir
python analysis/make_mpc_table.py data-best --output mpc_cem_success.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_mpc import (  # noqa: E402
    IncompatibleNpz,
    collect_npz_paths,
    load_records,
    parse_method_seed,
    rate,
)

MODEL_ORDER = ('pub', 'rep', 'tdv')
MODEL_LABELS = {
    'pub': r'LeWM$_{\mathrm{PUB}}$',
    'rep': r'LeWM$_{\mathrm{REP}}$',
    'tdv': 'TDV-LeWM',
}
MODEL_COLORS = {
    'pub': '#1F77B4',
    'rep': '#FF7F0E',
    'tdv': '#2CA02C',
}
MODEL_HATCH_COLORS = {
    'pub': '#AEC7E8',
    'rep': '#FFBB78',
    'tdv': '#98DF8A',
}
TASK_ORDER = ('cube', 'grip', 'both')
TASK_LABELS = {
    'cube': 'Cube',
    'grip': 'Gripper',
    'both': 'Both',
}
TASK_KEYS = {
    'cube': ('cube', 'cube_excl', 'n', 'n_excl'),
    'grip': ('grip', 'grip_excl', 'n', 'n_excl'),
    'both': ('both', 'both_excl', 'n', 'n_excl'),
}


def classify_model(model_name):
    key = model_name.lower()
    if 'quentinll' in key:
        return 'pub'
    if key.startswith('tdv') or 'tdv_' in key:
        return 'tdv'
    if key.startswith('lewm'):
        return 'rep'
    return None


def _hex_to_rgb(color):
    color = color.lstrip('#')
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mix(color, other='#FFFFFF', weight=0.42):
    rgb = np.array(_hex_to_rgb(color))
    other_rgb = np.array(_hex_to_rgb(other))
    mixed = (1.0 - weight) * rgb + weight * other_rgb
    return tuple(mixed)


def _edge(color):
    return _mix(color, '#1A1423', weight=0.28)


def pool_metrics(dataset):
    """Pool scenario-level success across files for one model."""
    cube = np.concatenate([data['cube_success'] for data in dataset])
    keep = np.concatenate([data['keep'] for data in dataset])
    grips = [data['gripper_success'] for data in dataset]
    boths = [data['both_success'] for data in dataset]
    grip = np.concatenate(grips) if all(g is not None for g in grips) else None
    both = np.concatenate(boths) if all(b is not None for b in boths) else None
    return {
        'n': int(len(cube)),
        'n_excl': int(keep.sum()),
        'cube': rate(cube),
        'cube_excl': rate(cube[keep]),
        'grip': rate(grip) if grip is not None else float('nan'),
        'grip_excl': rate(grip[keep]) if grip is not None else float('nan'),
        'both': rate(both) if both is not None else float('nan'),
        'both_excl': rate(both[keep]) if both is not None else float('nan'),
        'seeds': sorted(
            {
                parse_method_seed(data['file'])['seed']
                for data in dataset
                if parse_method_seed(data['file'])['seed'] is not None
            }
        ),
        'files': [data['file'] for data in dataset],
    }


def _fmt(value):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:.1f}'


def print_read_files(paths, skipped):
    print('files read')
    print('-' * 56)
    if not paths:
        print('  (none)')
    for path in paths:
        info = parse_method_seed(path.name)
        kind = classify_model(info['model']) or 'unknown'
        print(
            f'  {path.name}  model={info["model"]}  '
            f'method={info["method"]}  seed={info["seed"]}  -> {kind}'
        )
    other = []
    problems = []
    for path, reason in skipped:
        reason_s = str(reason)
        if reason_s.startswith('method='):
            other.append((path.name, reason_s.split('=', 1)[1]))
        else:
            problems.append((path, reason))
    if other:
        methods = sorted({method for _, method in other})
        print()
        print(
            f'skipped {len(other)} non-matching files '
            f'({", ".join(methods)})'
        )
    if problems:
        print()
        print('skipped (errors / unrecognized)')
        print('-' * 56)
        for path, reason in problems:
            print(f'  {path.name}: {reason}')


def print_aggregate_table(pooled):
    headers = (
        'model',
        'seeds',
        'n',
        'n ex-noop',
        'cube',
        'cube ex-noop',
        'grip',
        'grip ex-noop',
        'both',
        'both ex-noop',
    )
    cells = []
    for key in MODEL_ORDER:
        if key not in pooled:
            continue
        row = pooled[key]
        cells.append(
            (
                key,
                ','.join(str(s) for s in row['seeds']) or 'n/a',
                str(row['n']),
                str(row['n_excl']),
                _fmt(row['cube']),
                _fmt(row['cube_excl']),
                _fmt(row['grip']),
                _fmt(row['grip_excl']),
                _fmt(row['both']),
                _fmt(row['both_excl']),
            )
        )
    if not cells:
        return
    widths = [
        max(len(headers[i]), max(len(cell[i]) for cell in cells))
        for i in range(len(headers))
    ]
    aligns = ['<', '<', '>', '>', '>', '>', '>', '>', '>', '>']

    def fmt_row(vals):
        return '  '.join(
            f'{val:{align}{width}}'
            for val, align, width in zip(vals, aligns, widths)
        )

    print()
    print('aggregated CEM anytime success  (pooled across seeds)')
    print('=' * (sum(widths) + 2 * (len(widths) - 1)))
    print(fmt_row(headers))
    print('-' * (sum(widths) + 2 * (len(widths) - 1)))
    for cell in cells:
        print(fmt_row(cell))


def _annotate(ax, x, y, text, color, size=8.6, weight='semibold'):
    ax.text(
        x,
        y,
        text,
        ha='center',
        va='center',
        color=color,
        fontsize=size,
        fontweight=weight,
        zorder=6,
        bbox={
            'boxstyle': 'square,pad=0.18',
            'facecolor': 'white',
            'edgecolor': 'none',
        },
    )


def plot_stacked_bars(pooled, output, title=None):
    present = [key for key in MODEL_ORDER if key in pooled]
    if not present:
        raise SystemExit('no classified CEM models to plot')

    n_models = len(present)
    n_groups = len(TASK_ORDER)
    bar_w = 0.30
    gap = 0.018
    cluster = n_models * bar_w + (n_models - 1) * gap
    group_step = cluster + 0.38 * 0.70
    group_centers = np.arange(n_groups, dtype=float) * group_step
    offsets = (
        np.arange(n_models) * (bar_w + gap) - 0.5 * cluster + 0.5 * bar_w
    )

    ink = '#111111'
    fig, ax = plt.subplots(figsize=(7.4, 8.2), dpi=160)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.yaxis.grid(True, color='#D0D0D0', linestyle='--', linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 108)
    side_pad = 0.11
    ax.set_xlim(
        group_centers[0] + offsets[0] - 0.5 * bar_w - side_pad,
        group_centers[-1] + offsets[-1] + 0.5 * bar_w + side_pad,
    )
    ax.set_ylabel('Success rate (%)', fontsize=12, color=ink, labelpad=8)
    bar_xs = [
        group_centers[group_i] + offsets[model_i]
        for group_i in range(n_groups)
        for model_i in range(n_models)
    ]
    bar_labels = [MODEL_LABELS[model] for _ in range(n_groups) for model in present]
    ax.set_xticks(bar_xs)
    ax.set_xticklabels(
        bar_labels,
        fontsize=8.2,
        color=ink,
        rotation=45,
        ha='right',
        rotation_mode='anchor',
    )
    ax.tick_params(axis='y', labelsize=10, colors=ink, length=3.5, direction='out')
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(ink)
    ax.spines['bottom'].set_color(ink)

    plt.rcParams['hatch.linewidth'] = 1.05

    for group_i, task in enumerate(TASK_ORDER):
        total_key, excl_key, _, _ = TASK_KEYS[task]
        for model_i, model in enumerate(present):
            row = pooled[model]
            total = float(row[total_key])
            excl = float(row[excl_key])
            if not np.isfinite(total) or not np.isfinite(excl):
                continue
            noop = max(total - excl, 0.0)
            x = group_centers[group_i] + offsets[model_i]
            color = MODEL_COLORS[model]
            hatch_face = MODEL_HATCH_COLORS[model]

            ax.bar(
                x,
                excl,
                width=bar_w,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                zorder=3,
            )
            ax.bar(
                x,
                noop,
                width=bar_w,
                bottom=excl,
                facecolor=hatch_face,
                edgecolor=color,
                linewidth=0,
                hatch='///',
                zorder=3,
            )

            if excl >= 8:
                _annotate(ax, x, max(excl * 0.46, 10.0), f'{excl:.1f}', ink)
            if noop >= 4.5:
                _annotate(ax, x, excl + 0.52 * noop, f'+{noop:.1f}', ink, size=8.2)
                _annotate(ax, x, total + 3.2, f'{total:.1f}', ink, size=9.2)
            elif noop >= 1.2:
                _annotate(ax, x, total + 2.4, f'+{noop:.1f}', ink, size=8.0)
                _annotate(ax, x, total + 6.2, f'{total:.1f}', ink, size=9.2)
            else:
                _annotate(ax, x, total + 3.2, f'{total:.1f}', ink, size=9.2)

    fill_handles = [
        Patch(
            facecolor='#AEC7E8',
            edgecolor=ink,
            linewidth=0.9,
            hatch='///',
            label='Noop contribution (hatched)',
        ),
        Patch(
            facecolor='#FFFDF8',
            edgecolor=ink,
            linewidth=0.9,
            label='Ex-noop (solid)',
        ),
    ]
    model_handles = [
        Patch(
            facecolor=MODEL_COLORS[key],
            edgecolor=_edge(MODEL_COLORS[key]),
            linewidth=0.9,
            label=MODEL_LABELS[key],
        )
        for key in present
    ]
    fill_leg = ax.legend(
        handles=fill_handles,
        loc='upper left',
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        facecolor='white',
        edgecolor='#C8C8C8',
        fontsize=8.8,
        borderpad=0.55,
    )
    fill_leg.get_frame().set_linewidth(0.8)
    ax.add_artist(fill_leg)
    model_leg = ax.legend(
        handles=model_handles,
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        facecolor='white',
        edgecolor='#C8C8C8',
        fontsize=9.4,
        borderpad=0.55,
        handlelength=1.2,
    )
    model_leg.get_frame().set_linewidth(0.8)

    if title:
        ax.set_title(title, fontsize=15, color=ink, pad=12, fontweight='semibold')

    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.99))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_bottom = min(
        tick.get_window_extent(renderer).transformed(ax.transAxes.inverted()).y0
        for tick in ax.get_xticklabels()
    )
    for group_i, task in enumerate(TASK_ORDER):
        ax.text(
            group_centers[group_i],
            tick_bottom - 0.018,
            TASK_LABELS[task],
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=13.5,
            fontweight='medium',
            color=ink,
            clip_on=False,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f'\nsaved figure: {output}')
    return fig


def load_cem_dataset(path, method):
    paths = collect_npz_paths(path)
    loaded_paths = []
    skipped = []
    by_model = defaultdict(list)
    for file_path in paths:
        info = parse_method_seed(file_path.name)
        if info['method'].lower() != method.lower():
            skipped.append((file_path, f'method={info["method"] or "?"}'))
            continue
        try:
            data = load_records(file_path)
        except IncompatibleNpz as exc:
            skipped.append((file_path, exc))
            continue
        except (ValueError, OSError, KeyError) as exc:
            skipped.append((file_path, exc))
            continue
        kind = classify_model(info['model'])
        if kind is None:
            skipped.append((file_path, f'unrecognized model {info["model"]}'))
            continue
        loaded_paths.append(file_path)
        by_model[kind].append(data)
    return loaded_paths, skipped, by_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=Path,
        help='directory of .npz files, or a single .npz',
    )
    parser.add_argument(
        '--method',
        default='cem',
        help='planner to keep (default: cem)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('mpc_cem_success.png'),
        help='output figure path',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='display the figure after saving',
    )
    args = parser.parse_args()

    loaded_paths, skipped, by_model = load_cem_dataset(args.path, args.method)
    print_read_files(loaded_paths, skipped)
    if not by_model:
        raise SystemExit(f'no valid {args.method} eval .npz files to plot')

    pooled = {key: pool_metrics(dataset) for key, dataset in by_model.items()}
    print_aggregate_table(pooled)
    plot_stacked_bars(pooled, args.output)
    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
