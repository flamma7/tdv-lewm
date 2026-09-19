"""Analyze anytime-success records from cube eval scripts.

python analysis/analyze_mpc.py data/
python analysis/analyze_mpc.py data/ogb_cube_table.npz
python analysis/analyze_mpc.py analysis/scenarios --noop-table --seed 7 --method icem
"""

import argparse
import re
from pathlib import Path

import numpy as np

DISTANCE_EDGES = (0.04, 0.13, 0.2)
DISTANCE_LABELS = ('0-0.04', '0.04-0.13', '0.13-0.2', '0.2+')

# {model}_{method}_{seed}.npz from run_sequential.py
METHOD_SEED_RE = re.compile(
    r'^(?P<model>.+)_(?P<method>[A-Za-z][A-Za-z0-9]*)_(?P<seed>\d+)$'
)


def collect_npz_paths(path):
    path = Path(path)
    if path.is_file():
        if path.suffix != '.npz':
            raise SystemExit(f'not an .npz file: {path}')
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.npz'))
        if not files:
            raise SystemExit(f'no .npz files in {path}')
        return files
    raise SystemExit(f'path not found: {path}')


def success_by_distance_bin(distances, success):
    bin_idx = np.digitize(distances, DISTANCE_EDGES)
    rows = []
    for i, label in enumerate(DISTANCE_LABELS):
        mask = bin_idx == i
        n = int(mask.sum())
        n_ok = int(success[mask].sum()) if n else 0
        rate = 100.0 * n_ok / n if n else float('nan')
        rows.append((label, n, n_ok, rate))
    return rows


def print_success_chart(rows, title):
    width = 24
    print(title)
    for label, n, n_ok, rate in rows:
        if n == 0:
            bar = ''
            rate_s = '  n/a'
        else:
            bar = '█' * int(round(width * rate / 100.0))
            rate_s = f'{rate:5.1f}%'
        print(f'  {label:<12} | {bar:<{width}} {rate_s}  ({n_ok}/{n})')


def rate(success):
    success = np.asarray(success, dtype=bool)
    n = len(success)
    if n == 0:
        return float('nan')
    return 100.0 * float(success.sum()) / n


def print_gripper_cube_table(gripper_success, cube_success):
    gripper_success = np.asarray(gripper_success, dtype=bool)
    cube_success = np.asarray(cube_success, dtype=bool)
    n = len(gripper_success)

    both = int((gripper_success & cube_success).sum())
    gripper_only = int((gripper_success & ~cube_success).sum())
    cube_only = int((~gripper_success & cube_success).sum())
    neither = int((~gripper_success & ~cube_success).sum())

    def cell(label, count):
        pct = 100.0 * count / max(n, 1)
        return f'{label} {count} ({pct:.1f}%)'

    col_w = 28
    row_w = 18
    rule = '-' * (row_w + 2 * col_w)
    print()
    print(f'{"":<{row_w}} {"Cube success":<{col_w}} {"Cube failure":<{col_w}}')
    print(rule)
    print(
        f'{"Gripper success":<{row_w}} '
        f'{cell("both", both):<{col_w}} '
        f'{cell("gripper-only", gripper_only):<{col_w}}'
    )
    print(rule)
    print(
        f'{"Gripper failure":<{row_w}} '
        f'{cell("cube-only", cube_only):<{col_w}} '
        f'{cell("neither", neither):<{col_w}}'
    )


class IncompatibleNpz(ValueError):
    """npz is not a table-eval dump from eval_wm_cube_mpc.py."""


def load_records(path):
    loaded = np.load(path)
    keys = set(loaded.files)
    if 'cost_latent' in keys and 'records' not in keys:
        raise IncompatibleNpz(
            'plan-eval npz from eval_wm_cube_plan.py (no anytime-success records)'
        )
    if 'records' not in keys:
        raise IncompatibleNpz(
            f'not a table-eval npz (keys: {sorted(keys)})'
        )
    records = loaded['records']
    names = set(records.dtype.names or ())
    if 'cube_displacement' not in names:
        raise IncompatibleNpz("records missing 'cube_displacement'")

    if 'cube_success' in names:
        cube_success = np.asarray(records['cube_success'], dtype=bool)
        gripper_success = np.asarray(records['gripper_success'], dtype=bool)
        both_success = np.asarray(records['both_success'], dtype=bool)
    elif 'success' in names:
        cube_success = np.asarray(records['success'], dtype=bool)
        gripper_success = None
        both_success = None
    else:
        raise IncompatibleNpz("records missing 'cube_success' or 'success'")

    cube_d = np.asarray(records['cube_displacement'])
    arm_d = (
        np.asarray(records['arm_displacement'])
        if 'arm_displacement' in names
        else None
    )
    if 'scenario' in names:
        scenario = np.asarray(records['scenario'])
    else:
        scenario = np.arange(len(records), dtype=np.int32)
    cube_noop = cube_d < DISTANCE_EDGES[0]
    grip_noop = (
        arm_d < DISTANCE_EDGES[0]
        if arm_d is not None
        else np.zeros(len(records), dtype=bool)
    )
    both_noop = cube_noop & grip_noop
    keep = ~cube_noop
    if arm_d is not None:
        keep = keep & ~grip_noop

    return {
        'file': path.name,
        'n': len(records),
        'n_excl': int(keep.sum()),
        'scenario': scenario,
        'cube_d': cube_d,
        'arm_d': arm_d,
        'cube_success': cube_success,
        'gripper_success': gripper_success,
        'both_success': both_success,
        'cube_noop': cube_noop,
        'grip_noop': grip_noop,
        'both_noop': both_noop,
        'keep': keep,
    }


def print_file_charts(data, exclude_noop):
    cube_d = data['cube_d']
    arm_d = data['arm_d']
    cube_success = data['cube_success']
    gripper_success = data['gripper_success']
    both_success = data['both_success']
    cube_noop = data['cube_noop']
    keep = data['keep']

    print(f'loaded {data["n"]} scenarios from {data["file"]}')

    if exclude_noop:
        n_drop = data['n'] - data['n_excl']
        print(
            f'excluding {n_drop} no-op scenarios '
            f'(displacement < {DISTANCE_EDGES[0]}); '
            f'{data["n_excl"]} remaining'
        )
        cube_d = cube_d[keep]
        cube_success = cube_success[keep]
        if gripper_success is not None:
            gripper_success = gripper_success[keep]
            both_success = both_success[keep]
        if arm_d is not None:
            arm_d = arm_d[keep]

    print(f'cube displacement mean: {cube_d.mean():.6f}')
    print(f'cube displacement std:  {cube_d.std():.6f}')
    if arm_d is not None:
        print(f'arm displacement mean:  {arm_d.mean():.6f}')
        print(f'arm displacement std:   {arm_d.std():.6f}')

    n_cube_noop = int(cube_noop.sum())
    print()
    print(
        f'cube no-op:              {rate(cube_noop):5.1f}% '
        f'({n_cube_noop}/{data["n"]})'
    )
    print(f'cube anytime success:    {rate(data["cube_success"]):5.1f}%')
    print(
        f'cube anytime success (excluding no-op): '
        f'{rate(data["cube_success"][data["keep"]]):5.1f}%'
    )
    if data['gripper_success'] is not None:
        print(
            f'gripper anytime success: {rate(data["gripper_success"]):5.1f}%'
        )
        print(
            f'gripper anytime success (excluding no-op): '
            f'{rate(data["gripper_success"][data["keep"]]):5.1f}%'
        )
        print(f'both anytime success:    {rate(data["both_success"]):5.1f}%')
        print(
            f'both anytime success (excluding no-op): '
            f'{rate(data["both_success"][data["keep"]]):5.1f}%'
        )
    print()
    print_success_chart(
        success_by_distance_bin(cube_d, cube_success),
        'cube anytime success by cube displacement',
    )
    if gripper_success is not None and arm_d is not None:
        print()
        print_success_chart(
            success_by_distance_bin(arm_d, gripper_success),
            'gripper anytime success by arm displacement',
        )
        print()
        print_success_chart(
            success_by_distance_bin(cube_d, both_success),
            'both anytime success by cube displacement',
        )
        print()
        print_success_chart(
            success_by_distance_bin(arm_d, both_success),
            'both anytime success by arm displacement',
        )

    if gripper_success is not None:
        print_gripper_cube_table(gripper_success, cube_success)


def _fmt_rate(value):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:5.1f}%'


def parse_method_seed(filename):
    match = METHOD_SEED_RE.match(Path(filename).stem)
    if not match:
        return {'method': '', 'seed': None, 'model': Path(filename).stem}
    return {
        'method': match.group('method'),
        'seed': int(match.group('seed')),
        'model': match.group('model'),
    }


def _group_key(row):
    info = parse_method_seed(row['file'])
    seed = info['seed']
    return (
        info['method'] == '',
        info['method'],
        seed is None,
        -1 if seed is None else seed,
        info['model'],
        row['file'],
    )


def _group_label(row):
    info = parse_method_seed(row['file'])
    if not info['method'] or info['seed'] is None:
        return None
    return f"{info['method']}  seed={info['seed']}"


def print_comparison_table(rows):
    headers = (
        'file',
        'n',
        'n ex-noop',
        'cube',
        'cube ex-noop',
        'grip',
        'grip ex-noop',
        'both',
        'both ex-noop',
    )
    rows = sorted(rows, key=_group_key)
    cells = []
    for row in rows:
        info = parse_method_seed(row['file'])
        label = info['model'] if info['method'] else row['file']
        cells.append(
            (
                label,
                str(row['n']),
                str(row['n_excl']),
                _fmt_rate(row['cube']),
                _fmt_rate(row['cube_excl']),
                _fmt_rate(row['grip']),
                _fmt_rate(row['grip_excl']),
                _fmt_rate(row['both']),
                _fmt_rate(row['both_excl']),
            )
        )

    widths = [
        max(len(headers[i]), max(len(c[i]) for c in cells))
        for i in range(len(headers))
    ]
    aligns = ['<', '>', '>', '>', '>', '>', '>', '>', '>']

    def fmt_row(vals):
        return '  '.join(
            f'{val:{align}{width}}'
            for val, align, width in zip(vals, aligns, widths)
        )

    rule = '-' * (sum(widths) + 2 * (len(widths) - 1))
    unset = object()
    prev_label = unset
    for row, cell in zip(rows, cells):
        label = _group_label(row)
        if label != prev_label:
            if prev_label is not unset:
                print()
            if label is not None:
                print(label)
            print(fmt_row(headers))
            print(rule)
            prev_label = label
        print(fmt_row(cell))


def metrics_row(data):
    keep = data['keep']
    grip = data['gripper_success']
    both = data['both_success']
    return {
        'file': data['file'],
        'n': data['n'],
        'n_excl': data['n_excl'],
        'cube': rate(data['cube_success']),
        'cube_excl': rate(data['cube_success'][keep]),
        'grip': rate(grip) if grip is not None else float('nan'),
        'grip_excl': rate(grip[keep]) if grip is not None else float('nan'),
        'both': rate(both) if both is not None else float('nan'),
        'both_excl': rate(both[keep]) if both is not None else float('nan'),
    }


def _fmt_ids(ids):
    ids = [int(i) for i in np.asarray(ids).reshape(-1)]
    if not ids:
        return '(none)'
    return ', '.join(str(i) for i in ids)


def _model_label(filename):
    info = parse_method_seed(filename)
    return info['model'] if info['method'] else Path(filename).stem


def _same_scenarios(a, b):
    if a['n'] != b['n']:
        return False
    if not np.array_equal(a['scenario'], b['scenario']):
        return False
    if not np.allclose(a['cube_d'], b['cube_d'], atol=1e-6, equal_nan=True):
        return False
    if a['arm_d'] is None or b['arm_d'] is None:
        return a['arm_d'] is None and b['arm_d'] is None
    return np.allclose(a['arm_d'], b['arm_d'], atol=1e-6, equal_nan=True)


def print_noop_table(dataset, seed, method):
    """Print no-op scenario ids and per-model both-no-op successes."""
    dataset = sorted(dataset, key=lambda data: _model_label(data['file']))
    thresh = DISTANCE_EDGES[0]
    ref = dataset[0]
    mismatched = [
        data['file'] for data in dataset[1:] if not _same_scenarios(ref, data)
    ]
    print()
    print('=' * 72)
    print(
        f'no-op scenarios  method={method}  seed={seed}  '
        f'n={ref["n"]}  thresh={thresh}'
    )
    print('=' * 72)
    if mismatched:
        print(
            'warning: these dumps do not share the same scenarios as '
            f'{ref["file"]}: {", ".join(mismatched)}'
        )
        print('using no-op ids from the first matching file')

    scenario = ref['scenario']
    groups = (
        ('both', ref['both_noop']),
        ('cube', ref['cube_noop']),
        ('gripper', ref['grip_noop']),
    )
    print()
    print('no-op scenario numbers')
    label_w = max(len(name) for name, _ in groups)
    for name, mask in groups:
        ids = scenario[mask]
        print(f'  {name:<{label_w}}  ({len(ids):>3}):  {_fmt_ids(ids)}')

    n_cube_noop = int(ref['cube_noop'].sum())
    print(
        f'cube no-op: {rate(ref["cube_noop"]):5.1f}% '
        f'({n_cube_noop}/{ref["n"]})'
    )

    both_mask = ref['both_noop']
    n_both = int(both_mask.sum())
    print()
    print(f'both-no-op correct  ({n_both} both-no-op scenarios)')
    model_w = max(len(_model_label(data['file'])) for data in dataset)
    for data in dataset:
        both = data['both_success']
        if both is None:
            print(f'  {_model_label(data["file"]):<{model_w}}  n/a')
            continue
        if not _same_scenarios(ref, data):
            ids = data['scenario'][data['both_noop'] & both]
            print(
                f'  {_model_label(data["file"]):<{model_w}}  '
                f'(unaligned)  {_fmt_ids(ids)}'
            )
            continue
        ids = scenario[both_mask & both]
        print(
            f'  {_model_label(data["file"]):<{model_w}}  '
            f'({len(ids)}/{n_both}):  {_fmt_ids(ids)}'
        )

    aligned = [data for data in dataset if _same_scenarios(ref, data)]
    if len(aligned) < 2:
        return
    print()
    print('exclusive both-success (this approach only)')
    for data in aligned:
        label = _model_label(data['file'])
        both = data['both_success']
        if both is None:
            print(f'  {label:<{model_w}}  n/a')
            continue
        others = [
            other['both_success']
            for other in aligned
            if other is not data and other['both_success'] is not None
        ]
        exclusive = both.copy()
        for other in others:
            exclusive &= ~other
        ids = scenario[exclusive]
        ids_real = scenario[exclusive & data['keep']]
        print(f'  {label:<{model_w}}  all      ({len(ids)}):  {_fmt_ids(ids)}')
        print(
            f'  {"":<{model_w}}  ex-noop  ({len(ids_real)}):  {_fmt_ids(ids_real)}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=Path,
        help='directory of .npz files, or a single .npz',
    )
    parser.add_argument(
        '--exclude-noop',
        action='store_true',
        help=(
            'exclude no-op scenarios (cube or arm displacement < '
            f'{DISTANCE_EDGES[0]}) from per-file charts'
        ),
    )
    parser.add_argument(
        '--noop-table',
        action='store_true',
        help=(
            'print no-op scenario numbers and per-approach both-no-op '
            'successes for --seed and --method'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='filter dumps to this seed (required with --noop-table)',
    )
    parser.add_argument(
        '--method',
        default=None,
        help='filter dumps to this planner (required with --noop-table)',
    )
    args = parser.parse_args()
    if args.noop_table and (args.seed is None or not args.method):
        parser.error('--noop-table requires --seed and --method')

    paths = collect_npz_paths(args.path)
    rows = []
    loaded = []
    skipped = []
    for path in paths:
        try:
            data = load_records(path)
        except IncompatibleNpz as exc:
            skipped.append((path, exc))
            continue
        except (ValueError, OSError, KeyError) as exc:
            skipped.append((path, exc))
            continue
        info = parse_method_seed(data['file'])
        if args.seed is not None and info['seed'] != args.seed:
            continue
        if args.method and info['method'].lower() != args.method.lower():
            continue
        if rows:
            print()
        print('=' * 72)
        print(data['file'])
        print('=' * 72)
        print_file_charts(data, args.exclude_noop)
        rows.append(metrics_row(data))
        loaded.append(data)

    if not rows:
        if args.seed is not None or args.method:
            raise SystemExit(
                'no valid eval .npz files matched '
                f'seed={args.seed} method={args.method}'
            )
        raise SystemExit('no valid eval .npz files to analyze')

    print()
    print('=' * 72)
    print('anytime success comparison  (ex-noop: displacement < '
          f'{DISTANCE_EDGES[0]})')
    print('=' * 72)
    print_comparison_table(rows)
    if args.noop_table:
        print_noop_table(loaded, args.seed, args.method)
    if skipped:
        print()
        print('skipped:')
        for path, exc in skipped:
            print(f'  {path.name}: {exc}')


if __name__ == '__main__':
    main()
