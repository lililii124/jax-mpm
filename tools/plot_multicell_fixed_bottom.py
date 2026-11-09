"""Plot interval peak deflection at fixed initial bottom observation points."""
import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_multicell_binary_blast import compare, read_run
from tools.reconstruct_multicell_blast import mask_digest


def fixed_bottom_intervals(folder, geometry, fixed_yx):
    meta, rows, _ = read_run(folder, .02)
    if meta.get('bottom_point_peak_sampling') != 'every-step':
        raise ValueError('Every-step bottom point peaks are required')
    if hashlib.sha256(geometry.read_bytes()).hexdigest() != meta['geometry_sha256']:
        raise ValueError('Geometry differs from replay')
    with np.load(geometry, allow_pickle=False) as data:
        solid = np.asarray(data['solid_mask'], dtype=bool)
    if mask_digest(solid) != meta['solid_mask_sha256'] or not np.all(solid[0][fixed_yx[:, 0], fixed_yx[:, 1]]):
        raise ValueError('Geometry does not retain the observation points')
    with np.load(folder / 'reference.npz', allow_pickle=False) as data:
        columns = np.asarray(data['bottom_sample_columns'], dtype=np.int64)
    if columns.ndim != 2 or columns.shape[1] != 2 or len(np.unique(columns, axis=0)) != len(columns):
        raise ValueError('Invalid sample columns')
    lookup = {tuple(column): i for i, column in enumerate(columns)}
    occupied = np.argwhere(np.any(solid, axis=0))
    if set(lookup) != {(int(x), int(y)) for y, x in occupied}:
        raise ValueError('Sample columns differ from geometry')
    try:
        ids = np.array([lookup[(int(x), int(y))] for y, x in fixed_yx])
    except KeyError as error:
        raise ValueError('Observation point is missing') from error
    peaks = []
    for row in rows:
        with np.load(folder / f"frame-{row['step']:07d}.npz", allow_pickle=False) as data:
            values = np.asarray(data['bottom_interval_peak_m'])
        if (values.shape != (len(columns),) or not np.isfinite(values).all() or np.any(values < 0) or
                not np.isclose(values.max(), row['back_displacement_interval_peak_m'], rtol=1e-10, atol=1e-15)):
            raise ValueError('Invalid per-point interval peaks')
        peaks.append(float(values[ids].max()))
    return np.array([row['time_s'] for row in rows]), np.asarray(peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('baseline_geometry', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('candidate_geometry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    compare(args.baseline, args.candidate)
    with np.load(args.baseline_geometry, allow_pickle=False) as data:
        fixed_yx = np.argwhere(data['solid_mask'][0])
    if not len(fixed_yx):
        raise ValueError('Initial bottom observation set is empty')
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=160, layout='constrained')
    for folder, geometry, label, color in ((args.baseline, args.baseline_geometry, 'Initial', '#c27b29'),
                                           (args.candidate, args.candidate_geometry, 'Candidate', '#246a96')):
        time, peak = fixed_bottom_intervals(folder, geometry, fixed_yx)
        ax.stairs(peak[1:] * 1e3, time * 1e6, label=label, color=color, linewidth=1.7)
    ax.set(xlabel='Time (µs)', ylabel='Fixed bottom interval peak (mm)', xlim=(0, time[-1]*1e6), ylim=(0, None))
    ax.grid(color='#dddddd', linewidth=.5)
    ax.legend(frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
