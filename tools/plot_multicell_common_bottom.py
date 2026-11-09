"""Plot matched bottom-point response from a common-surface comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.comparison.read_text())
    sampling = result['sampling']
    if sampling not in ('every-step', 'saved-frames'):
        raise ValueError('Unknown sampling mode')
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=160, layout='constrained')
    for key, label, color in (('reference', 'Initial', '#c27b29'), ('candidate', 'Candidate', '#246a96')):
        rows = result['series'][key]
        times = np.asarray([row['time_s'] for row in rows]) * 1e6
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid comparison time series')
        field = 'interval_peak_m' if sampling == 'every-step' else 'saved_peak_m'
        peaks = np.asarray([row[field] for row in rows], dtype=float) * 1e3
        if not np.isfinite(peaks).all() or np.any(peaks < 0):
            raise ValueError('Invalid comparison response')
        if sampling == 'every-step':
            ax.stairs(peaks[1:], times, label=label, color=color, linewidth=1.7)
        else:
            ax.plot(times, peaks, label=label, color=color, linewidth=1.7)
    ax.set(xlabel='Time (µs)', ylabel='Matched bottom displacement (mm)', xlim=(0, times[-1]), ylim=(0, None))
    ax.grid(color='#dddddd', linewidth=.5)
    ax.legend(frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
