"""Plot the sampled back-face response of two validated binary replays."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_multicell_binary_blast import compare, read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite the plot')
    compare(args.baseline, args.candidate)
    meta, a, _ = read_run(args.baseline, .02)
    _, b, _ = read_run(args.candidate, .02)
    fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=160, layout='constrained')
    for rows, label, color in ((a, 'Initial', '#c27b29'), (b, 'Candidate', '#246a96')):
        t = np.array([r['time_s'] for r in rows])*1e6
        if meta.get('peak_sampling', 'saved-frames') == 'every-step':
            u = np.array([r['back_displacement_interval_peak_m'] for r in rows[1:]])*1e3
            ax.stairs(u, t, label=label, color=color, linewidth=1.7)
        else:
            u = np.array([r['back_displacement_max_m'] for r in rows])*1e3
            ax.plot(t, u, label=label, color=color, linewidth=1.7)
    ylabel = ('Interval peak displacement (mm)' if meta.get('peak_sampling', 'saved-frames') == 'every-step'
              else 'Back-face displacement (mm)')
    ax.set(xlabel='Time (µs)', ylabel=ylabel, xlim=(0, t[-1]), ylim=(0, None))
    ax.grid(color='#dddddd', linewidth=.5)
    ax.legend(frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
