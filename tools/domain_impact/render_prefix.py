"""Collect and render a hash-verified, incomplete accepted-state prefix."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pvpython', default='pvpython')
    parser.add_argument('--end-time', type=float, default=1e-6)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    collected = args.output / 'collected'
    rendered = args.output / 'render'
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    args.output.mkdir(parents=True)
    collect_cmd = [sys.executable, str(root / 'collect_history.py'), '--output',
                   str(collected), '--end-time', str(args.end_time), '--allow-prefix',
                   *map(str, args.run_dirs)]
    for attempt in range(8):
        result = subprocess.run(collect_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            print(result.stdout, end='', flush=True)
            break
        transient = 'run.json endpoint differs from accepted.npz' in result.stderr
        if not transient or attempt == 7:
            raise RuntimeError(result.stdout + result.stderr)
        time.sleep(.25)
    render_cmd = [args.pvpython, '--force-offscreen-rendering',
                  str(root / 'render_paraview.py'), str(collected), str(rendered),
                  '--allow-prefix', '--size', '400']
    result = subprocess.run(render_cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    main()
