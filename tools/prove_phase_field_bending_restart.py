"""Actual JAX 4-step versus 2+2 checkpoint proof; no crack qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np


def measured_run(command, name, output, timeout=240):
    start = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    usage = dict(scope='direct solver child; sampled /proc smaps_rollup every .25s',
                 peak_sampled_uss_bytes=0, peak_sampled_rss_bytes=0,
                 samples=0, timed_out=False, step_events=[])
    def monitor():
        while child.poll() is None:
            if time.monotonic()-start > timeout:
                usage['timed_out'] = True
                child.kill()
                return
            try:
                table = {}
                for line in Path(f'/proc/{child.pid}/smaps_rollup').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if value.strip().endswith('kB'):
                        table[key] = int(value.split()[0])*1024
                uss = sum(table.get(key, 0) for key in ('Private_Clean','Private_Dirty','Private_Hugetlb'))
                usage['peak_sampled_uss_bytes'] = max(usage['peak_sampled_uss_bytes'], uss)
                usage['peak_sampled_rss_bytes'] = max(usage['peak_sampled_rss_bytes'], table.get('Rss',0))
                usage['samples'] += 1
            except (OSError, ValueError):
                pass
            time.sleep(.25)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    with (output/f'{name}.log').open('w') as log:
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            if line.startswith('step='):
                usage['step_events'].append(dict(elapsed_seconds=time.monotonic()-start,
                                                 line=line.strip()))
    code = child.wait()
    watcher.join(timeout=1)
    usage.update(wall_seconds=time.monotonic()-start, returncode=code)
    (output/f'{name}.resources.json').write_text(json.dumps(usage, indent=2))
    if code != 0 or usage['timed_out']:
        raise RuntimeError(f'{name} failed; inspect exact log and resource evidence')
    return usage


def prove(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-m', 'examples.phase_field_bending_persistent_grid',
              '--dx', '.04', '--length', '.08', '--displacement', '.00005']
    resources = {}
    for name, steps, restart in [('whole', 4, None), ('prefix', 2, None),
                                ('continued', 4, output/'prefix/checkpoint_000002.npz')]:
        command = common+['--steps', str(steps), '--output', str(output/name)]
        if restart is not None:
            command += ['--restart', str(restart)]
        resources[name] = measured_run(command, name, output)
    checks = {}
    for filename in ('frame_000004.npz', 'checkpoint_000004.npz'):
        with np.load(output/'whole'/filename) as whole, np.load(output/'continued'/filename) as continued:
            assert set(whole.files) == set(continued.files)
            for key in whole.files:
                if key == 'history_rows':
                    # Prefix completion adds a diagnostic-only row at step 2.
                    left, right = whole[key], continued[key]
                    right = right[np.isin(right[:,0], left[:,0])]
                else:
                    left, right = whole[key], continued[key]
                assert np.array_equal(left, right), f'Exact restart parity failed: {filename}:{key}'
                checks[filename+':'+key] = True
    # Reject altered material/load and altered checkpoint binding before stepping.
    invalid = common+['--steps','4','--gc','.006','--restart',
        str(output/'prefix/checkpoint_000002.npz'),'--output',str(output/'rejected-binding')]
    rejected = subprocess.run(invalid, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
    (output/'binding-rejection.log').write_text(rejected.stdout)
    assert rejected.returncode != 0 and 'binding mismatch' in rejected.stdout
    if sys.platform.startswith('linux'):
        assert all(record['samples'] > 0 and record['peak_sampled_uss_bytes'] > 0
                   for record in resources.values()), 'Linux USS sampling did not produce evidence'
    result = dict(status='exact_checkpoint_continuation_parity_passed',
                  steps=4, split_step=2, dx=.04, production_mesh_qualified=False,
                  physical_crack_qualified=False, exact_array_checks=len(checks),
                  altered_binding_rejected=True, resources=resources)
    (output/'restart-parity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    prove(parser.parse_args().output)
