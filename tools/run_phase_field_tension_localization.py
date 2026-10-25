"""Guarded actual-JAX parity then bounded precrack tension pilot; no media."""
import argparse
import hashlib
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
                if uss > 6*1024**3 or table.get('Rss',0) > 6*1024**3:
                    usage['resource_limit_exceeded'] = True
                    child.kill()
                    return
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
    if code != 0 or usage['timed_out'] or usage['samples'] == 0 or usage.get('resource_limit_exceeded',False):
        raise RuntimeError(f'{name} failed; inspect exact log and resource evidence')
    return usage


def main(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    roots=[*Path('mpm_jax').rglob('*.py'),Path('examples/phase_field_tension_persistent_grid.py'),Path('examples/phase_field_tension_geometry.py'),Path(__file__),Path('.github/workflows/phase-field-tension-localization.yml')]
    (output/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in roots},indent=2))
    common=[sys.executable,'-m','examples.phase_field_tension_persistent_grid']
    for name,stop,restart in [('whole',4,None),('prefix',2,None),('continued',4,output/'prefix/checkpoint_000002.npz')]:
        cmd=common+['--steps',str(stop),'--output',str(output/name)]
        if restart:cmd+=['--restart',str(restart)]
        measured_run(cmd,name,output,timeout=450)
    checks={}
    for filename in ['checkpoint_000004.npz','frame_000004.npz']:
        with np.load(output/'whole'/filename) as a,np.load(output/'continued'/filename) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                av,bv=a[key],b[key]
                if key=='history_rows':bv=bv[np.isin(bv[:,0],av[:,0])]
                # Actual carry must be bitwise exact. Even derived exports are
                # strict here; any failure is preserved before pilot dispatch.
                assert np.array_equal(av,bv),filename+':'+key
                checks[filename+':'+key]=True
    bad=subprocess.run(common+['--steps','5','--gc','.0003','--restart',str(output/'whole/checkpoint_000004.npz'),'--output',str(output/'rejected')],capture_output=True,text=True,timeout=450)
    (output/'binding-rejection.log').write_text(bad.stdout+bad.stderr)
    assert bad.returncode!=0 and 'binding mismatch' in bad.stderr
    (output/'parity.json').write_text(json.dumps({'exact_fields':checks,'binding_rejected':True},indent=2))
    # Continue the proven four-step actual carry, never recompute its prefix.
    measured_run(common+['--steps','1200','--restart',str(output/'whole/checkpoint_000004.npz'),'--output',str(output/'pilot')],'pilot',output,timeout=1600)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
