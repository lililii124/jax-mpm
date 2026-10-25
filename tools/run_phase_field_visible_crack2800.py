"""Bounded source-bound 2400 to 2800 AT2 continuation; no publication."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil
from run_phase_field_visible_crack import owned_session_uss, terminate_owned_group, PINS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/phase-field-visible-crack2800'
PARENT_CONTAINER = ROOT/'outputs/phase-field-parent2400'
PARENT = PARENT_CONTAINER/'raw'
ARCHIVE = ROOT/'outputs/phase-field-parent2400.zip'
PARENT_DIGEST = '6abf7642b447cf9b9ab4c79f33fc07eac7f1dca8df01bea513fc3807bd0986bf'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def source_gate(require_dispatch=True):
    repo='lililii124/jax-mpm'
    ref='refs/heads/work/phase-field-visible-crack2800'
    if require_dispatch and (os.environ.get('GITHUB_REPOSITORY') != repo or os.environ.get('GITHUB_REF') != ref):
        raise RuntimeError('Exact continuation repository/ref required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if require_dispatch and head != os.environ['GITHUB_SHA']:
        raise RuntimeError('Dispatched source mismatch')
    own='.github/workflows/phase-field-visible-crack2800.yml'
    actual=os.environ['GITHUB_WORKFLOW_REF'].split('@',1)[0] if require_dispatch else repo+'/'+own
    if actual not in (repo+'/'+own,repo+'/.github/workflows/engineering-previews.yml'):
        raise RuntimeError('Unregistered continuation workflow')
    actual=actual[len(repo)+1:]
    if (ROOT/actual).read_bytes() != (ROOT/own).read_bytes():
        raise RuntimeError('Active workflow differs from template')
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'mpm_jax').rglob('*.py')]
    names += ['examples/__init__.py','examples/phase_field_visible_crack.py',
        'tools/phase_field_restart.py','tools/run_phase_field_visible_crack2800.py','tools/fetch_phase_field_parent2800.py',
        'tools/run_phase_field_visible_crack.py','tools/audit_phase_field_visible_crack.py',
        'tools/render_phase_field_visible_crack.py','tools/encode_phase_field_visible_crack.py',
        'tools/phase_field_bspline_sampling.py','tools/run_woven_preview.py',own,actual]
    files={}
    for name in sorted(set(names)):
        raw=(ROOT/name).read_bytes()
        if raw != subprocess.check_output(['git','show',head+':'+name],cwd=ROOT):
            raise RuntimeError('Required source differs from committed blob: '+name)
        files[name]=hashlib.sha256(raw).hexdigest()
    return dict(commit=head,repository=repo,ref=ref,workflow=actual,files=files)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    context=source_gate()
    versions={name:metadata.version(name) for name in PINS}
    if versions != PINS or platform.python_version() != '3.12.13':
        raise RuntimeError('Exact parent runtime required')
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != PARENT_DIGEST:
        raise RuntimeError('Parent archive differs from official artifact12586985399')
    PARENT_CONTAINER.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            if Path(entry.filename).is_absolute() or '..' in Path(entry.filename).parts:
                raise RuntimeError('Unexpected parent archive path')
        archive.extractall(PARENT_CONTAINER)
    parent=json.loads((PARENT/'summary.json').read_text())
    provenance=json.loads((PARENT/'source-runtime.json').read_text())
    if parent['steps'] != 2400 or provenance['commit'] != '9ccb037790f3a897b0ef3562ea46eefdb1bd9a2e':
        raise RuntimeError('Wrong source-bound parent')
    upstream=json.loads((PARENT_CONTAINER/'continuation-source-runtime.json').read_text())
    if upstream['commit'] != '4c8b0e3123a0111622cd266e64cc1eaad51fe174' or upstream['parent_sha256'] != '1db40386c761461444965fcf8df90de62a608f3d3422101919ad15ad40d61e84':
        raise RuntimeError('Wrong continuation parent lineage')
    guard=json.loads((PARENT_CONTAINER/'resource-guard.json').read_text())
    if not guard['completed'] or guard['returncode'] != 0 or guard['guard_reason'] is not None:
        raise RuntimeError('Parent solve incomplete')
    admission=json.loads((PARENT/'crack-checks.json').read_text())
    if not all(value is True for key,value in admission['criteria'].items() if key!='contiguous_crack'):
        raise RuntimeError('Parent failed a non-geometric admission gate')
    if sorted(p.name for p in PARENT.glob('frame_*.npz')) != [f'frame_{i:06d}.npz' for i in range(0,2401,25)]:
        raise RuntimeError('Parent97 global snapshots incomplete')
    with np.load(PARENT/'particles.npz') as particles, np.load(PARENT/'nodal_state.npz') as nodes, np.load(PARENT/'frame_002400.npz') as last:
        if 'displacement' not in particles.files or not np.array_equal(particles['reference']+particles['displacement'],particles['position']):
            raise RuntimeError('Exact parent displacement checkpoint required')
        if not np.array_equal(last['position'],particles['position']) or not np.array_equal(last['nodal_damage'],nodes['damage']) or not np.array_equal(last['damage'],particles['damage']):
            raise RuntimeError('Parent final snapshot and restart state differ')
        if not all(np.isfinite(particles[key]).all() for key in particles.files):
            raise RuntimeError('Nonfinite parent checkpoint')
    save('continuation-source-runtime.json',dict(**context,versions=versions,
        python=platform.python_version(),parent_artifact=12586985399,parent_sha256=PARENT_DIGEST))
    command=[sys.executable,'-m','examples.phase_field_visible_crack','--dx','.025',
        '--length','.05','--gc','.005','--speed','.15','--damage-interval','1',
        '--steps','2800','--snapshot-every','25','--resume',str(PARENT),'--output',str(OUT/'raw')]
    record=dict(completed=False,command=command,peak_tree_uss_bytes=0,
        uss_limit_bytes=int(5.5*1024**3),seconds_limit=5400,
        minimum_available_bytes=2*1024**3,guard_reason=None)
    import ctypes
    if sys.platform != 'linux' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) != 0:
        raise RuntimeError('Linux subreaper setup required')
    started=time.monotonic()
    with (OUT/'solver.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                uss=owned_session_uss(process.pid,psutil)
                record['peak_tree_uss_bytes']=max(record['peak_tree_uss_bytes'],uss)
                reason=('USS limit' if uss>record['uss_limit_bytes'] else
                    'available memory limit' if psutil.virtual_memory().available<record['minimum_available_bytes'] else
                    'wall time limit' if time.monotonic()-started>record['seconds_limit'] else None)
                if reason:
                    record['guard_reason']=reason
                    break
                save('resource-guard.json',record)
                time.sleep(1)
            code=process.poll()
        except BaseException as error:
            record['guard_reason']='Observer/controller failure: '+repr(error)
            raise
        finally:
            try:
                terminate_owned_group(process)
            finally:
                record.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-started)
                save('resource-guard.json',record)
    if code or record['guard_reason']:
        raise RuntimeError('Continuation failed/stopped; raw preserved')
    raw=OUT/'raw'
    summary=json.loads((raw/'summary.json').read_text())
    expected=[f'frame_{i:06d}.npz' for i in range(0,2801,25)]
    if sorted(p.name for p in raw.glob('frame_*.npz')) != expected:
        raise RuntimeError('All113 authentic global snapshots required')
    previous=np.zeros(28800)
    previous_time=-1.
    for name in expected:
        with np.load(raw/name) as frame:
            position,damage,now=frame['position'],frame['damage'],float(frame['time'])
            if position.shape!=(28800,3) or damage.shape!=(28800,) or not np.isfinite(position).all() or not np.isfinite(damage).all():
                raise RuntimeError('Invalid saved arrays')
            if damage.min() < -1e-12 or damage.max()>1+1e-12 or np.min(damage-previous)<-1e-10 or now<=previous_time:
                raise RuntimeError('Bounds/irreversibility/global time failure')
            previous,previous_time=damage.copy(),now
    if summary['steps']!=2800 or summary['particles']!=28800 or summary['maximum_kkt']>1e-7 or not np.isclose(previous_time,summary['time'],rtol=0,atol=1e-14):
        raise RuntimeError('Continuation completion metadata differs')
    record['completed']=True
    save('resource-guard.json',record)
    subprocess.run([sys.executable,'tools/audit_phase_field_visible_crack.py',str(raw)],cwd=ROOT,check=True)
    (raw/'complete-checks.json').write_text(json.dumps(dict(completed=True,frames=113,
        summary=summary,interpretation='Source-bound reconstructed-state continuation; original .99 crack and energy gates passed. No publication.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-preflight',action='store_true')
    if parser.parse_args().source_preflight:
        print(json.dumps(source_gate(require_dispatch=False),indent=2))
    else:
        main()
