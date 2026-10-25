"""Bounded source-bound 2000 to 2400 AT2 continuation; no publication."""
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
OUT = ROOT/'outputs/phase-field-visible-crack-continue'
PARENT = ROOT/'outputs/phase-field-visible-crack-parent'
ARCHIVE = ROOT/'outputs/phase-field-visible-crack-parent.zip'
PARENT_DIGEST = '1db40386c761461444965fcf8df90de62a608f3d3422101919ad15ad40d61e84'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def source_gate(require_dispatch=True):
    repo='lililii124/jax-mpm'
    ref='refs/heads/work/phase-field-visible-crack-continue'
    if require_dispatch and (os.environ.get('GITHUB_REPOSITORY') != repo or os.environ.get('GITHUB_REF') != ref):
        raise RuntimeError('Exact continuation repository/ref required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if require_dispatch and head != os.environ['GITHUB_SHA']:
        raise RuntimeError('Dispatched source mismatch')
    own='.github/workflows/phase-field-visible-crack-continue.yml'
    actual=os.environ['GITHUB_WORKFLOW_REF'].split('@',1)[0] if require_dispatch else repo+'/'+own
    if actual not in (repo+'/'+own,repo+'/.github/workflows/engineering-previews.yml'):
        raise RuntimeError('Unregistered continuation workflow')
    actual=actual[len(repo)+1:]
    if (ROOT/actual).read_bytes() != (ROOT/own).read_bytes():
        raise RuntimeError('Active workflow differs from template')
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'mpm_jax').rglob('*.py')]
    names += ['examples/__init__.py','examples/phase_field_visible_crack.py',
        'tools/phase_field_restart.py','tools/run_phase_field_visible_crack_continue.py','tools/fetch_phase_field_parent.py',
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
        raise RuntimeError('Parent archive differs from official artifact12586283132')
    PARENT.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            if Path(entry.filename).name != entry.filename:
                raise RuntimeError('Unexpected parent archive path')
        archive.extractall(PARENT)
    parent=json.loads((PARENT/'summary.json').read_text())
    provenance=json.loads((PARENT/'source-runtime.json').read_text())
    if parent['steps'] != 2000 or provenance['commit'] != '9ccb037790f3a897b0ef3562ea46eefdb1bd9a2e':
        raise RuntimeError('Wrong source-bound parent')
    guard=json.loads((PARENT/'resource-guard.json').read_text())
    if not guard['completed'] or guard['returncode'] != 0 or guard['guard_reason'] is not None:
        raise RuntimeError('Parent solve incomplete')
    save('continuation-source-runtime.json',dict(**context,versions=versions,
        python=platform.python_version(),parent_artifact=12586283132,parent_sha256=PARENT_DIGEST))
    command=[sys.executable,'-m','examples.phase_field_visible_crack','--dx','.025',
        '--length','.05','--gc','.005','--speed','.15','--damage-interval','1',
        '--steps','2400','--snapshot-every','25','--resume',str(PARENT),'--output',str(OUT/'raw')]
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
    expected=[f'frame_{i:06d}.npz' for i in range(0,2401,25)]
    if sorted(p.name for p in raw.glob('frame_*.npz')) != expected:
        raise RuntimeError('All97 authentic global snapshots required')
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
    if summary['steps']!=2400 or summary['particles']!=28800 or summary['maximum_kkt']>1e-7 or not np.isclose(previous_time,summary['time'],rtol=0,atol=1e-14):
        raise RuntimeError('Continuation completion metadata differs')
    record['completed']=True
    save('resource-guard.json',record)
    subprocess.run([sys.executable,'tools/audit_phase_field_visible_crack.py',str(raw)],cwd=ROOT,check=True)
    (raw/'complete-checks.json').write_text(json.dumps(dict(completed=True,frames=97,
        summary=summary,interpretation='Source-bound reconstructed-state continuation; original .99 crack and energy gates passed. No publication.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-preflight',action='store_true')
    if parser.parse_args().source_preflight:
        print(json.dumps(source_gate(require_dispatch=False),indent=2))
    else:
        main()
