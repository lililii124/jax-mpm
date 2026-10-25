"""Frozen production-mesh discrete-precrack tension step1200 to3600 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import numpy as np
from run_phase_field_tension_localization import measured_run
PARENT_BYTES=50272877
PARENT_ARTIFACT=12591972996
PARENT_ZIP_SHA='64196b8e97bf16cdfca33fe43bf0031d95a1b439b356faadf96f1e9aabb55ba9'
PARENT_HEAD='cf66f924c14e438787181b5eb956db4d1d938c9a'
CHECKPOINT_SHA='2547ab6cbb66701665369ef0a65041e16ec15602143c7f2a20d754cbfe3522d1'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(destination):
    # Authenticated API request and uncredentialed signed-blob requests use
    # separate request objects. Remove the token before starting the solver.
    token = os.environ.pop('GH_TOKEN')
    request = urllib.request.Request(
        f'https://api.github.com/repos/lililii124/jax-mpm/actions/artifacts/{PARENT_ARTIFACT}/zip',
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    try:
        urllib.request.build_opener(NoRedirect).open(request,timeout=30)
        raise RuntimeError('Expected official artifact redirect')
    except urllib.error.HTTPError as response:
        if response.code != 302:
            response.close()
            raise
        location = response.headers['Location']
        response.close()
    parsed = urllib.parse.urlsplit(location)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError('Invalid signed artifact URL')
    partial = destination.with_suffix('.partial')
    deadline = time.monotonic()+180
    digest = hashlib.sha256()
    with partial.open('wb') as stream:
        for start in range(0,PARENT_BYTES,1024*1024):
            end = min(PARENT_BYTES,start+1024*1024)-1
            for attempt in range(3):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Parent download exceeded180seconds')
                try:
                    request = urllib.request.Request(location,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=min(30,deadline-time.monotonic())) as response:
                        assert response.status == 206
                        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{PARENT_BYTES}'
                        block = response.read(end-start+2)
                    assert len(block)==end-start+1
                    break
                except Exception:
                    if attempt==2:
                        raise
            digest.update(block)
            stream.write(block)
    assert partial.stat().st_size==PARENT_BYTES and digest.hexdigest()==PARENT_ZIP_SHA
    partial.replace(destination)


def compare_restart_frame(parent, resumed):
    """Only elastic density may differ within a strain-scaled roundoff budget."""
    assert set(parent.files)==set(resumed.files)
    exact=[]
    for key in parent.files:
        if key!='elastic_energy_density':
            assert np.array_equal(parent[key],resumed[key]), f'Actual restart field changed: {key}'
            exact.append(key)
    strain=parent['strain'];d=parent['damage']
    lam=1000*.2/(1.2*.6);mu=1000/(2*1.2)
    values=np.linalg.eigvalsh(.5*(strain+strain.swapaxes(-1,-2)))
    trace=values.sum(axis=-1)
    positive=.5*lam*np.maximum(trace,0)**2+mu*np.sum(np.maximum(values,0)**2,axis=-1)
    negative=.5*lam*np.minimum(trace,0)**2+mu*np.sum(np.minimum(values,0)**2,axis=-1)
    reference=((1-1e-8)*(1-d)**2+1e-8)*positive+negative
    budget=np.maximum(128*np.finfo(float).eps*(lam+2*mu)*np.sum(strain*strain,axis=(-2,-1)),np.finfo(float).tiny)
    a,b=parent['elastic_energy_density'],resumed['elastic_energy_density']
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert (abs(a-b)<=budget).all(), 'Elastic export exceeds roundoff budget'
    assert (abs(a-reference)<=budget).all() and (abs(b-reference)<=budget).all()
    return dict(exact_fields=exact,elastic_density_bitwise_equal=bool(np.array_equal(a,b)),
        elastic_density_maximum_absolute_difference=float(abs(a-b).max()),
        elastic_density_maximum_roundoff_budget=float(budget.max()),
        rule='Only elastic_energy_density:128*eps*(lambda+2mu)*||strain||_F^2; all other fields exact')



def main(output,parent_zip=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    token=os.environ.get('GH_TOKEN')
    if not parent_zip:
        request=urllib.request.Request('https://api.github.com/repos/lililii124/jax-mpm/actions/runs/40175008288',headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(request,timeout=30) as response:official=json.load(response)
        assert official['conclusion']=='success' and official['head_sha']==PARENT_HEAD
        (output/'official-parent-run.json').write_text(json.dumps(official,indent=2))
    archive=Path(parent_zip) if parent_zip else output/'parent.zip'
    if not parent_zip:fetch(archive)
    assert archive.stat().st_size==PARENT_BYTES
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PARENT_ZIP_SHA
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()==PARENT_HEAD
        source=json.loads(z.read('proof/source.json'))
        for name,expected in source.items():
            name=name.removeprefix('/home/runner/work/jax-mpm/jax-mpm/')
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
        for name in z.namelist():
            assert (output/'parent'/name).resolve().is_relative_to((output/'parent').resolve())
        z.extractall(output/'parent')
    parent=output/'parent/proof/pilot/checkpoint_001200.npz'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==CHECKPOINT_SHA
    if parent_zip:
        print('offline source and parent admission passed')
        return
    command=[sys.executable,'-m','examples.phase_field_tension_persistent_grid','--steps','3600','--restart',str(parent),'--output',str(output/'raw')]
    measured_run(command,'continued',output,timeout=2200)
    assert parent.read_bytes()==(output/'raw/checkpoint_001200.npz').read_bytes()
    with np.load(output/'parent/proof/pilot/frame_001200.npz') as a,np.load(output/'raw/frame_001200.npz') as b:
        admission=compare_restart_frame(a,b)
    (output/'restart-admission.json').write_text(json.dumps(admission,indent=2))
    summary=json.loads((output/'raw/summary.json').read_text())
    assert summary['steps']==3600 and summary['maximum_kkt']<=1e-7 and summary['maximum_strain']<=.02
    rows=[json.loads(line) for line in (output/'raw/step-telemetry.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows]==list(range(1201,3601))
    assert all(row['nominal_cfl']<=.040000000001 for row in rows)
    (output/'numerical-admission.json').write_text(json.dumps({'continuous_steps':2400,'source_frozen':True,'physical_localization_validated':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--parent-zip')
    a=p.parse_args();main(a.output,a.parent_zip)
