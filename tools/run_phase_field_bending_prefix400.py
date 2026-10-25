"""Frozen-source, frozen-state 100->400 AT2 bending prefix; no medium admission."""
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
from prove_phase_field_bending_restart import measured_run

PARENT_RUN = 40161406959
PARENT_ARTIFACT = 12587055824
PARENT_HEAD = '217dc422cae6d847fd4b2a9288521799e89e1581'
PARENT_ZIP_SHA = '3c078e7e44644746e054cec38ce18b10379ae47329f62c0da35dda328cd67865'
PARENT_BYTES = 44448405
CHECKPOINT_SHA = 'fe677e65c1fe74ee6d8443f7dca80ced0d429eb262dc46f98343397e2b7cc057'


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


def prepare(output, parent_zip=None):
    output.mkdir(parents=True,exist_ok=False)
    archive = output/'parent-official.zip'
    if parent_zip is None:
        fetch(archive)
    else:
        shutil.copyfile(parent_zip,archive)
    assert archive.stat().st_size==PARENT_BYTES
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PARENT_ZIP_SHA
    source_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()==PARENT_HEAD
        for line in z.read('source-sha256.txt').decode().splitlines():
            expected,name=line.split(maxsplit=1)
            path=(source_root/name).resolve()
            assert path.is_relative_to(source_root)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
        result=json.loads(z.read('pilot/capacity-pilot.json'))
        assert result['status']=='production_mesh_exact_parity_resource_pilot_passed'
        assert result['exact_array_checks']==36 and result['segment_sampled_uss_reserve_gate']
        for name in ['source-head.txt','source-sha256.txt','pilot/capacity-pilot.json',
                     'pilot/whole/checkpoint_000100.npz','pilot/whole/frame_000100.npz',
                     'pilot/whole/history.csv','pilot/whole/summary.json']:
            destination=output/'parent'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(z.read(name))
    checkpoint=output/'parent/pilot/whole/checkpoint_000100.npz'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(checkpoint,allow_pickle=False) as state:
        binding=json.loads(str(state['binding']))
        assert int(state['step'])==100
        for key,value in dict(dx=.025,length=.05,density=1.,gc=.005,speed=.12,
                              displacement=.035,cfl=.04,required_steps=9923,particles=53144).items():
            assert binding[key]==value,(key,binding[key])
        for name,digest in binding['source_hashes'].items():
            assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,name
        assert all(np.isfinite(state[key]).all() for key in state.files if key!='binding')
        assert float(state['maximum_kkt'])<=1e-7 and float(state['maximum_strain'])<=.1
        assert state['nodal_damage'].min()>=0 and state['nodal_damage'].max()<=1
        assert state['history'].min()>=0 and abs(state['volume'].sum()-.098304)<1e-14
    proof=dict(parent_run=PARENT_RUN,parent_artifact=PARENT_ARTIFACT,parent_head=PARENT_HEAD,
        parent_zip_sha256=PARENT_ZIP_SHA,checkpoint_sha256=CHECKPOINT_SHA,
        start_step=100,absolute_stop_step=400,advanced_steps=300,
        unchanged_physics_binding=binding,source_closure_verified=True,
        medium_admitted=False,crack_validated=False)
    (output/'preflight.json').write_text(json.dumps(proof,indent=2))
    return checkpoint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parent-zip')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    checkpoint=prepare(output,args.parent_zip)
    os.environ.pop('GH_TOKEN',None)
    if args.preflight_only:
        print('Frozen ZIP, exact source closure, binding and numerical checkpoint preflight passed',flush=True)
        return
    command=[sys.executable,'-m','examples.phase_field_bending_persistent_grid',
        '--dx','.025','--length','.05','--density','1','--gc','.005','--speed','.12',
        '--cfl','.04','--displacement','.035','--steps','400','--restart',str(checkpoint),
        '--output',str(output/'raw')]
    usage=measured_run(command,'prefix400',output,timeout=2400)
    assert usage['samples']>0 and usage['peak_sampled_uss_bytes']>0
    assert max(usage['peak_sampled_uss_bytes'],usage['peak_sampled_rss_bytes'])<6*2**30
    assert usage['peak_sampled_uss_bytes']<=.8*6*2**30
    summary=json.loads((output/'raw/summary.json').read_text())
    assert summary['steps']==400 and summary['resumed_from_step']==100
    assert summary['maximum_kkt']<=1e-7 and summary['maximum_strain']<=.1
    assert hashlib.sha256((output/'raw/checkpoint_000100.npz').read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(output/'parent/pilot/whole/frame_000100.npz') as parent, np.load(output/'raw/frame_000100.npz') as resumed:
        assert set(parent.files)==set(resumed.files)
        assert all(np.array_equal(parent[key],resumed[key]) for key in parent.files)
    with np.load(output/'raw/checkpoint_000400.npz') as end:
        assert int(end['step'])==400 and all(np.isfinite(end[key]).all() for key in end.files if key!='binding')
    result=dict(status='bounded_prefix400_completed_unvalidated',resources=usage,
        start_step=100,stop_step=400,advanced_steps=300,
        reached_total_requested_loading=summary['reached_requested_displacement'],
        full_cfl_refinement_admitted=False,medium_admitted=False,crack_validated=False)
    (output/'prefix-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
