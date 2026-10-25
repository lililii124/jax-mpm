"""Case-specific seed-jump observer correction; strict carry3200 to3600 only."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import numpy as np
from run_phase_field_tension_localization import measured_run
from run_phase_field_tension_to3600 import compare_restart_frame
from phase_field_bspline_sampling import evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.phase_field_tension_qualification import POLICY,immutable_unit_seed,inspect
PARENT_BYTES=183481959
PARENT_ARTIFACT=12592306876
PARENT_ZIP_SHA='6095095ba0d5c8f1e1357db635a9022619eccfcc651f1b34d4c55319196dc8c4'
PARENT_HEAD='c6d538b2ebe1df425e733586ecae0382b963b4e6'
CHECKPOINT_SHA='2f6a9154fa19d47204272904918d40e7f160f94438b14d07a54ff27bb9038a7c'
PHYSICS_AST={
  "advance": "2892ca17ba878439f60d1635d97ff2b715a47612c38187f33f6a9d8daf74539e",
  "strain_of": "183e6d5a2d963e62f5dc3e5c8108a4c3a2f4a29d68fdadcc63419514d794b48d",
  "particle_state": "ade12d9dbdf68497d86d219835a34711ed9af4a73d76699e594d21b810582d8e",
  "potential": "b9b6ac1c53280f97f4c4be52455ad47898a98f779e0f2b7b28481737ca9c6539"
}
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



def prepare(output,parent_zip=None):
    output.mkdir(parents=True,exist_ok=False)
    archive=Path(parent_zip) if parent_zip else output/'parent.zip'
    if not parent_zip:
        request=urllib.request.Request('https://api.github.com/repos/lililii124/jax-mpm/actions/runs/40175930114',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
        with urllib.request.urlopen(request,timeout=30) as response:official=json.load(response)
        assert official['conclusion']=='failure' and official['head_sha']==PARENT_HEAD
        (output/'official-parent.json').write_text(json.dumps(official,indent=2))
        fetch(archive)
    assert archive.stat().st_size==PARENT_BYTES and hashlib.sha256(archive.read_bytes()).hexdigest()==PARENT_ZIP_SHA
    members={'continuation/raw/checkpoint_003200.npz':'original-checkpoint3200.npz',
        'continuation/raw/frame_003200.npz':'original-frame3200.npz',
        'continuation/raw/frame_003274.npz':'old-guard-frame3274.npz',
        'continuation/parent/proof/whole/frame_000000.npz':'initial-frame0.npz',
        'continuation/continued.resources.json':'original-resources.json'}
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()==PARENT_HEAD
        for old,new in members.items():(output/new).write_bytes(z.read(old))
    original=output/'original-checkpoint3200.npz'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(original,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    before=json.loads(str(arrays['binding']));after=dict(before);hashes=dict(before['source_hashes'])
    producer='examples/phase_field_tension_persistent_grid.py'
    assert hashes[producer]=='73f02d01035ec525fc0aa95890a81ceefdb9b4e613502a3787e128ef82810a49'
    def content(path):return Path(path).read_bytes().replace(b'\r\n',b'\n') if parent_zip else Path(path).read_bytes()
    for name,expected in hashes.items():
        if name!=producer:assert hashlib.sha256(content(name)).hexdigest()==expected,name
    tree=ast.parse(content(producer))
    for name,expected in PHYSICS_AST.items():
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==expected,name
    hashes[producer]=hashlib.sha256(content(producer)).hexdigest()
    helper='examples/phase_field_tension_qualification.py';hashes[helper]=hashlib.sha256(content(helper)).hexdigest()
    after['source_hashes']=hashes;after['strain_qualification_policy']=POLICY
    arrays['binding']=np.asarray(json.dumps(after,sort_keys=True))
    fork=output/'policy-checkpoint3200.npz';np.savez_compressed(fork,**arrays)
    with np.load(original,allow_pickle=False) as a,np.load(fork,allow_pickle=False) as b:
        exact=[k for k in a.files if k!='binding'];assert all(np.array_equal(a[k],b[k]) for k in exact)
    assert {k for k in after if k not in before or after[k]!=before[k]}=={'source_hashes','strain_qualification_policy'}
    initial=np.load(output/'initial-frame0.npz');f=np.load(output/'original-frame3200.npz');x=f['reference']
    seed=immutable_unit_seed(x,initial['damage'],before['dx'])
    gradient=np.stack([evaluate(x,f['grid_coordinates'],f['nodal_displacement'][:,i],before['dx'],gradient=True)[1] for i in range(3)],axis=1)
    admission=inspect(gradient,f['position']-x,f['damage'],f['volume'],seed);assert admission['passed'] and int(seed.sum())==376
    np.save(output/'immutable-initial-unit-seed.npy',seed)
    manifest=dict(old_checkpoint_sha256=CHECKPOINT_SHA,new_checkpoint_sha256=hashlib.sha256(fork.read_bytes()).hexdigest(),binding_changes=['source_hashes','strain_qualification_policy'],strictly_equal_carry_fields=exact,physical_step_time_unchanged=True,initial_qualification=admission,old_failure_preserved=True,source_physics_ast_unchanged=True)
    (output/'binding-fork.json').write_text(json.dumps(manifest,indent=2))
    return fork

def main(output,parent_zip=None):
    output=Path(output);fork=prepare(output,parent_zip)
    if parent_zip:
        print('offline source, carry, immutable mask and partition admission passed');return
    measured_run([sys.executable,'-m','examples.phase_field_tension_persistent_grid','--steps','3600','--restart',str(fork),'--output',str(output/'raw')],'continued',output,timeout=2100)
    assert fork.read_bytes()==(output/'raw/checkpoint_003200.npz').read_bytes()
    checks={}
    for step in [3200,3274]:
        old=output/('original-frame3200.npz' if step==3200 else 'old-guard-frame3274.npz')
        with np.load(old) as a,np.load(output/f'raw/frame_{step:06d}.npz') as b:checks[str(step)]=compare_restart_frame(a,b)
    rows=[json.loads(line) for line in (output/'raw/strain-qualification.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(3201,3601)) and all(r['passed'] for r in rows)
    (output/'numerical-admission.json').write_text(json.dumps(dict(strict_physical_state_comparisons=checks,accepted_through=3600,localized_new_crack_validated=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--parent-zip');a=p.parse_args();main(a.output,a.parent_zip)
