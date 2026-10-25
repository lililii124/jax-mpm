"""One two-step observer-on/off diagnostic only; no loading extension."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import numpy as np
from run_phase_field_tension_localization import measured_run
PARENT_BYTES=47475559
PARENT_ARTIFACT=12593179328
PARENT_ZIP_SHA='e71ea69ee751d5f6b9437963e40ec495eff3ad9e6d4e3e332dfe37e27fb061be'
CP_SHA='79a7329f144db7256aa6ada14d8c9db11c76250fe143aed504414a5104843beb'
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




def main(output):
    origin=Path.cwd();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    archive=output/'parent.zip';fetch(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()=='3fba4a1f6541da4c496c752493b33abec652f4b7'
        data=z.read('partitioned/raw/checkpoint_003600.npz')
    assert hashlib.sha256(data).hexdigest()==CP_SHA
    parent=output/'original3600.npz';parent.write_bytes(data)
    with np.load(parent,allow_pickle=False) as f:base={k:f[k].copy() for k in f.files}
    original_binding=json.loads(str(base['binding']))
    manifests={}
    for name in ['on','off']:
        tree=output/('source-'+name)
        for path,expected in original_binding['source_hashes'].items():
            data=(origin/path).read_bytes();assert hashlib.sha256(data).hexdigest()==expected,path
            target=tree/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (tree/'examples/__init__.py').write_bytes((origin/'examples/__init__.py').read_bytes())
        arrays={k:v.copy() for k,v in base.items()};binding=dict(original_binding)
        if name=='off':
            helper=tree/'examples/phase_field_tension_qualification.py'
            module=ast.parse(helper.read_text());immutable=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='immutable_unit_seed')
            helper.write_text("import numpy as np\nPOLICY='diagnostic-observer-off-v1'\n"+ast.unparse(immutable)+"\ndef inspect(*args):\n    return dict(passed=True,diagnostic_only=True,policy=POLICY)\n")
            hashes=dict(binding['source_hashes']);hashes['examples/phase_field_tension_qualification.py']=hashlib.sha256(helper.read_bytes()).hexdigest();binding['source_hashes']=hashes;binding['strain_qualification_policy']='diagnostic-observer-off-v1'
        arrays['binding']=np.asarray(json.dumps(binding,sort_keys=True));fork=output/(name+'-input.npz');np.savez_compressed(fork,**arrays)
        with np.load(parent) as a,np.load(fork) as b:assert all(np.array_equal(a[k],b[k]) for k in a.files if k!='binding')
        manifests[name]=dict(binding=binding,checkpoint_sha=hashlib.sha256(fork.read_bytes()).hexdigest(),diagnostic_only=True)
        os.chdir(tree)
        try:measured_run([sys.executable,'-m','examples.phase_field_tension_persistent_grid','--steps','3602','--restart',str(fork),'--output',str(output/name)],name,output,timeout=240)
        finally:os.chdir(origin)
    a=np.load(output/'on/checkpoint_003602.npz');b=np.load(output/'off/checkpoint_003602.npz');compare={}
    for key in ['nodal_displacement','nodal_velocity','history','nodal_damage']:
        delta=abs(a[key]-b[key]);scale=max(float(abs(a[key]).max()),float(abs(b[key]).max()),1e-30)
        compare[key]=dict(bitwise_equal=bool(np.array_equal(a[key],b[key])),maximum_absolute_difference=float(delta.max()),global_relative_difference=float(delta.max()/scale))
        assert float(delta.max()/scale)<1e-9,'Observer feedback above numerical-noise diagnostic budget'
    (output/'observer-diagnostic.json').write_text(json.dumps(dict(scope='exactly two steps each; no production continuation qualification',cases=manifests,comparison=compare),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);main(p.parse_args().output)
