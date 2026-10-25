"""Authenticated restore/deploy/guard/package controller; never dispatches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.local/metal-v6-relaxed125'
ADMISSION=ROOT/'tools/metal_v6_relaxed125_admission.json'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def context(a):
    if os.environ.get('GITHUB_REPOSITORY')!='lililii124/jax-mpm' or os.environ.get('GITHUB_REF')!=a['fixed_feature_ref']: raise ValueError('Exact private feature repository/ref required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=os.environ.get('GITHUB_SHA'): raise ValueError('Dispatch commit differs')
    workflow=os.environ.get('GITHUB_WORKFLOW_REF','').split('@')[0]
    template='.github/workflows/metal-v6-relaxed125.yml'
    alias='.github/workflows/engineering-previews.yml'
    if workflow not in ('lililii124/jax-mpm/'+template,'lililii124/jax-mpm/'+alias): raise ValueError('Exact isolated workflow or frozen registered alias required')
    if workflow.endswith('/engineering-previews.yml') and (ROOT/alias).read_bytes()!=(ROOT/template).read_bytes(): raise ValueError('Registered alias must equal frozen conservative relaxation template byte for byte')
    for n,h in a['source_sha256'].items():
        if sha(ROOT/n)!=h or subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)!=(ROOT/n).read_bytes(): raise ValueError('Exact committed source differs: '+n)
    for n in (str(ADMISSION.relative_to(ROOT)).replace('\\','/'),template,alias):
        if subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)!=(ROOT/n).read_bytes(): raise ValueError('Exact committed admission/workflow required')
    return dict(commit=commit,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],source_sha256=a['source_sha256'],admission_sha256=sha(ADMISSION),workflow_sha256=sha(ROOT/template),registered_alias_sha256=sha(ROOT/alias),workflow=workflow)


def restore(a):
    if DEST.exists(): raise FileExistsError('Fresh restoration required')
    identity=context(a);delivery=ROOT/'.local/metal-v6-relaxed-input'
    if sha(delivery/'transport.json')!=a['transport_sha256']: raise ValueError('Independent input manifest differs')
    manifest=json.loads((delivery/'transport.json').read_text())
    for name,info in manifest['parts'].items():
        if Path(name).name!=name or sha(delivery/name)!=info['sha256'] or (delivery/name).stat().st_size!=info['bytes']: raise ValueError('Whole input part differs')
    from tools.rht_impact_prefix import PartReader
    audit=DEST/'audit';audit.mkdir(parents=True)
    seen=set()
    with tarfile.open(fileobj=PartReader(delivery/n for n in manifest['parts']),mode='r|') as stream:
        for member in stream:
            rel=Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name or not member.isfile() or member.name in seen or member.name not in manifest['files']: raise ValueError('Unsafe/unexpected/duplicate member')
            info=manifest['files'][member.name]
            if member.size!=info['bytes']: raise ValueError('Member size differs')
            target=audit/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,target.open('xb') as out: shutil.copyfileobj(source,out)
            if sha(target)!=info['sha256']: raise ValueError('Whole raw input differs')
            seen.add(member.name)
    if seen!=set(manifest['files']): raise ValueError('Input member coverage incomplete')
    for name,h in a['input_sha256'].items():
        if sha(audit/name)!=h: raise ValueError('Independently frozen input differs')
    source=DEST/'runtime-source';shutil.copytree(audit/'runtime-source',source)
    for name in a['source_sha256']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    write(DEST/'control-source.json',identity)
    write(DEST/'source-before.json',{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
    shutil.copy2(ADMISSION,DEST/'admission.json')


def install(a):
    context(a)
    deps=DEST/'audit/dependencies'
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-r',str(deps/'requirements.txt'),str(deps/'jax-0.12.1.dev20261022-py3-none-any.whl')],check=True)


def run(a):
    if context(a)!=json.loads((DEST/'control-source.json').read_text()): raise ValueError('Controller context changed')
    import psutil
    from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group
    source=DEST/'runtime-source';command=[sys.executable,'-m','examples.metal_fragment_v6_relaxed125','--audit',str(DEST/'audit'),'--admission',str(DEST/'admission.json'),'--output',str(DEST/'result')]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    begun=time.monotonic();peak=0;failure=None;child=None
    try:
        with (DEST/'numerical.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,owned_session_uss(child.pid,psutil))
                if peak>6*1024**3 or time.monotonic()-begun>1600 or shutil.disk_usage(ROOT).free<3*1024**3: raise RuntimeError('Owned USS/wall/disk guard failed')
                if sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())>1.75*1024**3: raise RuntimeError('Raw1.75GiB cap')
                time.sleep(.3)
            if child.returncode: raise RuntimeError('Numerical diagnostic rejected or failed')
    except BaseException as error: failure=type(error).__name__+': '+str(error)
    finally:
        if child is not None: terminate_owned_group(child)
    before=json.loads((DEST/'source-before.json').read_text());after={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    unchanged=before==after
    write(DEST/'run-receipt.json',dict(admitted=failure is None and unchanged,failure=failure,source_unchanged=unchanged,peak_owned_uss_bytes=peak,elapsed_s=time.monotonic()-begun,production_restart_allowed=False))
    if failure or not unchanged: raise RuntimeError('Diagnostic failed; preserve raw')


def package(a):
    # Reuse whole-member, bounded-part transport; no lossy evidence selection.
    from tools import rht_impact_prefix as base
    base.ROOT=ROOT;base.DEST=DEST;base.RAW=DEST;base.DELIVERY=ROOT/'.local/metal-v6-relaxed-delivery'
    base.package()
    manifest=base.DELIVERY/'transport.json'
    print(json.dumps(dict(transport_sha256=sha(manifest),parts=json.loads(manifest.read_text())['parts']),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('restore','install','run','package'));args=p.parse_args()
    a=json.loads(ADMISSION.read_text());globals()[args.phase](a)
