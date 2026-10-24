"""Immutable multipart restoration and same-numerical-source f8 to f16 continuation."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.metal_sphere_impact_linear import digest, save_json, source_record
DEST=ROOT/'.local/metal-linear-continuation'
DELIVERY=ROOT/'.local/metal-linear-continuation-delivery'
ADMISSION=ROOT/'tools/metal_linear_f8_admission.json'
EXPECTED_RUNTIME=dict(python='3.12.13',numpy='2.5.4',jax='0.12.0.1',jaxlib='0.12.0.1',backend='cpu',x64=True)


def load(path): return json.loads(Path(path).read_text())


def validate_admission(a, allow_pending_readonly=False):
    if a['producer_run_id']!='40087609404' or a['start_frame']!=8 or a['stop_frame']!=16:
        raise ValueError('Only independently admitted original f8 producer may continue to f16')
    if a.get('independent_admission_complete') is not True and not allow_pending_readonly: raise ValueError('Independent whole-raw admission is still pending')
    artifacts=a['artifacts']
    if not 2<=len(artifacts)<=33 or len({str(x['id']) for x in artifacts})!=len(artifacts):
        raise ValueError('Invalid bounded unique artifact IDs')
    for x in artifacts:
        if not str(x['id']).isdigit() or len(x['sha256'])!=64: raise ValueError('Exact artifact ID/archive SHA required')
    if len(a['transport_sha256'])!=64 or len(a['receipt_sha256'])!=64:
        raise ValueError('Independent original manifest and receipt SHA required')


def request(url):
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    return urllib.request.urlopen(req)


def restore(*, allow_pending_readonly=False):
    a=load(ADMISSION);validate_admission(a,allow_pending_readonly=allow_pending_readonly)
    if os.environ.get('GITHUB_REF')!='refs/heads/work/metal-linear-full': raise ValueError('Fixed feature ref required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=os.environ.get('GITHUB_SHA'): raise ValueError('Exact dispatched commit required')
    for name in (*source_record()['files'], 'tools/run_metal_linear_continuation.py', 'tools/admit_metal_linear_f8.py', '.github/workflows/metal-linear-f8-admission.yml', 'tools/metal_linear_f8_admission.json', 'tools/rht_impact_prefix.py', '.github/workflows/metal-linear-continuation.yml', '.github/workflows/engineering-previews.yml'):
        if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)!=(ROOT/name).read_bytes(): raise ValueError('Git bytes differ: '+name)
    if DEST.exists(): raise FileExistsError('Original restored raw must never be overwritten')
    if shutil.disk_usage(ROOT).free<8*1024**3: raise ValueError('8GiB disk admission reserve required')
    DEST.mkdir(parents=True)
    parts=DEST/'producer-parts';parts.mkdir()
    repository=os.environ['GITHUB_REPOSITORY']
    for item in a['artifacts']:
        url=f'https://api.github.com/repos/{repository}/actions/artifacts/{item["id"]}'
        with request(url) as response: metadata=json.load(response)
        if (str(metadata['workflow_run']['id'])!=a['producer_run_id'] or metadata['expired'] or
            metadata['workflow_run']['head_sha']!=a['producer_commit'] or metadata['name']!=item['name']):
            raise ValueError('Artifact producer identity mismatch')
        if metadata['size_in_bytes']>129*1024**2: raise ValueError('Artifact ZIP exceeds bound')
        archive=DEST/f'archive-{item["id"]}.zip'
        with request(url+'/zip') as response,archive.open('wb') as target:
            shutil.copyfileobj(response,target,1024**2)
        if digest(archive)!=item['sha256']: raise ValueError('Original artifact archive differs')
        with zipfile.ZipFile(archive) as z:
            if len(z.infolist())!=1: raise ValueError('One exact payload per artifact required')
            entry=z.infolist()[0]
            name=Path(entry.filename).name
            if entry.filename!=name or (name!='transport.json' and not name.startswith('part-')) or (parts/name).exists():
                raise ValueError('Unexpected or duplicate artifact payload')
            if entry.file_size>128*1024**2: raise ValueError('Artifact payload exceeds 128MiB')
            with z.open(entry) as source,(parts/name).open('wb') as target: shutil.copyfileobj(source,target,1024**2)
        archive.unlink()
    if digest(parts/'transport.json')!=a['transport_sha256']: raise ValueError('Original transport manifest differs')
    from tools import rht_impact_prefix as transport
    transport.verify(parts)
    manifest=load(parts/'transport.json')
    restored=DEST/'producer';restored.mkdir()
    reader=transport.PartReader(parts/name for name in manifest['parts'])
    with tarfile.open(fileobj=reader,mode='r|') as tar:
        for member in tar:
            relative=Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile(): raise ValueError('Unsafe archive member')
            target=restored/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as out: shutil.copyfileobj(source,out,1024**2)
    if digest(restored/'receipt-8.json')!=a['receipt_sha256']: raise ValueError('Original f8 receipt differs')
    receipt=load(restored/'receipt-8.json')
    source=load(restored/'source.json');runtime=load(restored/'runtime.json')
    if source['commit']!=a['producer_commit'] or str(source['run_id'])!=a['producer_run_id'] or source['files']!=source_record()['files']:
        raise ValueError('Producer/source mismatch')
    if receipt['runtime']!=EXPECTED_RUNTIME or runtime!=dict(**EXPECTED_RUNTIME,psutil='7.2.2'):
        raise ValueError('Producer runtime differs')
    result=restored/'result';status=load(result/'status.json');cp=load(result/'checkpoint.json')
    if status['frame']!=8 or status['phase']!='checkpoint-ready' or status['state_valid'] is not True:
        raise ValueError('Producer is not a resumable accepted f8')
    if digest(result/'checkpoint.json')!=receipt['checkpoint_receipt_sha256']: raise ValueError('Checkpoint receipt differs')
    for name,sha in cp['files'].items():
        if digest(result/name)!=sha: raise ValueError('Checkpoint file differs: '+name)
    shutil.copytree(result,DEST/'result')
    # Exact producer artifacts remain immutable remotely; discard only verified transport copies.
    for name in manifest['parts']:
        (parts/name).unlink()
    # Original producer bytes are immutable; resume writes only the separate copy.
    save_json(DEST/'lineage.json',dict(admission_sha256=digest(ADMISSION),producer=a,source=source_record()))


def run():
    import psutil,jax,jaxlib,numpy as np,platform
    jax.config.update('jax_enable_x64',True)
    runtime=dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
    if runtime!=EXPECTED_RUNTIME or psutil.__version__!='7.2.2': raise ValueError('Exact reviewed runtime required')
    if (DEST/'monitor.json').exists(): raise FileExistsError('Single continuation attempt only')
    if load(DEST/'lineage.json')['source']!=source_record(): raise ValueError('Numerical source changed')
    begun=time.monotonic();samples=[];reason=None;peak=0
    cmd=[sys.executable,'-m','examples.metal_sphere_impact_linear','--resume','--stop-after-frame','16','--output',str(DEST/'result')]
    with (DEST/'solve.log').open('wb') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            try:
                parent=psutil.Process(proc.pid)
                usage=sum(p.memory_full_info().uss for p in [parent]+parent.children(recursive=True))
            except psutil.NoSuchProcess:
                if proc.poll() is not None: break
                reason='Live process USS vanished';usage=0
            except psutil.AccessDenied: reason='USS access denied';usage=0
            peak=max(peak,usage);raw=0
            for p in DEST.rglob('*'):
                try:
                    if p.is_file(): raw+=p.stat().st_size
                except FileNotFoundError: pass
            elapsed=time.monotonic()-begun
            samples.append(dict(elapsed_s=elapsed,tree_uss_bytes=usage,raw_bytes=raw))
            if usage>6*1024**3: reason='6GiB USS ceiling'
            # Restored original + independent resume copy + original multipart are all counted.
            if raw>4*1024**3-256*1024**2: reason='4GiB all-evidence watermark'
            if shutil.disk_usage(DEST).free<2*1024**3: reason='2GiB disk reserve'
            if elapsed>5100: reason='85 minute wall ceiling'
            if reason: os.killpg(proc.pid,signal.SIGKILL);break
            time.sleep(1)
        code=proc.wait()
    save_json(DEST/'monitor.json',dict(command=cmd,returncode=code,stop_reason=reason,peak_tree_uss_bytes=peak,samples=samples))
    if code or reason: raise RuntimeError('Failed raw retained without rollback')
    status=load(DEST/'result/status.json')
    if status['frame']!=16 or status['state_valid'] is not True: raise ValueError('f16 not attained')
    save_json(DEST/'receipt-16.json',dict(lineage=load(DEST/'lineage.json'),runtime=runtime,status=status,checkpoint_receipt_sha256=digest(DEST/'result/checkpoint.json'),physical_validation_passed=False))


def package():
    from tools import rht_impact_prefix as base
    base.DEST=DEST;base.DELIVERY=DELIVERY;base.package()


if __name__=='__main__':
    action=sys.argv[1]
    if action in ('restore','run','package'): globals()[action]()
    elif action=='verify':
        from tools.run_metal_linear_full import transport
        transport('verify',sys.argv[2],sys.argv[3])
    else: raise ValueError('Unsupported action')
