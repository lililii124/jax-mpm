"""Authenticated Linux single-2ps dynamic/replay controller; no media stage."""
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,tarfile,time
import urllib.request,urllib.error,urllib.parse,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.local/metal-affine-dynamic125'
CACHE=ROOT/'.local/metal-affine-dynamic-input'
ADMISSION=ROOT/'tools/metal_affine_dynamic125_cloud_admission.json'
GIB=1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def context(a):
    if os.environ.get('GITHUB_REPOSITORY')!='lililii124/jax-mpm' or os.environ.get('GITHUB_REF')!=a['fixed_ref']:raise ValueError('Exact isolated repository/ref required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=os.environ.get('GITHUB_SHA'):raise ValueError('Dispatch SHA differs')
    template='.github/workflows/metal-affine-dynamic125.yml';alias='.github/workflows/engineering-previews.yml'
    if (ROOT/template).read_bytes()!=(ROOT/alias).read_bytes():raise ValueError('Registered alias differs from bounded workflow')
    workflow=os.environ.get('GITHUB_WORKFLOW_REF','').split('@')[0]
    if workflow not in ('lililii124/jax-mpm/'+template,'lililii124/jax-mpm/'+alias):raise ValueError('Unexpected workflow')
    for name,pin in a['source_sha256'].items():
        if sha(ROOT/name)!=pin or (ROOT/name).read_bytes()!=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT):raise ValueError('Committed source differs: '+name)
    for name in (str(ADMISSION.relative_to(ROOT)),template,alias):
        if (ROOT/name).read_bytes()!=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT):raise ValueError('Uncommitted workflow/admission')
    return dict(commit=head,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],
        admission_sha256=sha(ADMISSION),workflow_sha256=sha(ROOT/template),source_sha256=a['source_sha256'])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def restore(a):
    identity=context(a)
    if DEST.exists() or CACHE.exists():raise FileExistsError('Fresh input and output required')
    DEST.mkdir(parents=True);CACHE.mkdir(parents=True)
    write(DEST/'control-source.json',identity)
    for name in list(a['source_sha256'])+[str(ADMISSION.relative_to(ROOT)),'.github/workflows/metal-affine-dynamic125.yml',
            '.github/workflows/engineering-previews.yml','evidence/metal-affine-local-resource-failures.json','evidence/metal-affine-original-be-correction-notice.json']:
        target=DEST/'committed-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    pin=a['parent_artifact'];api=f"https://api.github.com/repos/lililii124/jax-mpm/actions/artifacts/{pin['id']}"
    headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'}
    opener=urllib.request.build_opener(NoRedirect)
    with opener.open(urllib.request.Request(api,headers=headers),timeout=30) as response:meta=json.load(response)
    if meta['expired'] or meta['workflow_run']['id']!=pin['run'] or meta['workflow_run']['head_sha']!=pin['commit'] or meta['digest']!='sha256:'+pin['sha256'] or meta['size_in_bytes']!=pin['bytes']:raise ValueError('Authenticated parent artifact identity differs')
    try:
        opener.open(urllib.request.Request(api+'/zip',headers=headers),timeout=30);raise ValueError('Expected signed redirect')
    except urllib.error.HTTPError as response:
        if response.code!=302:raise
        location=response.headers['Location'];response.close()
    del headers
    parsed=urllib.parse.urlsplit(location)
    if parsed.scheme!='https' or parsed.username or parsed.password:raise ValueError('Invalid signed URL')
    archive=CACHE/'whole-parent.zip';complete=False
    for attempt in range(3):
        try:
            with urllib.request.urlopen(location,timeout=60) as response,archive.open('wb') as target:
                total=0
                for block in iter(lambda:response.read(1024**2),b''):
                    total+=len(block)
                    if total>pin['bytes']:raise ValueError('Parent exceeds exact size')
                    target.write(block)
            if archive.stat().st_size!=pin['bytes'] or sha(archive)!=pin['sha256']:raise ValueError('Whole parent ZIP differs')
            complete=True;break
        except Exception:
            if attempt<2:time.sleep(2**attempt)
    if not complete:raise RuntimeError('Signed artifact download failed bounded verification; token never forwarded')
    delivery=CACHE/'delivery';delivery.mkdir()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if Path(info.filename).name!=info.filename or info.is_dir():raise ValueError('Unexpected parent transport member')
            with z.open(info) as source,(delivery/info.filename).open('xb') as out:shutil.copyfileobj(source,out)
    manifest=json.loads((delivery/'transport.json').read_text())
    if sha(delivery/'transport.json')!=a['parent_transport_sha256']:raise ValueError('Parent transport manifest differs')
    for name,part in manifest['parts'].items():
        if Path(name).name!=name or sha(delivery/name)!=part['sha256'] or (delivery/name).stat().st_size!=part['bytes']:raise ValueError('Parent tarpart differs')
    from tools.rht_impact_prefix import PartReader
    raw=CACHE/'raw';raw.mkdir();seen=set()
    with tarfile.open(fileobj=PartReader(delivery/name for name in manifest['parts']),mode='r|') as stream:
        for member in stream:
            rel=Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name or not member.isfile() or member.name in seen or member.name not in manifest['files']:raise ValueError('Unsafe/incomplete input member')
            pin_member=manifest['files'][member.name]
            if member.size!=pin_member['bytes']:raise ValueError('Parent member size differs')
            target=raw/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,target.open('xb') as out:shutil.copyfileobj(source,out)
            if sha(target)!=pin_member['sha256']:raise ValueError('Parent member hash differs')
            seen.add(member.name)
    if seen!=set(manifest['files']):raise ValueError('Parent member coverage differs')
    for original,target in a['input_targets'].items():
        if sha(raw/original)!=a['input_sha256'][original]:raise ValueError('Frozen seed/runtime input differs')
        path=DEST/target;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw/original,path)
    source=DEST/'runtime-source';shutil.copytree(raw/'runtime-source',source)
    old=json.loads((DEST/'inputs/original-admission.json').read_text())
    for name in a['original_material_source']:
        if sha(source/name)!=old['source_sha256'][name]:raise ValueError('Original material source differs: '+name)
    for name in a['overlay_source']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    write(DEST/'source-before.json',{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
    write(DEST/'parent-archive-verification.json',dict(whole_archive_sha256=sha(archive),bytes=archive.stat().st_size,
        parent=meta,verified_parent_members=len(seen),all_parent_members_verified=True,token_forwarded_to_signed_url=False))
    shutil.copy2(delivery/'transport.json',DEST/'parent-transport.json');shutil.copy2(ADMISSION,DEST/'cloud-admission.json')


def install(a):
    context(a)
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-r',str(DEST/'dependencies/requirements.txt'),str(DEST/'dependencies/jax-0.12.1.dev20261022-py3-none-any.whl')],check=True)
    actual={name:importlib.metadata.version(name) for name in a['runtime_versions']}
    write(DEST/'installed-runtime.json',dict(python=platform.python_version(),packages=actual))
    if platform.python_version()!='3.12.13' or actual!=a['runtime_versions']:raise ValueError('Exact preserved runtime differs')


def run(a):
    if context(a)!=json.loads((DEST/'control-source.json').read_text()):raise ValueError('Controller identity changed')
    if sys.platform!='linux':raise ValueError('Bounded heavy computation requires Linux')
    import ctypes,psutil
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise RuntimeError('Subreaper required')
    from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group
    if Path('/proc/cpuinfo').exists():shutil.copy2('/proc/cpuinfo',DEST/'host-cpuinfo.txt')
    memory=psutil.virtual_memory();minimum=int(1.5*GIB);limit=min(8*GIB,memory.total-minimum)
    if memory.available<3*GIB or limit<3*GIB:raise RuntimeError('Insufficient real host memory before computation')
    source=DEST/'runtime-source';env=os.environ.copy()
    env.update(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    record=dict(completed=False,peak_owned_uss_bytes=0,minimum_observed_available_bytes=memory.available,uss_limit_bytes=limit,
        minimum_available_bytes=minimum,seconds_limit=900,host=dict(platform=platform.platform(),cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),memory_total_bytes=memory.total,memory_available_bytes=memory.available),runs=[],production_admitted=False,public_media_allowed=False)
    write(DEST/'pip-freeze.json',subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True).splitlines())
    started=time.monotonic();failure=None
    try:
        for label in ('result','replay'):
            command=[sys.executable,'-m','tools.run_metal_affine_dynamic125','--state',str(DEST/'inputs/initial.npz'),
                '--epoch',str(DEST/'inputs/epoch-initial.npz'),'--admission',str(DEST/'inputs/original-admission.json'),
                '--output',str(DEST/label),'--dt','2e-12','--steps','1']
            child=None
            try:
                with (DEST/f'{label}.log').open('xb') as log:
                    child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        uss=owned_session_uss(child.pid,psutil);available=psutil.virtual_memory().available
                        record['peak_owned_uss_bytes']=max(record['peak_owned_uss_bytes'],uss)
                        record['minimum_observed_available_bytes']=min(record['minimum_observed_available_bytes'],available)
                        reason=('owned USS cap' if uss>limit else 'available memory cap' if available<minimum else
                            'wall cap' if time.monotonic()-started>900 else 'disk cap' if shutil.disk_usage(DEST).free<3*GIB else
                            'complete new raw cap' if sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())>2*GIB else None)
                        if reason:raise RuntimeError(reason)
                        write(DEST/'resource-receipt.json',record);time.sleep(.5)
                    if child.returncode:raise RuntimeError(f'{label} rejected; full raw preserved')
            finally:
                if child is not None:terminate_owned_group(child)
            result=json.loads((DEST/label/'result.json').read_text())
            if result['status']!='bounded-window-passed' or result['accepted_steps']!=1:raise ValueError('No accepted real dynamic step')
            record['runs'].append(dict(label=label,status=result['status'],accepted_steps=result['accepted_steps']))
        import numpy as np
        first=sorted((DEST/'result').glob('*-accepted.npz'))[-1];second=sorted((DEST/'replay').glob('*-accepted.npz'))[-1]
        with np.load(first,allow_pickle=False) as x,np.load(second,allow_pickle=False) as y:
            equal=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            record['replay_bitwise_all_arrays']=equal;record['replay_array_count']=len(x.files)
            if not equal:raise ValueError('Same-source/dt replay differs')
        record['completed']=True
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        before=json.loads((DEST/'source-before.json').read_text());after={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
        record.update(failure=failure,source_unchanged=before==after,elapsed_seconds=time.monotonic()-started)
        write(DEST/'resource-receipt.json',record)
    if failure or not record['source_unchanged']:raise RuntimeError('Bounded first-step/replay failed; retain whole raw')


def package(a):
    from tools import rht_impact_prefix as base
    if not DEST.exists():DEST.mkdir(parents=True);write(DEST/'preflight-failure.json',dict(status='no-numerical-run'))
    base.ROOT=ROOT;base.DEST=DEST;base.RAW=DEST;base.DELIVERY=ROOT/'.local/metal-affine-dynamic-delivery'
    base.package()
    print(json.dumps(dict(transport_sha256=sha(base.DELIVERY/'transport.json'),parts=json.loads((base.DELIVERY/'transport.json').read_text())['parts'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('restore','install','run','package'));args=p.parse_args()
    try:globals()[args.phase](json.loads(ADMISSION.read_text()))
    except BaseException as error:
        if DEST.exists():write(DEST/f'{args.phase}-failure.json',dict(type=type(error).__name__,message=str(error),numerical_admission=False))
        raise
