"""Offline, allocated-container entry point; copied to capsule/run_cell.py."""
import argparse,ctypes,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
GIB=1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def memory_group():
    candidates=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2);relative=relative.lstrip('/')
        if not controllers:
            for p in (Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup')/relative):
                candidates.append((p/'memory.max',p/'memory.current'))
        elif 'memory' in controllers.split(','):
            for p in (Path('/sys/fs/cgroup/memory'),Path('/sys/fs/cgroup/memory')/relative):
                candidates.append((p/'memory.limit_in_bytes',p/'memory.usage_in_bytes'))
    values=[]
    for limit,current in candidates:
        if not limit.is_file() or not current.is_file():continue
        text=limit.read_text().strip()
        if text=='max':continue
        cap=int(text)
        if cap<2**60:values.append((cap,int(current.read_text()),str(limit)))
    if not values:raise RuntimeError('A finite Docker memory cgroup is required')
    return min(values,key=lambda x:x[0])
def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,pin in manifest['files'].items():
        path=root/name
        if path.stat().st_size!=pin['bytes'] or sha(path)!=pin['sha256']:raise ValueError('Capsule differs: '+name)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capsule',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--cell',choices=('dt1ps','dt0p5ps','both'),default='both');p.add_argument('--output',type=Path)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=a.capsule.resolve()
    manifest=verify(root);contract=json.loads((root/'contract.json').read_text())
    if a.verify_only:
        print(json.dumps(dict(capsule_verified=True,files=len(manifest['files']),source_commit=contract['source_commit'])));return
    if sys.platform!='linux' or platform.python_version()!='3.12.13':raise RuntimeError('Exact Linux/Python3.12.13 environment required')
    actual={name:importlib.metadata.version(name) for name in contract['runtime_versions']}
    if actual!=contract['runtime_versions']:raise RuntimeError('Dedicated dev-client image runtime differs')
    # Verify actual installed pure-Python client bytes against the preserved
    # original producer wheel, without retaining a JAX backend in this guard.
    dist=importlib.metadata.distribution('jax');client_files=0
    with zipfile.ZipFile(root/'wheels/jax-0.12.1.dev20261022-py3-none-any.whl') as z:
        for info in z.infolist():
            if info.filename.startswith('jax/') and info.filename.endswith('.py'):
                if sha(Path(dist.locate_file(info.filename)))!=hashlib.sha256(z.read(info)).hexdigest():raise RuntimeError('Installed dev client bytes differ')
                client_files+=1
    affinity=sorted(os.sched_getaffinity(0))
    if affinity!=contract['cpu_affinity']:raise RuntimeError('Use the assigned CPU40-43 allocation')
    cap,used,cap_path=memory_group()
    if cap>8*GIB or cap<6*GIB:raise RuntimeError('Use the assigned 8GiB Docker memory allocation')
    if a.output is None:raise ValueError('Fresh output directory required')
    output=a.output.resolve()
    if output.is_relative_to(root):raise ValueError('Output must be outside the immutable capsule')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Preserve prior output; choose a new empty directory')
    import psutil
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise RuntimeError('Linux subreaper required')
    source=root/'runtime-source';sys.path.insert(0,str(source))
    from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group
    env=os.environ.copy();env.update(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    summary=dict(source_commit=contract['source_commit'],expected_image_id=contract['image_id'],image_identity_supplied_by_launch_command=True,
        installed_runtime=actual,verified_client_python_files=client_files,cpu_affinity=affinity,cgroup_limit_bytes=cap,cgroup_limit_file=cap_path,
        host_platform=platform.platform(),host_cpu_count=os.cpu_count(),host_memory_total_bytes=psutil.virtual_memory().total,
        cells=[],production_admitted=False,public_media_allowed=False)
    if Path('/proc/cpuinfo').exists():(output/'host-cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
    cells=('dt1ps','dt0p5ps') if a.cell=='both' else (a.cell,)
    for cell in cells:
        dt,steps=contract['cells'][cell]['dt_s'],contract['cells'][cell]['steps']
        dest=output/cell;dest.mkdir();started=time.monotonic()
        record=dict(cell=cell,dt_s=dt,steps=steps,peak_owned_uss_bytes=0,minimum_cgroup_headroom_bytes=cap-used,
            uss_cap_bytes=4*GIB,minimum_headroom_bytes=2*GIB,wall_cap_s=900,status='running',source_commit=contract['source_commit'])
        try:
            for label in ('result',):
                command=[sys.executable,'-m','tools.run_metal_affine_dynamic125','--state',str(root/'inputs/initial.npz'),
                    '--epoch',str(root/'inputs/epoch-initial.npz'),'--admission',str(root/'inputs/original-admission.json'),
                    '--output',str(dest/label),'--dt',str(dt),'--steps',str(steps),'--source-commit',contract['source_commit']]
                child=None;last_print=0.
                try:
                    with (dest/f'{label}.log').open('xb') as log:
                        child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        while child.poll() is None:
                            uss=owned_session_uss(child.pid,psutil);cap,used,_=memory_group()
                            free=min(cap-used,psutil.virtual_memory().available)
                            record['peak_owned_uss_bytes']=max(record['peak_owned_uss_bytes'],uss)
                            record['minimum_cgroup_headroom_bytes']=min(record['minimum_cgroup_headroom_bytes'],free)
                            reason=('owned USS cap' if uss>4*GIB else 'cgroup/system memory headroom' if free<2*GIB else
                                'wall cap' if time.monotonic()-started>900 else 'disk free cap' if __import__('shutil').disk_usage(dest).free<3*GIB else
                                'raw cap' if sum(x.stat().st_size for x in dest.rglob('*') if x.is_file())>2*GIB else None)
                            if reason:raise RuntimeError(reason)
                            write(dest/'resource.json',record)
                            if time.monotonic()-last_print>30:
                                print(json.dumps(dict(cell=cell,phase=label,owned_uss_bytes=uss,available_cgroup_bytes=free,elapsed_s=time.monotonic()-started)),flush=True);last_print=time.monotonic()
                            time.sleep(.5)
                        if child.returncode:raise RuntimeError(f'{cell}/{label} rejected; whole prefix preserved')
                finally:
                    if child is not None:terminate_owned_group(child)
                result=json.loads((dest/label/'result.json').read_text())
                if result['status']!='bounded-window-passed' or result['accepted_steps']!=steps:raise RuntimeError('Complete physical window did not pass')
            record['same_dt_replay_performed']=False
            record['validation_scope']='Independent dt convergence against the preserved original 2ps reference; no repeated baseline or fine-cell replay'
            verify(root);record['status']='passed'
        except BaseException as error:
            record['status']='rejected';record['first_error']=dict(type=type(error).__name__,message=str(error))
        finally:
            record['elapsed_seconds']=time.monotonic()-started;write(dest/'resource.json',record)
            files={str(x.relative_to(dest)):dict(bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(dest.rglob('*')) if x.is_file() and x.name!='whole-raw.json'}
            write(dest/'whole-raw.json',dict(source_capsule_manifest_sha256=sha(root/'manifest.json'),files=files))
            summary['cells'].append(record);write(output/'batch-result.json',summary)
        if record['status']!='passed':raise SystemExit(2)
    print(json.dumps(dict(status='both-independent-cells-passed' if len(cells)==2 else 'cell-passed',source_commit=contract['source_commit'])),flush=True)


if __name__=='__main__':main()
