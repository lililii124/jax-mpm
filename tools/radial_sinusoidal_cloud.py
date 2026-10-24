"""Bounded fresh radial pull-off and separately invoked, read-only media finalization."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import rht_impact_prefix as transport

BASE = ROOT/'.local/radial-sinusoidal-full'
RAW = BASE/'result'
PARTS = ROOT/'.local/radial-sinusoidal-parts'
MEDIA = ROOT/'.local/radial-sinusoidal-media'
GIB = 1024**3
RAW_LIMIT = 8*GIB
MEMORY_LIMIT = int(5.5*GIB)
TOTAL_SECONDS = 4*3600
SEGMENT_SECONDS = 45*60
RUNTIME = ROOT/'tools/radial_sinusoidal_runtime.txt'
REGISTERED = '.github/workflows/engineering-previews.yml'
FEATURE_REF = 'refs/heads/work/radial-sinusoidal-full'
REPOSITORY = 'lililii124/jax-mpm'
WORKFLOWS = ('.github/workflows/radial-sinusoidal-full.yml','.github/workflows/radial-sinusoidal-media.yml')


def digest(path):
    return transport.digest(Path(path))


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def inventory(folder):
    files={}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink in evidence')
        if p.is_file():
            files[p.relative_to(folder).as_posix()]={'bytes':p.stat().st_size,'sha256':digest(p)}
    return files


def sources():
    paths=list((ROOT/'mpm_jax').glob('*.py'))
    paths += [ROOT/x for x in (REGISTERED,*WORKFLOWS,'examples/radial_sinusoidal_pull_off.py',
        'examples/adhesive_array.py','examples/common.py','examples/scenarios.py',
        'tools/radial_sinusoidal_cloud.py','tools/radial_sinusoidal_runtime.txt',
        'tools/rht_impact_prefix.py','tools/run_woven_preview.py','pyproject.toml')]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}


def check_sources(expected):
    if sources()!=expected:
        raise ValueError('Exact numerical/render/controller/workflow source differs')


def private_workflow_bytes(root=None):
    root=ROOT if root is None else root
    source=(Path(root)/WORKFLOWS[0]).read_bytes()
    first,rest=source.split(b'\n',1)
    ending=b'\r\n' if first.endswith(b'\r') else b'\n'
    if first.rstrip(b'\r')!=b'name: feature radial sinusoidal full numerical run':
        raise ValueError('Unexpected radial numerical workflow template')
    return b'name: private radial sinusoidal full numerical run'+ending+rest


def git_output(*arguments):
    return subprocess.check_output(['git',*arguments],cwd=ROOT)


def registered(name):
    """Bind the executed registered workflow, committed checkout and template bytes."""
    if os.environ.get('GITHUB_ACTIONS')!='true':
        return
    commit=git_output('rev-parse','HEAD').decode().strip()
    if (commit!=os.environ.get('SOURCE_SHA') or commit!=os.environ.get('GITHUB_SHA') or
        commit!=os.environ.get('GITHUB_WORKFLOW_SHA') or
        os.environ.get('GITHUB_REF')!=FEATURE_REF or
        os.environ.get('GITHUB_REPOSITORY')!=REPOSITORY):
        raise ValueError('Requires exact committed fixed feature source and workflow SHA')
    ref=os.environ.get('GITHUB_WORKFLOW_REF')
    direct=f'{REPOSITORY}/{name}@{FEATURE_REF}'
    override=f'{REPOSITORY}/{REGISTERED}@{FEATURE_REF}'
    if ref!=direct and not (name==WORKFLOWS[0] and ref==override):
        raise ValueError('Unexpected fixed registered workflow ref')
    paths=set(sources())
    for path in sorted(paths):
        if (ROOT/path).read_bytes()!=git_output('show',f'HEAD:{path}'):
            raise ValueError('Working source differs from Git HEAD: '+path)
    if ref==override and (ROOT/REGISTERED).read_bytes()!=private_workflow_bytes():
        raise ValueError('Actual registered workflow bytes differ from radial template override')


def guard(command,log,seconds,folder):
    import psutil
    start=time.monotonic(); peak=0; process=None; reason=None
    try:
        with Path(log).open('wb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=os.name!='nt',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            while process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    live=[parent]+parent.children(recursive=True)
                    used=0
                    for child in live:
                        try: used+=child.memory_full_info().uss
                        except psutil.NoSuchProcess: pass
                    peak=max(peak,used)
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    reason='live-worker-process-tree-inaccessible'
                    raise RuntimeError(reason)
                size=0
                for path in Path(folder).rglob('*'):
                    try:
                        if path.is_file(): size+=path.stat().st_size
                    except FileNotFoundError: pass
                if peak>MEMORY_LIMIT: reason='aggregate-memory-limit'
                elif psutil.virtual_memory().available<512*1024**2: reason='system-available-memory-watermark'
                elif time.monotonic()-start>seconds: reason='wall-time-limit'
                elif size>RAW_LIMIT-512*1024**2: reason='complete-raw-size-watermark'
                elif shutil.disk_usage(folder).free<4*GIB: reason='free-disk-watermark'
                if reason: raise RuntimeError(reason)
                time.sleep(2)
            code=process.wait()
            if code: raise RuntimeError(f'worker-exit-{code}')
        return {'seconds':time.monotonic()-start,'peak_aggregate_uss_bytes':peak,'returncode':code}
    finally:
        if process is not None and process.poll() is None:
            if os.name!='nt':
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            else:
                for child in psutil.Process(process.pid).children(recursive=True): child.kill()
                process.kill()
            process.wait()
        save(Path(log).with_suffix('.guard.json'),{'peak_aggregate_uss_bytes':peak,
            'elapsed_seconds':time.monotonic()-start,'stop_reason':reason,
            'returncode':process.returncode if process else None})


def worker(end):
    import jax
    jax.config.update('jax_enable_x64',True)
    from examples.radial_sinusoidal_pull_off import make_case
    from examples.common import run
    case,substeps=make_case()
    if case.states[0].x.shape[0]!=367744 or substeps*50!=3700:
        raise ValueError('Frozen resolution/step plan changed')
    run(case.states,case.materials,case.configs,frames=50,substeps=substeps,
        output=RAW,label='radial-sinusoidal-pull-off',forces=case.forces,
        observer=case.observer,metadata=case.description,render_output=False,
        resume=end>10,stop_after_frame=end)


def validate_complete(folder):
    import numpy as np
    summary=read(folder/'summary.json')
    if not (summary.get('completed') is True and summary['particles']==[367744]
            and summary['steps']==3700 and abs(summary['simulation_time']-1.3)<1e-10
            and summary['definition']['radial_sinusoid']['amplitude_m']==.008
            and summary['definition']['radial_sinusoid']['wavelength_m']==.08):
        raise ValueError('Incomplete or different numerical plan')
    frames=sorted((folder/'frames_data').glob('frame_*.npz'))
    if [p.name for p in frames]!=[f'frame_{i:05d}.npz' for i in range(51)]:
        raise ValueError('Expected exactly 51 saved states')
    for p in frames:
        with np.load(p,allow_pickle=False) as state:
            if state['x'].shape!=(367744,3) or not np.isfinite(state['x']).all():
                raise ValueError('Invalid saved positions')
    return summary


def validate_prefix(folder):
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    from examples.radial_sinusoidal_pull_off import make_case
    from examples.common import _load_state, _valid_checkpoint_states, _validate_resume_prefix
    from mpm_jax.particle_fields import material_metadata
    case,substeps=make_case()
    checkpoint=read(folder/'checkpoint.json')
    states=tuple(_load_state(folder/name) for name in checkpoint['states'])
    signature_data={'configs':[asdict(c) for c in case.configs],
        'materials':[material_metadata(m) for m in case.materials],
        'frames':50,'substeps':substeps,'label':'radial-sinusoidal-pull-off',
        'definition':case.description}
    signature=hashlib.sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest()
    segment=read(folder/'summary.json')
    if not (checkpoint['frame']==10 and substeps==74 and
        checkpoint['signature']==signature and len(states)==1 and
        len(states[0].x)==367744 and _valid_checkpoint_states(states) and
        segment.get('completed') is False and segment.get('segment_complete') is True and
        segment['frame']==10 and segment['steps_completed']==740):
        raise ValueError('Invalid exact f10 prefix checkpoint or plan signature')
    _validate_resume_prefix(folder,checkpoint,states,50,74,case.configs[0].dt,False)
    np.testing.assert_array_equal(states[0].mass,case.states[0].mass)
    save(folder.parent/'prefix-audit.json',{'status':'accepted-prefix','frame':10,
        'saved_states':11,'steps_completed':740,'full_steps':3700,'full_frames':50,
        'time_s':.26,'full_time_s':1.3,'full_plan_signature':signature,
        'production_eligible':False,'full_protocol_complete':False,'resumable':True})


def numerical(stop_after_frame=50, continue_from=None, parent_receipt_sha256=None):
    registered(WORKFLOWS[0])
    if BASE.exists(): raise ValueError('Fresh run refuses prior output')
    parent=None
    if stop_after_frame not in (10,50): raise ValueError('Only f10 prefix or full f50 is allowed')
    if continue_from is not None:
        if stop_after_frame!=50 or parent_receipt_sha256 is None:
            raise ValueError('Continuation requires full target and exact parent receipt SHA256')
        continue_from=Path(continue_from)
        if digest(continue_from/'receipt.json')!=parent_receipt_sha256:
            raise ValueError('Exact continuation receipt hash differs')
        parent=read(continue_from/'receipt.json')
        if os.environ.get('CONTINUATION_KEY') and parent.get('cache_key')!=os.environ['CONTINUATION_KEY']:
            raise ValueError('Restored parent cache identity differs')
        if parent.get('status')!='accepted-prefix': raise ValueError('Only accepted-prefix can continue')
        check_sources(parent['source_files'])
        if inventory(continue_from/'result')!=parent['raw_files']:
            raise ValueError('Exact continuation raw hashes differ')
        shutil.copytree(continue_from,BASE)
        shutil.copyfile(continue_from/'receipt.json',BASE/'parent-receipt.json')
    else:
        BASE.mkdir(parents=True)
    receipt={'status':'failedraw','run_id':os.environ.get('GITHUB_RUN_ID'),
        'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT'),
        'workflow_ref':os.environ.get('GITHUB_WORKFLOW_REF'),'workflow_sha':os.environ.get('GITHUB_WORKFLOW_SHA'),'commit':os.environ.get('SOURCE_SHA',os.environ.get('GITHUB_SHA')),
        'source_files':sources(),'python':platform.python_version(),
        'platform':platform.platform(),'machine':platform.machine(),
        'runtime_environment':{name:os.environ.get(name) for name in
            ('JAX_PLATFORMS','JAX_ENABLE_X64','XLA_PYTHON_CLIENT_PREALLOCATE',
             'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        'requested_stop_frame':stop_after_frame,'continuation_parent_sha256':parent_receipt_sha256,
        'continuation_parent_run_id':parent.get('run_id') if parent else None,
        'limits':{'raw_bytes':RAW_LIMIT,'aggregate_uss_bytes':MEMORY_LIMIT,
                  'total_seconds':TOTAL_SECONDS,'segment_seconds':SEGMENT_SECONDS},'segments':[]}
    source_hash=hashlib.sha256(json.dumps(receipt['source_files'],sort_keys=True).encode()).hexdigest()
    receipt['source_digest']=source_hash
    receipt['cache_key']=f"radial-sinusoidal-v1-{source_hash}-{receipt['run_id']}-{receipt['run_attempt']}"
    output=os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output,'a') as stream: stream.write('cache_key='+receipt['cache_key']+'\n')
    start=time.monotonic()
    try:
        pins={line.split('==')[0]:line.strip().split('==')[1] for line in RUNTIME.read_text().splitlines() if line.strip()}
        receipt['runtime']={name:importlib.metadata.version(name) for name in pins}
        receipt['installed_packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
        save(BASE/'receipt.json',receipt)
        if receipt['runtime']!=pins: raise ValueError('Exact pinned runtime mismatch')
        if parent and parent.get('runtime')!=receipt['runtime']:
            raise ValueError('Exact continuation runtime differs')
        if parent and os.environ.get('GITHUB_ACTIONS')=='true' and parent.get('commit')!=receipt['commit']:
            raise ValueError('Exact continuation source commit differs')
        if shutil.disk_usage(BASE).free<32*GIB: raise ValueError('32 GiB free disk admission required')
        endpoints=(20,30,40,50) if parent else ((10,) if stop_after_frame==10 else (10,20,30,40,50))
        for end in endpoints:
            check_sources(receipt['source_files'])
            budget=min(SEGMENT_SECONDS,TOTAL_SECONDS-(time.monotonic()-start))
            if budget<=0: raise RuntimeError('total-wall-limit')
            receipt['segments'].append({'end_frame':end,**guard(
                [sys.executable,str(Path(__file__).resolve()),'worker','--end',str(end)],
                BASE/f'segment-{end:02d}.log',budget,BASE)})
            save(BASE/'progress.json',receipt)
            save(BASE/'receipt.json',receipt)
        if stop_after_frame==10:
            receipt['prefix_validation']=guard([sys.executable,str(Path(__file__).resolve()),'audit-prefix'],
                BASE/'prefix-validation.log',600,BASE)
            receipt['status']='accepted-prefix'
            receipt['prefix_audit']=read(BASE/'prefix-audit.json')
        else:
            validate_complete(RAW)
            receipt['status']='complete'
    except BaseException as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start
        receipt['raw_files']=inventory(RAW) if RAW.exists() else {}
        save(BASE/'receipt.json',receipt)


def package():
    # Whole evidence, including a failed/incomplete numerical endpoint.
    if not (BASE/'receipt.json').is_file():
        save(BASE/'receipt.json',{'status':'failedraw','error':'controller did not produce receipt',
            'source_files':sources(),'run_id':os.environ.get('GITHUB_RUN_ID')})
    receipt=read(BASE/'receipt.json')
    if receipt.get('status') not in ('complete','accepted-prefix'):
        receipt['status']='failedraw'
        receipt['raw_files']=inventory(RAW) if RAW.exists() else {}
        save(BASE/'receipt.json',receipt)
    if PARTS.exists(): raise ValueError('Refusing prior multipart delivery')
    PARTS.mkdir(parents=True)
    files=inventory(BASE)
    bound=sum(((v['bytes']+511)//512+8)*512 for v in files.values())+10240
    if bound>RAW_LIMIT or shutil.disk_usage(PARTS).free<bound+512*1024**2:
        save(PARTS/'transport.json',{'status':'rejected-complete-raw-exceeds-budget','files':files})
        raise ValueError('Cannot preserve complete raw transport')
    sink=transport.PartWriter(PARTS,128*1024**2)
    try:
        with tarfile.open(fileobj=sink,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name in files: archive.add(BASE/name,arcname=name,recursive=False)
    finally: sink.close()
    parts={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sink.parts}
    if len(parts)>64: raise ValueError('Too many transport parts')
    save(PARTS/'transport.json',{'status':'complete-raw-packaged','files':files,
        'part_limit_bytes':128*1024**2,'parts':parts})


def retrieve(folder,extract=False):
    transport.verify(folder)
    manifest=read(folder/'transport.json')
    if extract:
        if BASE.exists(): raise ValueError('Extraction requires fresh numerical destination')
        BASE.mkdir(parents=True)
        reader=transport.PartReader(folder/name for name in manifest['parts'])
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                path=Path(member.name)
                if path.is_absolute() or '..' in path.parts or not member.isfile():
                    raise ValueError('Unsafe archive member')
                destination=BASE/path
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,destination.open('wb') as target:
                    shutil.copyfileobj(source,target)
    return manifest


def encode_saved_frames(folder, count=51):
    """Write one GIF image block per saved PNG, retaining identical hold states."""
    from PIL import Image
    first=Image.open(folder/'frames'/'frame_00000.png').convert('RGB')
    palette=first.quantize(colors=256)
    first.close()
    header=None
    with (folder/'simulation.gif').open('wb') as destination:
        for i in range(count):
            with Image.open(folder/'frames'/f'frame_{i:05d}.png') as frame:
                mapped=frame.convert('RGB').quantize(palette=palette)
                buffer=io.BytesIO()
                mapped.save(buffer,format='GIF',duration=60,optimize=False,
                            disposal=2,**({'loop':0} if i==0 else {}))
                payload=buffer.getvalue()
                size=13+(3*(2**((payload[10]&7)+1)) if payload[10]&128 else 0)
                if i==0:
                    header=payload[:size]
                    destination.write(header)
                elif payload[:size]!=header:
                    raise ValueError('GIF global palette/header changed')
                destination.write(payload[size:-1])
                mapped.close()
        destination.write(b';')
    palette.close()


def media():
    registered(WORKFLOWS[1])
    receipt=read(BASE/'receipt.json')
    if receipt['status']!='complete': raise ValueError('Failedraw cannot become media')
    expected_run=os.environ.get('RAW_RUN_ID')
    if expected_run and str(receipt['run_id'])!=expected_run: raise ValueError('Raw run identity differs')
    if os.environ.get('RAW_RUN_ATTEMPT') and str(receipt['run_attempt'])!=os.environ['RAW_RUN_ATTEMPT']:
        raise ValueError('Raw attempt differs')
    if os.environ.get('SOURCE_SHA') and receipt['commit']!=os.environ['SOURCE_SHA']:
        raise ValueError('Raw source commit differs')
    identity=read(ROOT/'.local/radial-sinusoidal-raw-artifact-identities.json')
    if not (str(identity['run_id'])==str(receipt['run_id']) and
            str(identity['attempt'])==str(receipt['run_attempt']) and
            identity['head_sha']==receipt['commit'] and identity['conclusion']=='success' and
            identity['workflow_path'] in (WORKFLOWS[0],REGISTERED)):
        raise ValueError('Actual registered numerical run provenance differs')
    transport_manifest=read(ROOT/'.local/radial-sinusoidal-input/transport.json')
    prefix=f"radial-raw-{receipt['run_id']}-{receipt['run_attempt']}-"
    names={prefix+'manifest'}|{prefix+Path(name).stem for name in transport_manifest['parts']}
    if {x['name'] for x in identity['artifacts']}!=names:
        raise ValueError('Actual raw artifact set differs from complete transport')
    check_sources(receipt['source_files'])
    if inventory(RAW)!=receipt['raw_files']: raise ValueError('Raw hashes differ')
    validate_complete(RAW)
    if MEDIA.exists(): raise ValueError('Media requires fresh destination')
    MEDIA.mkdir(parents=True)
    work=ROOT/'.local/radial-sinusoidal-render'
    if work.exists(): raise ValueError('Render workspace must be fresh')
    shutil.copytree(RAW,work)
    # This mode imports rendering/export helpers only; it never invokes worker/run/step.
    guard([sys.executable,'-m','mpm_jax.visualization',str(work),'--gif',
           '--field','body','--resolution','960'],MEDIA/'render-process.log',75*60,work)
    encode_saved_frames(work)
    subprocess.run([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',
        str(Path(__file__).resolve()),'vtk','--folder',str(work)],check=True,timeout=600)
    from PIL import Image
    record=read(work/'render.json')
    if not (record['renderer']=='ParaView' and record['frames_rendered']==51 and record['particles']==367744
        and record['axes'] is False and record['annotations'] is False):
        raise ValueError('Renderer provenance differs')
    geometry=[r for r in record['auxiliary_geometry'] if r['type']=='prescribed_radial_sinusoidal_rigid_sphere']
    if len(geometry)!=1 or geometry[0]['radial_sinusoid']!=read(work/'summary.json')['definition']['radial_sinusoid']:
        raise ValueError('Visible rigid geometry differs')
    if len(list((work/'frames').glob('frame_*.png')))!=51: raise ValueError('PNG count differs')
    for i in range(51):
        with Image.open(work/'frames'/f'frame_{i:05d}.png') as png:
            png.load()
            if png.size!=(960,960) or all(lo==hi for lo,hi in png.convert('RGB').getextrema()):
                raise ValueError('Invalid or blank PNG')
    with Image.open(work/'simulation.gif') as gif:
        if gif.n_frames!=51 or gif.size!=(960,960): raise ValueError('GIF frame/size differs')
        for i in range(51):
            gif.seek(i); gif.load()
            if gif.info.get('duration')!=60: raise ValueError('GIF timing differs')
    if inventory(RAW)!=receipt['raw_files']: raise ValueError('Sealed raw mutated during rendering')
    for name in receipt['raw_files']:
        if digest(work/name)!=receipt['raw_files'][name]['sha256']: raise ValueError('Copied raw mutated')
    for name in ('simulation.gif','render.json','summary.json','vtk-full-audit.json'):
        shutil.copyfile(work/name,MEDIA/name)
    save(MEDIA/'provenance.json',{'status':'rendered-from-exact-saved-raw',
        'numerical_receipt_sha256':digest(BASE/'receipt.json'),'raw_run_id':receipt['run_id'],
        'raw_run_attempt':receipt['run_attempt'],'commit':receipt['commit'],
        'raw_artifact_identities':identity,
        'source_files':receipt['source_files'],'raw_files':receipt['raw_files'],
        'saved_states':51,'solver_rerun':False,'raw_unchanged':True,
        'renderer_executable_sha256':digest(__import__('mpm_jax.visualization',fromlist=['find_pvpython']).find_pvpython()),
        'renderer_runtime':{name:importlib.metadata.version(name) for name in ('numpy','pillow','psutil')},
        'paraview_install':read(ROOT/'.local/radial-sinusoidal-pv-install/paraview-install.json')})
    if sum(p.stat().st_size for p in MEDIA.rglob('*') if p.is_file())>99_000_000:
        raise ValueError('Media delivery exceeds 100 MB')
    save(MEDIA/'delivery.json',inventory(MEDIA))


def vtk_audit(folder):
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    for i in range(51):
        reader=vtkXMLPolyDataReader()
        reader.SetFileName(str(folder/'vtk'/f'particles_{i:05d}.vtp'))
        reader.Update()
        data=reader.GetOutput()
        if data.GetNumberOfPoints()!=367744: raise ValueError('Actual VTK particle count differs')
        with np.load(folder/'frames_data'/f'frame_{i:05d}.npz',allow_pickle=False) as raw:
            np.testing.assert_allclose(vtk_to_numpy(data.GetPoints().GetData()),raw['x'],rtol=1e-6,atol=1e-7)
    save(folder/'vtk-full-audit.json',{'passed':True,'frames_checked':51,
        'particles_per_frame':367744,'comparison':'actual VTK positions against each saved raw state'})


def verify_media(folder):
    expected=read(folder/'delivery.json')
    actual=inventory(folder); actual.pop('delivery.json',None)
    if actual!=expected: raise ValueError('Retrieved media bytes differ')
    save(ROOT/'.local/radial-sinusoidal-media-retrieval.json',{
        'status':'all-delivery-files-verified','artifact_id':os.environ.get('MEDIA_ARTIFACT_ID'),
        'raw_run_id':read(folder/'provenance.json')['raw_run_id'],'files':actual})


def install():
    import run_woven_preview as pv
    pv.LOGS=ROOT/'.local/radial-sinusoidal-pv-install'
    pv.LOGS.mkdir(parents=True,exist_ok=True)
    pv.install()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','worker','package','retrieve','extract','media','verify-media','install','vtk','audit-prefix','stage-parent'])
    parser.add_argument('--end',type=int,choices=[10,20,30,40,50])
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--stop-after-frame',type=int,choices=[10,50],default=50)
    parser.add_argument('--continue-from',type=Path)
    parser.add_argument('--parent-receipt-sha256')
    args=parser.parse_args()
    if args.mode=='worker': worker(args.end)
    elif args.mode=='run': numerical(args.stop_after_frame,args.continue_from,args.parent_receipt_sha256)
    elif args.mode=='package': package()
    elif args.mode=='retrieve': retrieve(args.folder)
    elif args.mode=='extract': retrieve(args.folder,True)
    elif args.mode=='media': media()
    elif args.mode=='verify-media': verify_media(args.folder)
    elif args.mode=='install': install()
    elif args.mode=='vtk': vtk_audit(args.folder)
    elif args.mode=='audit-prefix': validate_prefix(RAW)
    elif args.mode=='stage-parent':
        destination=ROOT/'.local/radial-sinusoidal-resume-input'
        if destination.exists() or not BASE.is_dir(): raise ValueError('Invalid exact parent staging paths')
        if digest(BASE/'receipt.json')!=os.environ.get('PARENT_SHA'):
            raise ValueError('Parent staging receipt hash differs')
        BASE.rename(destination)


if __name__=='__main__': main()
