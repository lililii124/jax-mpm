"""Dual-checkout continuation/media for the independently admitted immutable f10 producer."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'.local/radial-immutable-producer'
PARENT=ROOT/'.local/radial-sinusoidal-full'
WORK=PRODUCER/'.local/radial-sinusoidal-full'
RAW=WORK/'result'
PARTS=ROOT/'.local/radial-continuation-parts'
MEDIA=ROOT/'.local/radial-continuation-media'
PROOF=ROOT/'.local/radial-independent-proof/independent-admission.json'
NUMERICAL_CONTROLLER=ROOT/'.local/radial-numerical-controller'
PRODUCER_COMMIT='de2bfea446e66e6245e33329a663996c5d5b9e84'
PARENT_RUN='40090780381'
PARENT_CACHE='radial-sinusoidal-v1-2363a43cf6afbd771833e035e0b7703ecd6ff6651306984ca2e7ebf828f2961d-40090780381-1'
PARENT_RECEIPT='d0af82b261b0433c024b80472cb5da67ee5d0113f11d2876f0510ee8f631c591'
PROOF_SHA='74840b9e9bf12f479787bdfb359bd34098354402fdcf3c13deccee698043836f'
REF='refs/heads/work/radial-sinusoidal-full'
REPOSITORY='lililii124/jax-mpm'
REGISTERED='.github/workflows/engineering-previews.yml'
CONTINUATION='.github/workflows/radial-sinusoidal-continue.yml'
MEDIA_WORKFLOW='.github/workflows/radial-sinusoidal-continuation-media.yml'
CONTROLLER='tools/radial_sinusoidal_continuation.py'
LIMIT=8*1024**3


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def inventory(folder):
    result={}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():raise ValueError('Evidence symlink rejected')
        if path.is_file():result[path.relative_to(folder).as_posix()]={'bytes':path.stat().st_size,'sha256':digest(path)}
    return result


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def head(root):return git(root,'rev-parse','HEAD').decode().strip()


def own_files(root= None):
    root=ROOT if root is None else Path(root)
    return {name:digest(root/name) for name in (CONTROLLER,CONTINUATION,MEDIA_WORKFLOW,REGISTERED)}


def private_workflow_bytes(root=None,media=False):
    root=ROOT if root is None else Path(root)
    template=MEDIA_WORKFLOW if media else CONTINUATION
    data=(root/template).read_bytes();first,rest=data.split(b'\n',1)
    ending=b'\r\n' if first.endswith(b'\r') else b'\n'
    expected=b'name: feature radial sinusoidal continuation'+(b' media' if media else b'')
    if first.rstrip(b'\r')!=expected:raise ValueError('Unexpected continuation template first line')
    return expected.replace(b'feature',b'private',1)+ending+rest


def bind_controller(media=False):
    commit=head(ROOT)
    if os.environ.get('GITHUB_ACTIONS')=='true':
        template=MEDIA_WORKFLOW if media else CONTINUATION
        valid={f'{REPOSITORY}/{template}@{REF}',f'{REPOSITORY}/{REGISTERED}@{REF}'}
        if (os.environ.get('GITHUB_REPOSITORY')!=REPOSITORY or os.environ.get('GITHUB_REF')!=REF
            or os.environ.get('GITHUB_WORKFLOW_REF') not in valid
            or commit!=os.environ.get('GITHUB_SHA') or commit!=os.environ.get('GITHUB_WORKFLOW_SHA')):
            raise ValueError('Actually registered new controller/workflow identity differs')
        if os.environ['GITHUB_WORKFLOW_REF']==f'{REPOSITORY}/{REGISTERED}@{REF}' and (ROOT/REGISTERED).read_bytes()!=private_workflow_bytes(media=media):
            raise ValueError('Actual registered new workflow bytes differ')
    files=own_files()
    for name in files:
        if (ROOT/name).read_bytes()!=git(ROOT,'show',f'HEAD:{name}'):
            raise ValueError('New controller source differs from its committed HEAD')
    return commit,files


def proof_and_producer():
    if digest(PROOF)!=PROOF_SHA:raise ValueError('Independent proof hash differs')
    proof=read(PROOF)
    if not (proof['admitted_prefix'] is True and proof['raw_unchanged'] is True
        and proof['solver_executed'] is False and proof['producer_commit']==PRODUCER_COMMIT
        and str(proof['producer_run'])==PARENT_RUN and proof['producer_receipt_sha256']==PARENT_RECEIPT
        and proof['cache_key']==PARENT_CACHE and proof['prefix_audit']['saved_states']==11):
        raise ValueError('Independent f10 admission differs')
    if head(PRODUCER)!=PRODUCER_COMMIT:raise ValueError('Immutable numerical producer HEAD differs')
    for name,expected in proof['source_files'].items():
        path=PRODUCER/name
        if digest(path)!=expected or path.read_bytes()!=git(PRODUCER,'show',f'{PRODUCER_COMMIT}:{name}'):
            raise ValueError('Immutable producer source differs: '+name)
    runtime={name:importlib.metadata.version(name) for name in proof['runtime']}
    if runtime!=proof['runtime']:raise ValueError('Immutable producer runtime differs')
    return proof


def producer_helpers():
    # Import only authenticated immutable producer helpers; its worker runs unchanged.
    sys.path.insert(0,str(PRODUCER));sys.path.insert(0,str(PRODUCER/'tools'))
    spec=importlib.util.spec_from_file_location('immutable_radial_producer',PRODUCER/'tools/radial_sinusoidal_cloud.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_parent(proof):
    if digest(PARENT/'receipt.json')!=PARENT_RECEIPT:raise ValueError('Exact f10 receipt differs')
    receipt=read(PARENT/'receipt.json')
    if not (receipt['status']=='accepted-prefix' and receipt['commit']==PRODUCER_COMMIT
        and str(receipt['run_id'])==PARENT_RUN and receipt['cache_key']==PARENT_CACHE
        and receipt['source_files']==proof['source_files'] and receipt['runtime']==proof['runtime']):
        raise ValueError('Exact admitted f10 parent differs')
    if inventory(PARENT/'result')!=proof['all_raw_files'] or receipt['raw_files']!=proof['all_raw_files']:
        raise ValueError('Whole original f10 raw differs')
    return receipt


def verify_prefix_frames(folder,proof):
    for name,expected in proof['all_raw_files'].items():
        if name.startswith('frames_data/'):
            path=folder/name
            if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
                raise ValueError('Original admitted prefix frame changed: '+name)


def continue_run():
    commit,files=bind_controller()
    if WORK.exists():raise ValueError('Fresh immutable producer work directory required')
    WORK.mkdir(parents=True)
    receipt={'schema':'radial-dual-producer-continuation-v1','status':'failedraw',
        'controller_commit':commit,'controller_files':files,'producer_commit':PRODUCER_COMMIT,
        'parent_run':PARENT_RUN,'parent_cache':PARENT_CACHE,'parent_receipt_sha256':PARENT_RECEIPT,
        'independent_proof_sha256':PROOF_SHA,'run_id':os.environ.get('GITHUB_RUN_ID'),
        'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT'),'segments':[]}
    receipt['cache_key']=f"radial-dual-continuation-v1-{commit}-{receipt['run_id']}-{receipt['run_attempt']}"
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:stream.write('cache_key='+receipt['cache_key']+'\n')
    save(WORK/'continuation-receipt.json',receipt)
    started=time.monotonic()
    try:
        proof=proof_and_producer();verify_parent(proof)
        if shutil.disk_usage(ROOT).free<32*1024**3:raise ValueError('32 GiB free disk required')
        for name in ('result',):shutil.copytree(PARENT/name,WORK/name)
        shutil.copyfile(PARENT/'receipt.json',WORK/'receipt.json')
        shutil.copyfile(PROOF,WORK/'independent-admission.json')
        receipt['producer_source_files']=proof['source_files'];receipt['runtime']=proof['runtime']
        old=producer_helpers()
        receipt['prefix_revalidation']=old.guard([sys.executable,str(PRODUCER/'tools/radial_sinusoidal_cloud.py'),'audit-prefix'],
            WORK/'prefix-revalidation.log',600,WORK)
        for end in (20,30,40,50):
            if own_files()!=files:raise ValueError('New controller source changed mid-run')
            proof_and_producer()
            remaining=4*3600-(time.monotonic()-started)
            if remaining<=0:raise RuntimeError('Continuation total wall budget exceeded')
            receipt['segments'].append({'end_frame':end,**old.guard(
                [sys.executable,str(PRODUCER/'tools/radial_sinusoidal_cloud.py'),'worker','--end',str(end)],
                WORK/f'continuation-{end}.log',min(45*60,remaining),WORK)})
            save(WORK/'continuation-receipt.json',receipt)
        receipt['full_validation']=old.guard([sys.executable,str(Path(__file__).resolve()),'audit-full'],
            WORK/'full-validation.log',600,WORK)
        verify_parent(proof)
        verify_prefix_frames(RAW,proof)
        receipt['status']='complete';receipt['parent_raw_unchanged']=True
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        receipt['raw_files']=inventory(RAW) if RAW.exists() else {}
        save(WORK/'continuation-receipt.json',receipt)


def audit_full(write_receipt=True):
    proof=proof_and_producer();old=producer_helpers()
    summary=old.validate_complete(RAW)
    if not isinstance(summary.get('min_J'),(int,float)) or summary['min_J']<=0:
        raise ValueError('Full protocol has nonpositive or invalid deformation determinant')
    import jax
    jax.config.update('jax_enable_x64',True)
    import examples.common as common
    if Path(common.__file__).resolve()!=(PRODUCER/'examples/common.py').resolve():
        raise ValueError('Full validator imported a different physical producer')
    checkpoint=read(RAW/'checkpoint.json')
    if checkpoint['frame']!=50 or checkpoint['signature']!=proof['prefix_audit']['full_plan_signature']:
        raise ValueError('Full endpoint checkpoint plan differs from admitted physical protocol')
    states=tuple(common._load_state(RAW/name) for name in checkpoint['states'])
    if not common._valid_checkpoint_states(states):raise ValueError('Full endpoint numerical state invalid')
    common._validate_resume_prefix(RAW,checkpoint,states,50,74,1.3/3700,False)
    verify_prefix_frames(RAW,proof)
    if write_receipt:save(WORK/'full-numerical-audit.json',{'passed':True,'saved_states':51,'steps':3700,
        'particles':367744,'simulation_time_s':1.3,'full_plan_signature':checkpoint['signature'],
        'all_checkpoint_history_checked':True,'prefix_frames_unchanged':True,
        'physical_comparison_effect_accepted':False})


def package():
    if not (WORK/'continuation-receipt.json').exists():
        save(WORK/'continuation-receipt.json',{'status':'failedraw','error':'preflight did not produce receipt'})
    if PARTS.exists():raise ValueError('Fresh multipart directory required')
    PARTS.mkdir(parents=True)
    files=inventory(WORK)
    bound=sum(((value['bytes']+511)//512+8)*512 for value in files.values())+10240
    if bound>LIMIT or shutil.disk_usage(ROOT).free<bound+512*1024**2:
        save(PARTS/'transport.json',{'status':'rejected-no-complete-delivery','files':files})
        raise ValueError('Whole failed/raw evidence exceeds transport or disk admission')
    old=producer_helpers();sink=old.transport.PartWriter(PARTS,128*1024**2)
    try:
        with tarfile.open(fileobj=sink,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name in files:archive.add(WORK/name,arcname=name,recursive=False)
    finally:sink.close()
    if len(sink.parts)>64:raise ValueError('Multipart count exceeds64')
    save(PARTS/'transport.json',{'status':'complete-raw-packaged','files':files,
        'part_limit_bytes':128*1024**2,'parts':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sink.parts}})


def retrieve(folder,extract=False):
    old=producer_helpers();old.transport.verify(folder)
    manifest=read(folder/'transport.json')
    if extract:
        if WORK.exists():raise ValueError('Fresh exact extraction destination required')
        WORK.mkdir(parents=True)
        reader=old.transport.PartReader(folder/name for name in manifest['parts'])
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts or not member.isfile():raise ValueError('Unsafe archive member')
                destination=WORK/name;destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,destination.open('wb') as target:shutil.copyfileobj(source,target)


def media():
    commit,files=bind_controller(media=True);proof=proof_and_producer()
    receipt=read(WORK/'continuation-receipt.json')
    if not (receipt['status']=='complete' and receipt['producer_commit']==PRODUCER_COMMIT
        and receipt['parent_receipt_sha256']==PARENT_RECEIPT and receipt['independent_proof_sha256']==PROOF_SHA
        and receipt['producer_source_files']==proof['source_files'] and receipt['runtime']==proof['runtime']):
        raise ValueError('Completed dual-producer evidence differs')
    if receipt['controller_commit']!=head(NUMERICAL_CONTROLLER):raise ValueError('Numerical controller checkout differs')
    for name,expected in receipt['controller_files'].items():
        if digest(NUMERICAL_CONTROLLER/name)!=expected or (NUMERICAL_CONTROLLER/name).read_bytes()!=git(NUMERICAL_CONTROLLER,'show',f'HEAD:{name}'):
            raise ValueError('Numerical controller source differs')
    if str(receipt['run_id'])!=os.environ.get('RAW_RUN_ID') or str(receipt['run_attempt'])!=os.environ.get('RAW_RUN_ATTEMPT'):
        raise ValueError('Exact continuation raw run/attempt differs')
    if digest(WORK/'receipt.json')!=PARENT_RECEIPT or inventory(RAW)!=receipt['raw_files']:
        raise ValueError('Original parent receipt or completed raw differs')
    identities=read(ROOT/'.local/radial-cont-artifact-identities.json')
    if not (str(identities['run_id'])==str(receipt['run_id']) and
        identities['head_sha']==receipt['controller_commit'] and identities['conclusion']=='success'
        and identities['workflow_path'] in (CONTINUATION,REGISTERED)):
        raise ValueError('Actual numerical continuation run provenance differs')
    transport=read(ROOT/'.local/radial-continuation-input/transport.json')
    prefix=f"radial-cont-raw-{receipt['run_id']}-{receipt['run_attempt']}-"
    expected={prefix+'manifest'}|{prefix+Path(name).stem for name in transport['parts']}
    if {item['name'] for item in identities['artifacts']}!=expected:
        raise ValueError('Actual numerical continuation artifact set differs')
    verify_prefix_frames(RAW,proof)
    old=producer_helpers();audit_full(write_receipt=False)
    if MEDIA.exists():raise ValueError('Fresh media delivery required')
    MEDIA.mkdir(parents=True)
    render=PRODUCER/'.local/radial-dual-render'
    if render.exists():raise ValueError('Fresh render work directory required')
    shutil.copytree(RAW,render)
    old.guard([sys.executable,'-m','mpm_jax.visualization',str(render),'--gif','--field','body','--resolution','960'],
        MEDIA/'render.log',75*60,render)
    old.encode_saved_frames(render)
    subprocess.run([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(PRODUCER/'tools/radial_sinusoidal_cloud.py'),
        'vtk','--folder',str(render)],cwd=PRODUCER,check=True,timeout=600)
    record=read(render/'render.json')
    if not (record['renderer']=='ParaView' and record['frames_rendered']==51 and record['particles']==367744
        and record['axes'] is False and record['annotations'] is False):raise ValueError('Actual51-frame renderer evidence differs')
    geometry=[g for g in record['auxiliary_geometry'] if g['type']=='prescribed_radial_sinusoidal_rigid_sphere']
    if len(geometry)!=1 or geometry[0]['radial_sinusoid']!=read(render/'summary.json')['definition']['radial_sinusoid']:
        raise ValueError('Visible corrugated geometry differs')
    from PIL import Image
    for i in range(51):
        with Image.open(render/'frames'/f'frame_{i:05d}.png') as image:
            image.load()
            if image.size!=(960,960) or all(lo==hi for lo,hi in image.convert('RGB').getextrema()):raise ValueError('PNG size differs or image is blank')
    with Image.open(render/'simulation.gif') as image:
        if image.n_frames!=51 or image.size!=(960,960):raise ValueError('GIF saved-state count differs')
        for i in range(51):
            image.seek(i);image.load()
            if image.info.get('duration')!=60:raise ValueError('GIF timing differs')
    if inventory(RAW)!=receipt['raw_files']:raise ValueError('Completed raw changed during read-only rendering')
    for name,expected in receipt['raw_files'].items():
        if digest(render/name)!=expected['sha256']:raise ValueError('Copied raw changed')
    for name in ('simulation.gif','render.json','summary.json','vtk-full-audit.json'):shutil.copyfile(render/name,MEDIA/name)
    shutil.copyfile(WORK/'full-numerical-audit.json',MEDIA/'full-numerical-audit.json')
    save(MEDIA/'provenance.json',{'schema':'radial-dual-media-v1','producer_commit':PRODUCER_COMMIT,
        'numerical_controller_commit':receipt['controller_commit'],'media_controller_commit':commit,
        'media_controller_files':files,'continuation_receipt_sha256':digest(WORK/'continuation-receipt.json'),
        'raw_run':receipt['run_id'],'raw_attempt':receipt['run_attempt'],'raw_files':receipt['raw_files'],
        'raw_artifact_identities':identities,
        'parent_receipt_sha256':PARENT_RECEIPT,'proof_sha256':PROOF_SHA,'solver_executed':False,
        'raw_unchanged':True,'saved_states':51,'renderer_source_files':proof['source_files'],
        'renderer_executable_sha256':digest(os.environ['PARAVIEW_PYTHON'])})
    save(MEDIA/'delivery.json',inventory(MEDIA))
    if sum(p.stat().st_size for p in MEDIA.iterdir() if p.is_file())>100_000_000:raise ValueError('Media delivery exceeds100MB')


def install():
    proof_and_producer();old=producer_helpers();old.install()


def verify_media(folder):
    actual=inventory(folder);actual.pop('delivery.json',None)
    if actual!=read(folder/'delivery.json'):raise ValueError('Retrieved media differs')
    save(ROOT/'.local/radial-continuation-media-retrieval.json',{'passed':True,
        'artifact_id':os.environ.get('MEDIA_ARTIFACT_ID'),'files':actual})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['continue','package','retrieve','extract','media','install','verify-media','audit-full'])
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    if args.mode=='continue':continue_run()
    elif args.mode=='package':package()
    elif args.mode=='retrieve':retrieve(args.folder)
    elif args.mode=='extract':retrieve(args.folder,True)
    elif args.mode=='media':media()
    elif args.mode=='install':install()
    elif args.mode=='verify-media':verify_media(args.folder)
    elif args.mode=='audit-full':audit_full()


if __name__=='__main__':main()
