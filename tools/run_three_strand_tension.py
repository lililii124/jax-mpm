"""Guard, seal and verify source-bound whole-frame three-strand stages."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.three_strand_tension import digest,plan_case,save_json
LIMIT=6*1024**3
FILES=('examples/three_strand_tension.py','tools/audit_three_strand_tension.py','tools/run_three_strand_tension.py',
       'mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py','mpm_jax/rope_hertz_contact.py',
       'mpm_jax/fibre_coupling.py','mpm_jax/fibre_contact.py','mpm_jax/fibre_patch_contact.py','mpm_jax/contact_patches.py',
       'mpm_jax/fibre_jkr_regions.py','mpm_jax/fibre_region_identity.py','examples/single_fibre_tension.py','tools/package_three_strand_raw.py','tools/finish_fibre_pull_off.py','tools/package_fibre_final_media.py','.github/workflows/three-strand-tension.yml')


PINNED_RUNTIME={'jax':'0.12.0.1','jaxlib':'0.12.0.1','ml-dtypes':'0.6.0','numpy':'2.5.4','opt-einsum':'3.4.0','pillow':'13.0.0','psutil':'7.2.2','scipy':'1.18.1'}


def source():
    expected=os.environ.get('GITHUB_SHA','')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
    if len(head)!=40 or expected!=head:raise ValueError('Actual Git HEAD must equal complete GitHub source SHA')
    own='.github/workflows/three-strand-tension.yml'
    ref=os.environ.get('GITHUB_WORKFLOW_REF','')
    active='.github/workflows/'+ref.split('/.github/workflows/',1)[1].split('@',1)[0] if '/.github/workflows/' in ref else own
    if active not in (own,'.github/workflows/engineering-previews.yml'):raise ValueError('Unregistered active workflow')
    if os.environ.get('GITHUB_ACTIONS')=='true' and not ref:raise ValueError('Actual registered workflow reference is required')
    names=sorted(set(FILES)|{active,'pyproject.toml'})
    files={};blobs={}
    for name in names:
        path=ROOT/name
        if path.is_symlink() or not path.is_file():raise ValueError('Nonregular source path')
        raw=path.read_bytes();committed=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+name])
        if raw!=committed:raise ValueError('Source bytes differ from actual Git HEAD: '+name)
        files[name]=hashlib.sha256(raw).hexdigest();blobs[name]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if (ROOT/active).read_bytes()!=(ROOT/own).read_bytes():raise ValueError('Registered engineering override must exactly equal frozen three-strand workflow bytes')
    tree=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',head+'^{tree}']).decode().strip()
    return {'files':files,'git_blob_sha':blobs,'commit':head,'tree':tree,'active_workflow':active,'run_id':os.environ.get('GITHUB_RUN_ID','local'),'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT','1')}


def inventory(root):
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'receipt.json'}


def environment():
    return {'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in PINNED_RUNTIME}}


def verify(root,expected):
    if len(expected)!=64 or digest(root/'receipt.json')!=expected:raise ValueError('Exact original stage receipt required')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['kind']!='accepted-prefix' or receipt['never_resumable'] or receipt['source']['files']!=source()['files'] or receipt['source']['commit']!=source()['commit']:raise ValueError('Rejected stage/source cannot resume')
    if receipt['files']!=inventory(root):raise ValueError('Complete raw/runtime inventory changed')
    current=receipt;seen=set()
    while current['parent_receipt_sha256']:
        parent_sha=current['parent_receipt_sha256'];path=root/'parents'/(parent_sha+'.json')
        if parent_sha in seen or digest(path)!=parent_sha:raise ValueError('Changed or cyclic parent ancestry')
        seen.add(parent_sha);old=json.loads(path.read_text())
        if old['cache_key']!=current['parent_cache_key'] or old['frame']>=current['frame'] or old['source']['files']!=receipt['source']['files']:raise ValueError('Parent source/frame lineage differs')
        current=old
    if current['frame']!=2:raise ValueError('Accepted ancestry must begin at frame2')
    return receipt


def guarded(arguments,log):
    import psutil
    import signal
    peak=0;minimum=2**63-1;start=time.monotonic();reason=None;error=None;samples=0;process=None;known={};code=None
    try:
        with log.open('w') as stream:
            options={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP}
            process=subprocess.Popen(arguments,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,**options)
            parent=psutil.Process(process.pid);known[parent.pid]=parent
            while process.poll() is None:
                measured=False
                for attempt in range(3):
                    try:
                        members=[parent]+parent.children(recursive=True)
                        for member in members:known[member.pid]=member
                        uss=sum(member.memory_full_info().uss for member in members)
                        measured=True;break
                    except psutil.NoSuchProcess:
                        # Refresh the complete tree. Never silently omit an
                        # unmeasured live child or publish a partial USS sum.
                        if process.poll() is not None:break
                if not measured:
                    if process.poll() is not None:break
                    reason='monitor_incomplete_process_tree';break
                samples+=1;available=psutil.virtual_memory().available;peak=max(peak,uss);minimum=min(minimum,available)
                if uss>=LIMIT:reason='process_tree_uss_6GiB'
                if available<2*1024**3:reason='system_available_below2GiB'
                if time.monotonic()-start>260*60:reason='260minute_wall_guard'
                if reason:break
                time.sleep(1)
    except BaseException as exc:
        reason='monitor_'+type(exc).__name__;error=type(exc).__name__
    finally:
        if process is not None:
            if reason or process.poll() is None:
                if os.name!='nt':
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    except BaseException as exc:reason=reason or 'cleanup_'+type(exc).__name__
                else:
                    try:subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15)
                    except BaseException as exc:reason=reason or 'cleanup_'+type(exc).__name__
            for member in reversed(list(known.values())):
                try:
                    if member.is_running():member.kill()
                except psutil.NoSuchProcess:continue
                except BaseException as exc:reason=reason or 'cleanup_'+type(exc).__name__
            try:code=process.wait(timeout=30)
            except BaseException:
                reason=reason or 'cleanup_wait_failed'
                try:process.kill();code=process.wait(timeout=10)
                except BaseException:code=-999
        if not samples:reason=reason or 'monitor_no_complete_sample'
        record={'peak_tree_uss_bytes':peak,'minimum_system_available_bytes':minimum if samples else None,'complete_samples':samples,'uss_limit_bytes':LIMIT,'reason':reason,'monitor_error':error,'returncode':code,'seconds':time.monotonic()-start}
        save_json(log.with_suffix('.json'),record)
    if reason or code:raise RuntimeError('Guarded stage failed; diagnostic retained: '+str(record))
    return record


def parent_artifact_identity(artifact_id,run_id):
    if not artifact_id.isdecimal() or not run_id.isdecimal():raise ValueError('Exact numeric parent run and artifact IDs required')
    repository=os.environ['GITHUB_REPOSITORY'];token=os.environ['GH_TOKEN']
    def api(suffix):
        request=urllib.request.Request('https://api.github.com/repos/'+repository+'/'+suffix,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    run=api('actions/runs/'+run_id);artifact=api('actions/artifacts/'+artifact_id);binding=source()
    if run['status']!='completed' or run['conclusion']!='success' or run['head_sha']!=binding['commit']:raise ValueError('Exact parent run must be terminal SUCCESS at the unchanged source')
    if artifact['id']!=int(artifact_id) or artifact['expired'] or artifact['workflow_run']['id']!=int(run_id) or artifact['workflow_run']['head_sha']!=binding['commit']:raise ValueError('Parent artifact run/source/expiry mismatch')
    if not artifact['name'].startswith('three-strand-complete-raw-f') or not artifact.get('digest','').startswith('sha256:') or artifact['size_in_bytes']>2*1024**3:raise ValueError('Unexpected or unbounded parent artifact identity')
    return {'run_id':int(run_id),'run_attempt':run['run_attempt'],'head_sha':run['head_sha'],'artifact':artifact,'backend':'github-actions-artifact-only'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','restore-parent','verify','run','render'));p.add_argument('--cache',type=Path,default=ROOT/'.local/three-strand-tension-cache');p.add_argument('--stop-after-frame',type=int,default=2);p.add_argument('--parent-receipt',default='');p.add_argument('--restore-key',default='');p.add_argument('--model',choices=['hertz','jkr'],default='hertz');p.add_argument('--length',type=float,default=.024);p.add_argument('--diameter',type=float,default=.002);p.add_argument('--pitch',type=float,default=.008);p.add_argument('--compactness',type=float,default=.985);p.add_argument('--dx',type=float,default=.0002);p.add_argument('--spacing',type=float,default=.0001);p.add_argument('--parent-artifact-id',default=os.environ.get('PARENT_ARTIFACT_ID',''));p.add_argument('--parent-run-id',default=os.environ.get('PARENT_RUN_ID',''));a=p.parse_args();source();root=a.cache.resolve();geometry,plan=plan_case(length=a.length,diameter=a.diameter,pitch=a.pitch,compactness=a.compactness,dx=a.dx,spacing=a.spacing,model=a.model)
    if a.phase=='prepare':
        if bool(a.restore_key)!=bool(a.parent_artifact_id) or bool(a.restore_key)!=bool(a.parent_run_id):raise ValueError('Continuation requires exact parent artifact and run IDs; fresh f2 requires neither')
        if a.restore_key:
            identity=parent_artifact_identity(a.parent_artifact_id,a.parent_run_id)
            save_json(ROOT/'.local/three-strand-parent-artifact.json',identity)
        if a.stop_after_frame not in (2,10,20,30,40,50,60):raise ValueError('Whole-frame stage endpoint required')
        if bool(a.parent_receipt)!=bool(a.restore_key) or not a.restore_key and a.stop_after_frame!=2:raise ValueError('Fresh stage must stop at frame2; continuation requires exact key and receipt')
        if a.restore_key and not a.restore_key.startswith('three-strand-tension-v1-'+source()['commit']+'-'):raise ValueError('Parent cache source differs')
        key='three-strand-tension-v1-'+source()['commit']+'-'+source()['run_id']+'-'+source()['run_attempt']+f'-f{a.stop_after_frame:02d}'
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('cache_key='+key+'\n')
        return
    if a.phase=='restore-parent':
        from tools.package_three_strand_raw import restore
        identity=parent_artifact_identity(a.parent_artifact_id,a.parent_run_id)
        restore(ROOT/'.local/three-strand-parent-delivery',root,a.parent_receipt)
        receipt=verify(root,a.parent_receipt)
        if receipt['cache_key']!=a.restore_key or receipt['request']!=plan:raise ValueError('Restored exact parent capsule key/physical request differs')
        expected_name='three-strand-complete-raw-f'+str(receipt['frame'])+'-'+str(identity['run_id'])+'-'+str(receipt['source']['run_attempt'])
        if identity['artifact']['name']!=expected_name or str(receipt['source']['run_id'])!=str(identity['run_id']):raise ValueError('Restored receipt differs from exact artifact/run identity')
        if receipt['environment']['python']!='3.12.13' or receipt['environment']['packages']!=PINNED_RUNTIME:raise ValueError('Parent capsule runtime differs from fixed numerical lock')
        save_json(ROOT/'.local/three-strand-parent-artifact.json',identity)
        return
    if a.phase=='verify':verify(root,a.parent_receipt);return
    if a.phase=='render':
        receipt=verify(root,a.parent_receipt)
        if receipt['frame']!=60 or not receipt['independent_review']['numerical_loading_accepted']:raise ValueError('Full independently admitted fracture required before rendering')
        import shutil
        destination=ROOT/'.local/three-strand-tension-media'
        if destination.exists():raise ValueError('Media output must be new')
        shutil.copytree(root/'data',destination)
        # Original numerical cache remains byte-identical; render separate copy.
        command=[sys.executable,'-m','mpm_jax.visualization',str(destination),'--gif','--field','family_id','--resolution','1200']
        guarded(command,ROOT/'.local/three-strand-render.log')
        guarded([sys.executable,str(ROOT/'tools/encode_video.py'),str(destination)],ROOT/'.local/three-strand-video.log')
        guarded([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(ROOT/'tools/check_vtk.py'),str(destination)],ROOT/'.local/three-strand-vtk.log')
        render=json.loads((destination/'render.json').read_text());video=json.loads((destination/'video.json').read_text())
        from PIL import Image
        with Image.open(destination/'simulation.gif') as gif:
            if gif.n_frames!=61:raise ValueError('Missing actual GIF frames')
        if render['renderer']!='ParaView' or render['field']!='family_id' or render['frames_rendered']!=61 or video['frames']!=61 or render['axes'] or render['annotations'] or render.get('geometry_scale',1)!=1 or video['render_sha256']!=digest(destination/'render.json'):raise ValueError('Actual media provenance mismatch')
        from PIL import Image
        frames=sorted((destination/'frames').glob('frame_*.png'))
        if len(frames)!=61:raise ValueError('Missing actual ParaView PNG frames')
        for index,path in enumerate(frames):
            if path.name!=f'frame_{index:05d}.png':raise ValueError('Saved PNG frame sequence differs')
            with Image.open(path) as picture:
                if picture.size!=(1200,1200):raise ValueError('Unexpected actual media resolution')
                picture.verify()
        if inventory(root)!=receipt['files']:raise ValueError('Rendering changed original numerical cache')
        save_json(destination/'media-check.json',{'passed':True,'actual_paraview_frames':61,'geometry_scale':1,'axes':False,'annotations':False,'public_visual_review_passed':False,'original_receipt_sha256':a.parent_receipt})
        return
    if platform.python_version()!='3.12.13' or environment()['packages']!=PINNED_RUNTIME:raise ValueError('Initial and restored numerical runtime must match fixed Python3.12.13 and exact pinned distributions')
    parent=None
    if a.restore_key:
        parent=verify(root,a.parent_receipt)
        if parent['cache_key']!=a.restore_key or parent['frame']>=a.stop_after_frame:raise ValueError('Parent key or advancing endpoint mismatch')
        if environment()!=parent['environment']:raise ValueError('Install original exactly pinned dependencies before continuing')
        endpoints=(2,10,20,30,40,50,60)
        index=endpoints.index(parent['frame'])
        if index+1>=len(endpoints) or a.stop_after_frame!=endpoints[index+1]:raise ValueError('Admit sequential measured stages2,10,20,30,40,50,60; no blind full-run jump')
        first=0 if not parent['parent_receipt_sha256'] else json.loads((root/'parents'/(parent['parent_receipt_sha256']+'.json')).read_text())['frame']
        measured_steps=plan['frame_steps'][parent['frame']]-plan['frame_steps'][first]
        seconds_per_step=parent['resource']['seconds']/max(measured_steps,1)
        projected=2*seconds_per_step*(plan['frame_steps'][a.stop_after_frame]-plan['frame_steps'][parent['frame']])+600
        if projected>260*60:raise ValueError('Measured preceding-stage conservative time estimate exceeds260min; revise stage plan explicitly')
        if parent['resource']['peak_tree_uss_bytes']>=LIMIT:raise ValueError('Previous USS measurement lacks resource admission')
    else:
        if a.stop_after_frame!=2:raise ValueError('First actual resource prefix must end at frame2')
        if root.exists():raise ValueError('Fresh stage needs new isolated cache')
        root.mkdir(parents=True)
    import shutil
    if shutil.disk_usage(root).free<4*1024**3:raise ValueError('Insufficient disk for complete raw/checkpoint evidence')
    logs=ROOT/'.local/three-strand-tension-logs';logs.mkdir(parents=True,exist_ok=True)
    try:
        args=[sys.executable,'-m','examples.three_strand_tension','--output',str(root/'data'),'--stop-after-frame',str(a.stop_after_frame)]
        args+=['--model',a.model,'--length',str(a.length),'--diameter',str(a.diameter),'--pitch',str(a.pitch),'--compactness',str(a.compactness),'--dx',str(a.dx),'--spacing',str(a.spacing)]
        if parent:
            if parent['request']!=plan:raise ValueError('Original full helical physical request differs')
            args+=['--resume']
        resource=guarded(args,logs/'solver.log')
        guarded([sys.executable,str(ROOT/'tools/audit_three_strand_tension.py'),str(root/'data')],logs/'audit.log')
        review=json.loads((root/'data/independent-review.json').read_text());summary=json.loads((root/'data/summary.json').read_text())
        if not review['prefix_verified'] or review['frame']!=a.stop_after_frame:raise ValueError('Independent saved-stage audit failed')
        if parent:
            receipts=root/'parents';receipts.mkdir(exist_ok=True);(receipts/(a.parent_receipt+'.json')).write_bytes((root/'receipt.json').read_bytes())
        runtime=environment();save_json(root/'environment.json',runtime)
        (root/'requirements.txt').write_text(''.join(n+'=='+v+'\n' for n,v in sorted(runtime['packages'].items())))
        key='three-strand-tension-v1-'+source()['commit']+'-'+source()['run_id']+'-'+source()['run_attempt']+f'-f{a.stop_after_frame:02d}'
        files=inventory(root)
        if sum(v['bytes'] for v in files.values())>2*1024**3:raise ValueError('Complete raw cache exceeds2GiB')
        receipt={'storage':{'backend':'github-actions-artifact-only','parent_artifact_id':a.parent_artifact_id,'parent_run_id':a.parent_run_id},'schema':'three-strand-tension-cache-v1','kind':'accepted-prefix','never_resumable':False,'source':source(),'environment':runtime,'frame':a.stop_after_frame,'cache_key':key,'parent_receipt_sha256':a.parent_receipt,'parent_cache_key':a.restore_key,'request':plan,'files':files,'resource':resource,'independent_review':review,'production_eligible':False,'public_visual_review_passed':False}
        save_json(root/'receipt.json',receipt)
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('receipt_sha256='+digest(root/'receipt.json')+'\n')
        print(json.dumps({'cache_key':key,'receipt_sha256':digest(root/'receipt.json'),'independent_review':review}))
    except BaseException:
        save_json(root/'failure.json',{'kind':'diagnostic','never_resumable':True,'stop_after_frame':a.stop_after_frame})
        raise


if __name__=='__main__':main()
