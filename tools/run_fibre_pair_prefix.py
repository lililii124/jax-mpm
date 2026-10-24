"""Artifact-only guarded formal f2 for the reliable Hertz/JKR adsorption pair."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from examples.fibre_adhesion_contrast import SOURCE_FILES,digest,save_json
from examples.fibre_pair_dynamic_prefix import plan as plan_case
LIMIT=6*1024**3
PINNED_RUNTIME={'jax':'0.12.0.1','jaxlib':'0.12.0.1','ml-dtypes':'0.6.0','numpy':'2.5.4','opt-einsum':'3.4.0','pillow':'13.0.0','psutil':'7.2.2','scipy':'1.18.1'}
FILES=tuple(SOURCE_FILES)+('examples/fibre_pair_dynamic_prefix.py','tools/run_fibre_pair_prefix.py','tools/review_fibre_pair_prefix.py','tools/review_fibre_adhesion_contrast.py','tools/render_fibre_pair_preview.py','tools/paraview_fibre_pair_scene.py','tools/finish_fibre_pull_off.py','tools/package_fibre_final_media.py','tools/run_woven_preview.py','mpm_jax/paraview_scene.py','mpm_jax/export.py','mpm_jax/particle_fields.py','mpm_jax/visualization.py','pyproject.toml','.github/workflows/fibre-adhesion-pair-f2.yml')


def source():
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
    if len(head)!=40 or head!=os.environ.get('GITHUB_SHA'):raise ValueError('Actual HEAD differs from complete source SHA')
    own='.github/workflows/fibre-adhesion-pair-f2.yml';ref=os.environ.get('GITHUB_WORKFLOW_REF','')
    if '/.github/workflows/' not in ref:raise ValueError('Actual registered workflow identity required')
    active='.github/workflows/'+ref.split('/.github/workflows/',1)[1].split('@',1)[0]
    if active not in (own,'.github/workflows/engineering-previews.yml'):raise ValueError('Unregistered workflow override')
    files={};blobs={}
    for name in sorted(set(FILES)|{active}):
        path=ROOT/name
        if path.is_symlink():raise ValueError('Nonregular source path')
        raw=path.read_bytes();blob=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+name])
        if raw!=blob:raise ValueError('Source differs from exact HEAD blob: '+name)
        files[name]=hashlib.sha256(raw).hexdigest();blobs[name]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if (ROOT/active).read_bytes()!=(ROOT/own).read_bytes():raise ValueError('Registered override differs from frozen workflow')
    return {'commit':head,'files':files,'git_blob_sha':blobs,'active_workflow':active,'run_id':os.environ['GITHUB_RUN_ID'],'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT','1')}


def inventory(root):
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'receipt.json'}


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


def capsule_package(root,delivery):
    import finish_fibre_pull_off as transport
    import package_fibre_final_media as verifier
    if delivery.exists():raise ValueError('New complete raw artifact directory required')
    receipt=json.loads((root/'receipt.json').read_text());sha=digest(root/'receipt.json');delivery.mkdir(parents=True)
    archive=transport.archive_complete_cache(root,receipt,delivery/'raw-archive',sha,{'files':{n:{'sha256':digest(ROOT/n)} for n in ('tools/run_fibre_pair_prefix.py','tools/finish_fibre_pull_off.py','tools/package_fibre_final_media.py')}})
    if archive['compressed_bytes']>2*1024**3:raise ValueError('Complete raw archive exceeds2GiB')
    save_json(delivery/'delivery-manifest.json',{'schema':'fibre-pair-f2-complete-raw-artifact-v1','passed':True,'raw_case':'fibre-adhesion-pair-f2','complete_raw_archive_sha256':archive['sha256'],'original_receipt_sha256':sha,'files':{'raw-archive/archive-manifest.json':transport.file_record(delivery/'raw-archive/archive-manifest.json')},'never_resumable':receipt['never_resumable'],'diagnostic_only':receipt['never_resumable'],'production_eligible':False,'public_visual_review_passed':False})
    result=verifier.verify(delivery);save_json(delivery.parent/'fibre-pair-local-archive-verification.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','run','package','verify','preview'));p.add_argument('--model',choices=('hertz','jkr'),required=True);a=p.parse_args()
    root=ROOT/'.local/fibre-pair-f2-capsule';logs=ROOT/'.local/fibre-pair-f2-logs';delivery=ROOT/'.local/fibre-pair-f2-delivery'
    if a.phase=='verify':
        import package_fibre_final_media as verifier
        r=verifier.verify(ROOT/'.local/fibre-pair-f2-retrieved');save_json(ROOT/'.local/fibre-pair-cloud-retrieval.json',r);print(json.dumps(r));return
    if a.phase=='package':
        if not (root/'receipt.json').exists():raise ValueError('No sealed original prefix/diagnostic to package')
        capsule_package(root,delivery);return
    binding=source();geometry,plan=plan_case(a.model)
    if plan['particles']!=202240 or plan['material']['young']!=4e5 or plan['protocol']['actuator_y_m'][3]!=.00045 or plan['duration_s']!=.04 or not plan['group_scan_qualified']:raise ValueError('Exact selected formal pair physical plan required')
    if a.phase=='prepare':
        save_json(ROOT/'.local/fibre-pair-f2-plan.json',{'source':binding,'request':plan,'artifact_only':True,'resource':{'process_tree_uss_bytes':LIMIT,'local_directory_note':'Isolated working raw only; no Actions cache read/write or quota API'}});return
    if a.phase=='preview':
        receipt=json.loads((root/'receipt.json').read_text())
        if receipt['kind']!='accepted-prefix' or receipt['never_resumable']:raise ValueError('Only independently accepted real f2 can preview')
        guarded([sys.executable,str(ROOT/'tools/render_fibre_pair_preview.py'),str(root/'data'/a.model),str(ROOT/'.local/fibre-pair-f2-preview')],logs/'preview.log');return
    if platform.python_version()!='3.12.13' or {n:importlib.metadata.version(n) for n in PINNED_RUNTIME}!=PINNED_RUNTIME:raise ValueError('Exactly pinned initial numerical environment required')
    if root.exists():raise ValueError('Fresh f2 needs a new isolated capsule')
    root.mkdir(parents=True);logs.mkdir(parents=True,exist_ok=True)
    kind='diagnostic';resource=None;failure=None
    try:
        resource=guarded([sys.executable,'-m','examples.fibre_pair_dynamic_prefix','--model',a.model,'--stop-after-frame','2','--output',str(root/'data')],logs/'solver.log')
        guarded([sys.executable,str(ROOT/'tools/review_fibre_pair_prefix.py'),str(root/'data'/a.model)],logs/'independent-audit.log')
        review=json.loads((root/'data'/a.model/'independent-f2-review.json').read_text())
        if not review['passed'] or review['frame']!=2:raise ValueError('Independent actual f2 audit refused')
        if source()!=binding:raise ValueError('Exact source binding changed during solver')
        kind='accepted-prefix'
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);save_json(root/'failure.json',{'never_resumable':True,'error':failure})
        raise
    finally:
        files=inventory(root)
        if sum(r['bytes'] for r in files.values())>2*1024**3:raise ValueError('Complete original raw exceeds2GiB; no truncated artifact substituted')
        receipt={'schema':'fibre-pair-actual-f2-capsule-v1','kind':kind,'never_resumable':kind!='accepted-prefix','source':binding,'request':plan,'model':a.model,'common_request_sha256':plan['common_request_sha256'],'frame':2 if kind=='accepted-prefix' else None,'cache_key':'fibre-pair-artifact-v1-'+binding['commit']+'-'+a.model+'-'+binding['run_id'],'files':files,'resource':resource,'failure':failure,'backend':'github-actions-artifact-only','environment':{'python':platform.python_version(),'packages':PINNED_RUNTIME},'production_eligible':False,'public_visual_review_passed':False}
        save_json(root/'receipt.json',receipt)
        print(json.dumps({'kind':kind,'receipt_sha256':digest(root/'receipt.json'),'common_request_sha256':plan['common_request_sha256']}))


if __name__=='__main__':main()
