"""Authenticated cloud publisher for the bounded temporal diagnostic."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_metal_v6_onset as onset
from tools.run_metal_v6_dt_convergence import SOURCES, hashes

base = onset.base
TEMPLATE = '.github/workflows/metal-v6-dt-convergence.yml'
ADMISSION = ROOT/'tools/metal_v6_energy_probe125_admission.json'
MANIFEST = ROOT/'tools/metal_v6_dt_convergence_source_admission.json'
base.DEST = ROOT/'.local/metal-v6-dt-convergence'
base.RAW = base.DEST/'raw'
base.DELIVERY = ROOT/'.local/metal-v6-dt-delivery'


def git_context(template):
    if os.environ.get('GITHUB_REPOSITORY') != base.REPO or os.environ.get('GITHUB_REF') != 'refs/heads/work/metal-fragment-v6-cloud':
        raise ValueError('Exact repository/ref required')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit != os.environ.get('GITHUB_SHA'):raise ValueError('Checkout differs from dispatch')
    names = (*SOURCES,'tools/publish_metal_v6_dt_convergence.py',
             'tools/run_metal_v6_onset.py','tools/recover_metal_v6_cloud1000.py',
             'tools/rht_impact_prefix.py','tools/metal_v6_cloud_seed_admission.json',
             'tools/metal_v6_onset_admission.json',ADMISSION.relative_to(ROOT).as_posix(),
             MANIFEST.relative_to(ROOT).as_posix(),'recovery-seeds/metal-v6-750ns/transport.json',
             'recovery-seeds/metal-v6-750ns/part-000.tarpart',TEMPLATE,'.github/workflows/engineering-previews.yml')
    files = {}
    for name in names:
        if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('Git source differs: '+name)
        files[name] = base.sha(ROOT/name)
    if hashes() != base.load(MANIFEST):raise ValueError('Pinned diagnostic source differs')
    workflow = os.environ.get('GITHUB_WORKFLOW_REF','').split('@')[0]
    if workflow not in (base.REPO+'/'+TEMPLATE,base.REPO+'/.github/workflows/engineering-previews.yml'):
        raise ValueError('Unregistered workflow')
    if workflow.endswith('/engineering-previews.yml') and (ROOT/TEMPLATE).read_bytes() != (ROOT/'.github/workflows/engineering-previews.yml').read_bytes():
        raise ValueError('Registered template differs')
    return dict(commit=commit,repository=base.REPO,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],workflow=workflow,files=files)


onset.git_context = base.git_context = git_context


def run():
    import psutil
    from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group
    raw = base.RAW
    context = git_context(TEMPLATE)
    if context != base.load(raw/'control-source.json'):raise ValueError('Changed publisher source')
    if os.environ.get('STOP_NS') != '135':raise ValueError('First-stage125–135ns only; renewed admission required for250ns')
    source = raw/'runtime-source'
    shutil.copytree(base.DEST/'seed/runtime-source', source)
    for name in SOURCES:
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,source/name)
    shutil.copy2(ADMISSION,raw/'frozen-admission.json')
    shutil.copy2(MANIFEST,raw/'source-admission.json')
    before = {p.relative_to(source).as_posix():base.sha(p) for p in source.rglob('*') if p.is_file()}
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    smoke = [sys.executable,'-m','tools.smoke_metal_v6_dt_runtime','--runtime-source',str(source),'--source-admission',str(raw/'source-admission.json'),'--frozen-admission',str(raw/'frozen-admission.json'),'--output',str(raw/'smoke.json'),'--numerical-imports']
    solve = [sys.executable,'-m','tools.run_metal_v6_dt_convergence','--parent',str(base.DEST/'seed/prefix-proof'),'--admission',str(raw/'frozen-admission.json'),'--source-admission',str(raw/'source-admission.json'),'--smoke-receipt',str(raw/'smoke.json'),'--output',str(raw/'result'),'--stop-ns',os.environ['STOP_NS'],'--wall-seconds','1600']
    begun = time.monotonic(); child = None; peak = 0; failure = None
    def cleanup(process):
        # The runner owns separate numerical sessions. Kill those before its
        # controller so an outer guard failure cannot orphan a numerical cell.
        groups = set()
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
            for descendant in descendants:
                try:
                    group = os.getsid(descendant.pid)
                    if group != os.getsid(0):groups.add(group)
                except (psutil.NoSuchProcess,ProcessLookupError):pass
        except psutil.NoSuchProcess:pass
        import signal
        for group in groups:
            try:os.killpg(group,signal.SIGKILL)
            except ProcessLookupError:pass
        terminate_owned_group(process)
    try:
        for name,command,cap in (('smoke',smoke,120),('solve',solve,1660)):
            started = time.monotonic()
            with (raw/(name+'.log')).open('xb') as log:
                child = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while child.poll() is None:
                        peak = max(peak,owned_session_uss(child.pid,psutil))
                        if peak>6*1024**3:raise RuntimeError('6GiB owned-session USS cap')
                        if time.monotonic()-started>cap:raise RuntimeError('Bounded process wall cap')
                        if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('3GiB packaging reserve')
                        if sum(p.stat().st_size for p in raw.rglob('*') if p.is_file())>1.75*1024**3:raise RuntimeError('1.75GiB raw cap')
                        time.sleep(.3)
                    if child.returncode:raise RuntimeError(name+' failed')
                finally:cleanup(child)
    except BaseException as error:
        failure = type(error).__name__+': '+str(error)
    after = {p.relative_to(source).as_posix():base.sha(p) for p in source.rglob('*') if p.is_file()}
    accepted = failure is None and before == after and (raw/'result/comparison.json').exists()
    base.write(raw/'run-receipt.json',dict(admitted=accepted,failure=failure,elapsed_s=time.monotonic()-begun,peak_owned_session_uss_bytes=peak,source_unchanged=before==after,source_context=context,production_restart_allowed=False,public_media_allowed=False))
    if not accepted:raise RuntimeError('Temporal diagnostic failed; preserve all raw')


if __name__ == '__main__':
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('restore','install','run','package','verify-retrieval'))
    parser.add_argument('--folder',type=Path);parser.add_argument('--expected')
    args = parser.parse_args()
    if args.phase == 'restore':onset.restore()
    elif args.phase == 'run':run()
    elif args.phase == 'verify-retrieval':base.verify_retrieval(args.folder,args.expected)
    else:getattr(base,args.phase)()
