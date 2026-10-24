"""Independent bounded artifact-only original/probe comparison controller."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_metal_v6_cloud1000 as base

ADMISSION = ROOT/'tools/metal_v6_energy_probe_admission.json'
base.ADMISSION = ADMISSION
base.DEST = ROOT/'.local/metal-v6-energy-probe'
base.RAW = base.DEST/'raw'
base.DELIVERY = ROOT/'.local/metal-v6-energy-probe-delivery'
TEMPLATE = '.github/workflows/metal-v6-energy-probe.yml'


def git_context(template):
    a = base.load(ADMISSION)
    proof_name = 'evidence/metal-v6/1250ns-independent-admission.json'
    if a.get('independent_admission_file') != proof_name or base.sha(ROOT/proof_name) != a.get('independent_admission_sha256'):
        raise ValueError('Exact independent1250ns admission proof required')
    proof = base.load(ROOT/proof_name)
    if (proof.get('admitted_numerical_segment') is not True or proof.get('completed_1250ns') is not True
            or proof.get('checkpoint_sha256') != a['checkpoint_sha256']
            or proof.get('receipt_sha256') != a['parent_receipt_sha256']
            or proof.get('public_GIF_allowed') is not False):
        raise ValueError('Independent proof identity or scientific scope differs')
    if os.environ.get('GITHUB_REPOSITORY') != base.REPO or os.environ.get('GITHUB_REF') != a['fixed_feature_ref']:
        raise ValueError('Exact repository/ref required')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit != os.environ.get('GITHUB_SHA'):
        raise ValueError('Checkout differs from dispatched source')
    names = ('tools/run_metal_v6_energy_admission.py','tools/recover_metal_v6_cloud1000.py','tools/metal_v6_energy_probe_admission.json','examples/metal_fragment_v6_energy_admission.py','mpm_jax/fragment_v6_energy_probe.py','tools/rht_impact_prefix.py',proof_name,TEMPLATE,'.github/workflows/engineering-previews.yml')
    files = {}
    for name in names:
        if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('Exact Git source differs: '+name)
        files[name] = base.sha(ROOT/name)
    workflow = os.environ.get('GITHUB_WORKFLOW_REF','').split('@')[0]
    if workflow not in (base.REPO+'/'+TEMPLATE,base.REPO+'/.github/workflows/engineering-previews.yml'):
        raise ValueError('Unregistered probe workflow')
    if workflow.endswith('/engineering-previews.yml') and (ROOT/TEMPLATE).read_bytes() != (ROOT/'.github/workflows/engineering-previews.yml').read_bytes():
        raise ValueError('Registered template differs')
    return dict(commit=commit,repository=base.REPO,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],workflow=workflow,files=files)


base.git_context = git_context


def owned_session_uss(group_id, psutil, attempts=3):
    """Two stable session inventories; never accept a partial live-tree sum."""
    def inventory():
        members = {}
        for process in psutil.process_iter():
            # A vanished process during the scan invalidates this observation;
            # restart the entire scan rather than silently omit its descendants.
            if os.getsid(process.pid) == group_id:
                members[process.pid] = process.create_time()
        owner = psutil.Process(os.getpid())
        members[owner.pid] = owner.create_time()
        return members
    for attempt in range(attempts):
        try:
            before = inventory()
            total = 0
            for pid, created in before.items():
                process = psutil.Process(pid)
                if process.create_time() != created:
                    raise RuntimeError('PID identity changed during USS observation')
                total += process.memory_full_info().uss
            if before == inventory():
                return total
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        # AccessDenied and other observer errors propagate immediately.
        if attempt+1 < attempts:
            time.sleep(.01)
    raise RuntimeError('Incomplete owned-session USS after bounded rescan')


def terminate_owned_group(child):
    """The session belongs to start_new_session Popen, even after leader exit."""
    if child is None:
        return
    import psutil
    deadline = time.monotonic()+20
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(.001,deadline-time.monotonic()))
    while True:
        live = []
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned session still live after bounded cleanup')
        time.sleep(.02)


def run():
    import psutil
    raw = base.RAW; a = base.load(ADMISSION)
    if a.get('producer_admitted') is not True or a.get('parent_admission_complete') is not True:
        raise ValueError('Independent producer admission pending')
    context = git_context(TEMPLATE)
    if context != base.load(raw/'control-source.json'):
        raise ValueError('Changed controls after restore')
    source = raw/'runtime-source'; shutil.copytree(base.DEST/'seed/runtime-source',source)
    for name in ('mpm_jax/fragment_v6_energy_probe.py','examples/metal_fragment_v6_energy_admission.py'):
        shutil.copy2(ROOT/name,source/name)
    (raw/'probe-admission.json').write_bytes(ADMISSION.read_bytes())
    before = {p.relative_to(source).as_posix():base.sha(p) for p in source.rglob('*') if p.is_file()}
    env = os.environ.copy(); env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable,'-m','examples.metal_fragment_v6_energy_admission','--parent',str(base.parent_root()),'--admission',str(raw/'probe-admission.json'),'--output',str(raw/'result')]
    begun = time.monotonic(); peak = 0; reason = None; child = None
    try:
        with (raw/'solve.log').open('wb') as log:
            child = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                try:
                    uss = owned_session_uss(child.pid,psutil)
                    peak = max(peak,uss)
                    if uss > 6*1024**3: reason = '6GiB process-tree USS limit'
                    elif time.monotonic()-begun > 1780: reason = '1780s numerical wall limit'
                    elif shutil.disk_usage(ROOT).free < 3*1024**3: reason = '3GiB packaging disk reserve'
                    elif sum(p.stat().st_size for p in raw.rglob('*') if p.is_file()) > 1.75*1024**3: reason = '1.75GiB raw stop'
                except Exception as error:
                    reason = 'Failclosed monitor: '+type(error).__name__
                if reason:
                    terminate_owned_group(child)
                    break
                time.sleep(.3)
            child.wait(timeout=20)
    except BaseException as error:
        reason = reason or 'Controller exception: '+type(error).__name__
        raise
    finally:
        cleanup_error = None
        try:
            terminate_owned_group(child)
        except BaseException as error:
            cleanup_error = type(error).__name__
            reason = reason or 'Owned-group cleanup: '+cleanup_error
        after = {p.relative_to(source).as_posix():base.sha(p) for p in source.rglob('*') if p.is_file()}
        proof = raw/'result/equivalence.json'
        accepted = reason is None and child is not None and child.returncode == 0 and before == after and proof.exists() and base.load(proof).get('admitted') is True
        base.write(raw/'monitor.json',dict(reason=reason,returncode=None if child is None else child.returncode,peak_process_tree_uss_bytes=peak,elapsed_s=time.monotonic()-begun,source_changed=before!=after,cleanup_error=cleanup_error))
        base.write(raw/'run-receipt.json',dict(schema='v6-energy-probe-admission-v1',admitted=accepted,production_restart_allowed=False,public_media_allowed=False,parent_checkpoint_sha256=a['checkpoint_sha256'],source_context=context,scope='Attribution only; both final states retained, no production restart or refinement admission'))
    if not accepted:
        raise RuntimeError('Probe admission failed; preserve raw and original parent artifacts')


if __name__ == '__main__':
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('restore','install','run','package','verify-retrieval'))
    parser.add_argument('--folder',type=Path); parser.add_argument('--expected')
    args = parser.parse_args()
    try:
        if args.phase == 'run': run()
        elif args.phase == 'verify-retrieval': base.verify_retrieval(args.folder,args.expected)
        else: getattr(base,args.phase)()
    except Exception as error:
        base.RAW.mkdir(parents=True,exist_ok=True)
        base.write(base.RAW/(args.phase+'-failure.json'),dict(stage=args.phase,error_type=type(error).__name__,admitted=False,production_restart_allowed=False))
        raise
