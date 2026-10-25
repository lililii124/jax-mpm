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

ADMISSION = ROOT/'tools/metal_v6_onset_admission.json'
base.ADMISSION = ROOT/'tools/metal_v6_cloud_seed_admission.json'
base.DEST = ROOT/'.local/metal-v6-onset'
base.RAW = base.DEST/'raw'
base.DELIVERY = ROOT/'.local/metal-v6-onset-delivery'
TEMPLATE = '.github/workflows/metal-v6-onset.yml'


def git_context(template):
    a = base.load(ADMISSION)
    if os.environ.get('GITHUB_REPOSITORY') != base.REPO or os.environ.get('GITHUB_REF') != 'refs/heads/work/metal-fragment-v6-cloud':
        raise ValueError('Exact repository/ref required')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit != os.environ.get('GITHUB_SHA'):
        raise ValueError('Checkout differs from dispatched source')
    names = ('tools/run_metal_v6_onset.py','tools/run_metal_v6_energy_admission.py','tools/recover_metal_v6_cloud1000.py','tools/metal_v6_onset_admission.json','tools/metal_v6_cloud_seed_admission.json','examples/metal_fragment_v6_onset_probe.py','mpm_jax/fragment_v6_onset_probe.py','tools/audit_metal_onset_state.py','tools/benchmark_metal_onset_transfer.py','mpm_jax/fragment_v6_energy_probe.py','tools/rht_impact_prefix.py','recovery-seeds/metal-v6-750ns/transport.json','recovery-seeds/metal-v6-750ns/part-000.tarpart',TEMPLATE,'.github/workflows/engineering-previews.yml')
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


def restore():
    import tarfile
    if base.DEST.exists():raise ValueError('Keep prior diagnostic evidence')
    if shutil.disk_usage(ROOT).free<6*1024**3:raise ValueError('6GiB complete seed/run/delivery disk reserve required')
    context=git_context(TEMPLATE);base.RAW.mkdir(parents=True);base.write(base.RAW/'control-source.json',context)
    cfg=base.load(base.ADMISSION);seed_input=ROOT/'recovery-seeds/metal-v6-750ns'
    manifest=base.verify_delivery(seed_input)
    from tools import rht_impact_prefix as transport
    seed=base.DEST/'seed';seed.mkdir()
    with tarfile.open(fileobj=transport.PartReader(seed_input/n for n in manifest['parts']),mode='r|') as tar:
        seen=set()
        for member in tar:
            if not member.isfile() or member.name not in manifest['files'] or member.name in seen:raise ValueError('Wrong seed whole member')
            p=transport.safe(seed,member.name) if hasattr(transport,'safe') else seed/member.name
            if not p.resolve().is_relative_to(seed.resolve()):raise ValueError('Unsafe seed path')
            seen.add(member.name);p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
            item=manifest['files'][member.name]
            if p.stat().st_size!=item['bytes'] or base.sha(p)!=item['sha256']:raise ValueError('Seed member changed')
    if seen!=set(manifest['files']) or base.sha(seed/'seed-context.json')!=cfg['seed_context_sha256']:raise ValueError('Seed full inventory/context changed')
    admission=base.load(ADMISSION);parent=seed/'prefix-proof'
    for n,h in admission['parent_files'].items():
        if base.sha(parent/n)!=h:raise ValueError('Exact125ns parent metadata differs')
    if base.sha(parent/'result/frame-001.npz')!=admission['checkpoint_sha256']:raise ValueError('Exact125ns carry differs')
    for n,h in admission['numerical_source'].items():
        if base.sha(seed/'runtime-source'/n)!=h:raise ValueError('Original numerical source differs')
    base.write(base.RAW/'seed-restoration.json',dict(whole_seed_manifest_sha256=cfg['transport_sha256'],all_original_seed_members_verified=len(seen),exact125ns_parent_checkpoint_sha256=admission['checkpoint_sha256'],original_inputs_unchanged=True,mechanical_steps_executed=0))


def run():
    import psutil
    raw = base.RAW; a = base.load(ADMISSION)
    if a['start_ns']!=125 or a['maximum_adaptive_steps']!=512 or os.environ.get('STOP_NS') not in ('135','250'):raise ValueError('Exact bounded125ns onset probe required')
    context = git_context(TEMPLATE)
    if context != base.load(raw/'control-source.json'):
        raise ValueError('Changed controls after restore')
    source = raw/'runtime-source'; shutil.copytree(base.DEST/'seed/runtime-source',source)
    for name in ('mpm_jax/fragment_v6_energy_probe.py','examples/metal_fragment_v6_onset_probe.py','mpm_jax/fragment_v6_onset_probe.py','tools/audit_metal_onset_state.py','tools/benchmark_metal_onset_transfer.py'):
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,source/name)
    (raw/'probe-admission.json').write_bytes(ADMISSION.read_bytes())
    before = {p.relative_to(source).as_posix():base.sha(p) for p in source.rglob('*') if p.is_file()}
    env = os.environ.copy(); env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable,'-m','examples.metal_fragment_v6_onset_probe','--parent',str(base.DEST/'seed/prefix-proof'),'--admission',str(raw/'probe-admission.json'),'--stop-ns',os.environ['STOP_NS'],'--max-steps','512','--output',str(raw/'result')]
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
        elif args.phase=='restore': restore()
        else: getattr(base,args.phase)()
    except Exception as error:
        base.RAW.mkdir(parents=True,exist_ok=True)
        base.write(base.RAW/(args.phase+'-failure.json'),dict(stage=args.phase,error_type=type(error).__name__,admitted=False,production_restart_allowed=False))
        raise
