"""Opt-in RHT f3->f4 resource/parity qualification. No imports of JAX in controller.

The original conservative controller remains unchanged. This separate experiment
requires an explicitly selected resource policy and never admits penetration.
"""
import argparse
import ast
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from tools import rht_impact_continue as original

GIB = 1024**3
COMMON = 'bb5b3ff00b4d4edb54632a11f54b81d2edca767f440d9c6f4a7d52a8a3e5a46f'
REMOVED = ['warm, *_ = batch(states, 0.0, material_arguments)',
           'warm[0].x.block_until_ready()', 'del warm']


def derive(raw):
    if hashlib.sha256(raw).hexdigest() != COMMON:
        raise RuntimeError('Unrecognized original common.py')
    tree = ast.parse(raw)
    expected = copy.deepcopy(tree)
    run = next(n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    canonical = lambda n: ast.dump(n, include_attributes=False)
    wanted = [canonical(ast.parse(s).body[0]) for s in REMOVED]
    removed = [n for n in run.body if canonical(n) in wanted]
    if [canonical(n) for n in removed] != wanted:
        raise RuntimeError('Unexpected warmup structure')
    run.body = [n for n in run.body if n not in removed]
    lines = raw.decode().splitlines(keepends=True)
    for node in removed:
        if node.lineno != node.end_lineno:
            raise RuntimeError('Unexpected multiline warmup')
        lines[node.lineno-1] = '    # audited control-only removal: '+lines[node.lineno-1].strip()+'\n'
    derived = ''.join(lines).encode()
    if canonical(ast.parse(derived)) != canonical(expected):
        raise RuntimeError('Numerical AST changed')
    return derived


def snapshot():
    import psutil
    files = {}
    for name in ('cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.current',
                 'memory.peak', 'memory.events', 'cpu.stat'):
        p = Path('/sys/fs/cgroup')/name
        files[name] = p.read_text().strip() if p.exists() else None
    quota = files['cpu.max']
    effective = float(len(os.sched_getaffinity(0))) if hasattr(os, 'sched_getaffinity') else float(psutil.cpu_count() or 0)
    if quota and quota.split()[0] != 'max':
        q, period = map(int, quota.split())
        effective = min(effective, q/period)
    available = psutil.virtual_memory().available
    if files['memory.max'] and files['memory.max'] != 'max':
        available = min(available, int(files['memory.max'])-int(files['memory.current']))
    return {'logical_cpus': psutil.cpu_count(), 'effective_cpus': effective,
            'effective_available_bytes': available, 'cgroup_v2': files,
            'cgroup_membership': Path('/proc/self/cgroup').read_text() if Path('/proc/self/cgroup').exists() else None}


def admission(resources, policy):
    conservative = (resources['logical_cpus'] or 0)>=8 and resources['effective_available_bytes']>=12*GIB
    measured = resources['effective_cpus']>=2 and resources['effective_available_bytes']>=4*GIB
    accepted = conservative if policy == 'conservative' else measured
    record = {'policy': policy, 'conservative_pass': conservative,
              'measured_experimental_pass': measured, 'accepted': accepted,
              'experimental_uss_ceiling_bytes':3*GIB,
              'reason':'Measured ~2 GiB original peak motivates isolated <=3 GiB USS qualification; numerical gates unchanged.'}
    if not accepted:
        raise RuntimeError('Resource admission rejected: '+json.dumps(record))
    return record


def owned_session_uss(group_id, psutil, attempts=3):
    """Two stable session inventories; never accept a partial live-tree sum."""

    group_ids = {group_id} if isinstance(group_id,int) else set(group_id)

    def inventory():
        members = {}
        for process in psutil.process_iter():
            if os.getsid(process.pid) in group_ids:
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
        if attempt + 1 < attempts:
            time.sleep(0.01)
    raise RuntimeError('Incomplete owned-session USS after bounded rescan')

def terminate_owned_group(child):
    """The session belongs to start_new_session Popen, even after leader exit."""
    if child is None:
        return
    import psutil
    deadline = time.monotonic() + 20
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(0.001, deadline - time.monotonic()))
    while True:
        live = []
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if pid == 0:
                break
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned session still live after bounded cleanup')
        time.sleep(0.02)

def run_arm(parent, destination, policy, arm):
    # Keep the original exact Git/controller/workflow binding, runtime and archive gates.
    if sys.platform!='linux' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise RuntimeError('Linux owned-session subreaper required')
    resources = snapshot()
    decision = admission(resources, policy)
    binding = original.controller_identity()
    root = Path(__file__).resolve().parents[1]
    wrapper_paths = [Path(__file__), root/'tools/rht_qualification_child.py']
    wrapper_hashes = {str(p): original.transport.digest(p) for p in wrapper_paths}
    for wrapper in wrapper_paths:
        relative = wrapper.relative_to(root).as_posix()
        if wrapper.read_bytes()!=subprocess.check_output(['git','show',binding['commit']+':'+relative],cwd=root):
            raise RuntimeError('Qualification wrapper differs from registered Git blob: '+relative)
    membership = resources['cgroup_membership']
    if membership and resources['cgroup_v2']['memory.max'] is None:
        raise RuntimeError('Unsupported cgroup layout: effective memory cannot be established')
    old_admit, old_popen = original.resource_admission, subprocess.Popen
    processes = []
    process_lock = threading.RLock()
    stop = threading.Event()
    extra = {'resource_start':resources, 'decision':decision, 'arm':arm,
             'wrapper_hashes':wrapper_hashes, 'derived_common_sha256':None,
             'status':'preparing', 'peak_owned_session_uss_bytes':0, 'cleanup_errors':[]}
    def resource_gate(cpus, available):
        return dict(snapshot=snapshot(), decision=admission(snapshot(), policy))
    def popen(command, *args, **kwargs):
        if 'resource_guard_error' in extra:
            raise RuntimeError(extra['resource_guard_error'])
        if isinstance(command,list) and 'runpy.run_module' in ' '.join(command):
            source = Path(command[4])
            if arm == 'no-warmup':
                derived = Path(destination).resolve()/'derived-source'
                shutil.copytree(source, derived)
                common = derived/'examples/common.py'
                common.write_bytes(derive(common.read_bytes()))
                extra['derived_common_sha256'] = original.transport.digest(common)
                source = derived
            timing = Path(destination).resolve()/'phase-timings.jsonl'
            command = [sys.executable, '-I', str(root/'tools/rht_qualification_child.py'),
                       str(source), str(timing), *command[5:]]
            kwargs['cwd'] = source
        numerical = isinstance(command,list) and command and command[0]==sys.executable
        if numerical:
            kwargs['start_new_session'] = True
        with process_lock:
            if 'resource_guard_error' in extra:
                raise RuntimeError(extra['resource_guard_error'])
            p = old_popen(command,*args,**kwargs)
            if numerical:
                processes.append(p)
        return p
    def monitor():
        try:
            import psutil
            while not stop.wait(.5):
                with process_lock:
                    groups = list(processes)
                # Two stable inventories across ALL sessions plus owner once.
                # Finished leaders stay registered so orphan descendants count.
                total = owned_session_uss([p.pid for p in groups],psutil)
                extra['peak_owned_session_uss_bytes'] = max(extra['peak_owned_session_uss_bytes'],total)
                if total>3*GIB:
                    raise RuntimeError('Experimental 3 GiB owned-session USS ceiling exceeded')
        except BaseException as exc:
            extra['resource_guard_error'] = f'{type(exc).__name__}: {exc}'
            # Never leave original.run waiting on a live unobserved process.
            # Popen rejects all later numerical launches once this flag is set.
            with process_lock:
                for child in processes:
                    try:
                        os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as kill_exc:
                        extra['cleanup_errors'].append(f'observer kill {child.pid}: {type(kill_exc).__name__}: {kill_exc}')
    original.resource_admission = resource_gate
    subprocess.Popen = popen
    watcher = threading.Thread(target=monitor,daemon=True)
    watcher.start()
    try:
        extra['status']='running'
        original.run(parent,destination)
        if 'resource_guard_error' in extra:
            raise RuntimeError(extra['resource_guard_error'])
        if original.controller_identity()!=binding or any(original.transport.digest(p)!=wrapper_hashes[str(p)] for p in wrapper_paths):
            raise RuntimeError('Qualification controller changed')
        # Every prior frame must remain byte-identical, independently of parity.
        base = Path(destination)/'parent-evidence/raw/frames_data'
        for p in base.iterdir():
            if p.is_file() and original.transport.digest(p)!=original.transport.digest(Path(destination)/'raw/frames_data'/p.name):
                raise RuntimeError('Original frame prefix changed')
        extra['status']='qualified-arm'
    except BaseException as exc:
        extra.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop.set()
        watcher.join(timeout=10)
        if watcher.is_alive():
            extra['resource_guard_error']='Observer failed to stop within 10 seconds'
        # Cleanup runs for every owned group, including exited group leaders.
        # Restore monkeypatches even when group cleanup or resource sampling fails.
        try:
            with process_lock:
                groups=list(processes)
            for child in groups:
                try:
                    terminate_owned_group(child)
                except BaseException as exc:
                    extra['cleanup_errors'].append(f'{child.pid}: {type(exc).__name__}: {exc}')
        finally:
            original.resource_admission, subprocess.Popen = old_admit, old_popen
            try:
                extra['resource_end'] = snapshot()
            except BaseException as exc:
                extra['resource_end_error']=f'{type(exc).__name__}: {exc}'
            if extra.get('resource_guard_error') or extra['cleanup_errors'] or extra.get('resource_end_error'):
                extra['status']='failed'
            Path(destination).mkdir(parents=True,exist_ok=True)
            original.transport.write(Path(destination)/'qualification.json',extra)
        if extra['status']=='failed' and sys.exc_info()[0] is None:
            raise RuntimeError('Qualification observer/cleanup failed closed; see qualification.json')


def compare(a,b):
    roots=[Path(a),Path(b)]
    qualifications=[json.loads((p/'qualification.json').read_text()) for p in roots]
    if ({q.get('arm') for q in qualifications}!={'original','no-warmup'}
            or any(q.get('status')!='qualified-arm' or q.get('cleanup_errors')
                   or q.get('resource_guard_error') for q in qualifications)):
        raise RuntimeError('Require two independently qualified, cleaned original/no-warmup arms')
    if qualifications[0].get('wrapper_hashes')!=qualifications[1].get('wrapper_hashes'):
        raise RuntimeError('Parity controllers differ')
    for p in roots:
        receipt=json.loads((p/'receipt.json').read_text())
        checkpoint=json.loads((p/'raw/checkpoint.json').read_text())
        segment=json.loads((p/'raw/segment.json').read_text())
        if (receipt.get('status')!='valid-impact-prefix' or receipt.get('incremental_steps')!=35
                or checkpoint.get('frame')!=4 or segment.get('frame')!=4
                or segment.get('segment_complete') is not True):
            raise RuntimeError('Missing independently admitted 35-step f4 endpoint')
    import numpy as np
    a,b = roots[0]/'raw',roots[1]/'raw'
    checked=[]
    # Compare all numerical leaves of every NPZ, including checkpoint history,
    # impulse/work and full frame arrays. Timing is never stored in these arrays.
    left={p.relative_to(a) for p in a.rglob('*.npz')}
    right={p.relative_to(b) for p in b.rglob('*.npz')}
    if left!=right or not left:
        raise RuntimeError('Numerical file inventory differs or absent')
    for name in sorted(left):
        with np.load(a/name,allow_pickle=False) as x,np.load(b/name,allow_pickle=False) as y:
            if set(x.files)!=set(y.files):
                raise RuntimeError('NPZ leaf inventory differs: '+str(name))
            for key in x.files:
                u,v=x[key],y[key]
                if u.dtype!=v.dtype or u.shape!=v.shape or u.tobytes()!=v.tobytes():
                    raise RuntimeError('Bitwise parity failed: '+str(name)+':'+key)
        checked.append(str(name))
    # Compare checkpoint accounting/signature and deterministic CSV observations.
    def scrub(value):
        if isinstance(value,dict):
            return {k:scrub(v) for k,v in value.items() if k not in {'solver_seconds','compile_and_warmup_seconds'}}
        if isinstance(value,list):
            return [scrub(v) for v in value]
        return value
    if scrub(json.loads((a/'checkpoint.json').read_text())) != scrub(json.loads((b/'checkpoint.json').read_text())):
        raise RuntimeError('Checkpoint accounting differs')
    for name in ('observables.csv',):
        if (a/name).read_bytes()!=(b/name).read_bytes():
            raise RuntimeError('Observations differ: '+name)
    return {'status':'bitwise-f3-f4-parity-passed','npz_files':checked,
            'penetration_validated':False,'full_48_frame_plan_completed':False,
            'limitation':'Only one exact-plan frame qualified. No automatic next segment.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['resources','arm','compare'])
    p.add_argument('--policy',choices=['conservative','measured-experimental'],default='conservative')
    p.add_argument('--arm',choices=['original','no-warmup'])
    p.add_argument('--parent',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--other',type=Path)
    p.add_argument('--weekly-used',type=int)
    args=p.parse_args()
    if args.action=='resources':
        print(json.dumps(snapshot(),indent=2)); return
    if args.action=='arm':
        if args.weekly_used is None or not 0<=args.weekly_used<90:
            p.error('Fresh publisher-confirmed --weekly-used <90 required; recheck before every arm')
        if not args.parent or not args.destination or not args.arm:
            p.error('--parent --destination --arm required')
        run_arm(args.parent,args.destination,args.policy,args.arm)
    else:
        if not args.destination or not args.other:
            p.error('--destination and --other required')
        print(json.dumps(compare(args.destination,args.other),indent=2))

if __name__=='__main__':
    main()
