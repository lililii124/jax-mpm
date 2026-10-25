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
import re
from pathlib import Path, PurePosixPath
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


def bounded_read(path,limit=262144):
    with Path(path).open() as stream:
        value=stream.read(limit+1)
    if len(value)>limit:
        raise RuntimeError('Resource evidence exceeds bounded read: '+str(path))
    return value


def cgroup_constraints(membership,mountinfo,read=bounded_read):
    """Resolve controller mounts and ancestor limits, including cgroup v1/hybrid."""
    unescape=lambda value: re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),value)
    mounts=[]
    for line in mountinfo.splitlines():
        left,sep,right=line.partition(' - ')
        if not sep:
            raise RuntimeError('Malformed mountinfo')
        fields,tail=left.split(),right.split()
        if len(fields)<6 or len(tail)<3:
            raise RuntimeError('Malformed mountinfo fields')
        if tail[0] in ('cgroup','cgroup2'):
            mounts.append((tail[0],unescape(fields[3]),unescape(fields[4]),set(tail[2].split(','))))
    groups=[]
    for line in membership.splitlines():
        parts=line.split(':',2)
        if len(parts)!=3 or not parts[2].startswith('/'):
            raise RuntimeError('Malformed cgroup membership')
        groups.append((set(parts[1].split(',')) if parts[1] else set(),parts[2]))
    evidence=[]; memory=[]; cpu=[]; verified_memory=False
    for controllers,path in groups:
        kind='cgroup' if controllers else 'cgroup2'
        relevant=controllers & {'memory','cpu'}
        if kind=='cgroup' and not relevant:
            continue
        matches=[m for m in mounts if m[0]==kind and (kind=='cgroup2' or relevant<=m[3])]
        if len(matches)!=1:
            raise RuntimeError('Cannot uniquely resolve cgroup controller mount')
        _,root,mount,options=matches[0]
        root_path,member=PurePosixPath(root),PurePosixPath(path)
        if '..' in member.parts or '..' in root_path.parts:
            raise RuntimeError('Unsafe cgroup path')
        if member.is_relative_to(root_path):
            relative=member.relative_to(root_path)
        elif path=='/':
            # A cgroup namespace exposes its mount root as membership '/'.
            relative=PurePosixPath('.')
        else:
            raise RuntimeError('Cgroup membership outside visible mount root')
        base=Path(mount); leaf=base/str(relative)
        observed=[]; found_memory=False
        current=leaf
        while True:
            names=(('memory.max','memory.current','cpu.max') if kind=='cgroup2' else
                   tuple((['memory.limit_in_bytes','memory.usage_in_bytes'] if 'memory' in controllers else [])+
                         (['cpu.cfs_quota_us','cpu.cfs_period_us'] if 'cpu' in controllers else [])))
            values={}
            for name in names:
                try:
                    values[name]=read(current/name,16384).strip()
                except FileNotFoundError:
                    values[name]=None
            if kind=='cgroup2':
                limit,usage=values.get('memory.max'),values.get('memory.current')
                quota=values.get('cpu.max')
                if quota:
                    q,period=quota.split()
                    if int(period)<=0:raise RuntimeError('Invalid CPU period')
                    if q!='max':cpu.append(int(q)/int(period))
            else:
                limit,usage=values.get('memory.limit_in_bytes'),values.get('memory.usage_in_bytes')
                q,period=values.get('cpu.cfs_quota_us'),values.get('cpu.cfs_period_us')
                if q is not None:
                    if period is None or int(period)<=0:raise RuntimeError('Missing/invalid v1 CPU period')
                    if int(q)>=0:cpu.append(int(q)/int(period))
            if limit is not None:
                found_memory=True
                if usage is None:raise RuntimeError('Missing cgroup memory usage')
                if int(usage)<0:raise RuntimeError('Invalid negative memory usage')
                if limit!='max':
                    if int(limit)<0:raise RuntimeError('Invalid negative memory limit')
                    memory.append(max(0,int(limit)-int(usage)))
            observed.append({'path':str(current),'files':values})
            if current==base:break
            if not current.is_relative_to(base):raise RuntimeError('Cgroup path escaped mount')
            current=current.parent
        # The true unified root has no memory.max/cpu.max by kernel design.
        # Only this fully verified root may use host limits without memory.max.
        true_root=(kind=='cgroup2' and root=='/' and path=='/' and leaf==base)
        if kind=='cgroup2' or 'memory' in controllers:
            if not found_memory and not true_root:
                raise RuntimeError('Resolved memory controller has no readable limits')
            verified_memory=True
        evidence.append({'kind':kind,'controllers':sorted(controllers),'mount_root':root,
                         'mount_point':mount,'membership':path,'true_unified_root':true_root,
                         'ancestors':observed})
    if not verified_memory:
        raise RuntimeError('No verifiable memory cgroup controller/root')
    return {'available_bounds':memory,'cpu_bounds':cpu,'resolved':evidence}


def snapshot():
    import psutil
    membership=bounded_read('/proc/self/cgroup',65536)
    mountinfo=bounded_read('/proc/self/mountinfo')
    constraints=cgroup_constraints(membership,mountinfo)
    effective=float(len(os.sched_getaffinity(0)))
    available=psutil.virtual_memory().available
    effective=min([effective,*constraints['cpu_bounds']])
    available=min([available,*constraints['available_bounds']])
    return {'logical_cpus':psutil.cpu_count(),'effective_cpus':effective,
            'effective_available_bytes':available,'cgroup_constraints':constraints,
            'cgroup_membership':membership,'mountinfo':mountinfo}


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
        existed=args.destination.exists()
        try:
            run_arm(args.parent,args.destination,args.policy,args.arm)
        except BaseException as exc:
            if not existed:
                args.destination.mkdir(parents=True,exist_ok=True)
                diagnostic={'status':'preflight-or-arm-failed','arm':args.arm,
                            'error':f'{type(exc).__name__}: {exc}',
                            'physical_result_admitted':False,'perforation_validated':False}
                for name,limit in (('/proc/self/cgroup',65536),('/proc/self/mountinfo',262144)):
                    try:diagnostic[name]=bounded_read(name,limit)
                    except BaseException as read_exc:diagnostic[name]=f'{type(read_exc).__name__}: {read_exc}'
                original.transport.write(args.destination/'resource-diagnostic.json',diagnostic)
                if not (args.destination/'qualification.json').exists():
                    original.transport.write(args.destination/'qualification.json',dict(diagnostic,status='preflight-failed'))
                if not (args.destination/'receipt.json').exists():
                    original.transport.write(args.destination/'receipt.json',dict(diagnostic,status='preflight-failed'))
            raise
    else:
        if not args.destination or not args.other:
            p.error('--destination and --other required')
        print(json.dumps(compare(args.destination,args.other),indent=2))

if __name__=='__main__':
    main()
