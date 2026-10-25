"""Resource preflight precedes JAX import; original inputs are never modified."""
import argparse
import ctypes
import signal
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--resource-profile', choices=('large-original', 'standard-linux-qualification'), default='large-original')
    args = parser.parse_args()
    import psutil
    memory = psutil.virtual_memory()
    report = {'platform': platform.system(), 'cpu_count': os.cpu_count(),
              'available_bytes': memory.available, 'total_bytes': memory.total,
              'scope': 'diagnostic only; never admission or resumable trajectory'}
    if os.name == 'nt':
        class PerformanceInformation(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
                ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
                 'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
                 'KernelNonpaged', 'PageSize')] + [(k, ctypes.c_ulong) for k in
                ('HandleCount', 'ProcessCount', 'ThreadCount')]
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise OSError('Cannot obtain Windows commit headroom')
        report['commit_headroom_bytes'] = (info.CommitLimit-info.CommitTotal)*info.PageSize
    standard = args.resource_profile == 'standard-linux-qualification'
    cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else (os.cpu_count() or 0)
    reserve = (5 if standard else 12)*1024**3
    report.update(resource_profile=args.resource_profile, affinity_cpu_count=cpus,
                  required_available_bytes=reserve, required_cpu_count=2 if standard else 8,
                  qualification_only=standard)
    report['resource_gate'] = (memory.available >= reserve and
                              cpus >= report['required_cpu_count'] and
                              (not standard or platform.system() == 'Linux') and
                              report.get('commit_headroom_bytes', reserve) >= reserve)
    print(json.dumps(report), flush=True)
    if args.preflight_only:
        return 0 if report['resource_gate'] else 2
    if not report['resource_gate']:
        raise SystemExit('Insufficient headroom: no JAX imported, no replay attempted')
    repo, data, output = args.repo.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists() or output.with_suffix('.preflight.json').exists():
        raise FileExistsError('Output already exists; preserve prior diagnostic')
    if data == output.parent or data in output.parents:
        raise ValueError('Output must be outside original raw directory')
    payload = Path(__file__).with_name('exact_probe_guarded_payload.py')
    files = [data/n for n in ('checkpoint.npz', 'plan.json', 'trajectory_x.npy')]
    files += [payload]
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    before = {str(p): digest(p) for p in files}
    report['sha256'] = before
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.preflight.json').write_text(json.dumps(report, indent=2))
    env = dict(os.environ, MPM_REPO_ROOT=str(repo), MPM_ORIGINAL_DATA=str(data),
               MPM_PROBE_OUTPUT=str(output), JAX_PLATFORMS='cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2')
    if sys.platform != "linux" or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Linux owned-session subreaper required")
    started = time.monotonic()
    status = {'uss_limit_bytes': (4 if standard else 6)*1024**3, 'timeout_seconds': args.timeout_seconds,
              'minimum_available_bytes': memory.available, 'system_reserve_bytes': 1024**3,
              'peak_tree_uss_bytes': 0, 'observed_tree_cpu_seconds': 0., 'complete_samples': 0,
              'reason': None, 'resource_profile': args.resource_profile}
    def interrupted(signum, frame):
        raise RuntimeError('Termination signal '+str(signum))
    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    with output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(payload)], cwd=repo, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                uss = owned_session_uss(child.pid, psutil)
                cpu_seconds = 0.
                for process in psutil.process_iter():
                    try:
                        if os.getsid(process.pid) == child.pid:
                            cpu_seconds += sum(process.cpu_times()[:2])
                    except (psutil.NoSuchProcess, ProcessLookupError):
                        pass
                status['observed_tree_cpu_seconds'] = max(status['observed_tree_cpu_seconds'], cpu_seconds)
                status['complete_samples'] += 1
                status['peak_tree_uss_bytes'] = max(status['peak_tree_uss_bytes'], uss)
                status['minimum_available_bytes'] = min(status['minimum_available_bytes'], psutil.virtual_memory().available)
                if uss > status['uss_limit_bytes']:
                    status['reason'] = 'process-tree USS limit exceeded'
                    break
                if status['minimum_available_bytes'] < status['system_reserve_bytes']:
                    status['reason'] = 'system available reserve breached'
                    break
                if time.monotonic()-started > args.timeout_seconds:
                    status['reason'] = 'bounded probe timeout'
                    break
                time.sleep(.2)
        except BaseException as error:
            status['reason'] = 'monitor failed or interrupted: '+repr(error)
        finally:
            try:
                terminate_owned_group(child)
                status['owned_session_cleanup_verified'] = True
            except Exception as error:
                status['owned_session_cleanup_verified'] = False
                status['reason'] = 'cleanup failed: '+repr(error)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            status['returncode'] = child.poll()
            status['seconds'] = time.monotonic()-started
            output.with_suffix('.monitor.json').write_text(json.dumps(status, indent=2))
    if before != {str(p): digest(p) for p in files}:
        raise RuntimeError('Original input or payload changed during replay')
    return status['returncode'] if status['reason'] is None and status['returncode'] is not None else 3


if __name__ == '__main__':
    raise SystemExit(main())
