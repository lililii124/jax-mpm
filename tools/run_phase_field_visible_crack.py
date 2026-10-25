"""Guarded CPU producer; incomplete solves never receive a completion certificate."""
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import signal
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase-field-visible-crack'
PINS = {'jax': '0.12.0.1', 'jaxlib': '0.12.0.1', 'numpy': '2.5.4',
        'scipy': '1.18.1', 'ml-dtypes': '0.6.0', 'opt-einsum': '3.4.0',
        'psutil': '7.2.2', 'pillow': '13.0.0'}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


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
        # Descendants may create another process group in our owned session.
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Linux subreaper adopts orphaned grandchildren, allowing real reaping.
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
        time.sleep(.02)


def source_gate():
    repo="lililii124/jax-mpm"
    ref="refs/heads/work/phase-field-visible-crack"
    if os.environ.get("GITHUB_REPOSITORY") != repo or os.environ.get("GITHUB_REF") != ref:
        raise RuntimeError("Exact repository and feature ref required")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if head != os.environ.get("GITHUB_SHA"):
        raise RuntimeError("Dispatched HEAD differs")
    own=".github/workflows/phase-field-visible-crack.yml"
    actual=os.environ["GITHUB_WORKFLOW_REF"].split("@",1)[0]
    allowed=[repo+"/"+own,repo+"/.github/workflows/engineering-previews.yml"]
    if actual not in allowed:
        raise RuntimeError("Unregistered workflow")
    actual=actual[len(repo)+1:]
    if (ROOT/actual).read_bytes() != (ROOT/own).read_bytes():
        raise RuntimeError("Active workflow differs from frozen template")
    names=sorted(set([str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT/"mpm_jax").rglob("*.py")]+["examples/__init__.py","examples/phase_field_visible_crack.py","tools/run_phase_field_visible_crack.py","tools/render_phase_field_visible_crack.py","tools/encode_phase_field_visible_crack.py","tools/run_woven_preview.py","tools/audit_phase_field_visible_crack.py","tools/phase_field_bspline_sampling.py",own,actual]))
    files={}
    for name in names:
        raw=(ROOT/name).read_bytes()
        if raw != subprocess.check_output(["git","show",head+":"+name],cwd=ROOT):
            raise RuntimeError("Source differs from Git blob: "+name)
        files[name]=hashlib.sha256(raw).hexdigest()
    return dict(commit=head,repository=repo,ref=ref,workflow=actual,files=files)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if list(OUT.glob('frame_*.npz')) or (OUT / 'summary.json').exists():
        raise RuntimeError('Fresh output directory required')
    versions = {name: metadata.version(name) for name in PINS}
    if versions != PINS or platform.python_version() != '3.12.13':
        raise RuntimeError(f'Runtime differs: {versions}')
    context=source_gate()
    save("source-runtime.json", dict(**context, run_id=os.environ["GITHUB_RUN_ID"],
        python=platform.python_version(), versions=versions))
    # Exact undamaged initial geometry from the producer's quadrature and notch.
    axes = [np.arange(.00625, extent, .0125) for extent in (.6, 1., .1)]
    x = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    x = x[~((x[:, 0] < .25) & (np.abs(x[:, 1] - .5) < .08))]
    assert len(x) == 28800
    np.savez_compressed(OUT / 'frame_000000.npz', position=x, damage=np.zeros(len(x)), time=0.)
    command = [sys.executable, '-m', 'examples.phase_field_visible_crack', '--dx', '.025',
        '--length', '.05', '--gc', '.005', '--damage-interval', '1', '--steps', '2000',
        '--snapshot-every', '25', '--output', str(OUT)]
    started = time.monotonic()
    record = dict(command=command, completed=False, peak_tree_uss_bytes=0,
        uss_limit_bytes=int(5.5 * 1024**3), seconds_limit=16200,
        minimum_available_bytes=2 * 1024**3, guard_reason=None)
    with (OUT / 'solver.log').open('wb') as log:
        import ctypes
        if sys.platform != 'linux' or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError('Linux child subreaper setup required')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                uss = owned_session_uss(process.pid, psutil)
                record['peak_tree_uss_bytes'] = max(record['peak_tree_uss_bytes'], uss)
                elapsed = time.monotonic() - started
                reason = ('USS limit' if uss > record['uss_limit_bytes'] else
                    'available memory limit' if psutil.virtual_memory().available < record['minimum_available_bytes'] else
                    'wall time limit' if elapsed > record['seconds_limit'] else None)
                if reason:
                    record['guard_reason'] = reason
                    break
                save('resource-guard.json', record)
                time.sleep(1)
            code = process.poll()
        except BaseException as error:
            record['guard_reason'] = 'Observer/controller failure: '+repr(error)
            raise
        finally:
            try:
                terminate_owned_group(process)
            finally:
                record.update(returncode=process.returncode, elapsed_seconds=time.monotonic()-started)
                save('resource-guard.json', record)
    if code or record['guard_reason']:
        raise RuntimeError('Solve failed or guard stopped it; partial raw snapshots retained')
    summary = json.loads((OUT / 'summary.json').read_text())
    with np.load(OUT / 'particles.npz') as final:
        if not np.array_equal(final['reference'], x):
            raise RuntimeError('Initial snapshot differs from exact producer reference geometry')
    expected = [f'frame_{i:06d}.npz' for i in range(0, 2001, 25)]
    if sorted(p.name for p in OUT.glob('frame_*.npz')) != expected:
        raise RuntimeError('Full 81 snapshot series required')
    previous = np.zeros(28800)
    previous_time = -1.
    for name in expected:
        with np.load(OUT / name) as frame:
            position, damage, now = frame['position'], frame['damage'], float(frame['time'])
            if position.shape != (28800, 3) or damage.shape != (28800,) or not np.isfinite(position).all() or not np.isfinite(damage).all():
                raise RuntimeError('Invalid particle arrays')
            if damage.min() < -1e-12 or damage.max() > 1+1e-12 or np.min(damage-previous) < -1e-10 or now <= previous_time:
                raise RuntimeError('Invalid bounds, irreversibility or time')
            previous, previous_time = damage.copy(), now
    if summary['steps'] != 2000 or summary['particles'] != 28800 or summary['damage_interval'] != 1 or summary['snapshot_every'] != 25 or not np.isfinite(summary['maximum_kkt']) or summary['maximum_kkt'] > 1e-7:
        raise RuntimeError('Completion metadata differs')
    if not np.isclose(previous_time, summary['time'], rtol=0, atol=1e-14):
        raise RuntimeError('Last snapshot time differs from completion summary')
    record['completed'] = True
    save('resource-guard.json', record)
    subprocess.run([sys.executable, str(ROOT/"tools/audit_phase_field_visible_crack.py"), str(OUT)], check=True)
    save('complete-checks.json', dict(completed=True, frames=81, finite=True,
        damage_bounds=True, irreversible=True, summary=summary,
        interpretation='Fresh 2000-step experiment; crack admission requires separate crack-checks.json. No restart or prescribed damage.'))


if __name__ == '__main__':
    main()
