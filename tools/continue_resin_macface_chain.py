"""Persistent remote host controller for bounded conservative-inlet segments.

The controller runs on the compute host, outside individual exec sessions. Each
container is a fresh <=512-step segment from the last admitted checkpoint.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def npy_scalar(npz_path, name, fmt):
    with zipfile.ZipFile(npz_path) as archive:
        return struct.unpack(fmt, archive.read(name + '.npy')[-struct.calcsize(fmt):])[0]


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def shell(args, *, timeout=45, capture=True):
    result = subprocess.run(args, check=True, timeout=timeout,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None,
                            text=True)
    return result.stdout.strip() if capture else ''


def inspect_container(name):
    return json.loads(shell(['docker', 'inspect', '--format', '{{json .State}}', name]))


def verify_source(root, manifest):
    assert manifest['source_commit'] == 'e70265e03581e02692f21874283762fabbfdff4e'
    for rel, expected in manifest['files'].items():
        assert sha(root / rel) == expected, f'frozen source digest mismatch: {rel}'


def run_container(root, name, work, source, parent, forming, *, mode, target_step=None):
    container = 'jaxmpm-' + name
    assert len(container) < 120
    if mode == 'segment':
        (work / 'output').mkdir()
        parent_mount = str(parent / 'output')
        command = ['python', '/work/start.py']
    else:
        parent_mount = str(parent)
        command = ['python', '/source/tools/review_resin_volume.py', '/flow', '/forming',
                   '--output', '/work/midcourse-review.json']
    spec = [
        'docker', 'run', '-d', '--name', container, '--network', 'none', '--user', '1006:1007',
        '--cpuset-cpus', '48-51', '--cpus', '4', '--cpu-shares', '128',
        '--memory', '8g', '--memory-swap', '8g', '--pids-limit', '512', '--shm-size', '1g',
        '--mount', f'type=bind,src={source},dst=/source,readonly',
        '--mount', f'type=bind,src={work},dst=/work',
        '--mount', f'type=bind,src={forming},dst=/forming,readonly',
        '--env', 'HOME=/work', '--env', 'JAX_ENABLE_X64=true', '--env', 'PYTHONUNBUFFERED=1',
        '--env', 'PYTHONPATH=/source', '--env', 'PYTHONDONTWRITEBYTECODE=1',
    ]
    if mode == 'segment':
        spec += ['--mount', f'type=bind,src={parent_mount},dst=/parent,readonly']
    else:
        spec += ['--mount', f'type=bind,src={parent_mount},dst=/flow,readonly']
    spec += ['--workdir', '/work', source_image(root), 'python',
             '/source/remote_job_guard.py', '--seconds', '900', '--uss-gib', '5', '--', *command]
    cid = shell(spec, timeout=60)
    return container, cid


def source_image(root):
    return json.loads((root / 'chain-config.json').read_text())['image']


def admit_segment(task_root, expected_start, expected_end, forming_sha):
    output = task_root / 'output'
    guard = json.loads((task_root / 'job/status.json').read_text())
    assert guard['state'] == 'complete' and guard['returncode'] == 0
    assert guard['peak_uss_bytes'] <= 5 * 1024**3
    review = json.loads((output / 'review.json').read_text())
    rows = json.loads((output / 'observables.json').read_text())
    assert review['accepted_steps'] == [expected_start, expected_end]
    assert len(rows) == expected_end - expected_start
    assert rows[0]['step'] == expected_start + 1 and rows[-1]['step'] == expected_end
    assert finite_tree(rows)
    for row in rows:
        assert row['valid']
        assert row['pressure_coverage_valid'] and row['physical_pressure_domain_valid']
        assert row['transport_extension_valid']
        assert row.get('particle_wall_valid', True)
        assert row['pressure_relative_residual'] <= 1e-7
        assert row['pressure_stationarity_relative'] <= 1e-10
        assert max(abs(row['transport_logJ_min']), abs(row['transport_logJ_max'])) <= 1e-7
        assert row['inlet_face_remainder_valid']
        assert abs(row['inlet_volume_balance_error_m3']) <= 1e-20
        assert row['newborn_quantization_delay_max_s'] <= 4e-5 + 1e-15
        assert row['solid_cell_intrusions'] == 0 and row['escaped_particles'] == 0
        assert abs(row['mass_identity_kg']) <= 1e-12 * max(row['source_mass_kg'], 1e-30)
        energy_scale = max(row['source_energy_J'] + row['fixture_work_cumulative_J'],
                           row['active_kinetic_J'], 1e-30)
        assert abs(row['global_energy_identity_J']) <= 1e-8 * energy_scale
        assert abs(row['source_clock_error_s']) <= 1e-10
        assert math.isfinite(row['momentum_step_residual_Ns'])
    cp = output / 'checkpoint.npz'
    assert npy_scalar(cp, 'step', '<q') == expected_end
    state_time = npy_scalar(cp, 'time', '<d')
    nominal = expected_end * 4e-5
    assert abs(state_time - rows[-1]['actual_state_time_s']) < 1e-12
    assert abs(state_time - nominal) <= 8 * max(expected_end, 1) * abs(math.ulp(nominal))
    assert abs(npy_scalar(cp, 'inlet_source_time_s', '<d') - nominal) < 1e-15
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['forming']['checkpoint_sha256'] == forming_sha
    return {'task': task_root.name, 'step': expected_end, 'time_s': state_time,
            'checkpoint_sha256': sha(cp), 'row_count': len(rows),
            'peak_uss_bytes': guard['peak_uss_bytes'], 'last_row': rows[-1]}


def run_midpoint_review(root, source, forming, parent, step, state, *, gate):
    name = f'resin-macface-review-{step}'
    work = root / 'reviews' / str(step)
    work.mkdir(parents=True, exist_ok=False)
    container, cid = run_container(root, name, work, source, parent, forming, mode='review')
    atomic_json(root / 'chain-state.json', dict(state='midpoint_review_running',
        active_container=container, container_id=cid, active_task=str(work), last_accepted=state))
    while True:
        current = inspect_container(container)
        if not current['Running']:
            break
        time.sleep(5)
    (work / 'docker.log').write_text(shell(['docker', 'logs', container], timeout=60))
    assert current['ExitCode'] == 0, f'midpoint reviewer failed: {container} {current}'
    guard = json.loads((work / 'job/status.json').read_text())
    assert guard['state'] == 'complete' and guard['returncode'] == 0
    report = json.loads((work / 'midcourse-review.json').read_text())
    assert report['step'] == step and report['checkpoint_sha256'] == state['checkpoint_sha256']
    kernel_mean = report['bulk_density_over_rho']['mean']
    reference_mean = 1.0014552977848605
    ref_volume = report['physical_particle_volume_m3']
    true_volume = report['integrated_incremental_transport_volume_m3']
    assert ref_volume > 0 and true_volume > 0
    volume_relative_error = (true_volume - ref_volume) / ref_volume
    if gate:
        assert 0.95 <= kernel_mean <= 1.05
        assert abs(kernel_mean - reference_mean) <= 0.05
        assert abs(volume_relative_error) <= 1e-7
    return {'step': step, 'checkpoint_sha256': state['checkpoint_sha256'],
            'review': report, 'kernel_reference_mean_over_rho': reference_mean,
            'kernel_mean_delta_from_reference': kernel_mean - reference_mean,
            'true_volume_relative_error': volume_relative_error,
            'gate_applied': gate, 'gate_passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)
    root = args.root.resolve()
    lock = root / 'controller.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.fsync(descriptor)
    os.close(descriptor)
    config = json.loads((root / 'chain-config.json').read_text())
    manifest = json.loads((root / 'source-manifest.json').read_text())
    verify_source(root, manifest)
    base = Path(config['remote_tasks_root'])
    forming = base / config['forming_task'] / 'output'
    forming_sha = config['forming_checkpoint_sha256']
    assert sha(forming / 'checkpoint.npz') == forming_sha
    source = root
    parent = config['initial_parent_task']
    parent_root = base / parent
    initial_cp = parent_root / 'output/checkpoint.npz'
    assert sha(initial_cp) == config['initial_checkpoint_sha256']
    assert npy_scalar(initial_cp, 'step', '<q') == config['start_step']
    current = {'task': parent, 'step': config['start_step'], 'checkpoint_sha256': sha(initial_cp),
               'time_s': npy_scalar(initial_cp, 'time', '<d')}
    atomic_json(root / 'chain-state.json', dict(state='running', source_commit=manifest['source_commit'],
        source_tree_sha256=manifest['source_tree_sha256'], last_accepted=current,
        target_step=config['target_step'], segment_max_steps=512,
        resources=config['resources'], controller_pid=os.getpid()))
    while current['step'] < config['target_step']:
        start = current['step']
        end = min(start + 512, config['target_step'])
        task_name = f"{config['task_prefix']}-{end}"
        task_root = base / task_name
        if task_root.exists():
            raise RuntimeError(f'refusing to reuse existing segment directory {task_root}')
        task_root.mkdir(parents=True)
        work = task_root
        (work / 'output').mkdir()
        wrapper = ("import hashlib,runpy,sys\nfrom pathlib import Path\n"
                   f"assert hashlib.sha256(Path('/parent/checkpoint.npz').read_bytes()).hexdigest()=={current['checkpoint_sha256']!r}\n"
                   "sys.argv=['examples/three_strand_resin_rt0.py','--checkpoint','/forming/checkpoint.npz',"
                   "'--forming-plan','/forming/plan.json','--output','/work/output','--steps',"
                   f"{str(end-start)!r},'--resume-from','/parent']\n"
                   "runpy.run_module('examples.three_strand_resin_rt0',run_name='__main__')\n")
        (work / 'start.py').write_text(wrapper)
        segment_record = dict(source_commit=manifest['source_commit'],
            source_tree_sha256=manifest['source_tree_sha256'], parent_task=current['task'],
            parent_checkpoint_sha256=current['checkpoint_sha256'], start_step=start, target_step=end,
            forming_checkpoint_sha256=forming_sha, physical_steps=end-start)
        atomic_json(work / 'segment-source.json', segment_record)
        container, cid = run_container(root, task_name, work, source, base / current['task'],
                                       forming, mode='segment', target_step=end)
        atomic_json(root / 'chain-state.json', dict(state='segment_running',
            source_commit=manifest['source_commit'], source_tree_sha256=manifest['source_tree_sha256'],
            controller_pid=os.getpid(), active_task=task_name, active_container=container,
            container_id=cid, segment_start_step=start, segment_target_step=end,
            last_accepted=current, resources=config['resources'], target_step=config['target_step']))
        while True:
            status = inspect_container(container)
            if not status['Running']:
                break
            time.sleep(5)
        (work / 'container.log').write_text(shell(['docker', 'logs', container], timeout=90))
        if status['ExitCode'] != 0:
            raise RuntimeError(f'segment container failed: {container} {status}')
        accepted = admit_segment(work, start, end, forming_sha)
        current = accepted
        atomic_json(root / 'chain-state.json', dict(state='segment_accepted',
            source_commit=manifest['source_commit'], source_tree_sha256=manifest['source_tree_sha256'],
            controller_pid=os.getpid(), last_accepted=current, target_step=config['target_step'],
            resources=config['resources']))
        if end == config['midpoint_step']:
            mid = run_midpoint_review(root, source, forming, base / current['task'], end, current, gate=True)
            atomic_json(root / 'midpoint-review.json', mid)
            atomic_json(root / 'chain-state.json', dict(state='midpoint_passed',
                source_commit=manifest['source_commit'], source_tree_sha256=manifest['source_tree_sha256'],
                controller_pid=os.getpid(), last_accepted=current, midpoint=mid,
                target_step=config['target_step'], resources=config['resources']))
    final = run_midpoint_review(root, source, forming, base / current['task'], config['target_step'], current, gate=False)
    final['fill_qualification'] = 'pending terminal volume and occupancy review; numerical step 8000 is not a fill claim'
    atomic_json(root / 'final-review.json', final)
    atomic_json(root / 'chain-state.json', dict(state='numerical_target_complete_review_pending',
        source_commit=manifest['source_commit'], source_tree_sha256=manifest['source_tree_sha256'],
        controller_pid=os.getpid(), last_accepted=current, final_review=final,
        target_step=config['target_step'], resources=config['resources']))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        root = Path(os.environ.get('RESIN_CHAIN_ROOT', '.')).resolve()
        state_path = root / 'chain-state.json'
        old = json.loads(state_path.read_text()) if state_path.exists() else {}
        old.update(state='stopped_failure', error=repr(error), controller_pid=os.getpid())
        atomic_json(state_path, old)
        raise
