"""Isolated 202240-particle compliant JKR demonstration pilot; never publish.

Hertz/JKR share this pilot's physical request. This fixture differs from the
existing public displacement-controlled examples and must be labeled as such.
MD can receive a resource prefix, but is not admitted as visibly distinct JKR.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def prepare(root):
    from examples.fibre_adhesion_contrast import plan_case
    from tools.scan_fibre_pair_fixture import state
    if root.exists():
        raise ValueError('Fresh isolated destination required')
    snapshot = root/'source'
    files = []
    for folder in ('mpm_jax', 'examples', 'tools'):
        files.extend(p for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend((ROOT/'.github/workflows/jkr-compliant-pilot.yml', ROOT/'pyproject.toml'))
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert head == os.environ['GITHUB_SHA'], 'actual controller SHA differs'
        active = os.environ['GITHUB_WORKFLOW_REF'].split('/.github/workflows/', 1)[1].split('@', 1)[0]
        active_path = ROOT/'.github/workflows'/active
        assert active in ('engineering-previews.yml', 'jkr-compliant-pilot.yml')
        assert active_path.read_bytes() == (ROOT/'.github/workflows/jkr-compliant-pilot.yml').read_bytes()
        files.append(active_path)
    for path in files:
        target = snapshot/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files}
    for name, expected in manifest.items():
        assert sha(snapshot/name) == expected
    plans = {}
    for model in ('hertz', 'jkr', 'md'):
        geometry, plan = plan_case(model=model, young=4e5, work=.08, tabor=5.,
            fixture_stiffness=.7056029424546741, fixture_damping=.015, duration=.05, frames=8)
        assert plan['particles'] == 202240
        # All models use identical new timing: longer true hold, same camera.
        protocol = dict(plan['protocol'], times_s=[0., .006, .010, .020, .045, .050])
        plan['protocol'] = protocol
        plan['physical_common']['protocol'] = protocol
        plan['camera'] = dict(plan['camera'], direction=[1.4, .2, 1.8], projection='parallel',
            framing_policy='Same fixed focus bounds and0.56 projected-bounds scale as original low-camera fibre renderer')
        plan['physical_common']['camera'] = plan['camera']
        plan['common_request_sha256'] = hashlib.sha256(json.dumps(
            plan['physical_common'], sort_keys=True).encode()).hexdigest()
        plan['frame_steps'] = [0, 64]+[round(t/plan['dt']) for t in (.006, .010, .020, .030, .0375, .045, .05)]
        plan['comparison_group'] = 'clearly-labeled-compliant-pull-off-demonstration'
        plan['scope'] = ('Private new compliant-fixture demonstration; differs from original public direct-displacement '
                         'Hertz/MD fixtures. Hertz/JKR pilot share all physical inputs; no quantitative identity with old examples.')
        write(root/'plans'/f'{model}.json', plan)
        plans[model] = plan
    assert len({p['common_request_sha256'] for p in plans.values()}) == 1
    upper = geometry['fibre_id'] == 1
    mass = float(geometry['mass'][upper].sum())
    fixture = plans['jkr']['fixture']
    stiffness = fixture['spring_stiffness_N_per_m']
    effective = fixture['series_stiffness_estimate_N_per_m']
    damping = fixture['parallel_damper_Ns_per_m']
    from tools.review_jkr_compliant_pilot import projected_pixels
    qstate = state('jkr', 4e5, .08, effective, -.00014, True)
    screen = dict(scope='Analytical resource/dynamics screen only; actual hold and image separation unmeasured',
        source_files=manifest, frozen_source_root=str(snapshot),
        common_request_sha256=plans['jkr']['common_request_sha256'],
        upper_fibre_mass_kg=mass,
        rigid_upper_critical_damping_Ns_per_m=2*math.sqrt(stiffness*mass),
        series_stiffness_mass_critical_damping_Ns_per_m=2*math.sqrt(effective*mass),
        rigid_upper_damping_ratio=damping/(2*math.sqrt(stiffness*mass)),
        series_stiffness_mass_damping_ratio=damping/(2*math.sqrt(effective*mass)),
        warning='These are rigid/series modal estimates. Coupled beam/contact eigenmodes require measured ring-down.',
        contact=plans['jkr']['contact'], dt=plans['jkr']['dt'],
        cfl_wave_ratio=plans['jkr']['dt']*math.sqrt(4e5*.75/(1.25*.5*1000))/.0002,
        static_hold_difference_m=plans['jkr']['pair_reduced_screen']['actual_centre_separation_difference_m'],
        static_hold_difference_projected_pixels=projected_pixels(
            plans['jkr']['pair_reduced_screen']['actual_centre_separation_difference_m'], plans['jkr']['camera']),
        static_compressed_radius_over_R=qstate['radius_m']/.001,
        MD_applicability='Same mu=5 finite-range MD is physically screenable, but MD-vs-JKR distinction is subpixel; no visible three-model quantitative claim.',
        prefix_steps=64, hold_endpoint_s=.045, full_61_authorized=False,
        private_only=True, production_eligible=False, public_visual_review_passed=False)
    write(root/'preflight.json', screen)
    print(json.dumps({k:v for k,v in screen.items() if k != 'source_files'}, indent=2))


def worker(root, model, stage):
    from examples.fibre_adhesion_contrast import plan_case, run_case
    plan = json.loads((root/'plans'/f'{model}.json').read_text())
    if stage == 'hold' and model == 'md':
        raise ValueError('MD hold needs separate applicability/history review; not a visible JKR discriminator')
    geometry, _ = plan_case(model=model, young=4e5, work=.08, tabor=5.,
        fixture_stiffness=.7056029424546741, fixture_damping=.015, duration=.05, frames=8)
    run_case(geometry, plan, root/'data'/model, resume=stage == 'hold',
             stop_after_frame=1 if stage == 'prefix' else 7)


def guarded_run(root, model, stage, timeout):
    import psutil
    screen = json.loads((root/'preflight.json').read_text())
    for name, expected in screen['source_files'].items():
        assert sha(root/'source'/name) == expected, 'frozen source changed: '+name
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    log = logs/f'{model}-{stage}.log'
    env = dict(os.environ, JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    command = [sys.executable, str(root/'source/tools/run_jkr_compliant_pilot.py'),
        'worker', '--root', str(root), '--model', model, '--stage', stage]
    # Windows currently shares the machine with ParaView: USS alone does not
    # capture pagefile/commit exhaustion. Never start near commit exhaustion.
    def commit_headroom():
        if os.name != 'nt':
            return None
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)]+[
                (name, ctypes.c_ulonglong) for name in ('total_physical', 'available_physical',
                'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'available_extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError('Windows commit headroom unavailable')
        return status.available_pagefile
    if os.name == 'nt' and commit_headroom() < 4*1024**3:
        raise RuntimeError('Windows commit headroom below4GiB: numerical pilot must use isolated Linux runner')
    limit = int((1.5 if os.name == 'nt' else 4)*1024**3)
    start = time.monotonic()
    peak, samples, reason, known = 0, 0, None, {}
    with log.open('w', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=root/'source', env=env, stdout=stream, stderr=subprocess.STDOUT)
        parent = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                members = [parent]+parent.children(recursive=True)
                for member in members:
                    known[member.pid] = member
                try:
                    uss = sum(p.memory_full_info().uss for p in members)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    reason = 'incomplete_process_tree_measurement'
                    break
                samples += 1
                peak = max(peak, uss)
                if uss >= limit:
                    reason = 'process_tree_USS_limit'
                if os.name == 'nt' and commit_headroom() < 1.5*1024**3:
                    reason = 'Windows_commit_headroom_below1point5GiB'
                if psutil.virtual_memory().available < 2*1024**3:
                    reason = 'system_reserve_2GiB'
                if time.monotonic()-start > timeout:
                    reason = 'wall_timeout'
                if reason:
                    break
                time.sleep(.2)
        finally:
            if child.poll() is None:
                for process in reversed(list(known.values())):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.kill()
            code = child.wait(timeout=15)
    report = dict(model=model, stage=stage, seconds=time.monotonic()-start,
        peak_tree_uss_bytes=peak, complete_samples=samples, reason=reason,
        uss_limit_bytes=limit,
        returncode=code, passed=reason is None and code == 0,
        production_eligible=False, public_visual_review_passed=False)
    write(log.with_suffix('.monitor.json'), report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)
    from tools.review_jkr_compliant_pilot import review
    review(root/'data'/model, root/'logs'/f'{model}-{stage}.independent.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=('hertz', 'jkr', 'md'), default='jkr')
    parser.add_argument('--stage', choices=('prefix', 'hold'), default='prefix')
    parser.add_argument('--timeout', type=float, default=120.)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root)
    elif args.phase == 'worker':
        worker(root, args.model, args.stage)
    else:
        guarded_run(root, args.model, args.stage, args.timeout)


if __name__ == '__main__':
    main()
