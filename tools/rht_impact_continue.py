"""Fail-closed continuation of the immutable run-40071417301 frame-3 evidence.

No execution happens without explicit ``run``. ``admit`` validates transport,
source and runtime only. Numerical modules are imported solely from archived source.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from tools import rht_impact_prefix as transport

COMMIT = '0c99377fac0ff77b376958dff2e124e2a4691f21'
TRANSPORT = '9508e80e500dcd106a728db25161ffb770ed11e6f1f6c832a2e62a44d6d5038f'
CASE = 'a1fe8fdc3b0726097f7e8008fd55ad174dbe28d2ce58e566833c0168856fb8da'
GIB = 1024 ** 3
GUARD_SECONDS = 10800
END_FRAME = 4
MIN_LOGICAL_CPUS = 8
MIN_AVAILABLE_MEMORY = 12 * GIB
OUTPUT_BYTES = 3 * GIB
ENVIRONMENT = {'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'true',
               'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2',
               'PYTHONUNBUFFERED': '1'}


def resource_admission(cpus, available):
    if (cpus or 0) < MIN_LOGICAL_CPUS or available < MIN_AVAILABLE_MEMORY:
        raise RuntimeError('One-frame qualification requires >=8 logical CPUs and >=12 GiB available RAM')
    return {'logical_cpus': cpus, 'available_memory_bytes': available}


def prefix_engineering_audit(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = ('projectile_tip_y_m', 'projectile_center_vy_m_per_s', 'tip_past_rear_m',
            'target_failed_reference_volume_fraction', 'target_failed_count',
            'target_maximum_damage', 'corridor_intact_count', 'target_minimum_J')
    values = [{key: float(row[key]) for key in keys} for row in rows]
    if not values or any(not math.isfinite(v) for row in values for v in row.values()):
        raise RuntimeError('Nonfinite or missing engineering history')
    if any(row['target_minimum_J']<=0 or not 0<=row['target_maximum_damage']<=1
           or not 0<=row['target_failed_reference_volume_fraction']<=1 for row in values):
        raise RuntimeError('Physical engineering history rejected')
    return {'saved_rows':len(rows), 'last':values[-1],
            'tip_has_crossed_original_rear_plane':values[-1]['tip_past_rear_m']>0,
            'failed_points_present':values[-1]['target_failed_count']>0,
            'perforation_validated':False,'fragments_validated':False,
            'limitation':'Tip crossing and failed points alone do not establish a through hole or separated fragments.'}


def controller_identity():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if (commit != os.environ.get('GITHUB_SHA')
            or os.environ.get('GITHUB_REF') != 'refs/heads/work/rht-pointed-impact-prefix'
            or os.environ.get('GITHUB_WORKFLOW_REF', '').split('@')[0] !=
               'lililii124/jax-mpm/.github/workflows/engineering-previews.yml'):
        raise RuntimeError('Exact registered feature controller identity required')
    names = ['tools/rht_impact_continue.py', 'tools/rht_impact_prefix.py',
             '.github/workflows/rht-impact-continue.yml', '.github/workflows/engineering-previews.yml']
    files = {}
    for name in names:
        raw = (root/name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', commit+':'+name], cwd=root):
            raise RuntimeError('Controller differs from its exact Git blob: '+name)
        files[name] = {'sha256': transport.digest(root/name), 'bytes': len(raw)}
    return {'commit': commit, 'files': files}


def requirements(parent, output):
    parent = Path(parent).resolve()
    if transport.digest(parent/'transport.json') != TRANSPORT:
        raise RuntimeError('Wrong original transport')
    transport.verify(parent)
    index = json.loads((parent/'transport.json').read_text())
    text = None
    with tarfile.open(fileobj=transport.PartReader(parent/n for n in index['parts']), mode='r|') as archive:
        for member in archive:
            if member.name == 'runtime-freeze.txt':
                text = archive.extractfile(member).read().decode()
                break
    if text is None:
        raise RuntimeError('Original complete runtime freeze absent')
    pins = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-e ') or line.startswith('jax-mpm @ '):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.!+_-]+', line):
            raise RuntimeError('Unpinned original runtime line')
        pins.append(line)
    if not {'jax', 'jaxlib', 'numpy', 'pillow', 'psutil'}.issubset(p.split('==')[0].lower() for p in pins):
        raise RuntimeError('Incomplete original runtime pins')
    Path(output).write_text('\n'.join(pins)+'\n')


def admit(parent):
    parent = Path(parent).resolve()
    manifest = parent / 'transport.json'
    if transport.digest(manifest) != TRANSPORT:
        raise RuntimeError('Parent transport is not the immutable admitted run')
    receipt = json.loads((parent / 'retrieval-receipt.json').read_text())
    if (receipt.get('status') != 'all-parts-and-whole-raw-files-verified'
            or receipt.get('transport_sha256') != TRANSPORT):
        raise RuntimeError('Missing matching complete parent retrieval receipt')
    # Reverify all bytes now; a historical receipt alone cannot admit changed files.
    transport.verify(parent)
    index = json.loads(manifest.read_text())
    wanted = {'receipt.json', 'source-manifest.json', 'git-blobs.json',
              'raw/checkpoint.json', 'raw/status.json'}
    metadata = {}
    reader = transport.PartReader(parent / n for n in index['parts'])
    with tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            if member.name in wanted:
                metadata[member.name] = json.load(archive.extractfile(member))
    if set(metadata) != wanted:
        raise RuntimeError('Parent metadata incomplete')
    original = metadata['receipt.json']
    if (original.get('commit') != COMMIT or original.get('status') != 'failed'
            or original.get('error') != 'RuntimeError: 50 minute solver guard exceeded'):
        raise RuntimeError('Parent is not the expected guard-only failure')
    if metadata['git-blobs.json'].get('commit') != COMMIT:
        raise RuntimeError('Original source commit differs')
    sources = metadata['source-manifest.json']
    if sources.get('examples/rht_pointed_impact.py') != CASE:
        raise RuntimeError('Original physical case differs')
    for name, sha in sources.items():
        if index['files'].get('source/' + name, {}).get('sha256') != sha:
            raise RuntimeError('Source manifest differs from verified full archive')
    if original.get('source_manifest_sha256') != index['files']['source-manifest.json']['sha256']:
        raise RuntimeError('Original receipt source identity differs')
    checkpoint, status = metadata['raw/checkpoint.json'], metadata['raw/status.json']
    if checkpoint.get('frame') != 3 or status.get('frame') != 3 or status.get('steps_completed') != 105:
        raise RuntimeError('Parent is not the expected frame-3/105-step checkpoint')
    for state in checkpoint.get('states', []):
        if 'raw/' + state not in index['files']:
            raise RuntimeError('Checkpoint state absent from full archive')
    if len(checkpoint.get('states', [])) != 2:
        raise RuntimeError('Expected exactly two saved bodies')
    if original.get('platform') != sys.platform or original.get('python') != sys.version:
        raise RuntimeError('Exact original Python/platform runtime required')
    for name, version in original['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Exact original numerical package runtime required: ' + name)
    return index, metadata


VALIDATE = '''import json
from pathlib import Path
import jax
import numpy as np
from examples.common import _load_state, _valid_checkpoint_states, _validate_resume_prefix
from examples.rht_pointed_impact import make_case
from dataclasses import asdict
from mpm_jax.particle_fields import material_metadata
import hashlib
jax.config.update('jax_enable_x64', True)
p=Path(__import__('sys').argv[1]); c=json.loads((p/'checkpoint.json').read_text())
plan=make_case(); assert plan.description['frames']==48
assert plan.description['substeps_per_frame']==35 and plan.duration==8.e-5
states=tuple(_load_state(p/n) for n in c['states'])
assert _valid_checkpoint_states(states)
assert len(states)==2
for old,new in zip(states,plan.states):
 assert old.x.shape==new.x.shape
 assert np.array_equal(old.mass,new.mass) and np.array_equal(old.volume,new.volume)
_validate_resume_prefix(p,c,states,48,35,plan.configs[0].dt,True)
signature={'configs':[asdict(v) for v in plan.configs],
 'materials':[material_metadata(v) for v in plan.materials],
 'frames':48,'substeps':35,'label':'RHT pointed impact','definition':plan.description}
assert c['signature']==hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
assert c['frame']==int(__import__('sys').argv[2])
i=np.asarray(c['contact_impulse']); assert np.isfinite(i).all()
print(json.dumps({'frame':c['frame'],'contact_impulse':i.tolist(),'solver_seconds':c['solver_seconds']}))
'''


def run(parent, destination):
    index, metadata = admit(parent)
    binding = controller_identity()
    import psutil
    resources = resource_admission(psutil.cpu_count(logical=True), psutil.virtual_memory().available)
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError('Continuation destination must be absent')
    if shutil.disk_usage(destination.parent).free < 8 * GIB:
        raise RuntimeError('Need 8 GiB free to retain parent, continuation and transport')
    destination.mkdir()
    record = {'controller_source': binding, 'controller_sha256': transport.digest(Path(__file__)),
              'transport_code_sha256': transport.digest(Path(transport.__file__)),
              'status': 'preparing', 'parent_transport_sha256': TRANSPORT,
              'original_commit': COMMIT, 'guard_seconds': GUARD_SECONDS,
              'resources': resources, 'start_frame': 3, 'end_frame': END_FRAME,
              'incremental_steps': (END_FRAME-3)*35,
              'scope': 'Private exact-plan frame-3 to frame-4 resource qualification; no perforation or fragment validation claim'}
    process = None
    start = time.monotonic()
    peak = 0
    try:
        shutil.copy2(Path(__file__), destination / 'controller-source.py')
        shutil.copy2(Path(transport.__file__), destination / 'transport-source.py')
        evidence = destination / 'parent-evidence'
        evidence.mkdir()
        saved_transport = destination / 'parent-transport'
        saved_transport.mkdir()
        for name in ['transport.json', 'retrieval-receipt.json', *index['parts']]:
            shutil.copy2(Path(parent) / name, saved_transport / name)
        reader = transport.PartReader(saved_transport / n for n in index['parts'])
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                target = (evidence / member.name).resolve()
                if not target.is_relative_to(evidence.resolve()) or not member.isfile():
                    raise RuntimeError('Unsafe parent archive member')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    shutil.copyfileobj(archive.extractfile(member), output)
                if transport.digest(target) != index['files'][member.name]['sha256']:
                    raise RuntimeError('Extracted parent changed')
        source = evidence / 'source'
        raw = destination / 'raw'
        shutil.copytree(evidence / 'raw', raw)
        environment = dict(os.environ, **ENVIRONMENT)
        # Avoid ambient module injection; cwd supplies only frozen numerical source.
        environment.pop('PYTHONPATH', None)
        def validate(frame):
            return subprocess.check_output([sys.executable, '-I', '-c',
                "import sys;sys.path.insert(0,sys.argv.pop(1));" + VALIDATE,
                str(source), str(raw), str(frame)], cwd=source,
                env=environment, text=True, timeout=600)
        record['parent_validation'] = json.loads(validate(3))
        command = [sys.executable, '-I', '-c',
                   "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('examples.rht_pointed_impact',run_name='__main__')",
                   str(source), '--resume', '--stop-after-frame', str(END_FRAME), '--output', str(raw)]
        record['command'] = command
        record['status'] = 'running'
        with (destination / 'solver.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    tree = [psutil.Process(process.pid)]
                    tree += tree[0].children(recursive=True)
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    raise
                uss = 0
                for child in tree:
                    try:
                        uss += child.memory_full_info().uss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, uss)
                size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
                if uss > 6 * GIB or size > OUTPUT_BYTES or shutil.disk_usage(destination).free < 2 * GIB:
                    raise RuntimeError('Memory/output/free-disk guard exceeded')
                if time.monotonic() - start > GUARD_SECONDS:
                    raise RuntimeError('3 hour continuation guard exceeded')
                time.sleep(2)
        if process.returncode:
            raise RuntimeError('Continuation solver failed; all raw retained')
        record['endpoint_validation'] = json.loads(validate(END_FRAME))
        if record['endpoint_validation']['contact_impulse'][1] <= 0:
            raise RuntimeError('No positive projectile contact impulse at planned endpoint')
        segment = json.loads((raw / 'segment.json').read_text())
        if segment.get('frame') != END_FRAME or segment.get('segment_complete') is not True:
            raise RuntimeError('Missing valid planned endpoint')
        record['engineering_audit'] = prefix_engineering_audit(raw/'observables.csv')
        for name, expected in index['files'].items():
            if transport.digest(evidence / name) != expected['sha256']:
                raise RuntimeError('Immutable original evidence changed')
        record['status'] = 'valid-impact-prefix'
        record['engineering_claims'] = {'full_48_frame_plan_completed': False,
            'perforation_validated': False, 'fragments_validated': False,
            'required_before_publication': [
                'Complete exact 48-frame history with valid positive-J states and conserved body mass accounting',
                'Projectile rear-face crossing with residual forward velocity and through-thickness open-path audit',
                'Failed-particle ballistic escape and surviving ligament audit; failure count alone is not fragmentation',
                'Whole-history contact impulse, energy, bounded penetration and numerical-resolution audit']}
        if controller_identity() != binding:
            raise RuntimeError('Controller source changed during continuation')
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            finally:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        record.update(elapsed_seconds=time.monotonic()-start, peak_aggregate_uss_bytes=peak)
        transport.write(destination / 'receipt.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['admit', 'requirements', 'run', 'package', 'verify'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--delivery', type=Path)
    args = parser.parse_args()
    if args.action in ('admit', 'requirements', 'run') and args.parent is None:
        parser.error('--parent required')
    if args.action == 'requirements':
        if args.destination is None:
            parser.error('--destination output requirements path required')
        requirements(args.parent, args.destination)
    elif args.action == 'admit':
        admit(args.parent)
        print('Original full raw/source/runtime admitted; no solver launched')
    elif args.action == 'run':
        if args.destination is None:
            parser.error('--destination required')
        run(args.parent, args.destination)
    else:
        if args.delivery is None:
            parser.error('--delivery required')
        if args.action == 'verify':
            transport.verify(args.delivery)
        else:
            if args.destination is None:
                parser.error('--destination required')
            transport.DEST, transport.DELIVERY = args.destination.resolve(), args.delivery.resolve()
            transport.package()


if __name__ == '__main__':
    main()
