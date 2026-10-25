"""Read-only failed-state contact registry audit; never advance or resume.

Rebuild the exact NumPy reference/coverage registry and inspect only unseeded
close cells at the already rejected step. No JAX or mechanical solver imports.
"""
import argparse
import ast
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/work/input')
FROZEN = Path('/work/source')
NUMERICAL_COMMIT = 'b873d277711f5e706b2d57eea1b69f25e6375dc2'
FAILED_SHA = '13c6793f386795dee851076ad1b845944a52654013e94f0e6a6aabd44ec64a2c'
FEATURE_REF = 'refs/heads/work/triaxial-zero-contact-diagnostic'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if (len(names) != len(set(names)) or any('\\' in n or Path(n).is_absolute()
            or '..' in Path(n).parts for n in names) or z.testzip() is not None):
        z.close()
        raise ValueError('Unsafe, duplicate or CRC-invalid diagnostic ZIP')
    return z


def verify_controller():
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip()
    if commit != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF') != FEATURE_REF:
        raise ValueError('Read-only diagnostic must use its exact committed feature SHA')
    files = {}
    for name in ('tools/diagnose_triaxial_zero_contact.py', 'tools/run_triaxial_zero_recovery.py',
                 '.github/workflows/engineering-previews.yml', '.github/workflows/triaxial-zero-contact-diagnostic.yml'):
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']):
            raise ValueError('Diagnostic controller differs from committed bytes: ' + name)
        files[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'ref': FEATURE_REF, 'files': files}


def prepare(archive):
    if BASE.exists() or sha(archive) != FAILED_SHA:
        raise ValueError('Fresh diagnostic directory and exact rejected artifact 12571135742 required')
    controller = verify_controller()
    BASE.mkdir(parents=True)
    with checked_zip(archive) as outer:
        if set(outer.namelist()) != {'capsule.zip', 'receipt.json'}:
            raise ValueError('Expected exact failed-capsule transport wrapper')
        receipt = json.loads(outer.read('receipt.json'))
        item = receipt['capsule']
        inner = BASE / 'failed-capsule.zip'
        with outer.open('capsule.zip') as source, inner.open('xb') as target:
            shutil.copyfileobj(source, target, 1024**2)
        if inner.stat().st_size != item['bytes'] or sha(inner) != item['sha256']:
            raise ValueError('Failed inner capsule differs from whole transport receipt')
    with checked_zip(inner) as z:
        report = json.loads(z.read('failed-manifest.json'))
        if (report.get('schema') != 'triaxial-zero-failed-capsule-v1'
                or report.get('never_resumable') is not True or report.get('restart_valid') is not False
                or set(z.namelist()) != set(report['files']) | {'failed-manifest.json'}):
            raise ValueError('Only the exact explicitly nonresumable diagnostic is accepted')
        for name, item in report['files'].items():
            with z.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Failed diagnostic member changed: ' + name)
        selected = {'checkpoint.npz': 'diagnostic/checkpoint.npz', 'plan.json': 'records/plan.json',
                    'parent-context.json': 'records/context.json', 'summary.json': 'records/current-summary.json',
                    'contact-metrics.json': 'records/current-contact-metrics.json'}
        for name, member in selected.items():
            with z.open(member) as source, (BASE / name).open('xb') as target:
                shutil.copyfileobj(source, target, 1024**2)
    context = read(BASE / 'parent-context.json')
    pins = context['environment']['packages']
    (BASE / 'requirements.txt').write_text(f"numpy=={pins['numpy']}\npsutil=={pins['psutil']}\n")
    save(BASE / 'diagnostic-admission.json', {'controller': controller, 'failed_outer_sha256': FAILED_SHA,
        'checkpoint_sha256': sha(BASE / 'checkpoint.npz'), 'inner_sha256': sha(inner),
        'never_resumable': True, 'mechanical_steps_permitted': 0, 'original_inputs_unchanged': True})


def descriptor(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def reference_geometry(plan, kwargs, source):
    """Execute original pure NumPy sampling code, never an MPM/contact worker."""
    import numpy as np
    module = types.ModuleType('original_reference_geometry_readonly')
    exec(compile(source, 'immutable-original-reference-geometry', 'exec'), module.__dict__)
    names = ('dimensions', 'diameter', 'pitch', 'crimp', 'clearance', 'particle_spacing',
             'packing', 'density', 'grip_length', 'max_particles')
    geometry = module.make_geometry(theta_deg=-kwargs['theta'], **{k: kwargs[k] for k in names})
    differences=[]
    def compare(a,b,path='metadata'):
        if type(a)!=type(b):differences.append(dict(path=path,expected=a,actual=b));return
        if isinstance(a,dict):
            for key in a:compare(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,list):
            if len(a)!=len(b):differences.append(dict(path=path,expected_length=len(a),actual_length=len(b)))
            else:
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif a!=b:differences.append(dict(path=path,expected=a,actual=b))
    compare(plan['geometry'],geometry['metadata'])
    assert len(geometry['x'])==plan['sampling_hash']['shape'][0]
    save(Path('/work/output/reference-compatibility.json'),dict(expected=plan['sampling_hash'],actual=descriptor(geometry['x']),metadata_differences=differences))
    return geometry


def analyze():
    import numpy as np
    admission, context, plan = read(BASE / 'diagnostic-admission.json'), read(BASE / 'parent-context.json'), read(BASE / 'plan.json')
    if sha(BASE / 'checkpoint.npz') != admission['checkpoint_sha256']:raise ValueError('Raw changed')
    if platform.python_version() != '3.12.13' or np.__version__ != context['environment']['packages']['numpy']:
        raise ValueError('Original Python/NumPy versions required for exact registry rebuild')
    for name, expected in plan['source']['files'].items():
        if sha(FROZEN / name) != expected:
            raise ValueError('Original numerical blob changed: ' + name)
    geometry = reference_geometry(plan, context['kwargs'], (FROZEN / 'examples/triaxial_fibre_geometry.py').read_text())
    source = (FROZEN / 'mpm_jax/fibre_coupling.py').read_text()
    tree = ast.parse(source)
    names = {'_closest_segments', '_host_centres', '_coverage_neighbours'}
    original = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {'np': np, 'product': product}
    exec(compile(original, 'immutable-original-numpy-contact-kernels', 'exec'), scope)
    closest, centres_fn, coverage_fn = (scope[n] for n in ('_closest_segments', '_host_centres', '_coverage_neighbours'))
    fibres = geometry['metadata']['fibres']
    cross = geometry['metadata']['cross_section_particles']
    sections, owners, segments, bounds = [], [], [], []
    section_offset, segment_offset = 0, 0
    for fibre in fibres:
        n = fibre['longitudinal_sections']
        sections.extend(np.repeat(section_offset + np.arange(n), cross))
        owners.extend([fibre['id']] * (n - 1))
        segments.extend(zip(section_offset + np.arange(n - 1), section_offset + np.arange(1, n)))
        bounds.append((segment_offset, segment_offset + n - 1))
        section_offset += n
        segment_offset += n - 1
    sections, owners, segments, bounds = map(lambda v: np.asarray(v, np.int32), (sections, owners, segments, bounds))
    radii = np.asarray([f['radius_m'] for f in fibres])
    reference = centres_fn(geometry['x'], geometry['mass'], sections, section_offset)
    coverage = coverage_fn(reference, segments, owners, radii, plan['coverage_skin_m'], 0., 2_000_000)
    contact_model = read(BASE / 'summary.json')['definition']['contact_model']
    if descriptor(coverage) != contact_model['coverage_pairs']:
        raise ValueError('Whole reference coverage pair hash differs from actual rejected run')
    registry = {tuple(item['fibre_ids']): i for i, item in enumerate(geometry['contact_candidates'])}
    seed = np.asarray([registry.get(tuple(owners[pair]), -1) for pair in coverage], np.int32)
    with np.load(BASE / 'checkpoint.npz', allow_pickle=False) as cp:
        x, mass, failed = cp['state_x'], cp['state_mass'], cp['history_failed']
        aggregate, instant = cp['aggregate'].copy(), float(cp['state_time'])
        mechanical_valid = bool(cp['state_valid']) and bool(np.all(cp['history_valid']))
        if int(aggregate[0]) != 7218 or int(aggregate[5]) != 1 or bool(aggregate[4]):
            raise ValueError('Input is not the exact already rejected pre-step-7219 state')
        if not np.array_equal(mass, geometry['mass']):
            raise ValueError('Rejected state particle order/reference mass differs')
        centres = centres_fn(x, mass, sections, section_offset)
    failed_sections = np.bincount(sections, weights=failed.astype(np.int32), minlength=section_offset) > 0
    records = []
    unseeded = np.flatnonzero(seed < 0)
    for start in range(0, len(unseeded), 8192):
        cell = unseeded[start:start + 8192]
        pair = coverage[cell]
        endpoints = segments[pair]
        a, b, wa, wb, distance = closest(centres[endpoints[:, 0, 0]], centres[endpoints[:, 0, 1]],
            centres[endpoints[:, 1, 0]], centres[endpoints[:, 1, 1]], np)
        gap = distance - radii[owners[pair]].sum(axis=1)
        for local in np.flatnonzero(gap <= 0):
            seg = pair[local]
            fs = owners[seg]
            refdist = closest(reference[endpoints[local, 0, 0]], reference[endpoints[local, 0, 1]],
                reference[endpoints[local, 1, 0]], reference[endpoints[local, 1, 1]], np)[-1]
            records.append({'coverage_cell': int(cell[local]), 'segment_ids': seg.tolist(),
                'fibre_ids': fs.tolist(), 'family_ids': [fibres[i]['family_id'] for i in fs],
                'periodic_line_indices': [fibres[i]['periodic_line_index'] for i in fs],
                'closest_weights': [float(wa[local]), float(wb[local])],
                'current_closest_points_m': [a[local].tolist(), b[local].tolist()],
                'current_surface_gap_m': float(gap[local]),
                'reference_surface_gap_m': float(refdist - radii[fs].sum()),
                'segments_from_nearest_physical_end': np.minimum(seg - bounds[fs, 0], bounds[fs, 1] - 1 - seg).tolist(),
                'failed_endpoint_sections': failed_sections[endpoints[local]].tolist(),
                'failed_particles_per_fibre': [int(failed[geometry['fibre_id'] == i].sum()) for i in fs],
                'pair_registered_in_reference_crossings': False})
    metrics = read(BASE / 'contact-metrics.json')
    values = dict(zip(metrics['fields'], metrics['values']))
    count_matches = len(records) == int(values['uncovered_near_pairs'])
    report = {'schema': 'triaxial-zero-rejected-contact-registry-v1', 'rejected_attempt_step': 7219,
        'last_accepted_steps': 7218, 'time_s': instant, 'mechanical_state_valid': mechanical_valid,
        'driver_aggregate_valid_unchanged': False, 'reason_code': 1, 'saved_contact_metrics': values,
        'reference_geometry_hash_verified': descriptor(geometry['x']) == plan['sampling_hash'], 'whole_coverage_hash_verified': True,
        'reference_registered_fibre_pairs': len(registry), 'coverage_cells': len(coverage),
        'unseeded_coverage_cells': len(unseeded), 'unseeded_close_cells': records,
        'numpy_count_matches_saved_rejection': count_matches, 'mechanical_steps_executed': 0,
        'saved_final_macroscopic_observation': read(BASE / 'summary.json').get('observables', {}).get('final'),
        'never_resumable': True, 'production_eligible': False,
        'interpretation': 'Reason 1 is contact validity, while mechanical/grid rejection is reason 2. Unseeded close contact is a sufficient rejection gate; unexported budget/curvature/overlap gates are not independently evaluated here. A previously unregistered cross-family fibre pair is distinct from skin exhaustion or a failed-section contact. This audit never changes any gate or integrates the failed state.'}
    if any(name == 'jax' or name.startswith(('jax.', 'mpm_jax.')) for name in sys.modules):
        raise ValueError('Mechanical/JAX module unexpectedly imported by NumPy-only diagnostic')
    if sha(BASE / 'checkpoint.npz') != admission['checkpoint_sha256']:
        raise ValueError('Original rejected state changed during read-only audit')
    save(Path('/work/output/unseeded-contact-report.json'), report)
    print(json.dumps(report))
    if not count_matches:
        raise ValueError('NumPy close-cell count differs from saved original JAX metrics; do not infer matching cell identity')


def run():
    if verify_controller() != read(BASE / 'diagnostic-admission.json')['controller']:
        raise ValueError('Read-only diagnostic controller changed before execution')
    path = ROOT / 'tools/run_triaxial_zero_recovery.py'
    spec = importlib.util.spec_from_file_location('readonly_resource_guard', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.guarded_command([sys.executable, __file__, 'analyze'], BASE / 'diagnostic.log',
        BASE / 'diagnostic-memory.json', 0, wall_seconds=20 * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.archive)
    elif args.action == 'run':
        run()
    else:
        analyze()
