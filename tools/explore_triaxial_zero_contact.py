"""Exploratory integer-registry audit; no integration, restart, or production admission.

The strict diagnostic is unchanged. This separate route certifies integer topology
and coverage while explicitly declining bit-exact reference float certification.
All inputs are opened read-only; JSON is emitted on stdout only.
"""
import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'plan.json': 'b9cab9b6cf523d8377f3818071c5b7a45dc6510cda3d0622e11265a5e075697e',
    'summary.json': '7e6d22b66dec3895f4aef2b27a82150bd05d376dffc25335cecdc9223a7441b0',
    'contact-metrics.json': 'afa824d6c051be08aeb128ba0739e304f346883b462b16e17d8c0637d05d1a7a',
    'parent-context.json': 'a92958c2b33cc79c80e75efdc966ecd841effe1cb41b634ce33015b8af18feed',
}
CP_SHA = '3b3a6326c93b4d6d8c97a63278b2b946671726f7690da7742b1a794b522ec085'
REGISTRY_SHA = '64794394e00405e753373557e8e7df9a3880ee42f98b28afe2210abf4d698853'
COVERAGE_SHA = '9329579a6107ad99acdde4153112415c48fb25aefcc4858dc5ba4a1d75d6dd80'
CONTROLLER_PINS = {
    'tools/diagnose_triaxial_zero_contact.py': 'aa5cf874a40bea9761fa9a42e976c9702bd67334601c4acfd4a38d9a65715e87',
    'tools/run_triaxial_zero_recovery.py': '7b338d59cb67a9c013b5f998928710256bd61e97c4b43b62935ae906bec0ead0',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def registry(fibres, pitch):
    # Exact 0/60/120 degree directions; Decimal sqrt evaluation is repeated at
    # 60/80 digits below. All decisions must clear a conservative 1 nm margin.
    D = Decimal
    frames = [(D(1), D(0)), (D('.5'), D(3).sqrt()/2), (-D('.5'), D(3).sqrt()/2)]
    pairs, included, excluded = [], D(1), D(1)
    for left in fibres:
        for right in fibres[left['id']+1:]:
            if left['family_id'] == right['family_id']:
                continue
            u, v = frames[left['family_id']], frames[right['family_id']]
            n, m = (-u[1], u[0]), (-v[1], v[0])
            a = D.from_float(pitch)*(D(left['periodic_line_index'])+D('.5'))
            b = D.from_float(pitch)*(D(right['periodic_line_index'])+D('.5'))
            det = n[0]*m[1]-n[1]*m[0]
            x, y = (a*m[1]-n[1]*b)/det, (n[0]*b-a*m[0])/det
            margins = [min(t-D.from_float(f['parameter_range_m'][0]),
                           D.from_float(f['parameter_range_m'][1])-t)
                       for f, t in ((left, x*u[0]+y*u[1]), (right, x*v[0]+y*v[1]))]
            if min(margins) >= 0:
                pairs.append((left['id'], right['id']))
                included = min(included, *margins)
            else:
                excluded = min(excluded, -min(margins))
    require(min(included, excluded) > D('1e-9'), 'Crossing trim decision lacks robust margin')
    return np.asarray(pairs, dtype='<i4'), float(included), float(excluded)


def analyze(inputs, source, checkpoint=None):
    for name, expected in CONTROLLER_PINS.items():
        require(sha(ROOT/name) == expected, 'Strict controller/guard changed: '+name)
    records = {}
    for name, expected in PINS.items():
        require(sha(inputs/name) == expected, 'Original metadata differs: '+name)
        records[name] = json.loads((inputs/name).read_text())
    plan, context = records['plan.json'], records['parent-context.json']
    for name, expected in plan['source']['files'].items():
        require(sha(source/name) == expected, 'Original numerical source differs: '+name)
    meta, kwargs = plan['geometry'], context['kwargs']
    require(meta['theta_deg'] == 0 and len(meta['fibres']) == 72, 'Wrong original geometry')
    fibres, cross = meta['fibres'], meta['cross_section_particles']
    require(cross == 24 and sum(f['particle_count'] for f in fibres) == 683232, 'Wrong particle topology')
    for i, f in enumerate(fibres):
        require(f['id'] == i and f['particle_count'] == cross*f['longitudinal_sections'], 'Invalid ordered fibre topology')
    for precision in (60, 80):
        with localcontext() as dc:
            dc.prec = precision
            pairs, inside, outside = registry(fibres, meta['pitch_m'])
        require(pairs.shape == (980, 2) and digest(pairs) == REGISTRY_SHA, 'Ordered registry differs')
    geometry_module = types.ModuleType('immutable_exploratory_numpy_geometry')
    exec(compile((source/'examples/triaxial_fibre_geometry.py').read_text(), 'original-geometry', 'exec'), geometry_module.__dict__)
    names = ('dimensions', 'diameter', 'pitch', 'crimp', 'clearance', 'particle_spacing', 'packing', 'density', 'grip_length', 'max_particles')
    geometry = geometry_module.make_geometry(theta_deg=-kwargs['theta'], **{n: kwargs[n] for n in names})
    integer_keys = ('id', 'family_id', 'periodic_line_index', 'particle_count', 'longitudinal_sections')
    require(len(geometry['metadata']['fibres']) == len(fibres), 'Rebuilt fibre count differs')
    require(all(all(a[k] == b[k] for k in integer_keys) for a, b in zip(fibres, geometry['metadata']['fibres'])), 'Rebuilt integer topology differs')
    require(np.array_equal(np.asarray([c['fibre_ids'] for c in geometry['contact_candidates']], dtype='<i4'), pairs), 'Source registry disagrees with analytic margins')
    tree = ast.parse((source/'mpm_jax/fibre_coupling.py').read_text())
    scope = {'np': np, 'product': product}
    kernels = {'_closest_segments', '_host_centres', '_coverage_neighbours'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in kernels], type_ignores=[]), 'original-numpy-kernels', 'exec'), scope)
    sections, segments, owners, offset = [], [], [], 0
    for f in fibres:
        n = f['longitudinal_sections']
        sections.extend(np.repeat(offset+np.arange(n), cross))
        segments.extend(zip(offset+np.arange(n-1), offset+np.arange(1, n)))
        owners.extend([f['id']]*(n-1))
        offset += n
    sections, segments, owners = [np.asarray(a, dtype=np.int32) for a in (sections, segments, owners)]
    centres_fn, closest = scope['_host_centres'], scope['_closest_segments']
    reference = centres_fn(geometry['x'], geometry['mass'], sections, offset)
    radii = np.asarray([f['radius_m'] for f in fibres])
    coverage = scope['_coverage_neighbours'](reference, segments, owners, radii, plan['coverage_skin_m'], 0., 2000000)
    saved = records['summary.json']['definition']['contact_model']['coverage_pairs']
    require(saved == {'shape': [1200704, 2], 'dtype': 'int32', 'sha256': COVERAGE_SHA}, 'Original coverage descriptor differs')
    require(coverage.shape == (1200704, 2) and str(coverage.dtype) == 'int32' and digest(coverage) == COVERAGE_SHA, 'Exact coverage integer bytes differ')
    report = dict(schema='triaxial-zero-exploratory-integer-registry-v1', production_eligible=False,
                  never_resumable=True, mechanical_steps_executed=0, production_gates_changed=False,
                  reference_float_sampling_certified=False, definitive_failure_identity=False,
                  registry_sha256=REGISTRY_SHA, coverage_sha256=COVERAGE_SHA,
                  retained_crossing_min_trim_margin_m=inside, excluded_crossing_min_trim_clearance_m=outside,
                  regenerated_sampling_sha256=digest(geometry['x']), original_sampling_sha256=plan['sampling_hash']['sha256'],
                  checkpoint_inspected=False, exploratory_close_cells=[])
    if checkpoint is not None:
        require(sha(checkpoint) == CP_SHA, 'Wrong rejected checkpoint')
        registry_set = {tuple(p) for p in pairs.tolist()}
        cells = np.asarray([i for i, p in enumerate(owners[coverage]) if tuple(p) not in registry_set], dtype=np.int32)
        with np.load(checkpoint, allow_pickle=False) as cp:
            aggregate = cp['aggregate']
            require(int(aggregate[0]) == 7218 and int(aggregate[5]) == 1 and not bool(aggregate[4]), 'Wrong rejected state')
            x, mass = cp['state_x'], cp['state_mass']
            require(x.shape == (683232, 3) and mass.shape == (683232,) and np.isfinite(x).all() and np.isfinite(mass).all() and (mass > 0).all(), 'Invalid rejected arrays')
            # Use exact saved masses, never regenerated reference float masses.
            current = centres_fn(x, mass, sections, offset)
        for start in range(0, len(cells), 8192):
            ids = cells[start:start+8192]
            ends = segments[coverage[ids]]
            distance = closest(current[ends[:, 0, 0]], current[ends[:, 0, 1]], current[ends[:, 1, 0]], current[ends[:, 1, 1]], np)[-1]
            gaps = distance-radii[owners[coverage[ids]]].sum(axis=1)
            for j in np.flatnonzero(gaps <= 0):
                report['exploratory_close_cells'].append(dict(coverage_cell=int(ids[j]), segment_ids=coverage[ids[j]].tolist(), fibre_ids=owners[coverage[ids[j]]].tolist(), current_surface_gap_m=float(gaps[j]), gap_margin_from_zero_m=float(abs(gaps[j]))))
        metrics = records['contact-metrics.json']
        values = dict(zip(metrics['fields'], metrics['values']))
        require(int(values['uncovered_near_pairs']) == 1, 'Original rejection metrics differ')
        report.update(checkpoint_inspected=True, original_uncovered_near_pairs=1,
                      exploratory_numpy_count_matches_saved=(len(report['exploratory_close_cells']) == 1))
        require(sha(checkpoint) == CP_SHA, 'Rejected input changed')
    require(not any(n == 'jax' or n.startswith(('jax.', 'mpm_jax.')) for n in sys.modules), 'Unexpected mechanical import')
    report['interpretation'] = 'Exact integer registry and coverage are independent of reference float sampling admission. Current gaps use saved rejected-state arrays and NumPy arithmetic. Count agreement does not prove original JAX cell identity; no reference gap or original numerical restart is certified.'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.inputs, args.source, args.checkpoint), indent=2, allow_nan=False))
