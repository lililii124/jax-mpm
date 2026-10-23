"""Dimension-aware comparison of the predamage tiny continuation fixture.

This checks numerical continuation, not physical accuracy or failure envelopes.
The spectral roundoff allowances are engineering budgets, not certified bounds
for an eigensolver implementation. Active-contact and damaged fixtures require
a separate comparison contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RELATIVE = 2e-10
EPSILON = np.finfo(np.float64).eps
SPECTRAL_ALLOWANCE = 64
CONTACT_FIELDS = (
    'search_boundary_pairs', 'uncovered_near_pairs', 'same_family_near_pairs',
    'coverage_motion_exceeded', 'maximum_nonrigid_displacement_m', 'minimum_gap_m',
    'active_pairs', 'minimum_force_N', 'maximum_force_N', 'failed_section_contacts',
    'unsupported_end_contacts', 'degenerate_pairs', 'force_residual_N',
    'torque_residual_Nm', 'potential_J', 'valid',
)
STATE_FIELDS = {'x', 'v', 'F', 'C', 'mass', 'volume', 'fibre_id', 'tau', 'time',
    'valid', 'boundary_impulse', 'external_impulse', 'boundary_work'}
HISTORY_FIELDS = {'history_kappa', 'history_damage', 'history_failed', 'history_valid'}
FINAL_FIELDS = STATE_FIELDS | HISTORY_FIELDS
CHECKPOINT_FIELDS = {'state_'+key for key in STATE_FIELDS} | HISTORY_FIELDS | {
    'contact_active', 'contact_law_valid', 'contact_radius', 'contact_segments', 'contact_valid',
    'total_impulse', 'clamp_work', 'grid_ke_work', 'aggregate', 'last_contact', 'initial_momentum',
    'frame', 'signature', 'solver_seconds', 'jax_version', 'tables'}
EXACT_OBSERVABLES = {
    'time', 'accepted_steps', 'prescribed_engineering_strain', 'peak_step_dynamic_grip_force_time_s',
    'first_material_failure_time_s', 'failed_particles', 'max_damage',
    'damaged_reference_volume_fraction', 'active_contact_pairs',
    'sampled_section_graph_connects_grips', 'active_fibre_x_support_cuts',
}
ENERGIES = {'kinetic_energy_J', 'kinetic_translational_energy_J', 'kinetic_apic_affine_energy_J',
    'elastic_energy_J', 'actual_clamp_work_J', 'grid_kinetic_correction_J'}
FORCES = {'mean_grip_tensile_force_N', 'peak_step_dynamic_grip_force_N'}
POSITIONS = {'minimum_contact_gap_m', 'coverage_nonrigid_displacement_m', 'grip_position_error_m'}
INACTIVE_OBSERVABLES = {'contact_potential_J', 'contact_force_balance_N', 'contact_torque_balance_Nm'}
VECTOR_COLUMNS = {f'{side}_{kind}_{axis}_{unit}' for side in ('left', 'right') for axis in 'xyz'
    for kind, unit in (('applied_force', 'N'), ('accumulated_impulse', 'Ns'))}
OBSERVABLE_FIELDS = (EXACT_OBSERVABLES | ENERGIES | FORCES | POSITIONS | INACTIVE_OBSERVABLES
    | VECTOR_COLUMNS | {'kinetic_to_elastic_ratio', 'grip_velocity_error_m_per_s'})
DIAGNOSTIC_FIELDS = {'time', 'mass', 'reference_volume', 'current_volume', 'min_J', 'max_J',
    'max_damage', 'failed_mass', 'mass_relative_drift', 'momentum_balance_residual_Ns',
    'state_valid', 'coupling_valid', 'failure_reason_code'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def magnitude(value):
    return float(np.max(np.abs(np.asarray(value)), initial=0.))


class Comparison:
    def __init__(self):
        self.metrics = []
        self.errors = []

    def require(self, name, condition, message):
        if not bool(condition):
            self.errors.append({'field': name, 'message': message})
        return bool(condition)

    def exact(self, name, left, right, *, units='discrete'):
        a, b = np.asarray(left), np.asarray(right)
        ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        if a.dtype.kind in 'fc':
            ok = ok and bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))
        self.metrics.append({'field': name, 'units': units, 'method': 'exact', 'passed': bool(ok)})
        return self.require(name, ok, 'Exact value, dtype, shape or finite check failed')

    def numeric(self, name, left, right, *, units, scale=None, floor=0., method='field Linf'):
        a, b = np.asarray(left), np.asarray(right)
        if not self.require(name, a.shape == b.shape and a.dtype == b.dtype,
                            'Numerical shape or dtype differs'):
            return False
        if not self.require(name, a.dtype.kind in 'fc' and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
                            'Numerical field must be finite floating-point data'):
            return False
        scale = max(magnitude(a), magnitude(b)) if scale is None else float(scale)
        if not self.require(name, np.isfinite(scale) and scale >= 0 and np.isfinite(floor) and floor >= 0,
                            'Invalid comparison budget'):
            return False
        difference = magnitude(a-b)
        budget = RELATIVE*scale+floor
        ok = difference <= budget
        self.metrics.append({'field': name, 'units': units, 'method': method,
            'max_abs_error': difference, 'scale': scale, 'relative_tolerance': RELATIVE,
            'roundoff_budget': float(floor), 'allowed_error': float(budget), 'passed': bool(ok)})
        return self.require(name, ok, 'Numerical difference exceeds registered dimensional budget')

    def identity(self, name, left, right, *, units, budget):
        a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        good = a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        error = magnitude(a-b) if good else None
        ok = good and error <= budget
        self.metrics.append({'field': name, 'units': units, 'method': 'within-run identity',
            'max_abs_error': error, 'allowed_error': float(budget), 'passed': bool(ok)})
        return self.require(name, ok, 'Derived value does not match its own base quantities')


def context(left, right, plan):
    material = plan['material']
    E, nu = float(material['young']), float(material['poisson'])
    dx = float(plan['grid']['dx_m'])
    if not (np.all(np.isfinite([E, nu, dx])) and E > 0 and -1 < nu < .5 and dx > 0):
        raise ValueError('Invalid elastic constants or grid spacing')
    smax, smin = 1., np.inf
    for data in (left, right):
        F = data['state_F']
        if F.ndim != 3 or len(F) == 0 or F.shape[1:] != (3, 3) or F.dtype != np.dtype('float64') or not np.all(np.isfinite(F)):
            raise ValueError('Expected finite binary64 deformation gradients')
        for start in range(0, len(F), 16384):
            singular = np.linalg.svd(F[start:start+16384], compute_uv=False)
            smax = max(smax, float(singular.max()))
            smin = min(smin, float(singular.min()))
    if smin < .5 or smax > 2 or abs(nu) > .45:
        raise ValueError('Spectral budget is restricted to the well-conditioned tiny elastic fixture')
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    condition = (3*abs(lam)+2*mu)/E
    result = {'dx_m': dx, 'young_Pa': E, 'poisson': nu, 'stiffness_condition': condition,
        'maximum_singular_stretch': smax, 'minimum_singular_stretch': smin,
        'F_roundoff': SPECTRAL_ALLOWANCE*EPSILON*smax,
        'stress_roundoff_Pa': SPECTRAL_ALLOWANCE*EPSILON*E*condition*smax**2,
        'kappa_roundoff': SPECTRAL_ALLOWANCE*EPSILON*condition*smax}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite material condition or spectral budget')
    return result


def _array_shapes(c, label, data):
    """Validate whole arrays before any fixed-index/group comparison."""
    plan, cp = data['plan'], data['checkpoint.npz']
    n, frames = plan['particles'], plan['saved_states']
    contact = data['summary']['definition']['contact_model']
    pairs = contact['candidate_pairs']
    if not c.require(label+'.array-dimensions', type(n) is int and n > 0
        and type(frames) is int and frames >= 2 and type(pairs) is int and pairs >= 0
        and contact['particles'] == n, 'Invalid plan particle/frame/contact counts'):
        return
    state = {key: ((n, 3), 'float64') for key in ('x', 'v')}
    state.update({key: ((n, 3, 3), 'float64') for key in ('F', 'C', 'tau')})
    state.update({key: ((n,), 'float64') for key in ('mass', 'volume', 'history_kappa', 'history_damage')})
    state.update({key: ((n,), 'bool') for key in ('history_failed', 'history_valid')})
    state.update(fibre_id=((n,), 'int32'), time=((), 'float64'), valid=((), 'bool'),
        boundary_impulse=((3,), 'float64'), external_impulse=((3,), 'float64'), boundary_work=((), 'float64'))
    checkpoint = {('state_'+key if key in STATE_FIELDS else key): spec for key, spec in state.items()}
    checkpoint.update(contact_active=((pairs,), 'bool'), contact_law_valid=((pairs,), 'bool'),
        contact_radius=((pairs,), 'float64'), contact_segments=((pairs, 2), 'int64'), contact_valid=((), 'bool'),
        total_impulse=((2, 3), 'float64'), clamp_work=((2,), 'float64'), grid_ke_work=((2,), 'float64'),
        aggregate=((6,), 'float64'), last_contact=((16,), 'float64'), initial_momentum=((3,), 'float64'),
        frame=((), 'int64'), solver_seconds=((), 'float64'), signature=((), 'text'),
        jax_version=((), 'text'), tables=((), 'text'))
    schemas = {'checkpoint.npz': checkpoint, 'final_state_0.npz': state,
        'particle_fields.npz': {'phase': ((n,), 'int8'), 'yarn_id': ((n,), 'int32'), 'family_id': ((n,), 'int16')},
        'trajectory.npz': {'time': ((frames,), 'float64'), 'body': ((n,), 'int32'), 'failed': ((frames, n), 'uint8')}}
    for archive, schema in schemas.items():
        for key, (shape, dtype) in schema.items():
            value = data[archive][key]
            type_ok = value.dtype.kind == 'U' if dtype == 'text' else value.dtype == np.dtype(dtype)
            c.require(f'{label}.{archive}:{key}.layout', value.shape == shape and type_ok,
                f'Expected complete shape {shape} and dtype {dtype}')
            if value.dtype.kind in 'fc':
                c.require(f'{label}.{archive}:{key}.finite', np.all(np.isfinite(value)), 'Array contains nonfinite values')
    values = np.asarray(data['contact']['values'])
    c.require(label+'.contact-metrics.layout', values.shape == (16,) and values.dtype == np.dtype('float64')
        and np.all(np.isfinite(values)), 'Expected 16 finite binary64 contact metric values')


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate CSV field')
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key is None:
                    raise ValueError('Extra CSV values')
                if value == '': parsed[key] = None
                elif value in ('True', 'False'): parsed[key] = value == 'True'
                else: parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _column(rows, key):
    return np.asarray([row[key] for row in rows])


def _group_scale(left, right, side, kind, unit):
    return max(magnitude(_column(rows, f'{side}_{kind}_{axis}_{unit}'))
        for rows in (left, right) for axis in 'xyz')


def _contact(c, name, a, b, ctx):
    if not c.require(name, a.shape == b.shape == (16,), 'Contact metric shape is not the registered 16 fields'):
        return
    for index, field in enumerate(CONTACT_FIELDS):
        label = name+'.'+field
        if index in (4, 5):
            c.numeric(label, a[index], b[index], units='m', scale=ctx['dx_m'], method='dx')
        else:
            c.exact(label, a[index], b[index])
            if index in (6, 7, 8, 12, 13, 14):
                c.require(label+'.inactive', a[index] == b[index] == 0,
                    'This contract does not admit active contact or nonzero contact force/potential')
            if index in (0, 1, 2, 3, 9, 10, 11):
                c.require(label+'.zero-errors', a[index] == b[index] == 0,
                    'Contact search, coverage and geometry error counts must be zero')
            if index == 15:
                c.require(label+'.valid', a[index] == b[index] == 1,
                    'Contact metrics must affirm validity, not merely agree across runs')


def _state_field(c, name, key, a, b, ctx, left, right):
    if key == 'x':
        c.numeric(name, a, b, units='m', scale=ctx['dx_m'], method='dx')
    elif key in ('v', 'C'):
        c.numeric(name, a, b, units='m/s' if key == 'v' else '1/s')
    elif key == 'F':
        c.numeric(name, a, b, units='1', scale=max(magnitude(a-np.eye(3)), magnitude(b-np.eye(3))),
            floor=ctx['F_roundoff'], method='F-I Linf plus spectral roundoff')
    elif key == 'tau':
        c.numeric(name, a, b, units='Pa', floor=ctx['stress_roundoff_Pa'], method='stress Linf plus E-scaled spectral roundoff')
    elif key == 'history_kappa':
        c.numeric(name, a, b, units='1', floor=ctx['kappa_roundoff'], method='kappa Linf plus stretch roundoff')
    elif key == 'boundary_impulse':
        for axis in range(3):
            scale = max(max(np.sum(np.abs(d['state_mass']*d['state_v'][:, axis])),
                np.sum(np.abs(d['total_impulse'][:, axis]))) for d in (left, right))
            c.numeric(f'{name}[{axis}]', a[axis], b[axis], units='N s', scale=scale,
                method='same-axis uncancelled momentum/fixture contributions')
    elif key == 'boundary_work':
        c.numeric(name, a, b, units='J', method='separate grid kinetic-correction scalar')
    else:
        c.exact(name, a, b)


def _observables(c, left, right, ctx):
    if not c.require('observables.columns', all(set(row) == OBSERVABLE_FIELDS for row in left+right),
                     'Missing or unregistered observable field'):
        return
    for key in sorted(OBSERVABLE_FIELDS):
        a, b = _column(left, key), _column(right, key)
        label = 'observables.'+key
        if key in EXACT_OBSERVABLES or key in INACTIVE_OBSERVABLES:
            c.exact(label, a, b)
            if key in INACTIVE_OBSERVABLES:
                c.require(label+'.inactive', np.all(a == 0) and np.all(b == 0), 'Inactive fixture must have zero contact quantity')
            if key in ('active_contact_pairs', 'failed_particles', 'max_damage', 'damaged_reference_volume_fraction'):
                c.require(label+'.scope', np.all(a == 0) and np.all(b == 0), 'Tiny saved history must be inactive and predamage')
        elif key in POSITIONS:
            c.numeric(label, a, b, units='m', scale=ctx['dx_m'], method='dx')
        elif key in ENERGIES:
            c.numeric(label, a, b, units='J', method='independent energy/work column')
        elif key in FORCES:
            c.numeric(label, a, b, units='N')
        elif key in VECTOR_COLUMNS:
            side = key.split('_')[0]
            kind, unit = ('applied_force', 'N') if '_applied_force_' in key else ('accumulated_impulse', 'Ns')
            c.numeric(label, a, b, units='N' if unit == 'N' else 'N s',
                scale=_group_scale(left, right, side, kind, unit), method='one fixture xyz over saved time')
        elif key == 'grip_velocity_error_m_per_s':
            scale = max(ctx['velocity_scale'], magnitude(a), magnitude(b))
            c.numeric(label, a, b, units='m/s', scale=scale, method='velocity-field scale')
        elif key == 'kinetic_to_elastic_ratio':
            for i, (ra, rb) in enumerate(zip(left, right)):
                ka, kb = ra['kinetic_energy_J'], rb['kinetic_energy_J']
                ua, ub = ra['elastic_energy_J'], rb['elastic_energy_J']
                if ua <= 1e-30 or ub <= 1e-30:
                    c.require(f'{label}[{i}]', a[i] is None and b[i] is None and ua <= 1e-30 and ub <= 1e-30,
                        'Ratio undefined semantics differ')
                else:
                    budget = RELATIVE*max(abs(ka), abs(kb))/min(ua, ub)
                    budget += max(abs(ka), abs(kb))*RELATIVE*max(abs(ua), abs(ub))/min(ua, ub)**2
                    c.identity(f'{label}[{i}].propagated', a[i], b[i], units='1', budget=budget)


def _identities(c, name, checkpoint, final, index, obs, diag, plan):
    n = len(checkpoint['state_mass'])
    gamma = (n+16)*EPSILON/(1-(n+16)*EPSILON)
    for key in FINAL_FIELDS:
        ck = 'state_'+key if key in STATE_FIELDS else key
        c.exact(f'{name}.checkpoint-final.{key}', checkpoint[ck], final[key])
    times = index['time']
    c.exact(name+'.saved-times', _column(obs, 'time'), times)
    c.exact(name+'.diagnostic-times', _column(diag, 'time'), times)
    c.require(name+'.frame-count', len(obs) == len(diag) == len(times) == int(plan['saved_states']), 'Incomplete saved frame prefix')
    c.exact(name+'.frame-steps', _column(obs, 'accepted_steps'), np.asarray(plan['frame_steps'], dtype=float))
    c.require(name+'.final-step', checkpoint['aggregate'][0] == plan['stop_steps'] == plan['frame_steps'][-1], 'Final step is not the full plan')
    c.require(name+'.final-frame', int(checkpoint['frame']) == len(times)-1, 'Checkpoint is not the final saved frame')
    c.exact(name+'.final-clock', checkpoint['state_time'], times[-1])
    for row_index, row in enumerate(diag):
        c.require(f'{name}.diagnostics[{row_index}].valid', row['state_valid'] is True
            and row['coupling_valid'] is True and row['failure_reason_code'] == 0,
            'Saved diagnostic validity must affirm an accepted state')
    for row_index, row in enumerate(obs):
        prefix = f'{name}.observables[{row_index}]'
        k = row['kinetic_translational_energy_J']+row['kinetic_apic_affine_energy_J']
        c.identity(prefix+'.kinetic-sum', row['kinetic_energy_J'], k, units='J', budget=8*EPSILON*abs(k))
        U = row['elastic_energy_J']
        if U > 1e-30:
            ratio = row['kinetic_energy_J']/U
            c.identity(prefix+'.ratio', row['kinetic_to_elastic_ratio'], ratio, units='1', budget=8*EPSILON*abs(ratio))
        else:
            c.require(prefix+'.ratio', row['kinetic_to_elastic_ratio'] is None, 'Undefined energy ratio must be null')
        mean = .5*(row['right_applied_force_x_N']-row['left_applied_force_x_N'])
        c.identity(prefix+'.mean-force', row['mean_grip_tensile_force_N'], mean, units='N',
            budget=8*EPSILON*(abs(row['right_applied_force_x_N'])+abs(row['left_applied_force_x_N'])))
        if row_index:
            dt = (plan['frame_steps'][row_index]-plan['frame_steps'][row_index-1])*plan['dt']
            old = obs[row_index-1]
            for side in ('left', 'right'):
                scale = max(abs(r[f'{side}_accumulated_impulse_{axis}_Ns']) for r in obs for axis in 'xyz')
                for axis in 'xyz':
                    current, previous = row[f'{side}_accumulated_impulse_{axis}_Ns'], old[f'{side}_accumulated_impulse_{axis}_Ns']
                    predicted = (current-previous)/dt
                    c.identity(prefix+f'.{side}-force-{axis}', row[f'{side}_applied_force_{axis}_N'], predicted,
                        units='N', budget=32*EPSILON*scale/dt)
    for group, side in enumerate(('left', 'right')):
        c.exact(name+f'.final-{side}-impulse', checkpoint['total_impulse'][group],
            np.asarray([obs[-1][f'{side}_accumulated_impulse_{axis}_Ns'] for axis in 'xyz']))
    for key, column in (('clamp_work', 'actual_clamp_work_J'), ('grid_ke_work', 'grid_kinetic_correction_J')):
        value = checkpoint[key]
        c.identity(name+'.'+key+'-sum', obs[-1][column], np.sum(value), units='J', budget=8*EPSILON*np.sum(np.abs(value)))
    c.identity(name+'.boundary-work', checkpoint['state_boundary_work'], np.sum(checkpoint['grid_ke_work']),
        units='J', budget=32*EPSILON*np.sum(np.abs(checkpoint['grid_ke_work'])))
    pterms = checkpoint['state_mass'][:, None]*checkpoint['state_v']
    momentum = np.sum(pterms, axis=0)
    residual = momentum-checkpoint['initial_momentum']-checkpoint['state_boundary_impulse']-checkpoint['state_external_impulse']
    base = (np.sum(np.abs(pterms), axis=0)+np.abs(checkpoint['initial_momentum'])
        +np.abs(checkpoint['state_boundary_impulse'])+np.abs(checkpoint['state_external_impulse']))
    c.identity(name+'.final-momentum-residual', diag[-1]['momentum_balance_residual_Ns'],
        np.linalg.norm(residual), units='N s', budget=float(2*gamma*np.linalg.norm(base)))
    c.identity(name+'.boundary-impulse-sum', checkpoint['state_boundary_impulse'], np.sum(checkpoint['total_impulse'], axis=0),
        units='N s', budget=float(32*EPSILON*np.max(np.sum(np.abs(checkpoint['total_impulse']), axis=0))))
    actual_k = .5*np.sum(checkpoint['state_mass']*np.sum(checkpoint['state_v']**2, axis=1))
    affine = plan['grid']['dx_m']**2/8*np.sum(checkpoint['state_mass']*np.sum(checkpoint['state_C']**2, axis=(1, 2)))
    for key, value in [('kinetic_translational_energy_J', actual_k), ('kinetic_apic_affine_energy_J', affine)]:
        c.identity(name+'.final-'+key, obs[-1][key], value, units='J', budget=2*gamma*abs(value))
    c.exact(name+'.peak-force-record', checkpoint['aggregate'][2], np.asarray(obs[-1]['peak_step_dynamic_grip_force_N']))
    c.exact(name+'.peak-time-record', checkpoint['aggregate'][3], np.asarray(obs[-1]['peak_step_dynamic_grip_force_time_s']))
    expected_first = None if checkpoint['aggregate'][1] < 0 else float(checkpoint['aggregate'][1])
    c.require(name+'.first-failure-record', obs[-1]['first_material_failure_time_s'] == expected_first, 'First-failure identity differs')


def compare_runs(segmented, reference):
    """Return all available comparison failures without writing either dataset."""
    c = Comparison()
    folders = {'segmented': Path(segmented), 'reference': Path(reference)}
    required = ['plan.json', 'summary.json', 'checkpoint.npz', 'final_state_0.npz', 'particle_fields.npz',
        'trajectory.npz', 'observables.csv', 'diagnostics.csv', 'contact-metrics.json']
    required += [f'trajectory_{key}.npy' for key in ('x', 'speed', 'damage', 'failed', 'temperature')]
    inputs, data, ctx = {}, {}, {}
    for label, folder in folders.items():
        inputs[label] = {}
        for filename in required:
            if c.require(label+'.'+filename, (folder/filename).is_file(), 'Required comparison input is missing'):
                inputs[label][filename] = digest(folder/filename)
    if c.errors:
        return _result(c, ctx, inputs)
    try:
        for label, folder in folders.items():
            d = {'plan': json.loads((folder/'plan.json').read_text()),
                 'summary': json.loads((folder/'summary.json').read_text()),
                 'contact': json.loads((folder/'contact-metrics.json').read_text()),
                 'obs': _read_csv(folder/'observables.csv'), 'diag': _read_csv(folder/'diagnostics.csv')}
            for filename in ('checkpoint.npz', 'final_state_0.npz', 'particle_fields.npz', 'trajectory.npz'):
                with np.load(folder/filename, allow_pickle=False) as z:
                    d[filename] = {key: z[key] for key in z.files}
            data[label] = d
        a, b = data['segmented'], data['reference']
        c.require('plan', a['plan'] == b['plan'], 'Physical plans differ')
        for label, d in data.items():
            c.require(label+'.completed', d['summary'].get('completed') is True, 'A partial/probe result is not a full comparison')
            c.require(label+'.scope', d['plan']['scope'] == 'smoke', 'Only the predamage smoke continuation contract is supported')
            expected = {'checkpoint.npz': CHECKPOINT_FIELDS, 'final_state_0.npz': FINAL_FIELDS,
                'particle_fields.npz': {'phase', 'yarn_id', 'family_id'}, 'trajectory.npz': {'time', 'body', 'failed'}}
            for archive, keys in expected.items():
                c.require(label+'.'+archive+'.schema', set(d[archive]) == keys, 'Missing or unregistered archive field')
            c.require(label+'.obs-count', len(d['obs']) == len(d['diag']) == d['plan']['saved_states'], 'Missing saved row')
            c.require(label+'.diagnostics-schema', all(set(row) == DIAGNOSTIC_FIELDS for row in d['diag']), 'Missing or unregistered diagnostic field')
            c.require(label+'.contact-schema', d['contact'].get('fields') == list(CONTACT_FIELDS), 'Unknown contact-metric schema')
            c.require(label+'.summary-fields', {'materials', 'configs', 'source', 'definition'} <= d['summary'].keys(),
                'Summary is missing numerical definition/source fields')
            c.require(label+'.summary-material', d['summary'].get('materials') == [d['plan']['material']],
                'Summary material is not the actual plan material')
            c.require(label+'.summary-source', d['summary'].get('source') == d['plan'].get('source'),
                'Summary source is not the actual plan source')
        if c.errors:
            return _result(c, ctx, inputs)
        for label, d in data.items():
            _array_shapes(c, label, d)
        if c.errors:
            return _result(c, ctx, inputs)
        left, right = a['checkpoint.npz'], b['checkpoint.npz']
        ctx = context(left, right, a['plan'])
        ctx['velocity_scale'] = max(magnitude(left['state_v']), magnitude(right['state_v']))
        for key in ('materials', 'configs', 'source', 'definition'):
            c.require('summary.'+key, a['summary'].get(key) == b['summary'].get(key), 'Summary numerical definition/source differs')
        for label, d in data.items():
            cp = d['checkpoint.npz']
            c.require(label+'.predamage-inactive', np.all(cp['history_damage'] == 0)
                and not np.any(cp['history_failed']) and not np.any(cp['contact_active'])
                and np.all(cp['contact_radius'] == 0), 'This contract is restricted to zero damage and inactive contact')
            c.require(label+'.binary64', all(cp[key].dtype == np.dtype('float64') for key in
                ('state_x', 'state_v', 'state_C', 'state_tau', 'history_kappa')), 'Expected binary64 physical state')
            c.require(label+'.valid', bool(cp['state_valid']) and np.all(cp['history_valid'])
                and bool(cp['contact_valid']) and np.all(cp['contact_law_valid'])
                and cp['aggregate'][4] == 1 and cp['aggregate'][5] == 0,
                'An invalid or refused state cannot satisfy the continuation contract')
            c.require(label+'.contact-valid-identity', float(cp['contact_valid']) == cp['aggregate'][4]
                == cp['last_contact'][15] == 1., 'Contact, aggregate and metric validity must all be true')
        for archive in ('checkpoint.npz', 'final_state_0.npz'):
            for key in sorted(data['segmented'][archive]):
                x, y = data['segmented'][archive][key], data['reference'][archive][key]
                label = archive+':'+key
                field = key.removeprefix('state_') if archive == 'checkpoint.npz' else key
                if key == 'solver_seconds':
                    continue
                if key == 'tables':
                    continue
                if field in FINAL_FIELDS:
                    _state_field(c, label, field, x, y, ctx, left, right)
                elif key in ('total_impulse', 'clamp_work', 'grid_ke_work'):
                    for group in range(2):
                        c.numeric(f'{label}[{group}]', x[group], y[group], units='N s' if key == 'total_impulse' else 'J',
                            method='one fixture xyz' if key == 'total_impulse' else 'one independent work channel')
                elif key == 'aggregate':
                    c.require(label+'.shape', x.shape == y.shape == (6,), 'Unknown aggregate shape')
                    for index in range(6):
                        if index == 2: c.numeric(f'{label}[{index}]', x[index], y[index], units='N')
                        else: c.exact(f'{label}[{index}]', x[index], y[index])
                elif key == 'last_contact':
                    _contact(c, label, x, y, ctx)
                else:
                    c.exact(label, x, y)
        for archive in ('particle_fields.npz', 'trajectory.npz'):
            for key in data['segmented'][archive]:
                c.exact(archive+':'+key, data['segmented'][archive][key], data['reference'][archive][key])
        _observables(c, a['obs'], b['obs'], ctx)
        for key in sorted(DIAGNOSTIC_FIELDS):
            x, y = _column(a['diag'], key), _column(b['diag'], key)
            if key == 'current_volume': c.numeric('diagnostics.'+key, x, y, units='m^3')
            elif key in ('min_J', 'max_J'):
                c.numeric('diagnostics.'+key, x, y, units='1', scale=max(magnitude(x-1), magnitude(y-1)),
                    floor=SPECTRAL_ALLOWANCE*EPSILON*ctx['maximum_singular_stretch']**3, method='J-1 plus determinant roundoff')
            elif key == 'momentum_balance_residual_Ns':
                scale = max(_group_scale(a['obs'], b['obs'], side, 'accumulated_impulse', 'Ns') for side in ('left', 'right'))
                c.numeric('diagnostics.'+key, x, y, units='N s', scale=scale, method='uncancelled fixture-impulse scale; final identity separately checked')
            else: c.exact('diagnostics.'+key, x, y)
        _contact(c, 'contact-metrics', np.asarray(a['contact']['values']), np.asarray(b['contact']['values']), ctx)
        for label, d in data.items():
            c.exact(label+'.contact-record', d['checkpoint.npz']['last_contact'], np.asarray(d['contact']['values']))
            tables = json.loads(str(d['checkpoint.npz']['tables']))
            c.require(label+'.checkpoint-tables', tables == {'observables': d['obs'], 'diagnostics': d['diag'],
                'times': d['trajectory.npz']['time'].tolist()}, 'Checkpoint tables do not equal the saved CSV/index')
            _identities(c, label, d['checkpoint.npz'], d['final_state_0.npz'], d['trajectory.npz'], d['obs'], d['diag'], d['plan'])
        _histories(c, folders, data, ctx)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
        c.require('read-or-schema', False, str(error))
    return _result(c, ctx, inputs)


def _histories(c, folders, data, ctx):
    for key in ('x', 'speed', 'damage', 'failed', 'temperature'):
        maps = []
        try:
            for folder in folders.values():
                maps.append(np.load(folder/f'trajectory_{key}.npy', mmap_mode='r', allow_pickle=False))
            a, b = maps
            frames, particles = data['segmented']['plan']['saved_states'], len(data['segmented']['checkpoint.npz']['state_mass'])
            shape = (frames, particles, 3) if key == 'x' else (frames, particles)
            dtype = np.dtype('uint8' if key == 'failed' else 'float32' if key == 'temperature' else 'float64')
            if not c.require('trajectory_'+key+'.shape', a.shape == b.shape == shape and a.dtype == b.dtype == dtype,
                             'History has missing frames, particles or dtype mismatch'):
                continue
            for frame in range(frames):
                name = f'trajectory_{key}[{frame}]'
                if key == 'x':
                    c.numeric(name, a[frame], b[frame], units='m', scale=ctx['dx_m'], method='dx')
                    if frame == 0: c.exact(name+'.initial', a[frame], b[frame])
                elif key == 'speed': c.numeric(name, a[frame], b[frame], units='m/s')
                else:
                    c.exact(name, a[frame], b[frame])
                    if key in ('damage', 'failed'):
                        c.require(name+'.predamage', np.all(a[frame] == 0) and np.all(b[frame] == 0), 'Tiny fixture must stay predamage')
            for label, values in zip(folders, maps):
                cp = data[label]['checkpoint.npz']
                if key in ('x', 'damage', 'failed'):
                    field = 'state_x' if key == 'x' else 'history_'+key
                    expected = cp[field].astype(values.dtype)
                    c.exact(label+'.final-history-'+key, values[-1], expected)
                if key == 'failed':
                    c.exact(label+'.failed-index', values, data[label]['trajectory.npz']['failed'])
                if key == 'speed':
                    expected = np.linalg.norm(cp['state_v'], axis=1)
                    c.identity(label+'.final-speed', values[-1], expected, units='m/s', budget=16*EPSILON*magnitude(expected))
        finally:
            for array in maps:
                array._mmap.close()


def _result(c, ctx, inputs):
    return {'schema': 'triaxial-tiny-cross-cpu-parity-v1', 'passed': not c.errors,
        'production_eligible': False, 'policy': {'relative_tolerance': RELATIVE,
            'float64_epsilon': EPSILON, 'spectral_allowance': SPECTRAL_ALLOWANCE,
            'allowance_scope': 'Engineering numerical budget; not a certified spectral backward-error bound.',
            'context': ctx}, 'metrics': c.metrics, 'errors': c.errors,
        'input_sha256': inputs, 'review_source_sha256': digest(__file__),
        'ignored_fields': ['checkpoint.npz:solver_seconds'],
        'scope': 'Predamage, inactive-contact tiny numerical continuation comparison only. Independent record/state/full-history audits remain required. Earlier saved scalar residuals are record comparisons, not independent reconstruction of missing v/F/C histories.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segmented', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', required=True, type=Path,
        help='Write a new comparison report; existing files are never replaced')
    args = parser.parse_args()
    report = compare_runs(args.segmented, args.reference)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'errors': len(report['errors']), 'output': str(args.output)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
