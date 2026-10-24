import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('triaxial_parity', Path(__file__).parents[1]/'tools/triaxial_parity.py')
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _replace(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(changes)
    np.savez(path, **values)


def _dataset(folder):
    folder.mkdir()
    n, frames, dt = 2, 3, 1e-8
    material = {'young': 1e10, 'poisson': .25, 'density': 1500., 'tensile_strength': 1e8,
                'fracture_energy': 100., 'characteristic_length': 75e-6}
    plan = {'scope': 'smoke', 'material': material, 'grid': {'dx_m': 75e-6}, 'particles': n,
        'saved_states': frames, 'frame_steps': [0, 1, 2], 'dt': dt, 'stop_steps': 2, 'source': {'commit': 'fixture'}}
    times = np.arange(frames, dtype=float)*dt
    state = {'x': np.array([[.001, .002, .003], [.002, .003, .004]]),
        'v': np.zeros((n, 3)), 'F': np.repeat(np.eye(3)[None], n, axis=0), 'C': np.zeros((n, 3, 3)),
        'mass': np.full(n, 1e-6), 'volume': np.full(n, 1e-6/1500), 'fibre_id': np.arange(n, dtype=np.int32),
        'tau': np.zeros((n, 3, 3)), 'time': np.asarray(times[-1]), 'valid': np.asarray(True),
        'boundary_impulse': np.zeros(3), 'external_impulse': np.zeros(3), 'boundary_work': np.asarray(0.),
        'history_kappa': np.zeros(n), 'history_damage': np.zeros(n), 'history_failed': np.zeros(n, bool),
        'history_valid': np.ones(n, bool)}
    contact = np.zeros(16)
    contact[5], contact[15] = 1e-5, 1.
    observations, diagnostics = [], []
    for frame, t in enumerate(times):
        row = dict.fromkeys(sorted(parity.OBSERVABLE_FIELDS), 0.)
        row.update(time=float(t), accepted_steps=float(frame), first_material_failure_time_s=None,
            kinetic_to_elastic_ratio=None, sampled_section_graph_connects_grips=True, minimum_contact_gap_m=1e-5)
        observations.append(row)
        diagnostic = dict.fromkeys(sorted(parity.DIAGNOSTIC_FIELDS), 0.)
        diagnostic.update(time=float(t), mass=float(state['mass'].sum()), reference_volume=float(state['volume'].sum()),
            current_volume=float(state['volume'].sum()), min_J=1., max_J=1., state_valid=True, coupling_valid=True)
        diagnostics.append(diagnostic)
    checkpoint = {'state_'+key: value for key, value in state.items() if key in parity.STATE_FIELDS}
    checkpoint.update({key: state[key] for key in parity.HISTORY_FIELDS})
    checkpoint.update(contact_active=np.zeros(1, bool), contact_law_valid=np.ones(1, bool),
        contact_radius=np.zeros(1), contact_segments=np.zeros((1, 2), np.int64), contact_valid=np.asarray(True),
        total_impulse=np.zeros((2, 3)), clamp_work=np.zeros(2), grid_ke_work=np.zeros(2),
        aggregate=np.array([2., -1., 0., 0., 1., 0.]), last_contact=contact, initial_momentum=np.zeros(3),
        frame=np.asarray(frames-1), signature=np.asarray('same-physical-plan'), solver_seconds=np.asarray(1.),
        jax_version=np.asarray('fixture'), tables=np.asarray(json.dumps({'observables': observations,
            'diagnostics': diagnostics, 'times': times.tolist()})))
    np.savez(folder/'checkpoint.npz', **checkpoint)
    np.savez(folder/'final_state_0.npz', **state)
    np.savez(folder/'particle_fields.npz', phase=np.ones(n, np.int8), yarn_id=np.arange(n, dtype=np.int32), family_id=np.arange(n, dtype=np.int16))
    np.savez(folder/'trajectory.npz', time=times, body=np.zeros(n, np.int32), failed=np.zeros((frames, n), np.uint8))
    for key, array in {'x': np.repeat(state['x'][None], frames, axis=0), 'speed': np.zeros((frames, n)),
        'damage': np.zeros((frames, n)), 'failed': np.zeros((frames, n), np.uint8),
        'temperature': np.full((frames, n), 293.15, np.float32)}.items():
        np.save(folder/f'trajectory_{key}.npy', array)
    _json(folder/'plan.json', plan)
    _json(folder/'summary.json', {'completed': True, 'materials': [material], 'configs': [{'dt': dt}],
        'source': plan['source'], 'definition': dict(plan, contact_model={'particles': n, 'candidate_pairs': 1})})
    _json(folder/'contact-metrics.json', {'fields': list(parity.CONTACT_FIELDS), 'values': contact.tolist()})
    _csv(folder/'observables.csv', observations)
    _csv(folder/'diagnostics.csv', diagnostics)
    return folder


@pytest.fixture
def pair(tmp_path):
    return _dataset(tmp_path/'segmented'), _dataset(tmp_path/'reference')


def test_same_complete_fixture_has_evidence_and_is_never_production(pair):
    report = parity.compare_runs(*pair)
    assert report['passed'], report['errors']
    assert not report['production_eligible']
    assert len(report['metrics']) > 150
    assert 'trajectory_failed.npy' in report['input_sha256']['segmented']
    assert report['policy']['relative_tolerance'] == 2e-10


def connected_pair(pair):
    for folder in pair:
        plan = json.loads((folder/'plan.json').read_text())
        plan.update(law='hertz', contact_regions=dict(mode='connected', capacity=16, max_patches=4, max_iterations=32))
        _json(folder/'plan.json', plan)
        summary = json.loads((folder/'summary.json').read_text())
        summary['definition'] = dict(plan, contact_model=dict(particles=2, candidate_pairs=1,
                                                              contact_regions=plan['contact_regions']))
        _json(folder/'summary.json', summary)
        _replace(folder/'checkpoint.npz', contact_active=np.zeros(4, bool), contact_law_valid=np.ones(4, bool),
            contact_radius=np.zeros(4), contact_segments=np.full((4, 2), -1, dtype=np.int32))
    return pair


def test_connected_hertz_history_slots_do_not_replace_the_true_seed_count(pair):
    pair = connected_pair(pair)
    report = parity.compare_runs(*pair)
    assert report['passed'], report['errors']
    assert report['policy']['relative_tolerance'] == 2e-10
    assert not report['production_eligible']


@pytest.mark.parametrize('defect', ['wrong_slots', 'wrong_segment_dtype', 'wrong_law', 'missing_budget',
                                  'different_summary', 'unknown_mode'])
def test_connected_contract_is_explicit_and_symmetric_corruption_is_still_rejected(pair, defect):
    pair = connected_pair(pair)
    for folder in pair:
        if defect == 'wrong_slots':
            _replace(folder/'checkpoint.npz', contact_radius=np.zeros(1))
        elif defect == 'wrong_segment_dtype':
            _replace(folder/'checkpoint.npz', contact_segments=np.full((4, 2), -1, dtype=np.int64))
        else:
            plan = json.loads((folder/'plan.json').read_text())
            summary = json.loads((folder/'summary.json').read_text())
            if defect == 'wrong_law':
                plan['law'] = 'jkr'
            elif defect == 'missing_budget':
                del plan['contact_regions']['capacity']
            elif defect == 'different_summary':
                summary['definition']['contact_model']['contact_regions']['max_patches'] = 1
            else:
                plan['contact_regions']['mode'] = 'unknown'
            _json(folder/'plan.json', plan)
            _json(folder/'summary.json', summary)
    report = parity.compare_runs(*pair)
    assert not report['passed']


def test_near_zero_vector_components_use_the_same_vector_scale():
    c = parity.Comparison()
    assert c.numeric('velocity', np.array([1., 1e-20, 0.]), np.array([1., 2e-20, 0.]), units='m/s')
    assert c.metrics[-1]['scale'] == 1.


@pytest.mark.parametrize('key,loss', [('total_impulse', 1e-14), ('clamp_work', 1e-16), ('clamp_work', 1e-30)])
def test_independently_zeroed_small_impulse_and_work_are_rejected(pair, key, loss):
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = z[key].copy()
    value.flat[0] = loss
    _replace(pair[0]/'checkpoint.npz', **{key: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    metric = next(m for m in report['metrics'] if m['field'] == f'checkpoint.npz:{key}[0]')
    assert not metric['passed']
    assert metric['allowed_error'] < loss/1e6


@pytest.mark.parametrize('field,change', [('F', 1e-11), ('tau', 1e-2), ('history_kappa', 1e-10)])
def test_spectral_budgets_reject_resolved_field_distortion(pair, field, change):
    key = 'state_'+field if field in parity.STATE_FIELDS else field
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = z[key].copy()
    value.flat[0] += change
    _replace(pair[0]/'checkpoint.npz', **{key: value})
    _replace(pair[0]/'final_state_0.npz', **{field: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    metric = next(m for m in report['metrics'] if m['field'] == 'checkpoint.npz:'+key)
    assert not metric['passed']


def test_F_budget_uses_strain_not_identity_diagonal(pair):
    report = parity.compare_runs(*pair)
    metric = next(m for m in report['metrics'] if m['field'] == 'checkpoint.npz:state_F')
    assert metric['scale'] == 0.
    assert metric['allowed_error'] == 64*np.finfo(float).eps


def test_position_budget_does_not_follow_coordinate_origin():
    for origin in (0., 1e3):
        c = parity.Comparison()
        assert not c.numeric('x', np.array([origin]), np.array([origin+1e-9]), units='m', scale=75e-6)
        assert c.metrics[0]['allowed_error'] == 2e-10*75e-6


@pytest.mark.parametrize('field', ['state_time', 'state_mass', 'state_fibre_id', 'contact_segments'])
def test_time_static_and_discrete_changes_are_exact(pair, field):
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = z[field].copy()
    value.flat[0] += 1 if value.dtype.kind in 'iu' else 1e-8
    _replace(pair[0]/'checkpoint.npz', **{field: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'checkpoint.npz:'+field for e in report['errors'])


@pytest.mark.parametrize('index', [0, 1, 3, 4, 5])
def test_mixed_aggregate_exact_fields_cannot_hide_beside_step_count(pair, index):
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = z['aggregate'].copy()
    value[index] += 1e-8
    _replace(pair[0]/'checkpoint.npz', aggregate=value)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == f'checkpoint.npz:aggregate[{index}]' for e in report['errors'])


def test_contact_potential_cannot_hide_beside_valid_flag(pair):
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = z['last_contact'].copy()
    value[14] = 1e-30
    _replace(pair[0]/'checkpoint.npz', last_contact=value)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'].endswith('last_contact.potential_J') for e in report['errors'])


@pytest.mark.parametrize('operation', ['missing', 'unknown'])
def test_archive_registry_fails_closed_even_when_both_datasets_match(pair, operation):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            values = {key: z[key] for key in z.files}
        if operation == 'missing': del values['state_C']
        else: values['unregistered_energy'] = np.asarray(1e-16)
        np.savez(folder/'checkpoint.npz', **values)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any('schema' in e['field'] for e in report['errors'])


def test_missing_file_is_reported_without_replacing_any_input(pair):
    before = parity.digest(pair[0]/'trajectory_x.npy')
    (pair[0]/'trajectory_damage.npy').unlink()
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert parity.digest(pair[0]/'trajectory_x.npy') == before


def test_missing_history_frame_rejected(pair):
    path = pair[0]/'trajectory_x.npy'
    np.save(path, np.load(path)[:-1])
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'trajectory_x.shape' for e in report['errors'])


@pytest.mark.parametrize('field,value', [('history_damage', .1), ('contact_active', True)])
def test_scope_does_not_silently_admit_damaged_or_active_contact(pair, field, value):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            modified = z[field].copy()
        modified.flat[0] = value
        _replace(folder/'checkpoint.npz', **{field: modified})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'].endswith('.predamage-inactive') for e in report['errors'])


def _change_tables(folder, category, mutate):
    with np.load(folder/'checkpoint.npz') as z:
        tables = json.loads(str(z['tables']))
    mutate(tables[category])
    _replace(folder/'checkpoint.npz', tables=np.asarray(json.dumps(tables)))
    _csv(folder/f'{category}.csv', tables[category])


def test_equal_corrupted_ratios_fail_their_own_identity(pair):
    def corrupt(rows):
        for row in rows:
            row.update(kinetic_energy_J=1e-12, kinetic_translational_energy_J=1e-12,
                       elastic_energy_J=1e-12, kinetic_to_elastic_ratio=2.)
    for folder in pair:
        _change_tables(folder, 'observables', corrupt)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'].endswith('.ratio') for e in report['errors'])


def test_equal_corrupted_momentum_residuals_fail_final_raw_state_identity(pair):
    for folder in pair:
        _change_tables(folder, 'diagnostics', lambda rows: rows[-1].update(momentum_balance_residual_Ns=1e-16))
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'].endswith('.final-momentum-residual') for e in report['errors'])


def test_unknown_csv_column_cannot_bypass_registry(pair):
    for folder in pair:
        _change_tables(folder, 'observables', lambda rows: [row.update(unknown_residual=0.) for row in rows])
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'observables.columns' for e in report['errors'])


def test_multiple_numerical_failures_are_retained(pair):
    with np.load(pair[0]/'checkpoint.npz') as z:
        tau, kappa = z['state_tau'].copy(), z['history_kappa'].copy()
    tau.flat[0], kappa.flat[0] = 1., 1e-8
    _replace(pair[0]/'checkpoint.npz', state_tau=tau, history_kappa=kappa)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    fields = {e['field'] for e in report['errors']}
    assert {'checkpoint.npz:state_tau', 'checkpoint.npz:history_kappa'} <= fields


def test_memory_maps_closed_after_comparison(pair):
    assert parity.compare_runs(*pair)['passed']
    # Windows refuses this rename while an mmap remains open.
    original = pair[0]/'trajectory_x.npy'
    replacement = pair[0]/'renamed.npy'
    original.rename(replacement)
    replacement.rename(original)


def test_large_other_work_channel_cannot_mask_small_work_loss(pair):
    _replace(pair[0]/'checkpoint.npz', clamp_work=np.array([1., 1e-16]))
    _replace(pair[1]/'checkpoint.npz', clamp_work=np.array([1., 0.]))
    report = parity.compare_runs(*pair)
    metric = next(m for m in report['metrics'] if m['field'] == 'checkpoint.npz:clamp_work[1]')
    assert not metric['passed']
    assert metric['scale'] == 1e-16


def test_identical_invalid_checkpoints_are_not_accepted(pair):
    for folder in pair:
        _replace(folder/'checkpoint.npz', state_valid=np.asarray(False))
        _replace(folder/'final_state_0.npz', valid=np.asarray(False))
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'segmented.valid' for e in report['errors'])


def test_identical_unknown_history_dtype_is_not_accepted(pair):
    for folder in pair:
        path = folder/'trajectory_failed.npy'
        np.save(path, np.load(path).astype(np.float64))
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'trajectory_failed.shape' for e in report['errors'])


def test_identical_missing_source_metadata_is_not_accepted(pair):
    for folder in pair:
        path = folder/'summary.json'
        value = json.loads(path.read_text())
        del value['source']
        _json(path, value)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'segmented.summary-fields' for e in report['errors'])


def test_saved_active_contact_cannot_hide_behind_final_inactive_state(pair):
    for folder in pair:
        _change_tables(folder, 'observables', lambda rows: rows[1].update(active_contact_pairs=1.))
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'observables.active_contact_pairs.scope' for e in report['errors'])


@pytest.mark.parametrize('key,shape', [
    ('clamp_work', (3,)), ('grid_ke_work', (3,)), ('total_impulse', (3, 3)),
    ('aggregate', (7,)), ('last_contact', (17,)), ('initial_momentum', (4,)),
    ('state_boundary_impulse', (4,)), ('state_external_impulse', (4,)),
    ('state_mass', (3,)), ('state_volume', (3,)), ('state_x', (3, 3)),
    ('state_v', (2, 4)), ('state_F', (2, 4, 3)), ('state_C', (2, 3, 4)),
    ('state_tau', (3, 3, 3)), ('history_kappa', (3,)), ('history_damage', (3,)),
    ('history_failed', (3,)), ('history_valid', (3,)), ('state_fibre_id', (3,)),
    ('contact_active', (2,)), ('contact_radius', (2,)), ('contact_law_valid', (2,)),
    ('contact_segments', (2, 2)), ('contact_valid', (1,)),
    ('state_time', (1,)), ('state_valid', (1,)), ('state_boundary_work', (1,)),
    ('frame', (1,)), ('solver_seconds', (1,)), ('signature', (1,)),
    ('jax_version', (1,)), ('tables', (1,)),
])
def test_complete_checkpoint_shapes_reject_extra_zero_storage(pair, key, shape):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            value = z[key]
        _replace(folder/'checkpoint.npz', **{key: np.resize(value, shape)})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == f'segmented.checkpoint.npz:{key}.layout' for e in report['errors'])


@pytest.mark.parametrize('key,shape', [('clamp_work', (3,)), ('total_impulse', (3, 3))])
def test_peer_one_sided_zero_tail_regressions(pair, key, shape):
    with np.load(pair[0]/'checkpoint.npz') as z:
        value = np.resize(z[key], shape)
    _replace(pair[0]/'checkpoint.npz', **{key: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == f'segmented.checkpoint.npz:{key}.layout' for e in report['errors'])


@pytest.mark.parametrize('archive,key,shape', [
    ('final_state_0.npz', 'boundary_impulse', (4,)),
    ('final_state_0.npz', 'mass', (3,)),
    ('particle_fields.npz', 'phase', (3,)),
    ('particle_fields.npz', 'yarn_id', (3,)),
    ('particle_fields.npz', 'family_id', (3,)),
    ('trajectory.npz', 'time', (4,)),
    ('trajectory.npz', 'body', (3,)),
    ('trajectory.npz', 'failed', (3, 3)),
])
def test_other_archives_enforce_plan_point_and_frame_counts(pair, archive, key, shape):
    for folder in pair:
        with np.load(folder/archive) as z:
            value = np.resize(z[key], shape)
        _replace(folder/archive, **{key: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == f'segmented.{archive}:{key}.layout' for e in report['errors'])


@pytest.mark.parametrize('key,dtype', [
    ('state_mass', 'float32'), ('state_F', 'float32'), ('state_fibre_id', 'int64'),
    ('history_failed', 'uint8'), ('contact_segments', 'int32'), ('contact_valid', 'uint8'),
    ('aggregate', 'float32'), ('total_impulse', 'float32'), ('frame', 'float64'),
])
def test_matching_wrong_checkpoint_dtypes_are_rejected(pair, key, dtype):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            value = z[key].astype(dtype)
        _replace(folder/'checkpoint.npz', **{key: value})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == f'segmented.checkpoint.npz:{key}.layout' for e in report['errors'])


@pytest.mark.parametrize('index', [0, 1, 2, 3, 9, 10, 11, 15])
def test_matching_invalid_contact_metrics_are_not_valid_results(pair, index):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            value = z['last_contact'].copy()
        value[index] = 0. if index == 15 else 1.
        _replace(folder/'checkpoint.npz', last_contact=value)
        _json(folder/'contact-metrics.json', {'fields': list(parity.CONTACT_FIELDS), 'values': value.tolist()})
    report = parity.compare_runs(*pair)
    assert not report['passed']
    suffix = '.valid' if index == 15 else '.zero-errors'
    assert any(e['field'] == 'checkpoint.npz:last_contact.'+parity.CONTACT_FIELDS[index]+suffix for e in report['errors'])
    if index == 15:
        assert any(e['field'] == 'segmented.contact-valid-identity' for e in report['errors'])


@pytest.mark.parametrize('field,value', [('young', np.inf), ('young', np.nan), ('poisson', np.inf), ('poisson', np.nan), ('dx_m', np.inf), ('dx_m', np.nan)])
def test_context_rejects_nonfinite_material_and_grid_values(field, value):
    plan = {'material': {'young': 1e10, 'poisson': .25}, 'grid': {'dx_m': 75e-6}}
    plan['grid' if field == 'dx_m' else 'material'][field] = value
    state = {'state_F': np.eye(3)[None]}
    with pytest.raises(ValueError, match='Invalid elastic'):
        parity.context(state, state, plan)


def test_context_rejects_empty_deformation_field():
    plan = {'material': {'young': 1e10, 'poisson': .25}, 'grid': {'dx_m': 75e-6}}
    with pytest.raises(ValueError, match='deformation gradients'):
        parity.context({'state_F': np.empty((0, 3, 3))}, {'state_F': np.eye(3)[None]}, plan)


def test_equal_invalid_diagnostic_flags_are_not_accepted(pair):
    for folder in pair:
        _change_tables(folder, 'diagnostics', lambda rows: rows[-1].update(coupling_valid=False))
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'segmented.diagnostics[2].valid' for e in report['errors'])


def test_fractional_accepted_step_cannot_be_truncated_into_validity(pair):
    for folder in pair:
        with np.load(folder/'checkpoint.npz') as z:
            value = z['aggregate'].copy()
        value[0] = 2.5
        _replace(folder/'checkpoint.npz', aggregate=value)
    report = parity.compare_runs(*pair)
    assert not report['passed']
    assert any(e['field'] == 'segmented.final-step' for e in report['errors'])
