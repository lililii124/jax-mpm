from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.triaxial_failure import output_directory, plan_case, restore_contact_history, restore_contact_model, run_case
from mpm_jax.fibre_coupling import build_contact_data, evaluate_contact, initial_contact_history
from mpm_jax.local_contact_models import local_contact_model

jax.config.update('jax_enable_x64', True)


class History(NamedTuple):
    failed: jax.Array


class State(NamedTuple):
    x: jax.Array
    mass: jax.Array
    history: History
    valid: jax.Array


def pair_fixture(*, proximity_margin=.0002, separation=.095, registered=True):
    t = np.linspace(-.4, .4, 9)
    first = np.column_stack((t, np.zeros(9), np.zeros(9)))
    second = np.column_stack((np.zeros(9), np.full(9, separation), t))
    offsets = np.array([[a, b] for a in (-.01, .01) for b in (-.01, .01)])
    x = np.concatenate(((first[:, None]+np.column_stack((np.zeros(4), offsets))).reshape(-1, 3),
        (second[:, None]+np.column_stack((offsets[:, 0], offsets[:, 1], np.zeros(4)))).reshape(-1, 3)))
    geometry = {'x': x, 'mass': np.ones(72), 'fibre_id': np.repeat(np.arange(2, dtype=np.int32), 36),
        'metadata': {'cross_section_particles': 4, 'fibres': [
            {'id': i, 'family_id': i, 'longitudinal_sections': 9, 'particle_count': 36, 'radius_m': .05} for i in range(2)]},
        'contact_candidates': [{'fibre_ids': [0, 1], 'reference_centreline_points_m': [[0., 0., 0.], [0., separation, 0.]]}] if registered else []}
    data = build_contact_data(geometry, coverage_skin=.1, search_window=2, proximity_margin=proximity_margin)
    state = State(jnp.asarray(x), jnp.ones(72), History(jnp.zeros(72, dtype=bool)), jnp.asarray(True))
    return state, data


@pytest.mark.parametrize('model', ['md', 'self-consistent'])
def test_finite_range_laws_reject_a_search_list_built_only_for_solid_touch(model):
    state, data = pair_fixture(proximity_margin=0.)
    law = local_contact_model(model, effective_modulus=1000., effective_radius=.05,
        work_of_adhesion=.002, interaction_range=.0002, outer_radius=.04, rings=16)
    history = initial_contact_history(data, law)
    _, _, metrics = jax.jit(evaluate_contact)(state, history, data, law)
    assert not bool(metrics.valid) and np.all(metrics.search_boundary)


@pytest.mark.parametrize('model', ['md', 'self-consistent'])
def test_unregistered_positive_gap_adhesive_patch_is_not_silently_ignored(model):
    state, data = pair_fixture(separation=.1001, registered=False)
    law = local_contact_model(model, effective_modulus=1000., effective_radius=.05,
        work_of_adhesion=.002, interaction_range=.0002, outer_radius=.04, rings=16)
    history = initial_contact_history(data, law)
    _, _, metrics = jax.jit(evaluate_contact)(state, history, data, law)
    assert not bool(metrics.valid) and int(metrics.uncovered_near_pairs) > 0


@pytest.mark.parametrize('model', ['md', 'self-consistent'])
def test_adhesive_models_act_on_independent_fibres_with_balanced_actual_forces(model):
    state, data = pair_fixture()
    law = local_contact_model(model, effective_modulus=1000., effective_radius=.05,
        work_of_adhesion=.002, interaction_range=.0002, outer_radius=.04, rings=32)
    history = initial_contact_history(data, law)
    force, current, metrics = jax.jit(evaluate_contact)(state, history, data, law)
    assert bool(metrics.valid) and np.all(current.law.valid)
    assert float(metrics.normal_force[0]) > 0
    np.testing.assert_allclose(force.sum(0), 0., atol=1e-13)
    np.testing.assert_allclose(np.cross(state.x, force).sum(0), 0., atol=1e-13)
    saved = {'contact_active': np.asarray(current.law.active), 'contact_law_valid': np.asarray(current.law.valid)}
    saved.update({'contact_local_'+key: np.asarray(value) for key, value in current.law._asdict().items()
                  if key not in ('active', 'valid')})
    if hasattr(law, 'base'):
        saved.update({'contact_bem_'+key: np.asarray(value) for key, value in law.base._asdict().items()})
        restored_law = restore_contact_model(law, saved)
        np.testing.assert_array_equal(restored_law.base.compliance, law.base.compliance)
        damaged = dict(saved)
        damaged['contact_bem_compliance'] = saved['contact_bem_compliance']*1.01
        with pytest.raises(ValueError):
            restore_contact_model(law, damaged)
    restored = restore_contact_history(initial_contact_history(data, law).law, saved, jnp)
    separated = state._replace(x=state.x.at[36:, 1].add(.0001))
    original = jax.jit(evaluate_contact)(separated, current, data, law)
    resumed = jax.jit(evaluate_contact)(separated, current._replace(law=restored), data, law)
    np.testing.assert_array_equal(original[0], resumed[0])


@pytest.mark.parametrize('model', ['md', 'self-consistent'])
def test_actual_model_selected_driver_can_stop_and_resume_its_full_saved_protocol(tmp_path, model):
    geometry, plan = plan_case(smoke=True, law=model, grid_storage='tube', load_transits=.03,
        max_strain=1e-7, frames=4, cohesive_rings=16)
    root = tmp_path/model
    first = run_case(geometry, plan, root, stop_after_frame=2)
    assert first['segment_complete'] and not first['completed']
    folder = output_directory(root, plan)
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as saved:
        if model == 'self-consistent':
            assert saved['contact_local_gap'].shape[1] == 16
            assert 'contact_bem_compliance' in saved
    final = run_case(geometry, plan, root, resume=True)
    assert final['completed'] and final['steps'] == plan['steps'] == 13
    jax.clear_caches()
