from typing import NamedTuple
from types import SimpleNamespace

import numpy as np
import pytest

from examples.fibre_contact_common import restore_contact_history
from examples.fibre_pull_off import plan_case


class RingHistory(NamedTuple):
    active: np.ndarray
    valid: np.ndarray
    gap: np.ndarray
    indentation: np.ndarray
    previous_profile: np.ndarray


def saved_history():
    template = RingHistory(np.zeros(2, dtype=bool), np.ones(2, dtype=bool),
        np.zeros((2, 4)), np.zeros(2), np.ones((2, 4)))
    saved = {'contact_active': np.array([True, False]), 'contact_law_valid': np.ones(2, dtype=bool),
        'contact_local_gap': np.arange(8, dtype=float).reshape(2, 4)*1e-6,
        'contact_local_indentation': np.array([1e-6, -2e-6]),
        'contact_local_previous_profile': np.arange(8, dtype=float).reshape(2, 4)*2e-6}
    return template, saved


def test_ring_history_restores_every_branch_leaf_including_previous_geometry():
    template, saved = saved_history()
    restored = restore_contact_history(template, saved)
    np.testing.assert_array_equal(restored.gap, saved['contact_local_gap'])
    np.testing.assert_array_equal(restored.indentation, saved['contact_local_indentation'])
    np.testing.assert_array_equal(restored.previous_profile, saved['contact_local_previous_profile'])
    np.testing.assert_array_equal(restored.active, [True, False])


@pytest.mark.parametrize('defect', [None, 'negative_profile', 'noncircular_profile', 'branch_flag'])
def test_saved_self_consistent_geometry_and_branch_must_match_the_ring_history(defect):
    template, saved = saved_history()
    base = np.arange(1, 5, dtype=float)
    saved['contact_local_previous_profile'] = np.array([base*1e-6, base*2e-6])
    parameters = SimpleNamespace(base=SimpleNamespace(profile=base),
        interaction_range=4e-6, work_of_adhesion=.1)
    if defect == 'negative_profile':
        saved['contact_local_previous_profile'][0] *= -1
    elif defect == 'noncircular_profile':
        saved['contact_local_previous_profile'][1, 2] *= 2
    elif defect == 'branch_flag':
        saved['contact_active'][0] = False
    if defect is None:
        restored = restore_contact_history(template, saved, parameters=parameters)
        np.testing.assert_array_equal(restored.active, [True, False])
    else:
        with pytest.raises(ValueError):
            restore_contact_history(template, saved, parameters=parameters)


@pytest.mark.parametrize('defect', ['missing_gap', 'missing_profile', 'bad_shape', 'nan_gap',
                                  'negative_gap', 'invalid_flag', 'wrong_flag_dtype'])
def test_incomplete_or_invalid_adhesive_history_is_not_reset_to_a_fresh_state(defect):
    template, saved = saved_history()
    if defect == 'missing_gap':
        del saved['contact_local_gap']
    elif defect == 'missing_profile':
        del saved['contact_local_previous_profile']
    elif defect == 'bad_shape':
        saved['contact_local_gap'] = np.zeros((2, 3))
    elif defect == 'nan_gap':
        saved['contact_local_gap'][0, 0] = np.nan
    elif defect == 'negative_gap':
        saved['contact_local_gap'][0, 0] = -1.
    elif defect == 'invalid_flag':
        saved['contact_law_valid'][0] = False
    else:
        saved['contact_active'] = saved['contact_active'].astype(float)
    with pytest.raises(ValueError):
        restore_contact_history(template, saved)


@pytest.mark.parametrize('law', ['md', 'self-consistent'])
def test_new_fibre_contact_plan_records_independent_contact_range_and_domain(law):
    _, plan = plan_case(model=law, dx=.002, spacing=.001, interaction_range=2e-6,
                        outer_radius=150e-6, rings=32)
    local = plan['contact']
    assert local['interaction_range_m'] == 2e-6
    assert local['outer_radius_m'] == 150e-6 and local['rings'] == 32
    assert local['work_of_adhesion_J_per_m2'] == 1.2
    assert {'mpm_jax/local_contact_models.py', 'mpm_jax/maugis_dugdale.py',
            'mpm_jax/cohesive_bem.py'} <= plan['source']['files'].keys()


@pytest.mark.parametrize('kwargs', [{'interaction_range': 0.}, {'outer_radius': np.nan},
                                   {'rings': 3}, {'rings': True}])
def test_invalid_local_contact_controls_are_refused_before_geometry_allocation(kwargs):
    with pytest.raises(ValueError):
        plan_case(model='self-consistent', **kwargs)
