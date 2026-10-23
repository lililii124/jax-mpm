from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.fibre_contact import fibre_contact_parameters
from mpm_jax.fibre_coupling import (
    build_contact_data, closest_segments, evaluate_contact, initial_contact_history,
    validate_contact,
)

jax.config.update('jax_enable_x64', True)


class History(NamedTuple):
    failed: jax.Array


class State(NamedTuple):
    x: jax.Array
    mass: jax.Array
    history: History
    valid: jax.Array


def fixture(*, height=.095, angle=np.pi/2, same_family=False, seed=True, end=False, **options):
    parameter = np.linspace(-.4, .4, 9)
    tangents = [np.array([1., 0., 0.]), np.array([np.cos(angle), 0., np.sin(angle)])]
    paths = [parameter[:, None]*tangents[0], parameter[:, None]*tangents[1]+[.4 if end else 0., height, 0.]]
    particles = []
    for path, tangent in zip(paths, tangents):
        first = np.array([0., 1., 0.])
        second = np.cross(tangent, first)
        offsets = np.array([a*first+b*second for a, b in ((-.01, -.01), (-.01, .01), (.01, -.01), (.01, .01))])
        particles.append((path[:, None, :]+offsets).reshape(-1, 3))
    x = np.concatenate(particles)
    ids = np.repeat(np.arange(2, dtype=np.int32), 36)
    geometry = {'x': x, 'mass': np.ones(72), 'fibre_id': ids,
        'metadata': {'cross_section_particles': 4, 'fibres': [
            {'id': i, 'family_id': 0 if same_family else i, 'longitudinal_sections': 9,
             'particle_count': 36, 'radius_m': .05} for i in range(2)]},
        'contact_candidates': ([{'fibre_ids': [0, 1], 'reference_centreline_points_m':
                                [[.4 if end else 0., 0., 0.], [.4 if end else 0., height, 0.]]}] if seed else [])}
    data = build_contact_data(geometry, coverage_skin=.1, search_window=2, **options)
    state = State(jnp.asarray(x), jnp.ones(72), History(jnp.zeros(72, dtype=bool)), jnp.asarray(True))
    return state, data, initial_contact_history(data), geometry


def law(model='hertz'):
    return fibre_contact_parameters(effective_modulus=1000., effective_radius=.05, work_of_adhesion=.02, model=model)


def test_finite_segment_closest_point_includes_endpoint_constraints():
    a, b, s, t, distance = closest_segments(jnp.array([0., 0., 0.]), jnp.array([1., 0., 0.]),
                                           jnp.array([2., 1., 0.]), jnp.array([2., 2., 0.]))
    np.testing.assert_allclose(a, [1., 0., 0.])
    np.testing.assert_allclose(b, [2., 1., 0.])
    np.testing.assert_allclose([s, t, distance], [1., 0., np.sqrt(2)])


@pytest.mark.parametrize('angle', [np.pi/2, np.pi/3])
def test_crossed_cylinder_equivalent_radius_uses_sum_of_curvatures(angle):
    state, data, history, _ = fixture(angle=angle)
    force, _, metrics = evaluate_contact(state, history, data, law())
    assert validate_contact(metrics)
    np.testing.assert_allclose(metrics.effective_radius, [.05/np.sin(angle)], rtol=2e-13)
    assert float(metrics.normal_force[0]) > 0
    assert float(force[:36, 1].sum()) < 0 < float(force[36:, 1].sum())
    np.testing.assert_allclose(metrics.force_residual, 0, atol=1e-14)
    np.testing.assert_allclose(metrics.torque_residual, 0, atol=1e-14)


def test_force_and_moment_balance_with_unequal_particle_masses():
    state, data, history, _ = fixture()
    state = state._replace(mass=jnp.tile(jnp.array([1., 2., 3., 4.]), 18))
    force, _, metrics = evaluate_contact(state, history, data, law())
    assert validate_contact(metrics)
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(np.cross(state.x, force).sum(axis=0), 0, atol=1e-14)
    for section in range(18):
        ids = np.arange(4*section, 4*section+4)
        np.testing.assert_allclose(force[ids]/state.mass[ids, None], np.broadcast_to(force[ids[0]]/state.mass[ids[0]], (4, 3)), atol=1e-14)


def test_rigid_motion_rotates_contact_normal_and_force_without_exhausting_skin():
    state, data, history, _ = fixture()
    force, _, old = evaluate_contact(state, history, data, law())
    axis = np.array([1., 2., 3.])/np.sqrt(14)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+np.sin(.7)*skew+(1-np.cos(.7))*(skew@skew)
    moved = state._replace(x=state.x@rotation.T+jnp.array([7., -2., 3.]))
    moved_force, _, new = jax.jit(evaluate_contact)(moved, history, data, law())
    assert validate_contact(new)
    assert float(new.maximum_nonrigid_displacement) < 1e-13
    np.testing.assert_allclose(new.gap, old.gap, atol=2e-15)
    np.testing.assert_allclose(new.normal, old.normal@rotation.T, atol=2e-13)
    np.testing.assert_allclose(moved_force, force@rotation.T, atol=3e-13)


def test_separation_and_jkr_attachment_history_are_distinct_states():
    state, data, history, _ = fixture()
    _, attached, _ = evaluate_contact(state, history, data, law('jkr'))
    separated = state._replace(x=state.x.at[36:, 1].add(.0051))
    force, _, bonded = evaluate_contact(separated, attached, data, law('jkr'))
    fresh_force, _, fresh = evaluate_contact(separated, history, data, law('jkr'))
    assert validate_contact(bonded) and validate_contact(fresh)
    assert bool(bonded.active[0]) and float(bonded.normal_force[0]) < 0
    assert not bool(fresh.active[0])
    assert np.linalg.norm(force) > 0
    np.testing.assert_array_equal(fresh_force, 0)


def test_search_window_exhaustion_is_a_refusal_not_zero_contact():
    state, data, history, _ = fixture()
    history = history._replace(segments=jnp.zeros_like(history.segments))
    _, updated, metrics = evaluate_contact(state, history, data, law())
    assert bool(metrics.search_boundary[0])
    assert not bool(updated.valid)
    with pytest.raises(ValueError, match='search_boundary'):
        validate_contact(metrics)


def test_previously_unseeded_same_family_contact_is_detected():
    state, data, history, _ = fixture(height=.12, angle=0., same_family=True, seed=False)
    _, _, separated = evaluate_contact(state, history, data, law())
    assert validate_contact(separated)
    moved = state._replace(x=state.x.at[36:, 1].add(-.04))
    _, _, metrics = evaluate_contact(moved, history, data, law())
    assert int(metrics.same_family_near_pairs) > 0
    assert int(metrics.uncovered_near_pairs) > 0
    assert not bool(metrics.coverage_motion_exceeded)
    assert not bool(metrics.valid)


def test_unseeded_cross_family_contact_is_not_treated_as_separation():
    state, data, history, _ = fixture(seed=False)
    _, _, metrics = evaluate_contact(state, history, data, law())
    assert int(metrics.uncovered_near_pairs) > 0
    assert int(metrics.same_family_near_pairs) == 0
    assert not bool(metrics.valid)


def test_large_nonrigid_motion_expires_neighbour_coverage():
    state, data, history, _ = fixture(height=.12, seed=False)
    moved = state._replace(x=state.x.at[36:, 1].add(.3))
    _, _, metrics = evaluate_contact(moved, history, data, law())
    assert bool(metrics.coverage_motion_exceeded)
    assert not bool(metrics.valid)


def test_failed_sections_and_physical_end_caps_do_not_receive_valid_contact():
    state, data, history, _ = fixture()
    damaged = state._replace(history=History(state.history.failed.at[16].set(True)))
    _, _, metrics = evaluate_contact(damaged, history, data, law())
    assert bool(metrics.failed_section_contact[0]) and not bool(metrics.valid)
    state, data, history, _ = fixture(end=True)
    _, _, metrics = evaluate_contact(state, history, data, law())
    assert bool(metrics.unsupported_end_contact[0]) and not bool(metrics.valid)


def test_section_membership_and_neighbour_budget_are_validated_before_allocation():
    _, _, _, geometry = fixture()
    geometry['fibre_id'][35] = 1
    with pytest.raises(ValueError, match='sections'):
        build_contact_data(geometry)
    _, _, _, geometry = fixture()
    with pytest.raises(ValueError, match='max_coverage_pairs'):
        build_contact_data(geometry, max_coverage_pairs=1)
