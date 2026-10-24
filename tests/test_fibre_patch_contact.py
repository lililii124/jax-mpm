import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.contact_patches import patch_neighbours
from mpm_jax.fibre_contact import fibre_contact_parameters
from mpm_jax.fibre_coupling import build_contact_data, evaluate_contact, initial_contact_history
from mpm_jax.fibre_patch_contact import initial_patch_history, evaluate_hertz_patches, _overlap_count

jax.config.update("jax_enable_x64", True)


class Damage(NamedTuple):
    failed: jax.Array


class State(NamedTuple):
    x: jax.Array
    mass: jax.Array
    history: Damage
    valid: jax.Array


def from_paths(paths, radius, *, masses=None, failed=None, families=(0, 1), seed=True, skin=None):
    offset = min(radius*.01, 1e-7)*np.array([[0., -1., -1.], [0., -1., 1.], [0., 1., -1.], [0., 1., 1.]])
    points = [(path[:, None]+offset).reshape(-1, 3) for path in paths]
    x = np.concatenate(points)
    masses = [np.ones(len(path)) for path in paths] if masses is None else masses
    particle_mass = np.concatenate([np.repeat(value/4, 4) for value in masses])
    flags = np.zeros(len(x), dtype=bool) if failed is None else np.concatenate([np.repeat(value, 4) for value in failed])
    geometry = {"x": x, "mass": particle_mass,
                "fibre_id": np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(points)]),
                "metadata": {"cross_section_particles": 4, "fibres": [
                    {"id": i, "family_id": families[i], "longitudinal_sections": len(paths[i]),
                     "particle_count": len(points[i]), "radius_m": radius} for i in range(2)]},
                "contact_candidates": [{"fibre_ids": [0, 1], "reference_centreline_points_m":
                                        [path[len(path)//2].tolist() for path in paths]}] if seed else []}
    data = build_contact_data(geometry, search_window=6, coverage_skin=10*radius if skin is None else skin)
    graph = patch_neighbours(np.asarray(data.coverage_pairs), np.asarray(data.segment_fibre))
    state = State(jnp.asarray(x), jnp.asarray(particle_mass), Damage(jnp.asarray(flags)), jnp.asarray(True))
    return state, data, graph


def crossed(height=.0999, angle=np.pi/2, **kwargs):
    t = np.linspace(-.4, .4, 9)
    paths = [np.column_stack((t, np.zeros_like(t), np.zeros_like(t))),
             np.column_stack((t*np.cos(angle), np.full_like(t, height), t*np.sin(angle)))]
    return from_paths(paths, .05, **kwargs)


def law(model="hertz", modulus=1000.):
    return fibre_contact_parameters(effective_modulus=modulus, effective_radius=.05, work_of_adhesion=.02, model=model)


def evaluate(state, data, graph, parameters=None, **kwargs):
    parameters = law() if parameters is None else parameters
    settings = {"capacity": 256, "max_patches": 8, "max_iterations": 128, **kwargs}
    history = initial_patch_history(data, settings["max_patches"])
    return evaluate_hertz_patches(state, history, data, parameters, graph, **settings)


@pytest.mark.parametrize("angle", [np.pi/2, np.pi/3])
def test_one_connected_contact_equals_existing_hertz_force(angle):
    state, data, graph = crossed(angle=angle)
    expected, _, old = evaluate_contact(state, initial_contact_history(data), data, law())
    force, history, metrics = evaluate(state, data, graph)
    assert old.valid and metrics.valid and history.valid
    assert metrics.patch_count == 1 and metrics.active_cell_count > 1
    np.testing.assert_allclose(force, expected, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(metrics.normal_force[metrics.active], old.normal_force, rtol=2e-12)
    np.testing.assert_array_equal(history.segments[~history.law.active], -1)


def test_action_reaction_moment_and_rigid_motion_with_unequal_section_masses():
    state, data, graph = crossed(masses=[np.arange(1., 10.), np.arange(9., 0., -1)])
    force, _, metrics = evaluate(state, data, graph)
    assert metrics.valid
    np.testing.assert_allclose(force.sum(0), 0., atol=1e-14)
    np.testing.assert_allclose(np.cross(state.x, force).sum(0), 0., atol=1e-14)
    theta = .37
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    moved = state._replace(x=state.x@rotation.T+jnp.array([2., -3., 4.]))
    transformed, _, result = jax.jit(lambda s: evaluate(s, data, graph))(moved)
    assert result.valid and not result.coverage_motion_exceeded
    np.testing.assert_allclose(transformed, force@rotation.T, rtol=2e-10, atol=1e-13)


def test_real_rejected_seed_909_has_two_positive_balanced_hertz_forces():
    path = Path(__file__).parent/"data/fibre_multipatch_regression.npz"
    proof = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == proof["fixture_sha256"]
    assert proof["saved_steps"] == 4980
    with np.load(path, allow_pickle=False) as data:
        paths = [data[f"centres_{i}"] for i in range(2)]
        masses = [data[f"section_mass_{i}"] for i in range(2)]
        failed = [data[f"section_failed_{i}"] for i in range(2)]
        radius = float(data["radius"])
    state, data, graph = from_paths(paths, radius, masses=masses, failed=failed, skin=.0005)
    parameters = law(modulus=10e9/(2*(1-.25**2)))
    _, _, old = evaluate_contact(state, initial_contact_history(data), data, parameters)
    assert not old.valid and old.uncovered_near_pairs > 0
    force, history, metrics = jax.jit(lambda s: evaluate(s, data, graph, parameters))(state)
    assert metrics.valid and history.valid and metrics.patch_count == 2
    assert int(metrics.active_cell_count) == 4
    assert np.all(np.asarray(metrics.normal_force)[np.asarray(metrics.active)] > 0)
    np.testing.assert_allclose(np.sort(np.asarray(metrics.gap)[np.asarray(metrics.active)]),
                               [-4.817676589562701e-7, -6.810158674941565e-9], rtol=0, atol=3e-17)
    np.testing.assert_allclose(force.sum(0), 0., atol=2e-14)
    np.testing.assert_allclose(np.cross(state.x, force).sum(0), 0., atol=2e-16)


def test_no_active_region_preserves_a_finite_real_coverage_minimum_and_zero_force():
    state, data, graph = crossed(height=.12)
    force, history, metrics = evaluate(state, data, graph)
    assert metrics.valid and metrics.patch_count == 0 and not np.any(history.law.active)
    assert float(metrics.minimum_coverage_gap) == pytest.approx(.02)
    np.testing.assert_array_equal(force, 0.)


@pytest.mark.parametrize("defect", ["active_capacity", "patch_capacity", "iterations", "adhesive", "radius", "failed", "unseeded", "same_family", "coverage"])
def test_unsupported_or_unresolved_contact_is_never_accepted(defect):
    state, data, graph = crossed(height=.095 if defect == "radius" else .0999,
                                seed=defect != "unseeded", families=(0, 0) if defect == "same_family" else (0, 1))
    options = {}
    parameters = law("jkr" if defect == "adhesive" else "hertz")
    if defect == "active_capacity":
        options["capacity"] = 1
    elif defect == "patch_capacity":
        graph[:] = -1
        options["max_patches"] = 1
    elif defect == "iterations":
        options["max_iterations"] = 1
    elif defect == "failed":
        state = state._replace(history=Damage(state.history.failed.at[16].set(True)))
    elif defect == "coverage":
        state = state._replace(x=state.x.at[36:, 1].add(2.))
    _, history, metrics = evaluate(state, data, graph, parameters, **options)
    assert not metrics.valid and not history.valid
    if defect == "radius":
        assert np.any(metrics.large_contact_radius)


def test_shared_fibre_footprints_are_checked_without_an_all_patch_tensor():
    centers = jnp.array([[[0., 0., 0.], [0., 0., 1.]], [[.1, 0., 0.], [.1, 0., -1.]],
                        [[10., 0., 0.], [10., 0., 1.]]])
    owners = jnp.array([[0, 1], [0, 2], [0, 3]])
    radius = jnp.array([.06, .06, .06])
    active = jnp.ones(3, dtype=bool)
    assert int(_overlap_count(centers, owners, radius, active, jnp.array(3), block=2)) == 1
    assert int(_overlap_count(centers, owners, radius*.5, active, jnp.array(3), block=2)) == 0


def test_empty_coverage_has_no_phantom_patch_and_an_explicit_zero_minimum():
    state, data, graph = crossed(height=2.)
    assert len(data.coverage_pairs) == 0
    force, history, metrics = evaluate(state, data, graph)
    assert metrics.valid and metrics.patch_count == 0 and metrics.minimum_coverage_gap == 0
    assert not np.any(history.law.active)
    np.testing.assert_array_equal(force, 0.)


def test_contact_at_a_physical_fibre_end_is_still_unsupported():
    t = np.linspace(-.4, .4, 9)
    paths = [np.column_stack((t, np.zeros_like(t), np.zeros_like(t))),
             np.column_stack((np.full_like(t, .4), np.full_like(t, .0999), t))]
    state, data, graph = from_paths(paths, .05)
    _, _, metrics = evaluate(state, data, graph)
    assert np.any(metrics.unsupported_end_contact) and not metrics.valid


@pytest.mark.parametrize("height,angle", [(0., np.pi/2), (.0999, 0.)])
def test_zero_normal_distance_or_degenerate_curvature_is_not_a_valid_patch(height, angle):
    state, data, graph = crossed(height=height, angle=angle)
    _, _, metrics = evaluate(state, data, graph)
    assert np.any(metrics.degenerate_geometry) and not metrics.valid


def test_an_invalid_previous_contact_cannot_be_reset_by_stateless_reselection():
    state, data, graph = crossed()
    history = initial_patch_history(data, 8)._replace(valid=jnp.asarray(False))
    _, updated, metrics = evaluate_hertz_patches(state, history, data, law(), graph, capacity=256, max_patches=8)
    assert not updated.valid and not metrics.valid
