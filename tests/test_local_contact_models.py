from functools import partial
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.contact_pull_off import DEFAULTS, displacement_cycle, run_reference
from mpm_jax.cohesive_bem import CohesiveBEMHistory, update_cohesive_bem
from mpm_jax.fibre_contact import (
    FibreContactParameters, fibre_contact_parameters, initial_fibre_contact_history,
    normal_pair_forces, update_fibre_contact,
)
from mpm_jax.local_contact_models import (
    SelfConsistentHistory, SelfConsistentParameters, initial_local_contact_history,
    local_contact_model, update_local_contact,
)
from mpm_jax.maugis_dugdale import MaugisHistory, MaugisParameters

jax.config.update("jax_enable_x64", True)


def make(name, **kwargs):
    return local_contact_model(name, **(dict(DEFAULTS, rings=16) | kwargs))


solve = jax.jit(partial(update_local_contact, max_sweeps=1200, tolerance=1e-9))


@pytest.mark.parametrize("name", ["hertz", "jkr"])
def test_legacy_types_and_call_results_are_exactly_preserved(name):
    p = make(name)
    assert type(p) is FibreContactParameters
    old = fibre_contact_parameters(effective_modulus=DEFAULTS["effective_modulus"],
                                   effective_radius=DEFAULTS["effective_radius"],
                                   work_of_adhesion=DEFAULTS["work_of_adhesion"], model=name)
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(old)):
        np.testing.assert_array_equal(a, b)
    delta = jnp.asarray([-4e-6, -1e-7, 0., 1e-6])
    history = initial_local_contact_history(p, delta.shape, active=True)
    baseline = initial_fibre_contact_history(delta.shape, active=True)
    assert type(history) is type(baseline)
    actual = jax.jit(update_local_contact)(history, delta, p)
    expected = jax.jit(update_fibre_contact)(baseline, delta, old)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


def test_factory_rejects_ambiguous_or_inconsistent_models():
    with pytest.raises(ValueError):
        make("unknown")
    with pytest.raises(ValueError):
        make("md", interaction_range=None)
    with pytest.raises(ValueError):
        make("self-consistent", outer_radius=None)
    with pytest.raises(ValueError):
        make("self-consistent", effective_modulus=[1e6, 2e6])


def test_md_initial_batch_has_the_scan_carry_shape():
    p = make("md", effective_radius=[.01, .012])
    assert type(p) is MaugisParameters
    history = initial_local_contact_history(p)
    assert type(history) is MaugisHistory and history.active.shape == (2,)
    values = jnp.asarray([[-4e-6, -4e-6], [1e-6, 1e-6], [-4e-6, -4e-6]])
    h, r = jax.jit(lambda: jax.lax.scan(lambda old, d: update_local_contact(old, d, p), history, values))()
    assert h.active.shape == (2,) and np.all(r.valid)
    np.testing.assert_array_equal(np.asarray(r.active)[[0, -1]], False)
    assert np.all(r.active[1])


@pytest.mark.parametrize("name", ["md", "self-consistent"])
def test_new_models_supply_balanced_forces_at_actual_pair_points(name):
    p = make(name)
    _, r = solve(initial_local_contact_history(p), 0., p)
    assert bool(r.valid) and float(r.normal_force) < 0
    first = jnp.asarray([.01, -.03, .02])
    second = jnp.asarray([.02, .01, -.01])
    forces = normal_pair_forces(second-first, r)
    assert forces.valid
    np.testing.assert_array_equal(forces.first+forces.second, 0)
    torque = jnp.cross(first, forces.first)+jnp.cross(second, forces.second)
    np.testing.assert_allclose(torque, 0, atol=1e-20, rtol=0)
    assert jnp.dot(forces.first, second-first) > 0


def test_sc_batched_radius_changes_preserve_the_old_elastic_displacement():
    p = make("self-consistent")
    assert type(p) is SelfConsistentParameters
    old = initial_local_contact_history(p, (2,))
    assert type(old) is SelfConsistentHistory
    old, loaded = solve(old, jnp.asarray([1e-6, .5e-6]), p)
    assert np.all(loaded.valid)
    changed = p._replace(effective_radius=jnp.asarray([.012, .008]))
    target = jnp.asarray([1.1e-6, .4e-6])
    actual, response = solve(old, target, changed)
    edges = p.base.edges
    profile = (edges[1:]**2+edges[:-1]**2)/(4*changed.effective_radius[:, None])
    warm = old.gap+profile-old.previous_profile
    np.testing.assert_allclose(warm-(profile-old.indentation[:, None]),
                               old.gap-(old.previous_profile-old.indentation[:, None]), atol=1e-20, rtol=0)
    for i in range(2):
        reference_old = CohesiveBEMHistory(warm[i], old.indentation[i], old.valid[i])
        local = p.base._replace(effective_radius=changed.effective_radius[i], profile=profile[i])
        expected, detail = jax.jit(partial(update_cohesive_bem, max_sweeps=1200, tolerance=1e-9))(
            reference_old, target[i], local,
        )
        assert response.valid[i] and detail.valid
        np.testing.assert_allclose(actual.gap[i], expected.gap, atol=1e-17, rtol=1e-10)
        np.testing.assert_allclose(response.normal_force[i], detail.normal_force, atol=1e-13, rtol=1e-9)
    np.testing.assert_allclose(actual.previous_profile, profile, atol=0, rtol=2e-15)


@pytest.mark.parametrize("field,value", [("effective_modulus", 2e6),
                                        ("work_of_adhesion", 0.), ("interaction_range", 1e-6)])
def test_sc_material_or_range_changes_are_rejected_without_rebuilding_history(field, value):
    p = make("self-consistent")
    history = initial_local_contact_history(p)
    changed = p._replace(**{field: jnp.asarray(value)})
    updated, response = solve(history, 1e-6, changed)
    assert not updated.valid and not response.valid
    np.testing.assert_array_equal(changed.base.compliance, p.base.compliance)


def test_sc_history_shape_and_domain_failure_are_explicit():
    p = make("self-consistent")
    with pytest.raises(ValueError):
        solve(initial_local_contact_history(p), jnp.asarray([1e-6, 1e-6]), p)
    small = make("self-consistent", outer_radius=1.2e-4)
    _, r = solve(initial_local_contact_history(small), 1e-6, small)
    assert not r.valid and not r.details.domain_clear


def test_sc_exact_open_solution_after_release_does_not_carry_residual_force():
    p = make("self-consistent")
    loaded, r = solve(initial_local_contact_history(p), 1e-6, p)
    assert r.valid and r.active
    released, r = solve(loaded, -4e-6, p)
    assert r.valid and not r.active and not r.open_exact and r.details.sweeps > 0
    clean, r = solve(released, -4e-6, p)
    assert r.valid and r.open_exact and r.details.sweeps == 0
    np.testing.assert_array_equal(r.details.ring_force, 0)
    assert r.normal_force == 0 and r.potential == 0
    np.testing.assert_allclose(clean.gap, clean.previous_profile+4e-6, rtol=2e-15)


def test_sc_exact_range_endpoint_is_not_classified_as_a_stable_open_state():
    p = local_contact_model("self-consistent", effective_modulus=1., effective_radius=1.,
                            work_of_adhesion=.001, interaction_range=.015625,
                            outer_radius=.5, rings=4)
    old = initial_local_contact_history(p)
    delta = -.01171875
    assert float(old.previous_profile[0])-delta == float(p.interaction_range)
    _, r = jax.jit(partial(update_local_contact, max_sweeps=1, tolerance=1e-12))(old, delta, p)
    assert not r.open_exact and int(r.details.sweeps) == 1


@pytest.mark.parametrize("name", ["hertz", "jkr", "md", "self-consistent"])
def test_reference_cycle_runs_real_histories_and_writes_auditable_curves(tmp_path, name):
    settings = dict(DEFAULTS, rings=16)
    summary = run_reference(name, output=tmp_path, samples=9, parameters=settings)
    assert summary["completed"] and not summary["final_active"]
    assert not summary["coupled_mpm"] and not summary["production_eligible"]
    folder = tmp_path/name
    for file, digest in summary["files"].items():
        assert hashlib.sha256((folder/file).read_bytes()).hexdigest() == digest
    with np.load(folder/"history.npz") as arrays:
        expected = displacement_cycle(samples=9)[1]
        np.testing.assert_array_equal(arrays["indentation_m"], expected)
        assert arrays["response_normal_force"].shape == (9,)
        assert np.all(arrays["response_valid"])
        if name == "self-consistent":
            assert arrays["details_gap"].shape == (9, 16)
    assert json.loads((folder/"summary.json").read_text())["completed"]


def test_reference_protocol_rejects_missing_peak_or_nonfinite_values():
    for kwargs in ({"samples": 8}, {"peak": -5e-6}, {"initial": np.nan}):
        with pytest.raises(ValueError):
            displacement_cycle(**kwargs)
