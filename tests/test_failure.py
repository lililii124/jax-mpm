from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, advance, block, coupled_step, elastic, fixed_box, johnson_cook, step
from mpm_jax.core import State, particle_stencil, particle_to_grid, stencil
from mpm_jax.initialization import diagnostics, validate_state
from mpm_jax.materials import History

jax.config.update("jax_enable_x64", True)


def mark_failed(state):
    return state._replace(
        history=state.history._replace(
            damage=jnp.ones_like(state.history.damage),
            failed=jnp.ones_like(state.history.failed),
            stress=jnp.zeros_like(state.history.stress),
        )
    )


def append_particles(a, b):
    history = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), a.history, b.history)
    return State(
        *(jnp.concatenate((getattr(a, key), getattr(b, key))) for key in State._fields[:6]),
        history,
        a.time,
        a.valid & b.valid,
    )


def sample(config, material=None, **kwargs):
    return block(
        config,
        (0.4,) * config.dim,
        (0.6,) * config.dim,
        particles_per_cell=1,
        material=material,
        dtype=jnp.float64,
        **kwargs,
    )


def test_ballistic_free_flight_preserves_mass_momentum_history_and_checkpoint(tmp_path):
    cfg = Config(
        dim=3,
        n_grid=12,
        dt=0.05,
        material="johnson-cook",
        failure_mode="ballistic",
        boundaries=(fixed_box((0, 0, 0), (1, 1, 1)),),
    )
    material = johnson_cook(density=1000.0)
    initial = mark_failed(sample(cfg, material, velocity=(1.7, -0.2, 0.4)))
    initial = initial._replace(C=jnp.full_like(initial.C, 0.3))
    acceleration = jnp.array([3.0, 1.5, -2.0])
    force = initial.mass[:, None] * acceleration
    count = 8
    final = advance(initial, material, cfg, count, force)
    acceleration = acceleration + jnp.asarray(cfg.gravity)
    expected_v = initial.v + count * cfg.dt * acceleration
    expected_x = initial.x + count * cfg.dt * initial.v
    expected_x += count * (count + 1) / 2 * cfg.dt**2 * acceleration
    np.testing.assert_allclose(final.v, expected_v, atol=1e-14)
    np.testing.assert_allclose(final.x, expected_x, atol=1e-14)
    np.testing.assert_array_equal(final.mass, initial.mass)
    np.testing.assert_array_equal(final.volume, initial.volume)
    np.testing.assert_array_equal(final.F, initial.F)
    np.testing.assert_array_equal(final.C, 0.0)
    for before, after in zip(initial.history, final.history):
        np.testing.assert_array_equal(before, after)
    impulse = count * cfg.dt * (force.sum(0) + initial.mass.sum() * jnp.asarray(cfg.gravity))
    change = jnp.sum(final.mass[:, None] * (final.v - initial.v), axis=0)
    np.testing.assert_allclose(change, impulse, atol=1e-12)
    assert float(final.x.max()) > cfg.domain_size
    validate_state(final, cfg)
    values = diagnostics(final, cfg, material)
    assert values["active_particles"] == 0
    assert values["debris_particles"] == len(initial.mass)
    assert values["active_mass"] == 0.0
    assert values["debris_mass"] == values["mass"]
    assert values["active_current_volume"] == 0.0

    path = tmp_path / "failed_state.npz"
    payload = {key: value for key, value in final._asdict().items() if key != "history"}
    payload.update({f"history_{key}": value for key, value in final.history._asdict().items()})
    np.savez(path, **payload)
    with np.load(path) as saved:
        history = History(*(jnp.asarray(saved[f"history_{key}"]) for key in History._fields))
        restored = State(
            *(history if key == "history" else jnp.asarray(saved[key]) for key in State._fields)
        )
    restarted = step(restored, material, cfg, force)
    uninterrupted = step(final, material, cfg, force)
    for actual, expected in zip(jax.tree.leaves(restarted), jax.tree.leaves(uninterrupted)):
        np.testing.assert_array_equal(actual, expected)
    assert asdict(cfg)["failure_mode"] == "ballistic"


def test_failed_particles_transfer_neither_mass_momentum_stress_nor_applied_force():
    cfg = Config(dim=3, n_grid=12, dt=1e-4, boundary_cells=0, failure_mode="ballistic")
    material = elastic()
    active = sample(cfg, velocity=(0.2, 0.1, -0.1))
    debris = mark_failed(active)._replace(
        v=jnp.full_like(active.v, 20.0),
        C=jnp.full_like(active.C, 3.0),
        F=active.F.at[:, 0, 0].set(1.2),
    )
    combined = append_particles(active, debris)
    active_force = active.mass[:, None] * jnp.array([0.5, -0.3, 0.1])
    debris_force = debris.mass[:, None] * jnp.array([400.0, 300.0, -200.0])
    force = jnp.concatenate((active_force, debris_force))
    expected_grid = particle_to_grid(active, material, cfg, force=active_force)
    # Even a caller-supplied unmasked stencil must not reactivate failed particles.
    actual_grid = particle_to_grid(combined, material, cfg, stencil(combined.x, cfg), force)
    for expected, actual in zip(expected_grid, actual_grid):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    expected = step(active, material, cfg, active_force)
    actual = step(combined, material, cfg, force)
    count = len(active.mass)
    for field in ("x", "v", "F", "C"):
        np.testing.assert_allclose(
            getattr(actual, field)[:count], getattr(expected, field), atol=1e-14
        )
    np.testing.assert_allclose(
        actual.v[count:],
        debris.v + cfg.dt * (jnp.asarray(cfg.gravity) + debris_force / debris.mass[:, None]),
        atol=1e-14,
    )


def test_inactive_positions_cannot_wrap_or_invalidate_active_grid_support():
    cfg = Config(dim=3, n_grid=12, gravity=(0, 0, 0), boundary_cells=0, failure_mode="ballistic")
    active = sample(cfg)
    debris = mark_failed(active)._replace(
        x=jnp.full_like(active.x, 1e30), v=jnp.zeros_like(active.v)
    )
    combined = append_particles(active, debris)
    support = particle_stencil(combined, cfg)
    assert bool(support.valid)
    np.testing.assert_array_equal(support.weights[len(active.mass) :], 0.0)
    np.testing.assert_array_equal(support.gradients[len(active.mass) :], 0.0)
    actual = step(combined, elastic(), cfg)
    validate_state(actual, cfg)
    np.testing.assert_allclose(actual.x[: len(active.mass)], active.x, atol=1e-14)
    invalid = active._replace(x=active.x.at[0, 0].set(0.0))
    with pytest.raises(FloatingPointError, match="stencil"):
        validate_state(invalid, cfg)
    assert not bool(step(invalid, elastic(), cfg).valid)
    with pytest.raises(FloatingPointError, match="stencil"):
        validate_state(combined, replace(cfg, failure_mode="retain"))


def test_ballistic_debris_exerts_no_two_body_contact_and_receives_its_force():
    cfg = Config(
        dim=3, n_grid=12, dt=1e-4, gravity=(0, 0, 0), boundary_cells=0, failure_mode="ballistic"
    )
    material = elastic()
    debris = mark_failed(sample(cfg, velocity=(2.0, 0.0, 0.0)))
    active = sample(cfg, velocity=(-1.0, 0.0, 0.0))
    force = debris.mass[:, None] * jnp.array([3.0, 2.0, 1.0])
    (actual_debris, actual_active), impulse = coupled_step(
        (debris, active), (material, material), (cfg, cfg), forces=(force, None)
    )
    expected_active = step(active, material, cfg)
    np.testing.assert_array_equal(impulse, 0.0)
    np.testing.assert_allclose(actual_active.v, expected_active.v, atol=1e-14)
    np.testing.assert_allclose(actual_active.F, expected_active.F, atol=1e-14)
    np.testing.assert_allclose(
        actual_debris.v, debris.v + cfg.dt * force / debris.mass[:, None], atol=1e-14
    )
    total_change = sum(
        jnp.sum(after.mass[:, None] * after.v - before.mass[:, None] * before.v, axis=0)
        for before, after in ((debris, actual_debris), (active, actual_active))
    )
    np.testing.assert_allclose(total_change, cfg.dt * force.sum(0), atol=1e-12)


def test_retain_is_default_and_preserves_grid_coupling():
    cfg = Config(dim=3, n_grid=12, dt=1e-4, gravity=(0, 0, 0), boundary_cells=0)
    active = sample(cfg, velocity=(0.0, 0.0, 0.0))
    debris = mark_failed(active)._replace(v=jnp.ones_like(active.v))
    combined = append_particles(active, debris)
    gm, _ = particle_to_grid(combined, elastic(), cfg)
    np.testing.assert_allclose(gm.sum(), combined.mass.sum(), rtol=1e-14)
    default = step(combined, elastic(), cfg)
    explicit = step(combined, elastic(), replace(cfg, failure_mode="retain"))
    for actual, expected in zip(jax.tree.leaves(default), jax.tree.leaves(explicit)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(default.v, 0.5, atol=1e-13)
    with pytest.raises(ValueError, match="failure_mode"):
        Config(failure_mode="delete")


def test_no_failure_ballistic_and_retain_have_identical_dynamics():
    cfg = Config(dim=3, n_grid=12, dt=1e-5, boundary_cells=0, material="johnson-cook")
    material = johnson_cook(density=1000.0)
    initial = sample(cfg, material, velocity=(0.03, -0.02, 0.01))
    retained = step(initial, material, cfg)
    ballistic = step(initial, material, replace(cfg, failure_mode="ballistic"))
    for actual, expected in zip(jax.tree.leaves(ballistic), jax.tree.leaves(retained)):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


def test_johnson_cook_threshold_transitions_to_ballistic_motion_next_step():
    cfg = Config(
        dim=3,
        n_grid=12,
        dt=1e-3,
        gravity=(0, 0, 0),
        boundary_cells=0,
        material="johnson-cook",
        failure_mode="ballistic",
    )
    material = johnson_cook(
        young=1e5,
        density=1000.0,
        A=10.0,
        B=0.0,
        C=0.0,
        failure=(1e-4, 0.0, 0.0, 0.0, 0.0),
    )
    initial = sample(cfg, material)
    gradient = jnp.diag(jnp.array([1.0, -0.3, -0.3]))
    initial = initial._replace(
        v=(initial.x - 0.5) @ gradient.T,
        C=jnp.broadcast_to(gradient, initial.C.shape),
    )
    failed = step(initial, material, cfg)
    assert bool(jnp.all(failed.history.failed))
    np.testing.assert_array_equal(failed.history.damage, 1.0)
    np.testing.assert_array_equal(failed.history.stress, 0.0)
    np.testing.assert_allclose(failed.v, initial.v, atol=1e-14)
    assert float(jnp.max(jnp.abs(failed.F - initial.F))) > 0
    gm, gp = particle_to_grid(failed, material, cfg)
    np.testing.assert_array_equal(gm, 0.0)
    np.testing.assert_array_equal(gp, 0.0)
    detached = step(failed, material, cfg)
    validate_state(detached, cfg)
    np.testing.assert_allclose(detached.x, failed.x + cfg.dt * failed.v, atol=1e-14)
    np.testing.assert_array_equal(detached.F, failed.F)
    np.testing.assert_array_equal(detached.v, failed.v)
    for actual, expected in zip(detached.history, failed.history):
        np.testing.assert_array_equal(actual, expected)
