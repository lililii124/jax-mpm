import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from mpm_jax import Config, block, elastic
from mpm_jax.core import step as intact_step
from mpm_jax.crack_mpm import (
    TensileTipLaw, advance_straight_tip, crack_fields, frictionless_contact,
    step as cracked_step,
)

jax.config.update('jax_enable_x64', True)


def test_crack_visibility_stops_at_the_tip():
    position = jnp.asarray([[.5, .1], [1.1, .1], [.5, -.1]])
    nodes = jnp.asarray([[.5, -.1], [1.1, -.1], [.5, 0.]])
    support = SimpleNamespace(dpos=(nodes - position)[:, None, :])
    fields = crack_fields(position, support, jnp.asarray([[0., 0.], [1., 0.]]))
    np.testing.assert_array_equal(np.asarray(fields[:, 0]), [1, 0, 1])


def test_remote_crack_preserves_intact_transfer():
    config = Config(dim=2, n_grid=16, dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.35, .35), (.65, .65), particles_per_cell=1, dtype=jnp.float64)
    state = state._replace(v=jnp.stack((state.x[:, 0], -state.x[:, 1]), axis=-1))
    segment = jnp.asarray([[.05, .1], [.2, .1]])
    reference = intact_step(state, elastic(), config)
    actual = cracked_step(state, elastic(), config, segment)
    np.testing.assert_allclose(actual.x, reference.x, atol=1e-13)
    np.testing.assert_allclose(actual.v, reference.v, atol=1e-13)


def test_crack_prevents_velocity_mixing_across_faces():
    config = Config(dim=2, n_grid=16, dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.375, .4375), (.625, .5625), particles_per_cell=2, dtype=jnp.float64)
    direction = jnp.where(state.x[:, 1] > .5, 1., -1.)
    state = state._replace(v=jnp.stack((jnp.zeros_like(direction), direction), axis=-1))
    segment = jnp.asarray([[.25, .5], [.75, .5]])
    intact = intact_step(state, elastic(), config)
    cracked = cracked_step(state, elastic(), config, segment)
    assert bool(cracked.valid)
    assert np.mean(np.abs(np.asarray(intact.v[:, 1]))) < .9
    np.testing.assert_allclose(cracked.v[:, 1], direction, atol=1e-13)
    np.testing.assert_allclose(np.sum(np.asarray(cracked.mass[:, None] * cracked.v), axis=0),
                               np.sum(np.asarray(state.mass[:, None] * state.v), axis=0), atol=1e-13)


def test_contact_preserves_momentum_and_tangential_slip_without_attraction():
    m0, m1 = jnp.asarray([2., 2., 0.]), jnp.asarray([3., 3., 3.])
    v0 = jnp.asarray([[4., -2.], [4., 2.], [4., -2.]])
    v1 = jnp.asarray([[-1., 1.], [-1., -1.], [-1., 1.]])
    n = jnp.asarray([[0., 1.]] * 3)
    a, b = frictionless_contact(m0, m1, v0, v1, n)
    np.testing.assert_allclose(m0[:, None] * a + m1[:, None] * b,
                               m0[:, None] * v0 + m1[:, None] * v1)
    np.testing.assert_allclose(a[:, 0], v0[:, 0])
    np.testing.assert_allclose(b[:, 0], v1[:, 0])
    np.testing.assert_allclose(a[0, 1], b[0, 1])
    np.testing.assert_array_equal(a[1:], v0[1:])
    np.testing.assert_array_equal(b[1:], v1[1:])
    assert float(jnp.sum(m0[:, None] * a*a + m1[:, None] * b*b)) < float(jnp.sum(m0[:, None] * v0*v0 + m1[:, None] * v1*v1))


def test_tip_grows_only_under_supercritical_tension_and_stops_at_limit():
    config = Config(dim=2, n_grid=32, dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.3, .3), (.7, .7), dtype=jnp.float64)
    material = elastic(young=1e6, poisson=.2)
    segment = jnp.asarray([[.2, .5], [.5, .5]])
    law = TensileTipLaw(strength=1000., radius=.1, max_speed=100., max_length=.301)
    for stretch in (.99, 1., 1.0001):
        F = state.F.at[:, 1, 1].set(stretch)
        next_segment, report = advance_straight_tip(state._replace(F=F), material, config, segment, law)
        np.testing.assert_array_equal(next_segment, segment)
        assert float(report.increment) == 0
    next_segment, report = advance_straight_tip(state._replace(F=state.F.at[:, 1, 1].set(1.02)), material, config, segment, law)
    assert 0 < float(report.increment) <= law.max_speed * config.dt
    assert float(next_segment[1, 0]) > .5
    np.testing.assert_array_equal(next_segment[:, 1], segment[:, 1])
    at_limit = segment.at[1, 0].set(.501)
    next_segment, _ = advance_straight_tip(state._replace(F=state.F.at[:, 1, 1].set(1.02)), material, config, at_limit, law)
    np.testing.assert_allclose(next_segment, at_limit, atol=1e-15)


def test_exited_crack_does_not_reconnect_through_exterior_nodes():
    config = Config(dim=2, n_grid=32, dt=1e-6, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.25, .4375), (.75, .5625), particles_per_cell=2, dtype=jnp.float64)
    direction = jnp.where(state.x[:, 1] > .5, 1., -1.)
    state = state._replace(v=jnp.stack((jnp.zeros_like(direction), direction), axis=-1))
    tip = float(state.x[:, 0].max()) + .001
    segment = jnp.asarray([[.15, .5], [tip, .5]])
    bridged = cracked_step(state, elastic(), config, segment)
    separated = cracked_step(state, elastic(), config, segment, end_exited=True)
    assert float(jnp.max(jnp.abs(bridged.v[:, 1] - direction))) > .01
    np.testing.assert_allclose(separated.v[:, 1], direction, atol=1e-13)
