from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, block, coupled_advance, elastic, step, validate_state
from mpm_jax.contact import contact_impulse
from mpm_jax.core import stencil

jax.config.update("jax_enable_x64", True)


def test_balanced_contact_coulomb_and_separation():
    ma, mb = jnp.array([2.0, 2.0]), jnp.array([3.0, 3.0])
    va = jnp.array([[2.0, 4.0, 0.0], [-2.0, 1.0, 0.0]])
    vb = jnp.zeros_like(va)
    n = jnp.array([[1.0, 0.0, 0.0]] * 2)
    a, b, impulse = contact_impulse(ma, mb, va, vb, n, 0.2)
    np.testing.assert_allclose(ma[:, None] * a + mb[:, None] * b, ma[:, None] * va)
    np.testing.assert_allclose(a[0, 0], b[0, 0], atol=1e-14)
    np.testing.assert_allclose(impulse[0, 1], 0.2 * impulse[0, 0])
    np.testing.assert_array_equal(a[1], va[1])
    np.testing.assert_array_equal(impulse[1], 0)
    old_energy = jnp.sum(ma[:, None] * va**2 + mb[:, None] * vb**2)
    new_energy = jnp.sum(ma[:, None] * a**2 + mb[:, None] * b**2)
    assert new_energy < old_energy


def test_shape_gradients_and_linear_reproduction():
    cfg = Config(dim=3, n_grid=16)
    x = jnp.array([[0.413, 0.517, 0.623]])
    s = stencil(x, cfg)
    np.testing.assert_allclose(s.gradients.sum(1), 0.0, atol=1e-13)
    moment = jnp.einsum("pki,pkj->pij", s.gradients, s.dpos)
    np.testing.assert_allclose(moment[0], np.eye(3), atol=1e-13)


def test_two_body_contact_conserves_total_momentum():
    cfg = Config(dim=3, n_grid=16, dt=1e-4, gravity=(0, 0, 0), boundary_cells=0)
    a = block(cfg, (0.25, 0.35, 0.35), (0.45, 0.65, 0.65), velocity=(1, 0.2, 0), dtype=jnp.float64)
    b = block(
        cfg, (0.47, 0.35, 0.35), (0.67, 0.65, 0.65), velocity=(-1, -0.1, 0), dtype=jnp.float64
    )
    momentum = sum(jnp.sum(s.mass[:, None] * s.v, axis=0) for s in (a, b))
    (a2, b2), impulse = coupled_advance((a, b), (elastic(), elastic()), (cfg, cfg), 10, 0.1)
    validate_state(a2, cfg)
    validate_state(b2, cfg)
    actual = sum(jnp.sum(s.mass[:, None] * s.v, axis=0) for s in (a2, b2))
    np.testing.assert_allclose(actual, momentum, atol=1e-11)
    assert impulse[0] > 0
    assert jnp.mean(a2.v[:, 0]) < 1 and jnp.mean(b2.v[:, 0]) > -1


def test_external_force_gives_correct_impulse():
    cfg = Config(dim=3, n_grid=12, gravity=(0, 0, 0), boundary_cells=0)
    state = block(cfg, particles_per_cell=1, dtype=jnp.float64)
    force = jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0]), state.v.shape)
    final = step(state, elastic(), cfg, force)
    momentum = jnp.sum(final.mass[:, None] * final.v, axis=0)
    np.testing.assert_allclose(momentum, cfg.dt * force.sum(0), rtol=1e-12)
