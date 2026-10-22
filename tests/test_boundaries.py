import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, PlaneWall, VelocityBox, advance, block, elastic, fixed_box

jax.config.update("jax_enable_x64", True)


def test_unilateral_wall_and_coulomb_sliding():
    wall = PlaneWall(1, 0.2, friction=0.25)
    x = jnp.array([[0.5, 0.1, 0.5], [0.5, 0.1, 0.5], [0.5, 0.3, 0.5]])
    v = jnp.array([[2.0, -4.0, 0.0], [2.0, 4.0, 0.0], [2.0, -4.0, 0.0]])
    result = wall.apply(v, x, 0.0)
    np.testing.assert_allclose(result[0], [1, 0, 0])
    np.testing.assert_array_equal(result[1:], v[1:])


def test_moving_wall_uses_relative_velocity_and_position():
    wall = PlaneWall(1, 0.2, velocity=(0.0, 0.1, 0.0))
    result = wall.apply(jnp.zeros((1, 3)), jnp.array([[0.5, 0.25, 0.5]]), 1.0)
    np.testing.assert_allclose(result, [[0.0, 0.1, 0.0]])


def test_velocity_mask_and_activation():
    bc = VelocityBox(
        (0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (2.0, 0.0, 0.0), (True, False, False), 0.1, 0.3
    )
    x = jnp.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    v = jnp.ones((2, 3))
    np.testing.assert_array_equal(bc.apply(v, x, 0.0), v)
    np.testing.assert_array_equal(bc.apply(v, x, 0.4), v)
    np.testing.assert_allclose(bc.apply(v, x, 0.2), [[2, 1, 1], [1, 1, 1]])


def test_fully_clamped_body_cancels_gravity_and_preserves_deformation():
    cfg = Config(
        dim=3, n_grid=12, boundary_cells=0, boundaries=(fixed_box((0.0,) * 3, (1.0,) * 3),)
    )
    s = block(cfg, particles_per_cell=1, dtype=jnp.float64)
    result = advance(s, elastic(), cfg, 10)
    np.testing.assert_allclose(result.x, s.x, atol=1e-14)
    np.testing.assert_allclose(result.v, 0, atol=1e-14)
    np.testing.assert_allclose(result.F, s.F, atol=1e-14)
    np.testing.assert_allclose(result.time, 10 * cfg.dt)
