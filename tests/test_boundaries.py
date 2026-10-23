import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import (
    Config, MovingVelocityBox, PlaneWall, VelocityBox, advance, block, elastic, fixed_box,
)

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


def test_moving_box_half_cosine_motion_and_velocity_integral():
    bc = MovingVelocityBox(
        (0.0,) * 3, (0.2,) * 3, (2.0, -1.0, 0.5), start=0.2, end=1.2, ramp_time=0.4
    )
    times = jnp.array([0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.5])
    displacement, velocity = jax.jit(jax.vmap(bc.motion))(times)
    distances = np.array([0.0, 0.0, 0.1 - 0.2 / np.pi, 0.2, 0.5, 0.8, 0.8])
    speeds = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(displacement, distances[:, None] * bc.velocity, atol=1e-15)
    np.testing.assert_allclose(velocity, speeds[:, None] * bc.velocity, atol=1e-15)
    for time in (0.3, 0.5, 0.8, 1.4):
        derivative = jax.jacfwd(lambda t: bc.motion(t)[0])(time)
        np.testing.assert_allclose(derivative, bc.motion(time)[1], atol=1e-14)


def test_moving_box_mask_tracks_motion_and_preserves_free_components():
    bc = MovingVelocityBox(
        (0.0, 0.2, 0.2), (0.2, 0.4, 0.4), (0.4, 0.0, 0.0),
        start=0.25, end=1.25, ramp_time=0.5,
    )
    coordinates = jnp.array([[0.05, 0.3, 0.3], [0.25, 0.3, 0.3], [0.45, 0.3, 0.3]])
    velocity = jnp.ones((3, 3))
    apply = jax.jit(bc.apply)
    np.testing.assert_array_equal(apply(velocity, coordinates, 0.2), velocity)
    np.testing.assert_allclose(apply(velocity, coordinates, 0.25), [[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    np.testing.assert_allclose(apply(velocity, coordinates, 0.75), [[1, 1, 1], [0.4, 0, 0], [1, 1, 1]])
    np.testing.assert_allclose(apply(velocity, coordinates, 1.25), [[1, 1, 1], [1, 1, 1], [0.4, 0, 0]])
    np.testing.assert_array_equal(apply(velocity, coordinates, 1.3), velocity)
    sliding = MovingVelocityBox(
        bc.lower, bc.upper, (0.4, 0.1, 0.0), components=(True, False, False),
        start=bc.start, end=bc.end, ramp_time=bc.ramp_time,
    )
    np.testing.assert_allclose(
        sliding.apply(velocity, coordinates, 0.75), [[1, 1, 1], [0.4, 1, 1], [1, 1, 1]]
    )


def test_moving_box_constant_speed_and_end_during_ramp():
    bc = MovingVelocityBox((0.0,) * 3, (0.2,) * 3, (2, 0, 0), start=0.4, end=0.9)
    displacement, velocity = jax.jit(bc.motion)(0.6)
    np.testing.assert_allclose(displacement, [0.4, 0, 0])
    np.testing.assert_allclose(velocity, [2, 0, 0])
    displacement, velocity = bc.motion(1)
    np.testing.assert_allclose(displacement, [1, 0, 0])
    np.testing.assert_allclose(velocity, 0)
    short = MovingVelocityBox(bc.lower, bc.upper, bc.velocity, end=0.5, ramp_time=1.0)
    np.testing.assert_allclose(short.motion(2.0)[0], [0.5 - 1 / np.pi, 0, 0])
    np.testing.assert_allclose(short.motion(2.0)[1], 0)


@pytest.mark.parametrize("ramp_time", [-0.1, np.nan, np.inf, -np.inf])
def test_moving_box_rejects_invalid_ramp_time(ramp_time):
    bc = MovingVelocityBox((0.0,) * 3, (0.2,) * 3, (1.0, 0.0, 0.0), ramp_time=ramp_time)
    with pytest.raises(ValueError, match="ramp_time"):
        bc.validate(3, 1.0)
