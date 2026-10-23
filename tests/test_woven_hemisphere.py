from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.woven_hemisphere import RigidHemisphereBoundary

jax.config.update("jax_enable_x64", True)


def tool():
    return RigidHemisphereBoundary((2., 2., 2.), 1., (0., -.2, 0.), .6)


def test_half_cosine_motion_and_stationary_hold_are_serializable():
    boundary = tool()
    boundary.validate(3, 4)
    assert asdict(boundary)["radius"] == 1
    np.testing.assert_allclose(boundary.motion(.3)[0], [2., 1.9, 2.])
    for t in (0., .6, 1.):
        np.testing.assert_array_equal(boundary.motion(t)[1], [0., 0., 0.])
    np.testing.assert_allclose(boundary.motion(1.)[0], [2., 1.8, 2.])
    derivative = (boundary.motion(.2 + 1e-6)[0] - boundary.motion(.2 - 1e-6)[0]) / 2e-6
    np.testing.assert_allclose(derivative, boundary.motion(.2)[1], rtol=1e-9)


def test_closed_cap_distance_includes_disk_rim_and_excludes_upper_half_sphere():
    points = jnp.array([[2., 1., 2.], [2., 2., 2.], [3., 2., 2.], [2., 2.5, 2.], [3.3, 2.4, 2.], [2., 1.5, 2.]])
    np.testing.assert_allclose(tool().signed_distance(points, 0.), [0., 0., 0., .5, .5, -.5], atol=1e-14)


def test_curved_contact_removes_only_closing_normal_velocity():
    points = jnp.array([[2., 1., 2.], [2., 1., 2.], [2., .9, 2.], [2., 2.5, 2.]])
    velocity = jnp.array([[1., 2., 3.], [1., -2., 3.], [1., 2., 3.], [1., -2., 3.]])
    expected = np.array([[1., 0., 3.], [1., -2., 3.], [1., 2., 3.], [1., -2., 3.]])
    np.testing.assert_array_equal(tool().apply(velocity, points, 0.), expected)


def test_flat_disk_contact_and_rim_separation_use_closed_solid_normals():
    points = jnp.array([[2., 2., 2.], [2., 2., 2.], [3., 2., 2.], [3., 2., 2.], [3., 2., 2.]])
    velocity = jnp.array([[1., -2., 3.], [1., 2., 3.], [-1., 1., 0.], [1., -1., 0.], [-1., -1., 0.]])
    result = np.asarray(tool().apply(velocity, points, 0.))
    np.testing.assert_array_equal(result[:4], [[1., 0., 3.], [1., 2., 3.], [-1., 1., 0.], [1., -1., 0.]])
    assert result[-1, 0] >= 0 or result[-1, 1] >= 0
    assert np.dot(result[-1], result[-1]) <= np.dot(velocity[-1], velocity[-1])


def test_oblique_curved_normal_preserves_tangent_and_cannot_add_relative_energy():
    normal = np.array([.6, -.8, 0.])
    tangent = np.array([.8, .6, 0.])
    point = jnp.asarray(np.array(tool().center) + normal)[None]
    velocity = jnp.asarray(-2 * normal + 3 * tangent)[None]
    result = np.asarray(tool().apply(velocity, point, 0.))[0]
    np.testing.assert_allclose(result, 3 * tangent, atol=1e-14)
    assert np.dot(result, result) <= float(jnp.sum(velocity**2))


def test_translating_cap_relative_velocity_and_float32_output():
    center, speed = tool().motion(.3)
    point = jnp.asarray(center + jnp.array([0., -1., 0.]), dtype=jnp.float32)[None]
    velocity = jnp.array([[.4, 0., -.2]], dtype=jnp.float32)
    result = tool().apply(velocity, point, jnp.asarray(.3, dtype=jnp.float64))
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, [[.4, float(speed[1]), -.2]], atol=1e-7)


@pytest.mark.parametrize("changes", [{"radius": 0}, {"loading_duration": -1}, {"start": -1}, {"radius": np.nan}, {"center": (2., 2.)}, {"displacement": (0., -2., 0.)}])
def test_invalid_or_outside_path_is_rejected(changes):
    arguments = asdict(tool())
    arguments.update(changes)
    with pytest.raises(ValueError):
        RigidHemisphereBoundary(**arguments).validate(3, 4)
