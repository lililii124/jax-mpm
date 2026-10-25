import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.fragment_v6_discrete_work import volume_piola

jax.config.update('jax_enable_x64', True)


def test_volume_work_and_torque_for_compression_tension_and_crossing():
    a = np.tile(np.eye(3), (6, 1, 1))
    b = a.copy()
    a[:, 0, 0] = [.8, 1.1, .9, 1.2, .75, 1.]
    b[:, 0, 0] = [.7, 1.3, 1.15, .85, .75, 1.]
    b[:, 0, 1] = [.12, -.2, .15, .2, 0., .3]
    damage = np.array([0., .3, .4, .6, .9, .2])
    bulk = 142e9
    P = np.asarray(jax.jit(volume_piola)(jnp.asarray(a), jnp.asarray(b), jnp.asarray(damage), bulk))
    def potential(F):
        z = np.linalg.det(F) - 1
        return .5*bulk*(np.minimum(z, 0)**2+(1-damage)*np.maximum(z, 0)**2)
    work = np.sum(P*(b-a), axis=(-2, -1))
    np.testing.assert_allclose(work, potential(b)-potential(a), rtol=5e-14, atol=1e-5)
    torque_tensor = P @ (.5*(a+b)).swapaxes(-1, -2)
    np.testing.assert_allclose(torque_tensor, torque_tensor.swapaxes(-1, -2), rtol=5e-14, atol=1e-5)
    Pzero = np.asarray(volume_piola(jnp.asarray(a), jnp.asarray(a), jnp.asarray(damage), bulk))
    J = np.linalg.det(a)
    expected = (bulk*(J-1)*np.where(J > 1, 1-damage, 1)*J)[:, None, None]*np.linalg.inv(a).swapaxes(-1, -2)
    np.testing.assert_allclose(Pzero, expected, rtol=5e-14, atol=1e-5)


def test_volume_work_is_objective_under_a_common_rotation():
    a = np.array([[[.9, .05, 0.], [0., 1., 0.], [0., 0., 1.]]])
    b = np.array([[[1.1, .2, .03], [0., .95, 0.], [0., 0., 1.05]]])
    angle = .7
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    P = np.asarray(volume_piola(jnp.asarray(a), jnp.asarray(b), jnp.array([.3]), 142e9))
    rotated = np.asarray(volume_piola(jnp.asarray(Q@a), jnp.asarray(Q@b), jnp.array([.3]), 142e9))
    np.testing.assert_allclose(rotated, Q@P, rtol=5e-14, atol=1e-5)
