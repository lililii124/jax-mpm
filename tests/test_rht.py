import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.materials import initial_history, rht
from mpm_jax.materials.history import equivalent
from mpm_jax.materials.rht import rht_eos, rht_failure, rht_lode, update_rht

jax.config.update("jax_enable_x64", True)


def test_rht_eos_initial_crushing_and_unloading():
    p = rht()
    rho = jnp.asarray([p.density, 1.1 * p.density, 1.4 * p.density])
    pressure, alpha, bulk, valid, active = jax.jit(rht_eos)(
        rho, jnp.zeros(3), jnp.full(3, p.alpha0), p
    )
    np.testing.assert_allclose(pressure[0], 0, atol=1e-5)
    assert bool(jnp.all(valid)) and bool(jnp.all(bulk > 0))
    assert p.alpha0 > float(alpha[1]) > 1
    assert bool(active[1])
    assert float(alpha[2]) == 1
    _, unloaded_alpha, _, _, _ = rht_eos(0.95 * rho, jnp.zeros(3), alpha, p)
    np.testing.assert_allclose(unloaded_alpha, alpha)


def test_rht_failure_surface_and_lode_meridians():
    p = rht()
    strength, apex = rht_failure(jnp.array([1 / 3, 1.0, 2.0]), 1.0, p)
    np.testing.assert_allclose(strength[0], 1.0, rtol=1e-12)
    assert bool(jnp.all(jnp.diff(strength) > 0)) and bool(jnp.all(apex < 0))
    tension = jnp.diag(jnp.array([2.0, -1.0, -1.0]))[None]
    np.testing.assert_allclose(rht_lode(jnp.array([0.0]), tension, p), p.Q0, atol=1e-12)
    np.testing.assert_allclose(rht_lode(jnp.array([0.0]), -tension, p), 1.0, atol=1e-12)


def test_rht_small_elastic_step_and_energy():
    p = rht()
    h = initial_history(1, jnp.float64, p, p.density)
    strain = jnp.diag(jnp.array([-1e-6, -1e-6, -1e-6]))[None]
    J = jnp.array([np.exp(-3e-6)])
    result = jax.jit(update_rht)(h, strain, J, 1e-6, p)
    assert bool(jnp.all(result.valid))
    np.testing.assert_allclose(result.pressure, p.A1 / p.alpha0 * 3e-6, rtol=2e-5)
    assert float(result.energy[0]) > 0
    np.testing.assert_allclose(equivalent(result.stress), 0, atol=1e-9)
    np.testing.assert_allclose(result.damage, 0, atol=1e-12)
    np.testing.assert_allclose(result.alpha, p.alpha0, atol=1e-12)


def test_rht_confined_shear_yields():
    p = rht()
    h = initial_history(1, jnp.float64, p, p.density)
    strain = jnp.array([[[-1e-4, 0.001, 0.0], [0.001, -1e-4, 0.0], [0.0, 0.0, -1e-4]]])
    result = jax.jit(update_rht)(h, strain, jnp.array([np.exp(-3e-4)]), 1e-6, p)
    assert bool(jnp.all(result.valid))
    assert float(result.plastic_strain[0]) > 0
    assert float(equivalent(result.stress)[0]) < float(equivalent(2 * p.shear * strain)[0])
