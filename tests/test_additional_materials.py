import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import bilinear_plastic, blatz_ko, thermal_plastic
from mpm_jax.materials import initial_history, kirchhoff_stress
from mpm_jax.materials.history import equivalent
from mpm_jax.materials.plastic import update_bilinear, update_thermal_plastic
from mpm_jax.materials.steinberg import steinberg_guinan, steinberg_strength, update_steinberg

jax.config.update("jax_enable_x64", True)


def test_mixed_hardening_yield_surface_and_backstress():
    strain = jnp.diag(jnp.array([0.01, -0.005, -0.005]))[None]
    for beta in (0.0, 0.5, 1.0):
        p = bilinear_plastic(isotropic_fraction=beta)
        h = initial_history(1, jnp.float64, p, p.density)
        result = update_bilinear(h, strain, jnp.ones(1), 1e-6, p)
        radius = p.yield_stress + beta * p.hardening * result.plastic_strain
        np.testing.assert_allclose(
            equivalent(result.stress - result.backstress), radius, rtol=1e-12
        )
        np.testing.assert_allclose(
            result.backstress,
            (2 / 3) * (1 - beta) * p.hardening * result.plastic_tensor,
            rtol=1e-12,
        )
        reverse = update_bilinear(result, -0.1 * strain, jnp.ones(1), 1e-6, p)
        assert jnp.all(reverse.plastic_strain >= result.plastic_strain)


def test_prescribed_temperature_expansion_and_fixed_temperature_plasticity():
    p = thermal_plastic(expansion=(12e-6, 12e-6))
    h = initial_history(1, jnp.float64, p, p.density)
    strain = 12e-6 * 100 * jnp.eye(3)[None]
    heated = update_thermal_plastic(
        h, strain, jnp.array([np.exp(0.0036)]), 1.0, p, temperature=393.0
    )
    np.testing.assert_allclose(heated.stress, 0, atol=1e-7)
    held = update_thermal_plastic(h, jnp.zeros((1, 3, 3)), jnp.ones(1), 1.0, p, temperature=393.0)
    assert float(held.pressure[0]) > 0
    strain = jnp.diag(jnp.array([0.01, -0.005, -0.005]))[None]
    plastic = update_thermal_plastic(h, strain, jnp.ones(1), 1e-6, p)
    np.testing.assert_allclose(
        equivalent(plastic.stress),
        p.yield_stress[0] + p.hardening[0] * plastic.plastic_strain,
        rtol=1e-12,
    )


def test_blatz_ko_energy_and_small_strain_limit():
    p = blatz_ko()
    F = jnp.array([[1.03, 0.02, 0], [0, 0.98, 0], [0, 0, 1.01]])
    power = 1 - 1 / (1 - 2 * p.poisson)

    def energy(F):
        return p.shear / 2 * (jnp.sum(F**2) - 3) - p.shear / power * (
            jnp.linalg.det(F) ** power - 1
        )

    np.testing.assert_allclose(
        kirchhoff_stress(F[None], p, "blatz-ko")[0], jax.grad(energy)(F) @ F.T, rtol=1e-12
    )
    np.testing.assert_allclose(kirchhoff_stress(jnp.eye(3)[None], p, "blatz-ko"), 0, atol=1e-12)


def test_steinberg_strength_and_energy():
    p = steinberg_guinan()
    g, y = steinberg_strength(0.0, 0.0, 300.0, 1.0, p)
    np.testing.assert_allclose(g, p.shear)
    np.testing.assert_allclose(y, p.yield_stress)
    assert steinberg_strength(0.1, 1e9, 300.0, 0.99, p)[1] > y
    assert steinberg_strength(0.0, 0.0, 500.0, 1.0, p)[1] < y
    h = initial_history(1, jnp.float64, p, p.density)
    strain = jnp.diag(jnp.array([0.01, -0.005, -0.005]))[None]
    result = update_steinberg(h, strain, jnp.ones(1), 1e-6, p)
    np.testing.assert_allclose(equivalent(result.stress), p.yield_stress, rtol=1e-12)
    assert float(result.plastic_strain[0]) > 0
    assert float(result.energy[0]) > 0
    np.testing.assert_allclose(result.temperature, 300 + result.energy / p.heat_capacity)
    assert bool(result.valid[0])
