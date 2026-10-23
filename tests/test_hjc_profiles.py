import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials import hjc, initial_history, update_hjc
from mpm_jax.materials.history import equivalent, equivalent_strain

jax.config.update("jax_enable_x64", True)


def shear_increment(amplitude=0.003, count=1):
    strain = jnp.zeros((count, 3, 3), dtype=jnp.float64)
    return strain.at[:, 0, 1].set(amplitude).at[:, 1, 0].set(amplitude)


def test_hjc_profile_selection_and_default_compatibility():
    default = hjc()
    explicit = hjc(profile="legacy-v3")
    for left, right in zip(default, explicit):
        np.testing.assert_array_equal(left, right)
    assert default.legacy_v3 is True
    assert hjc(profile="strain-floor").legacy_v3 is False
    with pytest.raises(ValueError, match="profile"):
        hjc(profile="unknown")


def test_hjc_zero_pressure_shear_uses_minimum_fracture_strain():
    p = hjc(profile="strain-floor", C=0)
    old = initial_history(1, jnp.float64, p, p.density)
    strain = shear_increment()
    result = jax.jit(update_hjc)(old, strain, jnp.ones(1), 1e-4, p)
    q_trial = 2 * p.shear * np.sqrt(3) * 0.003
    expected_dep = (q_trial - p.A * p.compressive_strength) / (3 * p.shear)
    assert p.D1 * (p.tensile_strength / p.compressive_strength) ** p.D2 < p.minimum_fracture_strain
    np.testing.assert_allclose(result.plastic_strain, expected_dep, rtol=1e-12)
    np.testing.assert_allclose(result.damage, expected_dep / p.minimum_fracture_strain, rtol=1e-12)
    np.testing.assert_array_equal(result.plastic_volume, 0)
    np.testing.assert_array_equal(result.pressure, 0)
    assert 0 < float(result.damage[0]) < 1
    legacy = jax.jit(update_hjc)(old, strain, jnp.ones(1), 1e-4, p._replace(legacy_v3=True))
    np.testing.assert_allclose(legacy.stress, result.stress, rtol=1e-12)
    np.testing.assert_array_equal(legacy.damage, 0)
    assert bool(jnp.all(result.valid))


def test_hjc_crushing_damage_includes_plastic_volume_above_strain_floor():
    p = hjc(profile="strain-floor", C=0)
    old = initial_history(1, jnp.float64, p, p.density)
    compression = 0.02
    J = jnp.array([1 / (1 + compression)])
    strain = jnp.log(J)[:, None, None] * jnp.eye(3) / 3
    result = update_hjc(old, strain, J, 1e-4, p)
    bulk = p.crush_pressure / p.crush_strain
    hardening = (p.lock_pressure - p.crush_pressure) / p.lock_strain
    expected_depv = (bulk * compression - p.crush_pressure) / (bulk + hardening)
    expected_pressure = bulk * (compression - expected_depv)
    fracture_strain = p.D1 * ((expected_pressure + p.tensile_strength) / p.compressive_strength) ** p.D2
    assert fracture_strain > p.minimum_fracture_strain
    np.testing.assert_allclose(result.plastic_strain, 0, atol=1e-14)
    np.testing.assert_allclose(result.plastic_volume, expected_depv, rtol=1e-12)
    np.testing.assert_allclose(result.damage, expected_depv / fracture_strain, rtol=1e-12)
    assert bool(jnp.all(result.valid))


def test_hjc_tensile_pressure_reduces_cohesive_strength():
    p = hjc(profile="strain-floor", C=0)
    pressures = p.tensile_strength * jnp.array([0.0, -0.2, -0.5, -0.8])
    J = 1 / (1 + pressures / (p.crush_pressure / p.crush_strain))
    old = initial_history(4, jnp.float64, p, p.density)._replace(damage=jnp.full(4, 0.2))
    strain = shear_increment(count=4) + jnp.log(J)[:, None, None] * jnp.eye(3) / 3
    result = jax.jit(update_hjc)(old, strain, J, 1e-4, p)
    expected = p.A * (1 + pressures / p.tensile_strength) * 0.8 * p.compressive_strength
    np.testing.assert_allclose(result.pressure, pressures, atol=3e-6)
    np.testing.assert_allclose(equivalent(result.stress), expected, rtol=2e-11)
    assert np.all(np.diff(np.asarray(equivalent(result.stress))) < 0)
    assert bool(jnp.all(result.valid))


def test_hjc_tensile_cutoff_zero_strength_still_accumulates_plasticity_and_damage():
    p = hjc(profile="strain-floor", C=0)
    old = initial_history(1, jnp.float64, p, p.density)
    J = jnp.array([1.001])
    strain = shear_increment() + jnp.log(J)[:, None, None] * jnp.eye(3) / 3
    result = jax.jit(update_hjc)(old, strain, J, 1e-4, p)
    expected_dep = 2 * np.sqrt(3) * 0.003 / 3
    np.testing.assert_array_equal(result.pressure, -p.tensile_strength)
    np.testing.assert_allclose(equivalent(result.stress), 0, atol=1e-9)
    np.testing.assert_allclose(result.plastic_strain, expected_dep, rtol=1e-12)
    np.testing.assert_allclose(result.damage, expected_dep / p.minimum_fracture_strain, rtol=1e-12)
    assert bool(jnp.all(result.valid))


def test_hjc_fully_damaged_material_flows_at_zero_strength_and_keeps_confinement_strength():
    p = hjc(profile="strain-floor", C=0)
    old = initial_history(3, jnp.float64, p, p.density)._replace(damage=jnp.ones(3))
    pressures = jnp.array([0.0, -p.tensile_strength / 2, p.tensile_strength])
    J = 1 / (1 + pressures / (p.crush_pressure / p.crush_strain))
    strain = shear_increment(count=3) + jnp.log(J)[:, None, None] * jnp.eye(3) / 3
    result = jax.jit(update_hjc)(old, strain, J, 1e-4, p)
    q_trial = 2 * p.shear * np.sqrt(3) * 0.003
    confined = p.B * (p.tensile_strength / p.compressive_strength) ** p.N * p.compressive_strength
    np.testing.assert_allclose(equivalent(result.stress), [0, 0, confined], rtol=2e-11)
    np.testing.assert_allclose(result.pressure, [0, 0, p.tensile_strength], atol=3e-6)
    np.testing.assert_allclose(result.plastic_strain[:2], q_trial / (3 * p.shear), rtol=1e-12)
    np.testing.assert_allclose(result.plastic_rate, result.plastic_strain / 1e-4, rtol=1e-12)
    np.testing.assert_allclose(result.plastic_tensor[:2], shear_increment(count=2), atol=1e-14)
    np.testing.assert_array_equal(result.damage, 1)
    np.testing.assert_array_equal(result.failed, False)
    assert bool(jnp.all(result.valid))


@pytest.mark.parametrize("rate", [1e-3, 1.0, 100.0])
def test_hjc_rate_multiplier_has_no_subreference_softening(rate):
    p = hjc(profile="strain-floor", C=0.1)
    old = initial_history(1, jnp.float64, p, p.density)
    strain = shear_increment()
    dt = float(equivalent_strain(strain)[0]) / rate
    result = update_hjc(old, strain, jnp.ones(1), dt, p)
    expected = p.A * p.compressive_strength * (1 + p.C * max(np.log(rate / p.reference_rate), 0))
    np.testing.assert_allclose(equivalent(result.stress), expected, rtol=1e-12)
    assert bool(jnp.all(result.valid))
