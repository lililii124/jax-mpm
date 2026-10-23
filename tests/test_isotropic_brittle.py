import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials.isotropic_brittle import (
    brittle_cauchy_stress,
    brittle_energy,
    brittle_first_piola,
    brittle_kirchhoff_stress,
    brittle_opening,
    brittle_scales,
    initial_brittle_history,
    isotropic_brittle,
    update_brittle,
)

jax.config.update("jax_enable_x64", True)


def uniaxial(strain, poisson=0.25):
    strain = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    stretches = jnp.stack((1 + strain, 1 - poisson * strain, 1 - poisson * strain), axis=-1)
    return jnp.eye(3) * stretches[..., None, :]


def history_path(strains, material):
    def advance(history, value):
        F = uniaxial(value, material.poisson)
        current, tau = update_brittle(history, F, material)
        P = tau @ jnp.swapaxes(jnp.linalg.inv(F), -1, -2)
        return current, (current, P[0, 0, 0])

    return jax.jit(lambda x: jax.lax.scan(advance, initial_brittle_history(1), x))(
        jnp.asarray(strains)
    )


@pytest.mark.parametrize("length", [0, -1e-5, 2e-4, 3e-4, np.nan, []])
def test_unresolvable_or_invalid_band_is_rejected(length):
    with pytest.raises(ValueError):
        isotropic_brittle(characteristic_length=length)


def test_scales_and_invalid_elastic_parameters():
    p = isotropic_brittle(characteristic_length=25e-6)
    scales = brittle_scales(p)
    np.testing.assert_allclose(scales, [.01, .25e-6, 2e-6, .08, 200e-6], rtol=1e-14)
    for kwargs in ({"young": 0}, {"poisson": .5}, {"poisson": -1},
                   {"density": np.inf}, {"fracture_energy": -1}, {"tensile_strength": 0}):
        with pytest.raises(ValueError):
            isotropic_brittle(characteristic_length=25e-6, **kwargs)


def test_actual_uniaxial_nominal_and_cauchy_stress():
    p = isotropic_brittle(characteristic_length=25e-6)
    strains = np.array([0, .002, .01, .025, .04, .06, .08, .09])
    _, (histories, nominal) = history_path(strains, p)
    expected = np.where(strains <= .01, p.young * strains,
                        p.tensile_strength * np.maximum(.08 - strains, 0) / .07)
    np.testing.assert_allclose(nominal, expected, rtol=3e-12, atol=2e-6)
    F = uniaxial(strains)
    flat_history = jax.tree.map(lambda value: value[:, 0], histories)
    P = brittle_first_piola(F, flat_history, p)
    sigma = brittle_cauchy_stress(F, flat_history, p)
    np.testing.assert_allclose(P[:, 1, 1], 0, atol=2e-6)
    np.testing.assert_allclose(P[:, 2, 2], 0, atol=2e-6)
    np.testing.assert_allclose(sigma[:, 0, 0], expected / (1 - .25 * strains)**2,
                               rtol=3e-12, atol=3e-6)
    assert bool(histories.failed[-1, 0])
    assert np.all(histories.valid)


@pytest.mark.parametrize("length", [10e-6, 25e-6, 100e-6])
def test_fracture_energy_and_opening_softening_area(length):
    p = isotropic_brittle(characteristic_length=length)
    scales = brittle_scales(p)
    delta = np.concatenate((np.linspace(0, float(scales.onset_displacement), 41),
                            np.linspace(float(scales.onset_displacement),
                                        float(scales.failure_displacement), 201)[1:]))
    _, (history, stress) = history_path(delta / length, p)
    opening = np.asarray(brittle_opening(history, p))[:, 0]
    np.testing.assert_allclose(np.trapezoid(stress, delta), p.fracture_energy, rtol=1e-11)
    np.testing.assert_allclose(np.trapezoid(np.asarray(stress)[40:], opening[40:]),
                               p.fracture_energy, rtol=1e-11)
    np.testing.assert_allclose(stress[40:], p.tensile_strength * (1 - opening[40:] / 2e-6),
                               atol=3e-6, rtol=2e-11)
    assert np.all(np.diff(np.asarray(history.damage)[:, 0]) >= -1e-14)


def test_unloading_reloading_has_fixed_damage_and_zero_closed_cycle_work():
    p = isotropic_brittle(characteristic_length=25e-6)
    strains = np.concatenate((np.linspace(0, .04, 81), np.linspace(.04, 0, 41)[1:],
                              np.linspace(0, .04, 41)[1:]))
    _, (history, nominal) = history_path(strains, p)
    np.testing.assert_allclose(history.kappa[80:], .04, rtol=1e-13)
    np.testing.assert_allclose(history.damage[80:], history.damage[80], rtol=1e-13)
    expected = (1 - float(history.damage[80, 0])) * p.young * strains[80:]
    np.testing.assert_allclose(nominal[80:], expected, rtol=1e-12, atol=3e-6)
    np.testing.assert_allclose(np.trapezoid(np.asarray(nominal)[80:], strains[80:]), 0, atol=1e-7)


def test_compression_does_not_initiate_damage_and_does_not_restore_it():
    p = isotropic_brittle(characteristic_length=25e-6)
    old = initial_brittle_history(1)
    compressed, tau = update_brittle(old, uniaxial(-.02), p)
    np.testing.assert_allclose(compressed.damage, 0, atol=0)
    np.testing.assert_allclose(tau[0, 0, 0] / .98, -.02 * p.young, rtol=1e-12)
    stretched, _ = update_brittle(old, uniaxial(.04), p)
    damaged, tau = update_brittle(stretched, uniaxial(-.02), p)
    np.testing.assert_array_equal(damaged.damage, stretched.damage)
    np.testing.assert_allclose(tau[0, 0, 0] / .98,
                               -(1 - damaged.damage[0]) * .02 * p.young, rtol=1e-12)
    failed, _ = update_brittle(old, uniaxial(.09), p)
    reclosed, tau = update_brittle(failed, uniaxial(-.02), p)
    np.testing.assert_array_equal(tau, 0)
    assert bool(reclosed.failed[0])


def test_rotation_reflection_and_rigid_motion():
    p = isotropic_brittle(characteristic_length=25e-6)
    old = initial_brittle_history(1)
    F = jnp.array([[[1.025, .008, 0], [0, .994, .003], [.002, 0, .997]]])
    angle = .73
    R = jnp.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    M = jnp.diag(jnp.array([-1., 1., 1.]))
    result, tau = update_brittle(old, F, p)
    rotated, rot_tau = update_brittle(old, R @ F, p)
    reflected, ref_tau = update_brittle(old, M @ F @ M, p)
    np.testing.assert_allclose(rotated.damage, result.damage, rtol=1e-11)
    np.testing.assert_allclose(reflected.damage, result.damage, rtol=1e-11)
    np.testing.assert_allclose(rot_tau, R @ tau @ R.T, rtol=1e-11, atol=3e-5)
    np.testing.assert_allclose(ref_tau, M @ tau @ M, rtol=1e-11, atol=3e-5)
    rigid, rigid_tau = update_brittle(old, R[None], p)
    np.testing.assert_allclose(rigid_tau, 0, atol=3e-5)
    np.testing.assert_array_equal(rigid.damage, 0)


@pytest.mark.parametrize("deformation", [np.eye(3), np.diag([1.005, .99875, .99875]),
                                         np.array([[1.02, .005, 0], [0, .996, .003], [.001, 0, 1.]])])
def test_fixed_damage_energy_derivative_is_first_piola(deformation):
    p = isotropic_brittle(characteristic_length=25e-6)
    old = initial_brittle_history(1)._replace(damage=jnp.array([.4]), kappa=jnp.array([.02]))
    F = jnp.asarray(deformation)
    gradient = jax.grad(lambda matrix: brittle_energy(matrix[None], old, p).sum())(F)
    expected = brittle_first_piola(F[None], old, p)[0]
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient, expected, rtol=3e-11, atol=3e-5)


def test_history_is_independent_of_monotonic_step_partition():
    p = isotropic_brittle(characteristic_length=25e-6)
    coarse, _ = history_path([0, .02, .04, .06], p)
    fine, _ = history_path(np.linspace(0, .06, 601), p)
    for a, b in zip(coarse, fine):
        np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        brittle_kirchhoff_stress(uniaxial(.06), coarse, p),
        brittle_kirchhoff_stress(uniaxial(.06), fine, p), rtol=1e-13,
    )


def test_particle_lengths_and_invalid_deformation_flags():
    p = isotropic_brittle(characteristic_length=np.array([10e-6, 25e-6, 100e-6]))
    history, _ = jax.jit(update_brittle)(initial_brittle_history(3), uniaxial([.03, .03, .03]), p)
    assert np.all(np.diff(history.damage) > 0)
    bad = jnp.stack((jnp.diag(jnp.array([-1., 1., 1.])),
                     jnp.diag(jnp.array([0., 1., 1.])), jnp.full((3, 3), jnp.nan)))
    invalid, tau = jax.jit(update_brittle)(initial_brittle_history(3), bad, p)
    np.testing.assert_array_equal(invalid.valid, False)
    np.testing.assert_array_equal(tau, 0)
