import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials.elastic import linear_stress, orthotropic
from mpm_jax.materials.woven import (
    woven_elastic,
    woven_energy,
    woven_kirchhoff_stress,
    woven_wave_speed_bound,
)

jax.config.update("jax_enable_x64", True)


def rotation(angle):
    return np.array(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
    )


def material(phase=(0, 1), orientation=None):
    if orientation is None:
        orientation = np.stack([rotation(0.37 * i) for i in range(len(phase))])
    return woven_elastic(
        matrix_young=2e9,
        matrix_poisson=0.3,
        matrix_density=1200,
        yarn_young=(40e9, 4e9, 4e9),
        yarn_poisson=(0.25, 0.25, 0.3),
        yarn_shear=(2e9, 2e9, 4e9 / 2.6),
        yarn_density=1600,
        orientation=orientation,
        phase=np.asarray(phase),
    )


def test_rigid_rotation_has_zero_strain_energy_and_stress():
    mat = material()
    F = jnp.asarray(np.stack((rotation(0.9), rotation(-0.6))))
    np.testing.assert_allclose(woven_energy(F, mat), 0, atol=1e-6)
    np.testing.assert_allclose(woven_kirchhoff_stress(F, mat), 0, atol=2e-5)


def test_stress_is_energy_derivative_for_both_phases():
    mat = material()
    F = jnp.array(
        [
            [[1.006, 0.009, 0.002], [0.002, 0.998, 0], [0, 0.003, 1.002]],
            [[0.997, 0.004, 0], [0.001, 1.007, 0.003], [0.002, 0, 0.999]],
        ]
    )
    first_piola = jax.grad(lambda deformation: jnp.sum(woven_energy(deformation, mat)))(
        F
    )
    expected = first_piola @ jnp.swapaxes(F, -1, -2)
    np.testing.assert_allclose(
        woven_kirchhoff_stress(F, mat), expected, rtol=1e-12, atol=1e-6
    )


def test_superposed_spatial_rotation_preserves_energy_and_rotates_stress():
    mat = material()
    F = jnp.broadcast_to(
        jnp.array([[1.008, 0.006, 0], [0, 0.997, 0.002], [0, 0, 1.001]]), (2, 3, 3)
    )
    R = jnp.asarray(rotation(0.83))
    np.testing.assert_allclose(
        woven_energy(R @ F, mat), woven_energy(F, mat), rtol=1e-11
    )
    np.testing.assert_allclose(
        woven_kirchhoff_stress(R @ F, mat),
        R @ woven_kirchhoff_stress(F, mat) @ R.T,
        rtol=1e-11,
        atol=1e-5,
    )


def test_reference_direction_changes_yarn_response_but_not_matrix():
    F = jnp.broadcast_to(jnp.diag(jnp.array([1.01, 1.0, 1.0])), (2, 3, 3))
    straight = material(orientation=np.stack((np.eye(3), np.eye(3))))
    turned = material(orientation=np.stack((rotation(np.pi / 2), rotation(np.pi / 2))))
    first, second = woven_kirchhoff_stress(F, straight), woven_kirchhoff_stress(
        F, turned
    )
    np.testing.assert_allclose(first[0], second[0])
    assert float(first[1, 0, 0]) > 8 * float(second[1, 0, 0])


def test_yarn_initial_tangent_matches_independent_orthotropic_linear_law():
    mat = material(phase=(1,), orientation=np.eye(3)[None])
    H = jnp.array([[0.005, 0.003, 0.002], [-0.001, -0.002, 0.001], [0, 0, 0.003]])
    _, tangent = jax.jvp(
        lambda F: woven_kirchhoff_stress(F, mat), (jnp.eye(3)[None],), (H[None],)
    )
    reference = orthotropic((40e9, 4e9, 4e9), (0.25, 0.25, 0.3), (2e9, 2e9, 4e9 / 2.6))
    np.testing.assert_allclose(
        tangent[0], linear_stress((H + H.T) / 2, reference), rtol=1e-12
    )


def test_initial_wave_bound_covers_directional_acoustic_speeds():
    mat = material(phase=(1,), orientation=np.eye(3)[None])
    tangent = np.asarray(
        jax.hessian(lambda F: woven_energy(F[None], mat)[0])(jnp.eye(3))
    )
    bound = woven_wave_speed_bound(mat)
    for angle in np.linspace(0, 2 * np.pi, 17):
        normal = np.array((np.cos(angle), np.sin(angle), 0.25))
        normal /= np.linalg.norm(normal)
        acoustic = np.einsum("ijkl,j,l->ik", tangent, normal, normal)
        speed = np.sqrt(np.linalg.eigvalsh(acoustic).max() / mat.yarn_density)
        assert speed <= bound * (1 + 1e-12)
    assert bound >= np.sqrt((mat.matrix_lam + 2 * mat.matrix_mu) / mat.matrix_density)


def test_invalid_phase_and_improper_local_frames_are_rejected():
    with pytest.raises(ValueError, match="phase must"):
        material(phase=(2,))
    with pytest.raises(ValueError, match="proper rotation"):
        material(phase=(1,), orientation=np.diag([1, 1, -1])[None])
