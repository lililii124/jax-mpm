import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials import linear_elastic, linear_stress, orthotropic, transverse_isotropic

jax.config.update("jax_enable_x64", True)


def test_linear_uniaxial_and_shear():
    E, nu, strain = 200e9, 0.3, 0.001
    material = linear_elastic(E, nu)
    eps = jnp.diag(jnp.array([strain, -nu * strain, -nu * strain]))
    np.testing.assert_allclose(linear_stress(eps, material), np.diag([E * strain, 0, 0]), atol=1e-6)
    eps = jnp.zeros((3, 3)).at[0, 1].set(strain / 2).at[1, 0].set(strain / 2)
    np.testing.assert_allclose(linear_stress(eps, material)[0, 1], E / (2 * (1 + nu)) * strain)


def test_orthotropic_engineering_constants():
    E = np.array([100e9, 50e9, 30e9])
    nu = np.array([0.2, 0.15, 0.25])
    G = np.array([20e9, 12e9, 10e9])
    mat = orthotropic(E, nu, G)
    S = np.linalg.inv(mat.stiffness)
    np.testing.assert_allclose(S[:3, :3].diagonal(), 1 / E)
    np.testing.assert_allclose(-S[[1, 2, 2], [0, 0, 1]] * E[[0, 0, 1]], nu)
    np.testing.assert_allclose(mat.stiffness[3:, 3:].diagonal(), G[::-1])
    assert np.linalg.eigvalsh(mat.stiffness).min() > 0
    for d in range(3):
        stress = np.zeros(6)
        stress[d] = 1e6
        strain = S @ stress
        result = linear_stress(jnp.diag(jnp.asarray(strain[:3])), mat)
        np.testing.assert_allclose(result, np.diag(stress[:3]), atol=1e-8)


def test_anisotropic_orientation():
    Q = jnp.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    args = ((100.0, 50.0, 30.0), (0.2, 0.15, 0.25), (20.0, 12.0, 10.0))
    local, rotated = orthotropic(*args), orthotropic(*args, orientation=Q)
    strain = jnp.array([[0.01, 0.002, 0.003], [0.002, -0.004, 0.001], [0.003, 0.001, -0.002]])
    np.testing.assert_allclose(
        linear_stress(Q @ strain @ Q.T, rotated), Q @ linear_stress(strain, local) @ Q.T, atol=1e-14
    )


def test_transverse_isotropy_and_isotropic_limit():
    mat = transverse_isotropic(100.0, 20.0, 0.25, 0.3, 10.0)
    t = 0.63
    Q = jnp.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    strain = jnp.array([[0.01, 0.002, 0.003], [0.002, -0.004, 0.001], [0.003, 0.001, -0.002]])
    np.testing.assert_allclose(
        linear_stress(Q @ strain @ Q.T, mat), Q @ linear_stress(strain, mat) @ Q.T, atol=1e-14
    )
    iso = transverse_isotropic(100.0, 100.0, 0.25, 0.25, 40.0)
    np.testing.assert_allclose(iso.stiffness, linear_elastic(100.0, 0.25).stiffness, atol=1e-12)


def test_invalid_compliance_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        orthotropic((1, 1, 1), (0.9, 0.9, 0.9), (1, 1, 1))


def test_linear_parameter_gradient():
    strain = jnp.diag(jnp.array([0.01, 0.0, 0.0]))
    derivative = jax.grad(lambda E: linear_stress(strain, linear_elastic(E, 0.25))[0, 0])(100.0)
    np.testing.assert_allclose(derivative, 0.012, rtol=1e-12)
