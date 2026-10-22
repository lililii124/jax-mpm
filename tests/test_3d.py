import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, advance, block, elastic, fluid, validate_state
from mpm_jax.core import grid_to_particle, grid_update, particle_to_grid, stencil
from mpm_jax.materials import kirchhoff_stress

jax.config.update("jax_enable_x64", True)


def test_3d_transfer_moments_and_conservation():
    cfg = Config(dim=3, n_grid=16, gravity=(0, 0, 0), boundary_cells=0)
    state = block(cfg, (0.3,) * 3, (0.6,) * 3, dtype=jnp.float64)
    rng = np.random.default_rng(8)
    state = state._replace(
        v=jnp.asarray(rng.normal(size=state.v.shape)),
        C=jnp.asarray(rng.normal(size=state.C.shape)),
        F=state.F.at[:, 0, 1].set(0.02),
    )
    support = stencil(state.x, cfg)
    assert support.weights.shape[1] == 27
    np.testing.assert_allclose(support.weights.sum(1), 1, atol=1e-14)
    moment = jnp.einsum("pk,pki,pkj->pij", support.weights, support.dpos, support.dpos)
    np.testing.assert_allclose(
        moment, np.broadcast_to(cfg.dx**2 / 4 * np.eye(3), moment.shape), atol=1e-14
    )
    gm, gp = particle_to_grid(state, elastic(), cfg)
    np.testing.assert_allclose(gm.sum(), state.mass.sum(), rtol=1e-14)
    np.testing.assert_allclose(gp.sum(0), (state.mass[:, None] * state.v).sum(0), atol=1e-12)
    gx = np.indices((17,) * 3).reshape(3, -1).T * cfg.dx
    orbital = np.sum(np.cross(state.x, state.mass[:, None] * state.v), axis=0)
    C = state.C
    spin = (
        cfg.dx**2
        / 4
        * jnp.stack(
            (C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0], C[:, 1, 0] - C[:, 0, 1]), axis=-1
        )
    )
    np.testing.assert_allclose(
        np.cross(gx, gp).sum(0), orbital + (state.mass[:, None] * spin).sum(0), atol=1e-12
    )


def test_3d_affine_reproduction():
    cfg = Config(dim=3, n_grid=16, gravity=(0, 0, 0), boundary_cells=0)
    state = block(cfg, dtype=jnp.float64)
    A = jnp.array([[0.1, -0.3, 0.1], [0.2, 0.1, -0.1], [-0.2, 0.4, 0.1]])
    state = state._replace(v=state.x @ A.T + 0.2, C=jnp.broadcast_to(A, state.C.shape))
    s = stencil(state.x, cfg)
    gm, gp = particle_to_grid(state, elastic(), cfg, s)
    v, C = grid_to_particle(grid_update(gm, gp, cfg), s, cfg)
    np.testing.assert_allclose(v, state.v, atol=1e-13)
    np.testing.assert_allclose(C, state.C, atol=1e-12)


@pytest.mark.parametrize("kind,material", [("elastic", elastic()), ("fluid", fluid())])
def test_3d_free_fall(kind, material):
    cfg = Config(dim=3, n_grid=12, material=kind, boundary_cells=0)
    state = block(cfg, velocity=(0.1, 0.2, -0.1), particles_per_cell=1, dtype=jnp.float64)
    n = 30
    final = advance(state, material, cfg, n)
    validate_state(final, cfg)
    expected = (
        state.x + n * cfg.dt * state.v + n * (n + 1) / 2 * cfg.dt**2 * jnp.array(cfg.gravity)
    )
    np.testing.assert_allclose(final.x, expected, atol=1e-12)
    np.testing.assert_allclose(state.volume.sum(), 0.3**3, atol=1e-14)


def test_3d_hyperelastic_energy_gradient_and_rotation():
    mat = elastic()
    F = jnp.array([[1.1, 0.03, -0.02], [0.02, 0.95, 0.01], [0, -0.01, 1.02]])

    def energy(F):
        logJ = jnp.log(jnp.linalg.det(F))
        return mat.mu / 2 * (jnp.sum(F**2) - 3) - mat.mu * logJ + mat.lam / 2 * logJ**2

    np.testing.assert_allclose(
        kirchhoff_stress(F[None], mat, "elastic")[0], jax.grad(energy)(F) @ F.T, rtol=1e-12
    )
    Q, _ = jnp.linalg.qr(F)
    np.testing.assert_allclose(kirchhoff_stress(Q[None], mat, "elastic"), 0, atol=1e-10)


def test_3d_material_parameter_gradient():
    cfg = Config(dim=3, n_grid=12, gravity=(0, 0, 0), boundary_cells=0)
    state = block(cfg, (0.4,) * 3, (0.6,) * 3, particles_per_cell=1, dtype=jnp.float64)
    state = state._replace(F=state.F.at[:, 0, 0].set(1.04))

    def response(E):
        return jnp.mean((advance(state, elastic(E), cfg, 20).x[:, 0] - 0.5) ** 2)

    derivative = jax.jit(jax.grad(response))(20000.0)
    fd = (response(20002.0) - response(19998.0)) / 4
    assert abs(float(derivative)) > 1e-12
    np.testing.assert_allclose(derivative, fd, rtol=2e-5)
