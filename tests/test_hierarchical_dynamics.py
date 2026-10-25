import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.boundaries import fixed_box
from mpm_jax.hierarchical_dynamics import (
    angular_momentum, elastic_step, initial_state,
)
from mpm_jax.hierarchical_grid import (
    grid_stencil, hierarchical_grid, moment_matrix, uniform_grid,
)
from mpm_jax.materials import neo_hookean

jax.config.update("jax_enable_x64", True)


def hierarchy():
    return hierarchical_grid(cells=16, domain_size=1., refined_lower=(3.5 / 16,) * 3,
                             refined_upper=(12.5 / 16,) * 3)


def test_uniform_reference_has_shifted_fine_and_coarse_moments():
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.uniform(.1, .9, (111, 3)))
    for cells, offset in ((16, 0.), (32, .5)):
        grid = uniform_grid(cells=cells, domain_size=1., offset=offset)
        support = grid_stencil(x, grid)
        assert bool(support.valid)
        np.testing.assert_allclose(support.weights.sum(1), 1., rtol=0, atol=1e-14)
        np.testing.assert_allclose((support.weights[..., None] * support.displacements).sum(1),
                                   0., rtol=0, atol=1e-15)
        np.testing.assert_allclose(moment_matrix(support),
                                   np.broadcast_to(np.eye(3) / (4 * cells**2), (len(x), 3, 3)),
                                   rtol=0, atol=1e-16)


def test_free_translation_crosses_both_refinement_interfaces():
    grid = hierarchy()
    x = np.stack(np.meshgrid(np.linspace(.12, .18, 6), np.linspace(.45, .55, 5),
                              np.linspace(.45, .55, 5), indexing="ij"), -1).reshape(-1, 3)
    state = initial_state(x, 1e-5, 1., grid, velocity=(.7, 0., 0.))
    material = neo_hookean(.05, .2)

    @jax.jit
    def run(s):
        return jax.lax.scan(lambda carry, _: elastic_step(carry, material, grid, .005),
                            s, None, length=200)

    final, records = run(state)
    assert bool(final.valid) and np.asarray(records["valid"]).all()
    np.testing.assert_allclose(final.x, x + np.array([.7, 0., 0.]), rtol=0, atol=2e-12)
    np.testing.assert_allclose(final.v, state.v, rtol=0, atol=2e-12)
    np.testing.assert_allclose(final.F, state.F, rtol=0, atol=2e-11)
    np.testing.assert_allclose(angular_momentum(final), angular_momentum(state), rtol=0, atol=1e-14)
    assert np.max(np.abs(records["ledger_error"])) < 1e-16
    assert np.max(np.abs(records["moment_advection"])) < 1e-16


def test_stressed_rotating_body_retains_momenta_and_accounts_for_energy():
    grid = hierarchy()
    x = np.random.default_rng(31).uniform(.35, .65, (200, 3))
    A = np.array([[0., -.2, 0.], [.2, 0., 0.], [0., 0., 0.]])
    F = np.array([[1.03, .02, 0.], [0., .98, 0.], [0., 0., 1.01]])
    state = initial_state(x, 1e-4, 10., grid, velocity=(x - .5) @ A.T,
                          deformation=F, affine=A)
    material = neo_hookean(20., .25)

    @jax.jit
    def run(s):
        return jax.lax.scan(lambda carry, _: elastic_step(carry, material, grid, .001),
                            s, None, length=20)

    final, records = run(state)
    assert np.asarray(records["valid"]).all()
    np.testing.assert_allclose((final.mass[:, None] * final.v).sum(0),
                               (state.mass[:, None] * state.v).sum(0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(angular_momentum(final), angular_momentum(state), rtol=0, atol=1e-13)
    assert np.max(np.abs(records["force_gradient_remainder"])) < 1e-15
    assert np.max(np.abs(records["ledger_error"])) < 1e-15
    assert np.min(records["boundary_loss"]) >= 0
    assert np.max(records["p2g_change"]) < 1e-14
    assert np.max(records["g2p_change"]) < 1e-14


def test_fixed_body_reaction_balances_gravity_kick():
    grid = uniform_grid(cells=16, domain_size=1.)
    x = np.random.default_rng(8).uniform(.3, .7, (100, 3))
    state = initial_state(x, 1e-4, 10., grid)
    dt, gravity = .001, (0., -9.81, 0.)
    final, record = elastic_step(state, neo_hookean(20., .25), grid, dt,
                                 gravity=gravity, boundaries=(fixed_box((0.,) * 3, (1.,) * 3),))
    np.testing.assert_array_equal(final.x, state.x)
    np.testing.assert_array_equal(final.v, state.v)
    np.testing.assert_array_equal(final.F, state.F)
    work = .5 * float(state.mass.sum()) * gravity[1]**2 * dt**2
    np.testing.assert_allclose(record["body_work"], work, rtol=1e-14)
    np.testing.assert_allclose(record["boundary_loss"], work, rtol=1e-14)
    assert float(record["boundary_work"]) == 0
    assert abs(float(record["ledger_error"])) < 1e-17
