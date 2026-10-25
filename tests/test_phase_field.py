import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.fibre_mpm import build_layout
from mpm_jax.phase_field import (material, spectral_split, reference_basis,
    damage_system, solve_damage, interpolate, State, explicit_step)


def fixture():
    x = np.array(list(__import__('itertools').product(np.arange(.025, .5, .05),
        np.arange(.025, .3, .05), np.arange(.025, .2, .05))))
    layout = build_layout(x, np.zeros(len(x), np.int32), dx=.1, padding_cells=2)
    basis = reference_basis(jnp.asarray(x), layout)
    return x, layout, basis, jnp.full(len(x), .05**3)


def test_spectral_split_and_rotation():
    law = material(1000, .2, 1, .1)
    e = jnp.diag(jnp.array([.01, .02, .03]))
    p, sp, sn = spectral_split(e, law)
    n, _, compression = spectral_split(-e, law)
    assert p > 0 and n == 0
    np.testing.assert_allclose(sn, 0, atol=1e-14)
    np.testing.assert_allclose(compression, -sp)
    q, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
    pp, spp, _ = spectral_split(q@e@q.T, law)
    np.testing.assert_allclose(pp, p, rtol=1e-12)
    np.testing.assert_allclose(spp, q@sp@q.T, atol=1e-12)


def test_basis_partition_and_affine_gradient():
    x, layout, b, volume = fixture()
    from mpm_jax.fibre_mpm import grid_coordinates
    coordinates, _ = grid_coordinates(layout, jnp.arange(layout.node_count))
    np.testing.assert_allclose(jnp.sum(b.weight, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(b.gradient, axis=1), 0, atol=1e-13)
    value, grad = interpolate(coordinates[:, 0], b)
    np.testing.assert_allclose(value, x[:, 0], atol=1e-14)
    np.testing.assert_allclose(grad, np.broadcast_to([1, 0, 0], grad.shape), atol=1e-13)


def test_operator_symmetry_energy_minimization_and_irreversibility():
    x, layout, b, volume = fixture()
    law = material(1000, .2, .1, .1)
    h = jnp.full(len(x), 5.)
    n = layout.node_count
    fixed = jnp.zeros(n, bool)
    zero = jnp.zeros(n)
    op, rhs, bound = damage_system(h, volume, b, n, law)
    rng = np.random.default_rng(42)
    a, c = map(jnp.asarray, rng.normal(size=(2, n)))
    np.testing.assert_allclose(a@op(c), c@op(a), atol=1e-13)
    assert a@op(a) >= 0
    solution = jax.jit(solve_damage, static_argnames=('max_iterations',))(h, volume, b, zero, law, fixed, zero)
    assert solution.converged
    assert .5*solution.damage@op(solution.damage)-solution.damage@rhs < 0
    dp, _ = interpolate(solution.damage, b)
    np.testing.assert_allclose(dp, 2*(1-law.residual)*5/(law.gc/law.length+2*(1-law.residual)*5), atol=5e-6)
    unload = solve_damage(jnp.zeros_like(h), volume, b, solution.damage, law, fixed, zero)
    assert unload.converged
    assert np.min(unload.damage-solution.damage) >= 0


def test_coupled_rigid_translation_and_uniform_tension():
    x, layout, b, volume = fixture()
    law = material(1000, .2, 1, .1)
    n = layout.node_count
    state = State(jnp.zeros_like(jnp.asarray(x)), jnp.broadcast_to(jnp.array([.1, 0, 0]), x.shape),
        jnp.zeros((len(x), 3, 3)), jnp.zeros(len(x)), jnp.zeros(n))
    mask = jnp.zeros((n, 3), bool)
    next_state, damage, reaction = explicit_step(state, volume, 1., b, law, .001,
        mask, jnp.zeros((n, 3)), jnp.zeros(n, bool), jnp.zeros(n))
    np.testing.assert_allclose(next_state.velocity, state.velocity, atol=1e-14)
    np.testing.assert_allclose(next_state.strain, 0, atol=1e-14)
    assert damage.converged
    state = state._replace(strain=jnp.broadcast_to(jnp.diag(jnp.array([.1, 0, 0])), (len(x), 3, 3)))
    next_state, damage, reaction = explicit_step(state, volume, 1., b, law, .0001,
        mask, jnp.zeros((n, 3)), jnp.zeros(n, bool), jnp.zeros(n))
    assert damage.converged and np.max(next_state.damage) > 0
    assert np.max(next_state.history) > 0
    np.testing.assert_allclose(jnp.sum(volume[:, None]*(next_state.velocity-state.velocity), axis=0), 0, atol=1e-13)



def test_stress_is_energy_derivative_and_compression_stays_intact():
    x, layout, b, volume = fixture()
    law = material(1000, .2, .1, .1)
    strain = np.array([[.012, .003, .001], [.003, -.005, .002], [.001, .002, .004]])
    damage = .4
    degradation = (1-law.residual)*(1-damage)**2+law.residual
    _, sp, sn = spectral_split(jnp.asarray(strain), law)
    def energy(e):
        plus, _, _ = spectral_split(jnp.asarray(e), law)
        total = .5*law.lam*np.trace(e)**2+law.mu*np.sum(e*e)
        return float(total+(degradation-1)*plus)
    direction = np.array([[.3, .2, .1], [.2, -.1, -.2], [.1, -.2, .4]])
    numerical = (energy(strain+1e-7*direction)-energy(strain-1e-7*direction))/2e-7
    np.testing.assert_allclose(numerical, np.sum(np.asarray(degradation*sp+sn)*direction), rtol=1e-8)
    history, _, _ = spectral_split(-jnp.broadcast_to(jnp.eye(3)*.1, (len(x), 3, 3)), law)
    zero = jnp.zeros(layout.node_count)
    result = solve_damage(history, volume, b, zero, law, jnp.zeros(layout.node_count, bool), zero)
    assert result.converged
    np.testing.assert_array_equal(result.damage, 0)


def test_transfer_damping_and_boundary_energy_audit():
    from mpm_jax.phase_field_diagnostics import audited_step
    x, layout, b, volume = fixture()
    n = layout.node_count
    law = material(1000, .2, .1, .1)
    rng = np.random.default_rng(7)
    state = State(jnp.zeros_like(jnp.asarray(x)), jnp.asarray(rng.normal(size=x.shape)*.1),
        jnp.broadcast_to(jnp.diag(jnp.array([.01, 0, 0])), (len(x), 3, 3)),
        jnp.zeros(len(x)), jnp.zeros(n))
    mask = jnp.zeros((n, 3), bool).at[:n//2, 1].set(True)
    updated, result, reaction, audit = audited_step(state, volume, 1., b, law,
        .0001, mask, jnp.full((n, 3), .02), jnp.zeros(n, bool), jnp.zeros(n), damping=5.)
    assert result.converged
    assert audit.p2g_loss >= -1e-15 and audit.g2p_loss >= -1e-15
    assert audit.damping_loss >= 0
    np.testing.assert_allclose(audit.kinetic_balance_error, 0, atol=1e-15)
    np.testing.assert_allclose(audit.total_balance_error, 0, atol=1e-15)
    assert np.isfinite(float(audit.constitutive_work_defect))


def test_damage_solver_at_tiny_loading_and_active_bounds():
    x, layout, b, volume = fixture()
    law = material(1000, .2, .1, .1)
    n = layout.node_count
    zero = jnp.zeros(n)
    history = jnp.where(jnp.asarray(x[:, 1]) > .2, 1e-12, 0.)
    result = solve_damage(history, volume, b, zero, law, jnp.zeros(n, bool), zero)
    assert result.converged
    assert float(result.relative_kkt) <= 1e-7
    assert np.min(result.damage) >= 0


def test_lagged_damage_retains_tensile_history():
    x, layout, b, volume = fixture()
    n = layout.node_count
    law = material(1000, .2, .1, .1)
    state = State(jnp.zeros_like(jnp.asarray(x)), jnp.zeros_like(jnp.asarray(x)),
        jnp.broadcast_to(jnp.eye(3)*.01, (len(x), 3, 3)),
        jnp.zeros(len(x)), jnp.zeros(n))
    args = (volume, 1., b, law, .0001, jnp.zeros((n, 3), bool),
        jnp.zeros((n, 3)), jnp.zeros(n, bool), jnp.zeros(n))
    held, result, _ = explicit_step(state, *args, update_damage=False)
    np.testing.assert_array_equal(held.damage, state.damage)
    assert np.min(held.history) > 0 and result.iterations == 0
    advanced, result, _ = explicit_step(held, *args, update_damage=True)
    assert result.converged and np.max(advanced.damage) > 0
    assert np.min(advanced.history-held.history) >= 0
