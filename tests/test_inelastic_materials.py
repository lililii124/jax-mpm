import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, advance, block, hjc, jh2, johnson_cook, validate_state
from mpm_jax.materials import initial_history, update_history
from mpm_jax.materials.history import equivalent
from mpm_jax.materials.hjc import update_hjc
from mpm_jax.materials.jh2 import update_jh2
from mpm_jax.materials.johnson_cook import flow_stress, update_johnson_cook

jax.config.update("jax_enable_x64", True)


def test_jc_elastic_and_exact_perfect_plastic_return():
    p = johnson_cook(B=0.0, C=0.0, heat_fraction=0.0)
    h = initial_history(1, jnp.float64, p, p.density)
    small = jnp.diag(jnp.array([1e-5, -5e-6, -5e-6]))[None]
    elastic = update_johnson_cook(h, small, jnp.ones(1), 1e-6, p)
    np.testing.assert_allclose(elastic.stress, 2 * p.mu * small, rtol=1e-12)
    assert float(elastic.plastic_strain[0]) == 0
    large = small * 1000
    plastic = update_johnson_cook(h, large, jnp.ones(1), 1e-6, p)
    np.testing.assert_allclose(equivalent(plastic.stress), p.A, rtol=1e-11)
    exact = (equivalent(2 * p.mu * large) - p.A) / (3 * p.mu)
    np.testing.assert_allclose(plastic.plastic_strain, exact, rtol=1e-11)
    unloaded = update_johnson_cook(plastic, -small, jnp.ones(1), 1e-6, p)
    np.testing.assert_allclose(unloaded.plastic_strain, plastic.plastic_strain)
    assert bool(jnp.all(plastic.valid))


def test_jc_hardening_rate_temperature_and_heat():
    p = johnson_cook()
    assert flow_stress(0.2, 1e3, 293.15, p) > flow_stress(0.1, 1e3, 293.15, p)
    assert flow_stress(0.1, 1e3, 293.15, p) > flow_stress(0.1, 1.0, 293.15, p)
    assert flow_stress(0.1, 1.0, 500.0, p) < flow_stress(0.1, 1.0, 293.15, p)
    assert flow_stress(0.1, 1.0, p.melt_temperature, p) == 0
    h = initial_history(1, jnp.float64, p, p.density)
    strain = jnp.diag(jnp.array([0.01, -0.005, -0.005]))[None]
    result = jax.jit(update_johnson_cook)(h, strain, jnp.ones(1), 1e-5, p)
    q = equivalent(result.stress)
    np.testing.assert_allclose(
        q, flow_stress(result.plastic_strain, result.plastic_rate, h.temperature, p), rtol=1e-10
    )
    np.testing.assert_allclose(
        result.temperature - h.temperature,
        p.heat_fraction * q * result.plastic_strain / (p.density * p.heat_capacity),
        rtol=1e-10,
    )
    assert bool(jnp.all(result.valid))


def test_jc_failure_latches():
    p = johnson_cook(failure=(0.001, 0, 0, 0, 0))
    h = initial_history(1, jnp.float64, p, p.density)
    strain = jnp.diag(jnp.array([0.01, -0.005, -0.005]))[None]
    h = update_johnson_cook(h, strain, jnp.ones(1), 1e-5, p)
    assert bool(h.failed[0])
    np.testing.assert_array_equal(h.stress, 0)
    again = update_johnson_cook(h, strain, jnp.ones(1), 1e-5, p)
    np.testing.assert_array_equal(again.stress, 0)
    np.testing.assert_array_equal(again.plastic_strain, h.plastic_strain)


def test_jh2_hydrostatic_eos_and_tension():
    p = jh2(K2=40e9, K3=15e9)
    h = initial_history(1, jnp.float64, p, p.density)
    strain = -1e-3 * jnp.eye(3)[None]
    J = jnp.array([np.exp(-0.003)])
    result = update_jh2(h, strain, J, 1e-6, p)
    mu = 1 / J - 1
    np.testing.assert_allclose(result.pressure, p.K1 * mu + p.K2 * mu**2 + p.K3 * mu**3)
    np.testing.assert_array_equal(result.damage, 0)
    tension = update_jh2(h, -strain, 1 / J, 1e-6, p)
    np.testing.assert_array_equal(tension.damage, 1)
    np.testing.assert_array_equal(tension.stress, 0)


def test_hjc_elastic_crush_unload_and_damage_gate():
    p = hjc()
    h = initial_history(1, jnp.float64, p, p.density)
    mu = 0.0003
    J = jnp.array([1 / (1 + mu)])
    de = jnp.log(J)[:, None, None] * jnp.eye(3) / 3
    elastic = update_hjc(h, de, J, 1e-6, p)
    np.testing.assert_allclose(elastic.pressure, p.crush_pressure / p.crush_strain * mu, rtol=1e-12)
    np.testing.assert_array_equal(elastic.plastic_volume, 0)
    compressed_J = jnp.array([1 / 1.02])
    compressed = update_hjc(
        h, jnp.log(compressed_J)[:, None, None] * jnp.eye(3) / 3, compressed_J, 1e-6, p
    )
    assert float(compressed.plastic_volume[0]) > 0
    assert float(compressed.compaction[0]) > 0
    released = update_hjc(
        compressed, -jnp.log(compressed_J)[:, None, None] * jnp.eye(3) / 3, jnp.ones(1), 1e-6, p
    )
    np.testing.assert_array_equal(released.plastic_volume, compressed.plastic_volume)
    np.testing.assert_array_equal(released.compaction, compressed.compaction)
    assert float(released.damage[0]) >= float(compressed.damage[0])


def test_incremental_rigid_rotation():
    angle = 0.43
    Q = jnp.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    for kind, p in (("johnson-cook", johnson_cook()), ("jh2", jh2()), ("hjc", hjc())):
        h = initial_history(1, jnp.float64, p, p.density)
        sigma = jnp.diag(jnp.array([1e4, -1e4, 0.0]))[None]
        h = h._replace(stress=sigma)
        final = update_history(h, Q[None], jnp.eye(3)[None], 1e-6, p, kind)
        np.testing.assert_allclose(final.stress, Q @ sigma @ Q.T, atol=1e-3)
        np.testing.assert_allclose(final.plastic_strain, 0, atol=1e-14)


def test_incremental_models_run_in_3d():
    for kind, p in (("johnson-cook", johnson_cook()), ("jh2", jh2()), ("hjc", hjc())):
        cfg = Config(dim=3, n_grid=12, dt=1e-7, gravity=(0, 0, 0), material=kind)
        h = block(
            cfg,
            (0.3,) * 3,
            (0.6,) * 3,
            density=p.density,
            material=p,
            dtype=jnp.float64,
            particles_per_cell=1,
        )
        A = jnp.diag(jnp.array([100.0, -50.0, -50.0]))
        h = h._replace(v=(h.x - 0.45) @ A.T, C=jnp.broadcast_to(A, h.C.shape))
        final = advance(h, p, cfg, 30)
        validate_state(final, cfg)
        assert float(jnp.max(equivalent(final.history.stress))) > 1e5
