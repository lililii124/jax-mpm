import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials import hjc, initial_history, update_hjc

jax.config.update('jax_enable_x64', True)


def source_table():
    return json.loads((Path(__file__).parent/'data/materials/hjc_compacted_unloading.json').read_text())


@pytest.mark.parametrize('profile', ['legacy-v3', 'strain-floor'])
def test_compacted_unloading_matches_compiled_mat111_block_before_tensile_cutoff(profile):
    table = source_table()
    points = np.asarray(table['values'])
    u, oracle = points[:, 0], points[:, 1]
    p = hjc(profile=profile, compacted_tension='mat111-linear', **table['parameters_Pa'])
    mu = p.lock_strain+(1+p.lock_strain)*u
    J = jnp.asarray(1/(1+mu))
    old = initial_history(len(J), jnp.float64, p, p.density)._replace(
        compaction=jnp.ones(len(J)), plastic_volume=jnp.full(len(J), p.lock_strain),
        max_compression=jnp.full(len(J), .2))
    new = jax.jit(update_hjc)(old, jnp.zeros((len(J), 3, 3)), J, 1e-6, p)
    expected = np.maximum(oracle, -p.tensile_strength)
    np.testing.assert_allclose(new.pressure, expected, atol=3e-5, rtol=3e-12)
    np.testing.assert_allclose(new.stress, -expected[:, None, None]*np.eye(3), atol=3e-5, rtol=3e-12)
    np.testing.assert_array_equal(new.plastic_volume, old.plastic_volume)
    np.testing.assert_array_equal(new.damage, old.damage)
    assert np.all(new.valid)
    assert np.all(points[:, 2] == 0)


@pytest.mark.parametrize('damage', [0., .25, 1.])
def test_compacted_source_branch_still_uses_old_damage_tensile_cutoff(damage):
    p = hjc(compacted_tension='mat111-linear')
    old = initial_history(2, jnp.float64, p, p.density)._replace(
        compaction=jnp.ones(2), damage=jnp.full(2, damage), max_compression=jnp.full(2, .2))
    u = np.array([-1e-5, -.001])
    J = jnp.asarray(1/(1+p.lock_strain+(1+p.lock_strain)*u))
    new = jax.jit(update_hjc)(old, jnp.zeros((2, 3, 3)), J, 1e-6, p)
    expected = np.maximum(p.K1*u, -p.tensile_strength*(1-damage))
    np.testing.assert_allclose(new.pressure, expected, atol=2e-5)
    np.testing.assert_array_equal(new.damage, old.damage)
    assert np.all(new.valid)


def test_unloading_choice_is_inactive_for_partial_compaction_and_default_is_preserved():
    p = hjc(profile='strain-floor')
    explicit = hjc(profile='strain-floor', compacted_tension='legacy-polynomial')
    for a, b in zip(p, explicit):
        np.testing.assert_array_equal(a, b)
    changed = p._replace(mat111_compacted_tension=True)
    old = initial_history(4, jnp.float64, p, p.density)._replace(
        compaction=jnp.array([0., .2, .8, .999]), plastic_volume=jnp.array([0., .01, .06, .09]),
        max_compression=jnp.array([0., .03, .08, .109]))
    mu = jnp.array([0., .009999, .059999, .089999])
    J = 1/(1+mu)
    strain = jnp.zeros((4, 3, 3))
    legacy = jax.jit(update_hjc)(old, strain, J, 1e-6, p)
    source = jax.jit(update_hjc)(old, strain, J, 1e-6, changed)
    for a, b in zip(legacy, source):
        np.testing.assert_array_equal(a, b)


def test_source_branch_differs_from_legacy_only_after_full_compaction():
    p = hjc()
    old = initial_history(1, jnp.float64, p, p.density)._replace(
        compaction=jnp.ones(1), max_compression=jnp.full(1, .2))
    u = -1e-5
    J = jnp.array([1/(1+p.lock_strain+(1+p.lock_strain)*u)])
    strain = jnp.zeros((1, 3, 3))
    legacy = jax.jit(update_hjc)(old, strain, J, 1e-6, p)
    source = jax.jit(update_hjc)(old, strain, J, 1e-6, p._replace(mat111_compacted_tension=True))
    np.testing.assert_allclose(source.pressure, -850000., atol=2e-5)
    np.testing.assert_allclose(legacy.pressure-source.pressure, -17.100208, atol=2e-5)
    assert legacy.valid.all() and source.valid.all()


def test_compacted_unloading_selector_rejects_unknown_conventions():
    with pytest.raises(ValueError, match='compacted_tension'):
        hjc(compacted_tension='unknown')
