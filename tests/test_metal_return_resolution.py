"""Focused return-resolution checks; no full impact/phase validation."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials.shock_metal import deviatoric_return, flow_stress, shock_metal

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('batched', [False, True])
def test_exact_replay_particle_meets_unchanged_return_tolerance(batched):
    # Exact small replay input exported by run40071942761, particle2986.
    # Fixture SHA2562674aacded8dbf6504c0b3b28efe7034c31457940e4d17295221e988defc4a6c.
    # This is a numerical regression, never evidence of the original run cause.
    with np.load(Path(__file__).parent/'fixtures/metal_return_replay_particle_2986.npz', allow_pickle=False) as f:
        case = {name: jnp.asarray(f[name]) for name in f.files}
    if batched:
        case = {name: value[None, ...] for name, value in case.items()}
    p = shock_metal()
    result = jax.jit(deviatoric_return)(*(case[name] for name in ('tau', 'ep', 'strain', 'rotation', 'J', 'dt', 'T')), p)
    strain = .5*(case['strain']+jnp.swapaxes(case['strain'], -1, -2))
    strain -= jnp.trace(strain, axis1=-2, axis2=-1)[..., None, None]*jnp.eye(3)/3
    trial = case['tau']+2*p.shear_modulus*strain
    qtrial = jnp.sqrt(1.5*jnp.sum(trial**2, axis=(-2, -1)))
    unchanged_tolerance = 256*jnp.finfo(trial.dtype).eps*jnp.maximum(p.A*case['J'], qtrial)+1e-6
    assert np.asarray(result.valid).all()
    assert np.asarray(jnp.abs(result.yield_residual) <= unchanged_tolerance).all()
    assert np.asarray(result.plastic_increment > 0).all()
    assert np.asarray(result.specific_return_dissipation >= 0).all()
    np.testing.assert_allclose(result.specific_plastic_heat+result.specific_cold_work,
                               result.specific_plastic_work, rtol=2e-15, atol=1e-20)


def test_near_initial_yield_with_fractional_hardening_converges():
    p = shock_metal()
    ep = jnp.asarray([0., 1e-20, 1e-18, 1e-16, 1e-14])
    q = .99*flow_stress(ep, 0., 300., p)+1e-5
    tau = q[:, None, None]*jnp.diag(jnp.array([2/3, -1/3, -1/3]))
    result = jax.jit(deviatoric_return)(tau, ep, jnp.zeros((5, 3, 3)), jnp.eye(3), .99, 3.2e-10, 300., p)
    tol = 256*jnp.finfo(tau.dtype).eps*jnp.maximum(p.A*.99, q)+1e-6
    assert np.asarray(result.valid).all()
    assert np.asarray(jnp.abs(result.yield_residual) <= tol).all()
