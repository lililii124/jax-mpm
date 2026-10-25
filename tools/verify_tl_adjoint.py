"""Finite-dimensional power-adjoint check for the TL contact kinematic map."""

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from mpm_jax.total_lagrangian import ReferenceGrid
from mpm_jax.metal_tl_midpoint import tl_velocity_map, tl_velocity_map_transpose


def main():
    rng = np.random.default_rng(20261026)
    particles, nodes = 7, 11
    indices = rng.integers(0, nodes, size=(particles, 4), dtype=np.int32)
    weights = rng.normal(size=(particles, 4))
    gradients = rng.normal(size=(particles, 4, 3))
    mass = rng.uniform(.2, 2., size=particles)
    reference = ReferenceGrid(
        positions=jnp.zeros((particles, 3)), indices=jnp.asarray(indices),
        weights=jnp.asarray(weights), gradients=jnp.asarray(gradients),
        volume=jnp.ones((particles,)), particle_mass=jnp.asarray(mass),
        nodal_mass=jnp.ones((nodes,)), nodes=jnp.zeros((nodes, 3)),
        valid=jnp.asarray(True))
    velocity = jnp.asarray(rng.normal(size=(nodes, 3)))
    # Deliberately non-symmetric affine impulses exercise component ordering.
    phase_impulse = jnp.asarray(rng.normal(size=(particles, 4, 3)))
    E0_cases = {
        'identity': jnp.eye(3),
        'nonsymmetric': jnp.asarray([[1.0, .2, -.1], [.3, .8, .05], [-.2, .4, 1.1]])}
    result = {}
    for name, E0 in E0_cases.items():
        Av = tl_velocity_map(velocity, reference, jnp.asarray(mass), E0)
        Aty = tl_velocity_map_transpose(phase_impulse, reference,
                                        jnp.asarray(mass), E0, nodes)
        lhs = float(jnp.vdot(Av, phase_impulse))
        rhs = float(jnp.vdot(velocity, Aty))
        error = abs(lhs-rhs)
        result[name] = {'dot_Av_y': lhs, 'dot_v_Aty': rhs,
                        'absolute_error': error,
                        'relative_error': error/max(abs(lhs), abs(rhs), 1e-30)}
        if error > 2e-12*max(1., abs(lhs), abs(rhs)):
            raise AssertionError(f'{name} A/A^T power identity failed: {error:g}')
    print(result)


if __name__ == '__main__':
    main()
