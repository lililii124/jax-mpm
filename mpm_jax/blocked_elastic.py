"""Bound temporary nodal-force storage for reference-grid Neo-Hookean MPM."""
import jax
import jax.numpy as jnp

from .materials.elastic import kirchhoff_stress


def make_internal_force(reference, material, block_size=4096):
    """Build an internal-force operator; external loads are added by the caller.

    This preserves the reference support, per-point material, volume and
    contribution ordering. The result is a function of the deformation
    gradient. It does not update material history or include body forces.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError('A positive integer point block size is required')
    count = len(reference.positions)
    pad = (-count) % block_size
    indices = jnp.pad(reference.indices, ((0, pad), (0, 0)))
    gradients = jnp.pad(reference.gradients, ((0, pad), (0, 0), (0, 0)))
    volume = jnp.pad(reference.volume, ((0, pad),))

    def internal_force(F):
        if F.shape != (count, 3, 3):
            raise ValueError('Deformation gradients must match the reference points')
        tau = kirchhoff_stress(F, material, 'elastic')
        cofactor = jnp.stack((jnp.cross(F[:, :, 1], F[:, :, 2]),
                             jnp.cross(F[:, :, 2], F[:, :, 0]),
                             jnp.cross(F[:, :, 0], F[:, :, 1])), axis=-1)
        J = jnp.sum(F[:, :, 0]*cofactor[:, :, 0], axis=-1)
        P = (tau@cofactor)/J[:, None, None]
        padded_P = jnp.pad(P, ((0, pad), (0, 0), (0, 0)))

        def one(block, nodal):
            offset = block*block_size
            stress = jax.lax.dynamic_slice_in_dim(padded_P, offset, block_size)
            grad = jax.lax.dynamic_slice_in_dim(gradients, offset, block_size)
            vol = jax.lax.dynamic_slice_in_dim(volume, offset, block_size)
            ids = jax.lax.dynamic_slice_in_dim(indices, offset, block_size)
            load = -vol[:, None, None]*jnp.einsum('pij,pkj->pki', stress, grad)
            return nodal.at[ids].add(load)

        return jax.lax.fori_loop(0, (len(F)+pad)//block_size, one,
                                 jnp.zeros_like(reference.nodes))

    return internal_force
