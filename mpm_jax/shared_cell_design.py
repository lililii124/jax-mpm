"""Differentiable shared-cell 3-D design map for explicit MPM geometry.

The design field is shared by all tiles. Smooth weights are optimization
surrogates; thresholded voxel geometry is the explicit material mask.
This module does not solve MPM dynamics or optimize a structural objective.
"""
from typing import NamedTuple
from numbers import Real
import math

import jax
import jax.numpy as jnp


class SharedCellMap(NamedTuple):
    density: jax.Array       # (nz, ny, nx), cell design variables in [0,1]
    design_xyz: jax.Array    # voxel-center coordinates in the single cell
    full_xyz: jax.Array      # repeated voxel centers for all tiles
    full_density: jax.Array  # tiled continuous proxy weights
    cell_id: jax.Array       # tile ID per full-domain voxel
    local_ijk: jax.Array     # local voxel index per full-domain voxel


def project_fixed_volume(logits, volume_fraction, beta=8.0, iterations=48):
    """Smooth projection with a scalar offset chosen for prescribed mean.

    The implicit root derivative accounts for the change in offset, so the
    projected total remains fixed under changes to the design field.
    """
    logits = jnp.asarray(logits)
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        logits = logits.astype(jnp.float32)
    vf = jnp.asarray(volume_fraction, dtype=logits.dtype)
    if logits.ndim != 3:
        raise ValueError("logits must have shape (nz, ny, nx)")
    if isinstance(volume_fraction, Real) and (not math.isfinite(volume_fraction) or not 0 < volume_fraction < 1):
        raise ValueError("volume_fraction must be strictly between 0 and 1")
    if beta <= 0 or iterations < 1:
        raise ValueError("beta and iterations must be positive")
    def residual(offset):
        return jnp.mean(jax.nn.sigmoid(beta * (logits-offset))) - vf
    def solve(function, initial):
        del initial
        lo, hi = jnp.min(logits)-40.0/beta, jnp.max(logits)+40.0/beta
        def body(_, bounds):
            a,b=bounds; mid=(a+b)*0.5
            return jnp.where(function(mid)>0,mid,a), jnp.where(function(mid)>0,b,mid)
        lo,hi=jax.lax.fori_loop(0,iterations,body,(lo,hi))
        return (lo+hi)*0.5
    offset=jax.lax.custom_root(residual,jnp.asarray(0,dtype=logits.dtype),solve,
                               lambda g,y:y/g(jnp.asarray(1,dtype=logits.dtype)))
    return jax.nn.sigmoid(beta*(logits-offset))


def build_shared_cell_map(density, cell_shape, repeats, cell_origin=(0., 0., 0.), cell_size=(1., 1., 1.)):
    """Tile one density array into a full explicit multi-cell voxel candidate.

    Array ordering is (z,y,x); repeat ordering is (nx,ny,nz). Cell boundaries
    are periodic for design identity in x/y through exact tiling. z is tiled
    geometrically too, while its within-cell field is unconstrained/aperiodic.
    """
    d = jnp.asarray(density)
    if d.ndim != 3:
        raise ValueError("density must have shape (nz, ny, nx)")
    if tuple(d.shape[::-1]) != tuple(cell_shape):
        raise ValueError("density shape must match cell_shape=(nx, ny, nz)")
    nxr, nyr, nzr = map(int, repeats)
    if min(nxr, nyr, nzr) < 1:
        raise ValueError("repeats must be positive")
    nx, ny, nz = map(int, cell_shape)
    origin, size = jnp.asarray(cell_origin), jnp.asarray(cell_size)
    axes = [origin[k] + (jnp.arange(n, dtype=d.dtype) + .5) * size[k] / n
            for k, n in enumerate((nx, ny, nz))]
    zz, yy, xx = jnp.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    xyz = jnp.stack((xx, yy, zz), axis=-1).reshape((-1, 3))
    iz,iy,ix=jnp.meshgrid(jnp.arange(nz),jnp.arange(ny),jnp.arange(nx),indexing='ij')
    local=jnp.stack((ix,iy,iz),axis=-1).reshape((-1,3))
    tz,ty,tx=jnp.meshgrid(jnp.arange(nzr),jnp.arange(nyr),jnp.arange(nxr),indexing='ij')
    tiles=jnp.stack((tx,ty,tz),axis=-1).reshape((-1,3))
    nt, nvox = tiles.shape[0], nx*ny*nz
    full_xyz=(xyz[None,:,:]+tiles[:,None,:]*size).reshape((-1,3))
    weights=jnp.broadcast_to(d.reshape((1,nvox)),(nt,nvox)).reshape(-1)
    ids=jnp.broadcast_to(tiles[:,None,:],(nt,nvox,3)).reshape((-1,3))
    indices=jnp.broadcast_to(local[None,:,:],(nt,nvox,3)).reshape((-1,3))
    return SharedCellMap(d,xyz,full_xyz,weights,ids,indices)


def threshold_material_mask(mapped, threshold=0.5):
    """Return explicit voxel positions above a threshold (not differentiable).

    Thresholding may change material volume; use a fixed-count mask when a
    binary reconstruction must preserve a voxel-quantized material amount.
    """
    mask = mapped.full_density >= threshold
    return mapped.full_xyz[mask], mask


def fixed_count_material_mask(mapped, solid_voxels_per_cell):
    """Select a fixed number of equal-volume voxels in every identical cell.

    Binary reconstruction is not differentiable and fixes volume only to one
    voxel of a cell. The continuous material fraction used during optimization
    must also be checked after reconstruction and during the MPM simulation.
    """
    count = int(solid_voxels_per_cell)
    nvox = mapped.density.size
    if count < 1 or count > nvox:
        raise ValueError("solid voxel count must lie between 1 and cell size")
    _, indices = jax.lax.top_k(mapped.density.reshape(-1), count)
    cell_mask = jnp.zeros(nvox, dtype=jnp.bool_).at[indices].set(True)
    repeats = mapped.full_density.shape[0] // nvox
    mask = jnp.tile(cell_mask, repeats)
    return mapped.full_xyz[mask], mask


def weighted_particle_candidates(mapped, particles_per_voxel=1):
    """Return repeated positions and proxy material weights for downstream MPM.

    These are candidate quadrature points; weights below one represent a
    relaxation and are not physical holes. Apply threshold_material_mask for
    actual binary voxel geometry.
    """
    if particles_per_voxel != 1:
        raise NotImplementedError("use one voxel-center candidate per voxel")
    return mapped.full_xyz, mapped.full_density
