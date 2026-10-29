"""Rectilinear reference quadrature for the existing 3D TL integrator."""
from itertools import product
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .materials.elastic import Material
from .materials.elastic import kirchhoff_stress
from .total_lagrangian import ReferenceGrid, TLState, deformation, forces
from types import SimpleNamespace


class Support(NamedTuple):
    positions: np.ndarray
    volume: np.ndarray
    nodes: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    spacing: np.ndarray
    shape: tuple


def rectilinear_support(positions, volume, spacing):
    x = np.asarray(positions, dtype=np.float64)
    h = np.asarray(spacing, dtype=np.float64)
    volume = np.broadcast_to(np.asarray(volume, dtype=np.float64), (len(x),)).copy()
    if (x.ndim != 2 or x.shape[1] != 3 or h.shape != (3,)
            or not np.isfinite(x).all() or not np.isfinite(h).all()
            or np.any(h <= 0) or np.any(volume <= 0) or not np.isfinite(volume).all()):
        raise ValueError('Finite 3D points, positive volumes and three positive grid spacings required')
    offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int32)
    q = x/h
    base = np.floor(q-.5).astype(np.int32)
    origin_index = base.min(axis=0)
    shape = tuple((base.max(axis=0)-origin_index+3).tolist())
    f = q-base
    w = np.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2), axis=1)
    dw = np.stack((f-1.5, -2*(f-1), f-.5), axis=1)/h
    selected = w[:, offsets, np.arange(3)]
    weights = selected.prod(axis=-1)
    derivative = dw[:, offsets, np.arange(3)]
    gradients = np.stack([derivative[..., a]*selected[..., (a+1)%3]*selected[..., (a+2)%3]
                          for a in range(3)], axis=-1)
    ijk = base[:, None, :]-origin_index+offsets
    indices = np.ravel_multi_index(tuple(ijk[..., a] for a in range(3)), shape).astype(np.int32)
    node_index = np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=-1)
    nodes = (node_index+origin_index)*h
    # Translation-invariant affine reproduction also checks anisotropic derivatives.
    affine = np.einsum('pki,pkj->pij', nodes[indices]-x[:, None, :], gradients)
    if (not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=2e-14)
            or not np.allclose(gradients.sum(axis=1)*h, 0, rtol=0, atol=2e-14)
            or not np.allclose(affine, np.eye(3), rtol=0, atol=2e-13)):
        raise ValueError('Reference partition or affine reproduction failed')
    return Support(x, volume, nodes, indices, weights, gradients, h, shape)


def reference_grid(support, density):
    """Keep geometric volume fixed; density alone sets particle/nodal inertia."""
    x = jnp.asarray(support.positions)
    volume = jnp.asarray(support.volume)
    density = jnp.broadcast_to(jnp.asarray(density), volume.shape)
    mass = density*volume
    indices = jnp.asarray(support.indices)
    weights = jnp.asarray(support.weights)
    nodes = jnp.asarray(support.nodes)
    nodal_mass = jnp.zeros(len(nodes), dtype=x.dtype).at[indices].add(mass[:, None]*weights)
    valid = jnp.all(jnp.isfinite(mass)) & jnp.all(mass > 0)
    return ReferenceGrid(x, indices, weights, jnp.asarray(support.gradients), volume,
                         mass, nodal_mass, nodes, valid)


def point_material(young, poisson):
    """Broadcast one isotropic elastic law per quadrature point, not per axis."""
    young, poisson = jnp.broadcast_arrays(jnp.asarray(young), jnp.asarray(poisson))
    young, poisson = young.reshape(-1), poisson.reshape(-1)
    return Material((young/(2*(1+poisson)))[:, None, None],
                    (young*poisson/((1+poisson)*(1-2*poisson)))[:, None, None],
                    (young/(3*(1-2*poisson)))[:, None, None])


def elastic_energy(F, reference, material):
    """Compressible Neo-Hookean reference energy, consistent with TL stress."""
    J = jnp.linalg.det(F)
    logJ = jnp.log(jnp.maximum(J, jnp.finfo(F.dtype).tiny))
    mu, lam = material.mu[:, 0, 0], material.lam[:, 0, 0]
    density = .5*mu*(jnp.sum(F*F, axis=(-2, -1))-3)-mu*logJ+.5*lam*logJ**2
    return jnp.sum(reference.volume*density)


def surface_weights(support, points):
    """Evaluate the volume grid basis at independent reference-face quadrature."""
    x = np.asarray(points)
    origin = support.nodes.min(axis=0)
    q = (x-origin)/support.spacing
    base = np.floor(q-.5).astype(np.int32)
    offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int32)
    f = q-base
    w = np.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2), axis=1)
    weights = w[:, offsets, np.arange(3)].prod(axis=-1)
    ijk = base[:, None, :]+offsets
    if np.any(ijk < 0) or np.any(ijk >= np.asarray(support.shape)):
        raise ValueError('Reference face lies outside the volume support')
    indices = np.ravel_multi_index(tuple(ijk[..., a] for a in range(3)), support.shape).astype(np.int32)
    return indices, weights


def initial_state(reference, material, load):
    u = jnp.zeros_like(reference.nodes)
    F = deformation(u, reference)
    zero = jnp.asarray(0., dtype=u.dtype)
    return TLState(u, u, load, F, None, zero, reference.valid,
                   jnp.zeros(3), jnp.zeros(3), zero, zero)


def step(state, reference, material, dt, load_start, load_end, fixed):
    """Velocity Verlet with actual nodal surface loads and fixed zero DOFs.

    The TLState body_work ledger holds external pressure work in this branch.
    There is no particle body acceleration, mass-scaled loading, or hidden damping.
    """
    mass = reference.nodal_mass[:, None]
    inverse = 1/jnp.where(mass > 0, mass, 1)
    a0 = jnp.where(fixed, 0., state.force*inverse)
    half = state.velocity+.5*dt*a0
    u = jnp.where(fixed, 0., state.displacement+dt*half)
    half = jnp.where(fixed, (u-state.displacement)/dt, half)
    F = deformation(u, reference)
    config = SimpleNamespace(material='elastic')
    internal = forces(F, None, reference, material, config, jnp.zeros_like(reference.positions))
    force = internal+load_end
    a1 = jnp.where(fixed, 0., force*inverse)
    velocity = jnp.where(fixed, 0., half+.5*dt*a1)
    J0 = jnp.where(fixed, mass*(half-state.velocity)-.5*dt*state.force, 0.)
    J1 = jnp.where(fixed, mass*(velocity-half)-.5*dt*force, 0.)
    reaction0 = jnp.where(fixed, mass*a0-state.force, 0.)
    reaction1 = jnp.where(fixed, mass*a1-force, 0.)
    external_impulse = .5*dt*jnp.sum(load_start+load_end, axis=0)
    boundary_impulse = jnp.sum(J0+J1, axis=0)
    x0, x1 = reference.nodes+state.displacement, reference.nodes+u
    torque = .5*dt*jnp.sum(jnp.cross(x0, load_start)+jnp.cross(x1, load_end), axis=0)
    torque += jnp.sum(jnp.cross(x0, J0)+jnp.cross(x1, J1), axis=0)
    work = jnp.sum(.5*(load_start+load_end)*(u-state.displacement))
    boundary_work = jnp.sum(.5*(reaction0+reaction1)*(u-state.displacement))
    valid = (state.valid & jnp.all(jnp.linalg.det(F) > 0) & jnp.all(jnp.isfinite(F))
             & jnp.all(jnp.isfinite(u)) & jnp.all(jnp.isfinite(velocity))
             & jnp.all(jnp.isfinite(force)))
    following = TLState(u, velocity, force, F, None, state.time+dt, valid,
                        state.impulse+external_impulse+boundary_impulse,
                        state.angular_impulse+torque, state.body_work+work,
                        state.boundary_work+boundary_work)
    return following, external_impulse, boundary_impulse


def face_stress(F, material):
    tau = kirchhoff_stress(F, material, 'elastic')
    stress = tau/jnp.linalg.det(F)[:, None, None]
    dev = stress-jnp.trace(stress, axis1=-2, axis2=-1)[:, None, None]*jnp.eye(3)/3
    return jnp.sqrt(1.5*jnp.sum(dev*dev, axis=(-2, -1))+1e-24)
