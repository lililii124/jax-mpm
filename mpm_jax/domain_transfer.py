"""Volume-averaged quadratic transfers on fixed fragment geometry."""

from functools import partial
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import stencil
from .cut_fragments import current_domain
from .cut_geometry import clip,bulk_quadrature,boundary_quadrature


class DomainStencil(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    gradients: jax.Array
    dpos: jax.Array
    moment: jax.Array
    inverse_moment: jax.Array
    volume: jax.Array
    valid: jax.Array


def _tiles(state,config,tile_radius):
    domain=current_domain(state)
    base=jnp.floor(state.x/config.dx-.5).astype(jnp.int32)
    offsets=jnp.asarray(list(product(range(-tile_radius,tile_radius+1),repeat=2)))
    lower=(base+offsets+.5)*config.dx;unit=jnp.eye(2,dtype=state.x.dtype)
    def tile(lo):
        q=domain
        for axis in (0,1):
            q=clip(q,unit[axis],lo[axis])
            q=clip(q,-unit[axis],-lo[axis]-config.dx)
        return q
    return domain,jax.vmap(tile)(lower),base,lower


def _nodes(base,config,tile_radius):
    width=2*tile_radius+3
    lower=base-tile_radius
    coordinates=lower+jnp.asarray(list(product(range(width),repeat=2)))
    bounded=jnp.clip(coordinates,0,config.n_grid)
    return coordinates,bounded[:,0]*(config.n_grid+1)+bounded[:,1],lower,width


def _covered(domain,lower,config):
    first=lower.min(axis=0);last=lower.max(axis=0)+config.dx
    tolerance=128*jnp.finfo(domain.vertices.dtype).eps*config.domain_size
    active=jnp.arange(len(domain.vertices))<domain.count
    return jnp.all(jnp.where(active[:,None],(domain.vertices>=first-tolerance)&(domain.vertices<=last+tolerance),True))


@partial(jax.jit,static_argnames=('config','tile_radius'))
def domain_stencil(state,config,tile_radius=1):
    """Integrate a fragment over a bounded tile patch split at spline knots.

    A fragment extending outside the tile patch is invalid, never truncated.
    The default has 3x3 knot tiles and 5x5 candidate grid nodes.
    """
    if config.dim!=2 or not isinstance(tile_radius,int) or tile_radius<0:
        raise ValueError('Require 2D geometry and a nonnegative integer tile radius')
    domain,pieces,base,lower=_tiles(state,config,tile_radius)
    dtype=state.x.dtype;unit=jnp.eye(2,dtype=dtype)
    points,quadrature=jax.vmap(bulk_quadrature)(pieces)
    points=points.reshape(-1,2);quadrature=quadrature.reshape(-1)
    basis=stencil(points,config,active=quadrature>0)
    coordinates,indices,node_lower,width=_nodes(base,config,tile_radius);count=width**2
    inside=jnp.all((coordinates>=0)&(coordinates<=config.n_grid),axis=1)
    actual=jnp.stack((basis.indices//(config.n_grid+1),basis.indices%(config.n_grid+1)),axis=-1)
    local=actual-node_lower
    destinations=local[:,:,0]*width+local[:,:,1]
    destinations=jnp.where(quadrature[:,None]>0,destinations,count)
    integral=jnp.zeros(count,dtype=dtype).at[destinations].add(quadrature[:,None]*basis.weights,mode='drop')
    derivative=jnp.zeros((count,2),dtype=dtype).at[destinations].add(quadrature[:,None,None]*basis.gradients,mode='drop')
    volume=state.reference_area*jnp.linalg.det(state.F)
    denominator=jnp.where(state.active,volume,1.)
    weights=integral/denominator;gradients=derivative/denominator
    dpos=coordinates*config.dx-state.x
    normalized=jnp.einsum('k,ki,kj->ij',weights,dpos/config.dx,dpos/config.dx)
    normalized=jnp.where(state.active,normalized,.25*unit)
    rank=(jnp.linalg.det(normalized)>1e-12)&jnp.all(jnp.diag(normalized)>0)
    inverse=jnp.linalg.inv(jnp.where(rank,normalized,unit))/config.dx**2
    area_error=jnp.abs(quadrature.sum()-volume)
    valid=state.valid&jnp.all(pieces.valid)&basis.valid&_covered(domain,lower,config)&rank
    valid &= (~state.active)|((volume>0)&(area_error<=2e-11*volume)&jnp.all((weights==0)|inside))
    return DomainStencil(indices,weights,gradients,dpos,normalized*config.dx**2,inverse,volume,valid)


@partial(jax.jit,static_argnames=('config','tile_radius'))
def domain_stencil_separable(state,config,tile_radius=1):
    """Integrate the same tiled cut domain with factored quadratic weights.

    Each knot tile's quadrature is contracted to its nine local nodes before
    scattering. This avoids constructing a core stencil for every quadrature
    point while retaining the exact polygon pieces and quadrature rule.
    """
    if config.dim!=2 or not isinstance(tile_radius,int) or tile_radius<0:
        raise ValueError('Require 2D geometry and a nonnegative integer tile radius')
    domain,pieces,base,lower=_tiles(state,config,tile_radius)
    dtype=state.x.dtype
    points,quadrature=jax.vmap(bulk_quadrature)(pieces)
    u=(points-lower[:,None,:])/config.dx
    ux,uy=u[...,0],u[...,1]
    wx=jnp.stack((.5*(1-ux)**2,.5+ux*(1-ux),.5*ux**2),axis=-1)
    wy=jnp.stack((.5*(1-uy)**2,.5+uy*(1-uy),.5*uy**2),axis=-1)
    dwx=jnp.stack((ux-1,1-2*ux,ux),axis=-1)/config.dx
    dwy=jnp.stack((uy-1,1-2*uy,uy),axis=-1)/config.dx
    tile_weights=jnp.einsum('tq,tqi,tqj->tij',quadrature,wx,wy)
    tile_grad_x=jnp.einsum('tq,tqi,tqj->tij',quadrature,dwx,wy)
    tile_grad_y=jnp.einsum('tq,tqi,tqj->tij',quadrature,wx,dwy)
    tile_gradients=jnp.stack((tile_grad_x,tile_grad_y),axis=-1)

    coordinates,indices,node_lower,width=_nodes(base,config,tile_radius)
    count=width**2
    local_axis=jnp.arange(3)
    tile_nodes=base[None,:]+jnp.asarray(list(product(range(-tile_radius,tile_radius+1),repeat=2)))
    tile_nodes=tile_nodes[:,:,None]+local_axis[None,None,:]
    # Arrange each tile's (x,y) tensor-product nodes in row-major grid order.
    local_x=tile_nodes[:,0,:]-node_lower[0]
    local_y=tile_nodes[:,1,:]-node_lower[1]
    destinations=(local_x[:,:,None]*width+local_y[:,None,:]).reshape(-1)
    integral=jnp.zeros(count,dtype=dtype).at[destinations].add(tile_weights.reshape(-1),mode='drop')
    derivative=jnp.zeros((count,2),dtype=dtype).at[destinations].add(tile_gradients.reshape(-1,2),mode='drop')

    volume=state.reference_area*jnp.linalg.det(state.F)
    denominator=jnp.where(state.active,volume,1.)
    weights=integral/denominator;gradients=derivative/denominator
    dpos=coordinates*config.dx-state.x
    normalized=jnp.einsum('k,ki,kj->ij',weights,dpos/config.dx,dpos/config.dx)
    unit=jnp.eye(2,dtype=dtype)
    normalized=jnp.where(state.active,normalized,.25*unit)
    rank=(jnp.linalg.det(normalized)>1e-12)&jnp.all(jnp.diag(normalized)>0)
    inverse=jnp.linalg.inv(jnp.where(rank,normalized,unit))/config.dx**2

    tile_active=jnp.sum(quadrature,axis=1)>0
    complete=jnp.all((tile_nodes>=0)&(tile_nodes<=config.n_grid),axis=(1,2))
    basis_valid=jnp.all((~tile_active)|complete)
    inside=jnp.all((coordinates>=0)&(coordinates<=config.n_grid),axis=1)
    area_error=jnp.abs(quadrature.sum()-volume)
    valid=state.valid&jnp.all(pieces.valid)&basis_valid&_covered(domain,lower,config)&rank
    valid &= (~state.active)|((volume>0)&(area_error<=2e-11*volume)&jnp.all((weights==0)|inside))
    return DomainStencil(indices,weights,gradients,dpos,normalized*config.dx**2,inverse,volume,valid)


class BoundaryIntegral(NamedTuple):
    indices: jax.Array
    vectors: jax.Array
    valid: jax.Array


@partial(jax.jit,static_argnames=('config','tile_radius'))
def plane_boundary_integral(state,normal,offset,config,tile_radius=1):
    """Integrate N_i n_out ds on domain edges lying on the specified plane."""
    if config.dim!=2 or not isinstance(tile_radius,int) or tile_radius<0:
        raise ValueError('Require 2D geometry and a nonnegative integer tile radius')
    domain,pieces,base,lower=_tiles(state,config,tile_radius)
    points,quadrature,outward=jax.vmap(boundary_quadrature)(pieces)
    points=points.reshape(-1,2);quadrature=quadrature.reshape(-1);outward=outward.reshape(-1,2)
    normal=jnp.asarray(normal,dtype=state.x.dtype)
    tolerance=128*jnp.finfo(state.x.dtype).eps*(jnp.abs(offset)+jnp.linalg.norm(normal)*config.domain_size)
    on_plane=jnp.abs(points@normal-offset)<=tolerance
    quadrature=jnp.where(on_plane,quadrature,0.)
    basis=stencil(points,config,active=quadrature>0)
    coordinates,indices,node_lower,width=_nodes(base,config,tile_radius);count=width**2
    actual=jnp.stack((basis.indices//(config.n_grid+1),basis.indices%(config.n_grid+1)),axis=-1)
    local=actual-node_lower;destinations=local[:,:,0]*width+local[:,:,1]
    destinations=jnp.where(quadrature[:,None]>0,destinations,count)
    values=quadrature[:,None,None]*basis.weights[:,:,None]*outward[:,None,:]
    vectors=jnp.zeros((count,2),dtype=state.x.dtype).at[destinations].add(values,mode='drop')
    valid=state.valid&jnp.all(pieces.valid)&basis.valid&_covered(domain,lower,config)
    valid &= (jnp.sum(normal**2)>0)&jnp.all(jnp.isfinite(normal))&jnp.isfinite(offset)
    return BoundaryIntegral(indices,vectors,valid)


def _batch(function,states,batch_size):
    if not isinstance(batch_size,int) or batch_size<1 or len(states.x)<1:
        raise ValueError('Require a nonempty fragment batch and a positive batch size')
    count=len(states.x);blocks=(count+batch_size-1)//batch_size
    indices=jnp.minimum(jnp.arange(blocks*batch_size).reshape(blocks,batch_size),count-1)
    batches=jax.tree.map(lambda a:a[indices],states)
    def advance(_,part):
        return None,jax.vmap(function)(part)
    _,result=jax.lax.scan(advance,None,batches)
    return jax.tree.map(lambda a:a.reshape((-1,)+a.shape[2:])[:count],result)


@partial(jax.jit,static_argnames=('config','tile_radius','batch_size'))
def batch_stencils(states,config,tile_radius=1,batch_size=32):
    """Scan bounded batches using the equivalent separable tile contraction."""
    return _batch(lambda s:domain_stencil_separable(s,config,tile_radius),states,batch_size)


@partial(jax.jit,static_argnames=('config','tile_radius','batch_size'))
def batch_boundary_integrals(states,normal,offset,config,tile_radius=1,batch_size=32):
    return _batch(lambda s:plane_boundary_integral(s,normal,offset,config,tile_radius),states,batch_size)


def particle_to_grid(state,support,size):
    """Scatter batched fragment mass and affine momentum."""
    weighted_mass=state.mass[:,None]*support.weights
    affine=jnp.einsum('pij,pkj->pki',state.C,support.dpos)
    mass=jnp.zeros(size,dtype=state.mass.dtype).at[support.indices].add(weighted_mass)
    momentum=jnp.zeros((size,2),dtype=state.v.dtype).at[support.indices].add(weighted_mass[:,:,None]*(state.v[:,None,:]+affine))
    return mass,momentum


def grid_to_fragment(velocity,support):
    gathered=velocity[support.indices]
    v=jnp.sum(support.weights[:,:,None]*gathered,axis=1)
    first=jnp.einsum('pk,pki,pkj->pij',support.weights,gathered,support.dpos)
    return v,first@support.inverse_moment


def velocity_gradient(velocity,support):
    return jnp.einsum('pki,pkj->pij',velocity[support.indices],support.gradients)


def internal_force(kirchhoff,reference_area,support,size):
    force=-reference_area[:,None,None]*jnp.einsum('pij,pkj->pki',kirchhoff,support.gradients)
    return jnp.zeros((size,2),dtype=kirchhoff.dtype).at[support.indices].add(force)
