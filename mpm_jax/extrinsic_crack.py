"""Prestressed cohesive insertion on a moving 2D polyline."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cohesive_crack import (Transfer, elastic_energy, face_velocity, kinetic_energy,
                             midpoint_bulk, scatter_face_force)
from .core import grid_boundaries, grid_to_particle, particle_stencil, stencil
from .materials import kirchhoff_stress
from .polyline_crack import link_fields, make_path, shifted_nodes, vertex_normals
from .crack_mpm import _cross


class CohesivePath(NamedTuple):
    plus: jax.Array
    minus: jax.Array
    count: jax.Array
    area: jax.Array
    birth_traction: jax.Array
    maximum: jax.Array
    bonded: jax.Array
    extension: jax.Array
    valid: jax.Array

    @property
    def vertices(self):
        return .5*(self.plus+self.minus)


def initialize(vertices, capacity=64):
    path = make_path(vertices, capacity)
    lengths = jnp.linalg.norm(jnp.diff(path.vertices, axis=0), axis=-1)
    return CohesivePath(path.plus, path.minus, path.count, lengths,
                        jnp.zeros((capacity-1, 2), dtype=path.plus.dtype),
                        jnp.zeros((capacity-1,2), dtype=path.plus.dtype),
                        jnp.zeros(capacity-1, dtype=bool), path.extension, path.valid)


def segment_positions(plus, minus):
    return jnp.stack((plus[:-1], plus[1:], minus[:-1], minus[1:]), axis=1)


def local_opening(positions):
    tangent = .5*(positions[:,1]+positions[:,3]-positions[:,0]-positions[:,2])
    length = jnp.sqrt(jnp.maximum(jnp.sum(tangent*tangent, axis=-1), 1e-100))
    tangent /= length[:,None]
    normal = jnp.stack((-tangent[:,1], tangent[:,0]), axis=-1)
    fraction = jnp.asarray([.5-.5/jnp.sqrt(3.),.5+.5/jnp.sqrt(3.)])
    jump = (1-fraction)[None,:,None]*(positions[:,0]-positions[:,2])[:,None,:]+fraction[None,:,None]*(positions[:,1]-positions[:,3])[:,None,:]
    return jnp.stack((jnp.sum(jump*tangent[:,None,:], axis=-1), jnp.sum(jump*normal[:,None,:], axis=-1)), axis=-1)


class SurfaceResponse(NamedTuple):
    stored: jax.Array
    dissipated: jax.Array
    maximum: jax.Array
    softening: jax.Array


def surface_response(positions, maximum, birth, bonded, law, update=True):
    """Regularized extrinsic potential, normalized to zero energy at insertion.

    The preload shift is birth/K in a co-rotating segment frame. In pure
    opening the descending branch starts at birth traction and its area is Gc.
    Its unloading law retains that shift; it is not an initially rigid law.
    """
    opening = local_opening(positions)
    strength = jnp.sqrt(jnp.maximum(jnp.sum(birth*birth, axis=-1), 1e-100))
    strength = jnp.where(bonded, strength, 1.)
    r0 = strength[:,None]/law.stiffness
    width = 2*law.energy/strength[:,None]
    shifted = opening+birth[:,None,:]/law.stiffness
    radius = jnp.sqrt(jnp.maximum(jnp.sum(shifted*shifted, axis=-1), 1e-100))
    history = jnp.maximum(maximum, r0)
    if update:
        history = jnp.maximum(history, radius)
    history = jnp.minimum(history, r0+width)
    history = jax.lax.stop_gradient(history)
    softening = jnp.clip((history-r0)/width, 0., 1.)
    remaining = r0/jnp.maximum(history, r0)*(1-softening)
    stored = .5*law.stiffness*remaining*radius**2-.5*strength[:,None]*r0*(1-softening)
    compression = .5*law.stiffness*jnp.minimum(opening[:,:,1], 0.)**2
    stored = jnp.where(bonded[:,None], stored, 0.)+compression
    dissipated = jnp.where(bonded[:,None], law.energy*softening, 0.)
    return SurfaceResponse(stored.mean(axis=1), dissipated.mean(axis=1), jnp.where(bonded[:,None], history, 0.),
                           jnp.where(bonded[:,None], softening, 0.))


def surface_gradient(positions, maximum, birth, bonded, law):
    return jax.grad(lambda q:jnp.sum(surface_response(q, maximum, birth, bonded, law, update=False).stored))(positions)


def discrete_surface_gradient(old, new, maximum, birth, bonded, law):
    before = surface_response(old, maximum, birth, bonded, law)
    after = surface_response(new, before.maximum, birth, bonded, law)
    gradient = .5*(surface_gradient(old, before.maximum, birth, bonded, law)
                   +surface_gradient(new, after.maximum, birth, bonded, law))
    change = new-old
    middle = .5*(new+old)
    relative = middle-middle.mean(axis=1, keepdims=True)
    rotation = jnp.stack((-relative[:,:,1], relative[:,:,0]), axis=-1)
    norm = jnp.sum(rotation*rotation, axis=(1,2))

    def project(value):
        value = value-value.mean(axis=1, keepdims=True)
        spin = jnp.sum(value*rotation, axis=(1,2))/jnp.maximum(norm, 1e-100)
        return value-spin[:,None,None]*rotation

    gradient, direction = project(gradient), project(change)
    energy = after.stored-before.stored+after.dissipated-before.dissipated
    remainder = energy-jnp.sum(gradient*change, axis=(1,2))
    square = jnp.sum(direction*direction, axis=(1,2))
    coordinate_scale = jnp.maximum(norm,jnp.sum(old*old+new*new,axis=(1,2)))
    resolution = (64*jnp.finfo(old.dtype).eps)**2*jnp.maximum(coordinate_scale,1e-100)
    correction = jnp.where(square > resolution, remainder/jnp.maximum(square, resolution), 0.)
    gradient += correction[:,None,None]*direction
    error = jnp.sum(gradient*change, axis=(1,2))-energy
    return gradient, after, energy, error


def transfers(position, normals, side, mass, config, vertices, count, nodes, mass_weighted=True):
    support = stencil(position, config)
    fields = link_fields(position+side*1e-6*config.dx*normals, support, vertices, count, nodes[support.indices])
    indices = support.indices+fields*((config.n_grid+1)**2)
    local_mass = mass[indices]
    if mass_weighted:
        weights = support.weights*local_mass/jnp.maximum(local_mass.max(axis=1,keepdims=True),1e-100)
    else:
        weights = jnp.where(local_mass>0,support.weights,0.)
    total = weights.sum(axis=1)
    weights /= jnp.where(total > 0, total, 1.)[:,None]
    basis = jnp.concatenate((jnp.ones_like(weights)[...,None],support.dpos/config.dx),axis=-1)
    moment = jnp.einsum('pk,pki,pkj->pij',weights,basis,basis)
    full_rank = jnp.linalg.det(moment)>1e-12
    matrix = jnp.where(full_rank[:,None,None],moment,jnp.eye(3))
    rhs = jnp.broadcast_to(jnp.asarray([1.,0.,0.]),(len(position),3))
    coefficient = jnp.linalg.solve(matrix,rhs[...,None])[...,0]
    weights *= jnp.einsum('pki,pi->pk',basis,coefficient)
    error = jnp.max(jnp.abs(jnp.sum(weights[...,None]*basis,axis=1)-rhs),axis=1)
    active = jnp.arange(len(position)) < count
    valid = support.valid & jnp.all(jnp.where(active,(total>0)&full_rank&(error<1e-9),True))
    return Transfer(indices, weights, valid)


def shared_tip(plus, minus, count):
    index = jnp.concatenate((plus.indices, minus.indices), axis=1)
    p = jnp.concatenate((plus.weights, jnp.zeros_like(minus.weights)), axis=1)
    m = jnp.concatenate((jnp.zeros_like(plus.weights), minus.weights), axis=1)
    common = .5*jnp.concatenate((plus.weights[count-1], minus.weights[count-1]))
    p, m = p.at[count-1].set(common), m.at[count-1].set(common)
    active = (jnp.arange(len(p)) < count)[:,None]
    return (Transfer(index, jnp.where(active,p,0.), plus.valid),
            Transfer(index, jnp.where(active,m,0.), minus.valid))


class Work(NamedTuple):
    boundary: jax.Array
    gravity: jax.Array
    transfer_loss: jax.Array
    boundary_projection_loss: jax.Array
    bulk_error: jax.Array
    surface_error: jax.Array
    iterations: jax.Array
    residual: jax.Array


@partial(jax.jit, static_argnames=('config','law','consistent_gradients'))
def step(state, path, material, config, law, consistent_gradients=False, dt=None):
    if config.dim != 2 or config.material not in ('elastic','neo-hookean'):
        raise ValueError('Extrinsic cohesive MPM requires 2D neo-Hookean elasticity')
    dt = config.dt if dt is None else dt
    vertices = path.vertices
    first = vertices[1]-vertices[0]
    visibility = vertices.at[0].add(-3*config.dx*first/jnp.maximum(jnp.linalg.norm(first),1e-100))
    support = particle_stencil(state, config)
    size = (config.n_grid+1)**2
    index = jnp.arange(size)
    nodes = config.dx*jnp.stack((index//(config.n_grid+1),index%(config.n_grid+1)),axis=-1)
    nodes = shifted_nodes(nodes, visibility, path.count)
    fields = link_fields(state.x, support, visibility, path.count, nodes[support.indices])
    remapped = support._replace(indices=support.indices+fields*size)
    mass = jnp.zeros(2*size, dtype=state.x.dtype).at[remapped.indices].add(support.weights*state.mass[:,None])
    affine = jnp.einsum('pij,pkj->pki',state.C,support.dpos)
    contribution = support.weights[...,None]*state.mass[:,None,None]*(state.v[:,None,:]+affine)
    momentum = jnp.zeros((2*size,2),dtype=state.x.dtype).at[remapped.indices].add(contribution)
    occupied = mass > 0
    denominator = jnp.where(occupied,mass,1.)[:,None]
    transferred = momentum/denominator

    def constrain(v,t):
        return jnp.concatenate([grid_boundaries(v[i*size:(i+1)*size],config,t,mass[i*size:(i+1)*size]) for i in (0,1)])

    initial = constrain(transferred,state.time)
    initial_impulse = mass[:,None]*(initial-transferred)
    normals = vertex_normals(vertices,path.count)
    plus = transfers(path.plus,normals,1.,mass,config,visibility,path.count,nodes,mass_weighted=not consistent_gradients)
    minus = transfers(path.minus,normals,-1.,mass,config,visibility,path.count,nodes,mass_weighted=not consistent_gradients)
    plus, minus = shared_tip(plus,minus,path.count)
    old_positions = segment_positions(path.plus,path.minus)

    def evaluate(velocity):
        middle = .5*(initial+velocity)
        vp, gradient = grid_to_particle(middle,remapped,config)
        if consistent_gradients:
            gradient = jnp.einsum('pki,pkj->pij',middle[remapped.indices],support.gradients)
        F,tau,valid = midpoint_bulk(state.F,gradient,dt,material)
        bulk_gradient = support.gradients if consistent_gradients else 4/config.dx**2*support.weights[...,None]*support.dpos
        bulk_contribution = -state.volume[:,None,None]*jnp.einsum('pij,pkj->pki',tau,bulk_gradient)
        bulk_force = jnp.zeros_like(momentum).at[remapped.indices].add(bulk_contribution)
        next_plus = path.plus+dt*face_velocity(middle,plus)
        next_minus = path.minus+dt*face_velocity(middle,minus)
        q = segment_positions(next_plus,next_minus)
        grad,after,energy,error = discrete_surface_gradient(old_positions,q,path.maximum,path.birth_traction,path.bonded,law)
        force = -path.area[:,None,None]*grad
        fp = jnp.zeros_like(path.plus).at[:-1].add(force[:,0]).at[1:].add(force[:,1])
        fm = jnp.zeros_like(path.minus).at[:-1].add(force[:,2]).at[1:].add(force[:,3])
        surface_force = scatter_face_force(fp,plus,momentum.shape)+scatter_face_force(fm,minus,momentum.shape)
        raw = jnp.where(occupied[:,None],initial+dt*((bulk_force+surface_force)/denominator+jnp.asarray(config.gravity)),0.)
        bounded = 2*constrain(.5*(initial+raw),state.time+.5*dt)-initial
        return bounded,(middle,vp,F,next_plus,next_minus,after,energy,error,bulk_force,surface_force,raw,valid)

    def iterate(carry):
        v,n,_ = carry
        following,_ = evaluate(v)
        residual = jnp.max(jnp.abs(following-v))/(1+jnp.max(jnp.abs(following)))
        return following,n+1,residual

    velocity,iterations,residual = jax.lax.while_loop(lambda c:(c[1]<32)&(c[2]>1e-9),iterate,
                                                    (initial,jnp.asarray(0),jnp.asarray(jnp.inf)))
    bounded,data = evaluate(velocity)
    middle,vp,F,next_plus,next_minus,after,energy,error,bulk_force,surface_force,raw,bulk_valid = data
    v,C = grid_to_particle(velocity,remapped,config)
    x = state.x+dt*vp
    valid = state.valid & path.valid & support.valid & plus.valid & minus.valid & bulk_valid & (residual<=1e-8)
    valid &= jnp.isfinite(dt) & (dt>0)
    valid &= jnp.max(jnp.abs(path.area*error)) <= 1e-10*(1+jnp.sum(path.area)*law.energy)
    for array in (x,v,C,F,next_plus,next_minus):
        valid &= jnp.all(jnp.isfinite(array))
    valid &= jnp.all((x>=.5*config.dx)&(x<config.domain_size-.5*config.dx))
    following = state._replace(x=x,v=v,C=C,F=F,time=state.time+dt,valid=valid)
    path = path._replace(plus=next_plus,minus=next_minus,maximum=after.maximum,valid=valid)
    boundary = jnp.sum(mass[:,None]*(bounded-raw)*middle)+jnp.sum(initial_impulse*initial)
    gravity = dt*jnp.sum(mass[:,None]*jnp.asarray(config.gravity)*middle)
    transfer = kinetic_energy(state,config)-.5*jnp.sum(mass[:,None]*transferred**2)+.5*jnp.sum(mass[:,None]*velocity**2)-kinetic_energy(following,config)
    projection = .5*jnp.sum(mass[:,None]*(initial-transferred)**2)
    bulk_error = dt*jnp.sum(bulk_force*middle)+elastic_energy(following,material,config)-elastic_energy(state,material,config)
    surface_error = dt*jnp.sum(surface_force*middle)+jnp.sum(path.area*energy)
    return following,path,Work(boundary,gravity,transfer,projection,bulk_error,surface_error,iterations,residual)


class InsertionReport(NamedTuple):
    inserted: jax.Array
    traction: jax.Array
    direction: jax.Array
    energy_jump: jax.Array
    capacity_exceeded: jax.Array
    intersection: jax.Array


def insert_tip(state,path,material,config,law,spacing,radius,max_extension,max_turn=jnp.pi/4):
    """Insert a finite segment at sufficient sampled traction; no imposed tip speed."""
    vertices = path.vertices
    tip = vertices[path.count-1]
    forward = tip-vertices[path.count-2]
    forward /= jnp.maximum(jnp.linalg.norm(forward),1e-100)
    offset = state.x-tip
    r2 = jnp.sum(offset*offset,axis=-1)/radius**2
    weights = state.volume*jnp.where((offset@forward>=0)&(r2<1),(1-r2)**2,0.)
    volume = jnp.sum(weights)
    stress = kirchhoff_stress(state.F,material,config.material)/jnp.linalg.det(state.F)[:,None,None]
    average = jnp.sum(weights[:,None,None]*stress,axis=0)/jnp.where(volume>0,volume,1.)
    values,vectors = jnp.linalg.eigh(average)
    normal = vectors[:,-1]
    candidate = jnp.array([normal[1],-normal[0]])
    candidate *= jnp.where(candidate@forward>=0,1.,-1.)
    angle = jnp.clip(jnp.arctan2(_cross(forward,candidate),forward@candidate),-max_turn,max_turn)
    direction = jnp.cos(angle)*forward+jnp.sin(angle)*jnp.array([-forward[1],forward[0]])
    direction = jnp.where(values[-1]-values[0]>1e-10*law.strength,direction,forward)
    normal = jnp.array([-direction[1],direction[0]])
    traction = average@normal
    birth = jnp.array([traction@direction,traction@normal])
    request = state.valid & path.valid & (volume>0) & (birth[1]>=law.strength) & (path.extension+spacing<=max_extension+1e-12)
    full = request & (path.count>=len(vertices))
    newtip = tip+spacing*direction

    def crossed(i,hit):
        a,b = vertices[i],vertices[i+1]
        ray,tangent = newtip-tip,b-a
        denominator = _cross(ray,tangent)
        safe = jnp.where(denominator!=0,denominator,1.)
        t,u = _cross(a-tip,tangent)/safe,_cross(a-tip,ray)/safe
        return hit|((denominator!=0)&(t>0)&(t<=1)&(u>=0)&(u<=1))

    intersection = request & jax.lax.fori_loop(0,jnp.maximum(path.count-2,0),crossed,jnp.asarray(False))
    accepted = request & ~full & ~intersection
    index = jnp.minimum(path.count,len(vertices)-1)
    segment = jnp.minimum(path.count-1,len(path.area)-1)
    plus = path.plus.at[index].set(jnp.where(accepted,newtip,path.plus[index]))
    minus = path.minus.at[index].set(jnp.where(accepted,newtip,path.minus[index]))
    following = path._replace(plus=plus,minus=minus,count=path.count+accepted.astype(jnp.int32),
                               area=path.area.at[segment].set(jnp.where(accepted,spacing,path.area[segment])),
                               birth_traction=path.birth_traction.at[segment].set(jnp.where(accepted,birth,path.birth_traction[segment])),
                               maximum=path.maximum.at[segment].set(jnp.where(accepted,jnp.linalg.norm(birth)/law.stiffness,path.maximum[segment])),
                               bonded=path.bonded.at[segment].set(path.bonded[segment]|accepted),
                               extension=path.extension+jnp.where(accepted,spacing,0.),valid=path.valid & ~full & ~intersection)
    old = surface_response(segment_positions(path.plus,path.minus),path.maximum,path.birth_traction,path.bonded,law)
    new = surface_response(segment_positions(plus,minus),following.maximum,following.birth_traction,following.bonded,law)
    jump = jnp.sum(following.area*(new.stored+new.dissipated))-jnp.sum(path.area*(old.stored+old.dissipated))
    return following,InsertionReport(accepted,birth[1],direction,jump,full,intersection)
