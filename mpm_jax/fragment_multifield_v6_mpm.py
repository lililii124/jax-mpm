"""Independent exploratory damage and fracture APIC prototype with plastic-work-only temperature.

Physical energy defects are reported explicitly, never deposited into temperature.
The signed numerical reservoir records discarded transfer/contact/viscous work
and constitutive/spatial work mismatch; it is accounting, not physical heat.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .contact import contact_impulse
from .fibre_mpm import PackedLayout, build_layout, grid_to_particle, stencil
from .materials.linear_metal import artificial_viscosity, deviatoric_return, eos_state, volume_energy


@partial(jax.tree_util.register_dataclass,
         data_fields=('layout', 'pairs', 'clamped', 'node_body', 'global_node_id', 'contact_colors', 'contact_color_active'), meta_fields=())
@dataclass(frozen=True)
class ImpactGrid:
    layout: PackedLayout
    pairs: jax.Array
    clamped: jax.Array
    node_body: jax.Array
    global_node_id: jax.Array
    contact_colors: jax.Array
    contact_color_active: jax.Array


@dataclass(frozen=True)
class ImpactConfig:
    cfl: float = .1
    strength: bool = True
    contact_iterations: int = 1024
    contact_tolerance: float = .005
    damage_onset_strain: float = .10
    damage_failure_strain: float = .30

    def __post_init__(self):
        if not np.isfinite(self.cfl) or not 0 < self.cfl <= .2:
            raise ValueError('cfl must lie in (0,.2]')
        if type(self.contact_iterations) is not int or not 1<=self.contact_iterations<=2048:
            raise ValueError('Require1..2048 contact sweeps')
        if not np.isfinite(self.contact_tolerance) or self.contact_tolerance<=0:
            raise ValueError('Positive finite contact tolerance required')
        if not 0 < self.damage_onset_strain < self.damage_failure_strain:
            raise ValueError('Require positive ordered damage thresholds')
        if not isinstance(self.strength, bool):
            raise ValueError('strength must be boolean')


class ImpactState(NamedTuple):
    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    mass: jax.Array
    volume: jax.Array
    body_id: jax.Array
    internal_energy: jax.Array
    tau: jax.Array
    plastic_strain: jax.Array
    cold_work: jax.Array
    time: jax.Array
    boundary_reservoir: jax.Array
    numerical_reservoir: jax.Array
    boundary_impulse: jax.Array
    boundary_projection_loss: jax.Array
    boundary_coupling_work: jax.Array
    contact_projection_loss: jax.Array
    contact_coupling_work: jax.Array
    initial_entropy: jax.Array
    minimum_entropy: jax.Array
    negative_entropy_increments: jax.Array
    ever_phase_extrapolated: jax.Array
    ever_tension_extrapolated: jax.Array
    maximum_temperature: jax.Array
    minimum_J: jax.Array
    maximum_J: jax.Array
    damage: jax.Array
    fracture_energy: jax.Array
    maximum_contact_closing_speed: jax.Array
    maximum_contact_complementarity_residual: jax.Array
    maximum_contact_iterations: jax.Array
    contact_lambda: jax.Array
    contact_cache_dt: jax.Array
    valid: jax.Array


class ImpactDiagnostics(NamedTuple):
    p2g_dissipation: jax.Array
    g2p_dissipation: jax.Array
    contact_projection_loss: jax.Array
    contact_coupling_work: jax.Array
    contact_compatible_energy: jax.Array
    contact_body_work: jax.Array
    stress_work: jax.Array
    boundary_energy: jax.Array
    energy_residual: jax.Array
    physical_energy_residual: jax.Array
    numerical_energy_increment: jax.Array
    viscosity_work: jax.Array
    constitutive_work_defect: jax.Array
    momentum_residual: jax.Array
    dt_limit: jax.Array
    maximum_viscosity: jax.Array
    minimum_J: jax.Array
    minimum_temperature: jax.Array
    minimum_entropy: jax.Array
    minimum_entropy_increment: jax.Array
    entropy_decrease_fraction: jax.Array
    minimum_entropy_change: jax.Array
    minimum_cumulative_negative_entropy: jax.Array
    phase_extrapolated: jax.Array
    tension_extrapolated: jax.Array
    valid: jax.Array


def matching_nodes(layout, max_pairs=2_000_000):
    """Separate velocity DOFs at coincident global nodes for every material group."""
    lower,shape,stride,offset=map(np.asarray,(layout.lower,layout.shape,layout.strides,layout.offsets))
    pairs=[];total=0
    for a in range(layout.fibre_count):
        for b in range(a+1,layout.fibre_count):
            first=np.maximum(lower[a],lower[b]);last=np.minimum(lower[a]+shape[a]-1,lower[b]+shape[b]-1)
            if np.any(last<first):continue
            count=int(np.prod(last-first+1));total+=count
            if total>max_pairs:raise ValueError('Contact pair budget exceeded; never merge fragments')
            indices=np.indices(tuple(last-first+1),dtype=np.int64).reshape(3,-1).T+first
            pairs.append(np.stack((offset[a]+np.sum((indices-lower[a])*stride[a],axis=1),offset[b]+np.sum((indices-lower[b])*stride[b],axis=1)),axis=1))
    return jnp.asarray(np.concatenate(pairs) if pairs else np.empty((0,2)),dtype=jnp.int32)


def build_grid(x, body_id, *, dx, padding_cells=3, reference_bounds=None,
               origin=(0.,0.,0.),max_nodes=2_000_000,clamped=None,field_physical_body=None,contact_sweeps=None):
    """Compact occupied/swept nodes; dense indices are only a bounded lookup map.

    Missing future support or contact coverage rejects a step. No fragment is
    merged and no pair is omitted to meet capacity. Virtual map <=32M int32.
    """
    dense=build_layout(x,body_id,dx=dx,padding_cells=padding_cells,reference_bounds=reference_bounds,origin=origin,max_nodes=32_000_000)
    lower,shape,stride,offset=map(np.asarray,(dense.lower,dense.shape,dense.strides,dense.offsets))
    physical=np.asarray(field_physical_body)
    if physical.shape!=(dense.fibre_count,) or np.any((physical<0)|(physical>1)):raise ValueError('Original physical body map required')
    start,end=map(np.asarray,contact_sweeps if contact_sweeps is not None else (x,x));ids=np.asarray(body_id)
    first=np.floor(np.minimum(start,end)/dx-.5).astype(np.int32)-1
    last=np.floor(np.maximum(start,end)/dx-.5).astype(np.int32)+3
    first=np.maximum(first,lower[ids]);last=np.minimum(last,lower[ids]+shape[ids]-1)
    spans=last-first+1
    if np.sum(np.prod(spans,axis=1,dtype=np.int64))>32_000_000:raise ValueError('Swept support candidate budget exceeded')
    all_indices=[];all_coordinates=[]
    for span in np.unique(spans,axis=0):
        selected=np.flatnonzero(np.all(spans==span,axis=1))
        increments=np.indices(tuple(span),dtype=np.int32).reshape(3,-1).T
        coordinates=(first[selected,None,:]+increments[None,:,:]).reshape(-1,3)
        field=np.repeat(ids[selected],len(increments))
        dense_indices=offset[field]+np.sum((coordinates-lower[field])*stride[field],axis=1)
        all_indices.append(dense_indices.astype(np.int32));all_coordinates.append(coordinates)
    candidate_indices=np.concatenate(all_indices);candidate_coordinates=np.concatenate(all_coordinates)
    dense_ids,unique_rows=np.unique(candidate_indices,return_index=True)
    if len(dense_ids)>max_nodes:raise ValueError('Compact occupied-node budget exceeded')
    coordinates=candidate_coordinates[unique_rows]
    node_field=np.searchsorted(offset,dense_ids,side='right')-1
    node_body=physical[node_field].astype(np.int32)
    _,global_id=np.unique(coordinates,axis=0,return_inverse=True)
    global_id=global_id.astype(np.int32)
    ordering=np.argsort(global_id,kind='stable')
    _,starts,counts=np.unique(global_id[ordering],return_index=True,return_counts=True)
    pair_count=np.sum(counts.astype(np.int64)*(counts-1)//2)
    if pair_count>2_000_000:raise ValueError('Complete compact contact-pair budget exceeded')
    pieces=[]
    for degree in np.unique(counts[counts>1]):
        groups=np.flatnonzero(counts==degree)
        packed=ordering[starts[groups,None]+np.arange(degree)[None,:]]
        a,b=np.triu_indices(degree,k=1)
        pieces.append(np.stack((packed[:,a].ravel(),packed[:,b].ravel()),axis=1))
    records=np.concatenate(pieces) if pieces else np.empty((0,2),dtype=np.int32)
    mapping=np.full(dense.node_count,-1,dtype=np.int32);mapping[dense_ids]=np.arange(len(dense_ids),dtype=np.int32)
    from .fibre_mpm import PackedLayout
    layout=PackedLayout(dense.lower,dense.shape,dense.strides,dense.offsets,dense.origin,dense.dx,len(dense_ids),dense.fibre_count,jnp.asarray(mapping),jnp.asarray(dense_ids,dtype=jnp.int32))
    mask=np.zeros((layout.node_count,3),bool) if clamped is None else np.asarray(clamped)
    pairs=jnp.asarray(np.asarray(records,dtype=np.int32).reshape(-1,2))
    colors,color_active=color_contact_pairs(np.asarray(pairs),global_id)
    return ImpactGrid(layout,pairs,jnp.asarray(mask),jnp.asarray(node_body),jnp.asarray(global_id),jnp.asarray(colors),jnp.asarray(color_active))


def color_contact_pairs(pairs,global_id):
    """Round-robin edge coloring of complete per-node material contact graphs."""
    if not len(pairs):return np.zeros((1,1),np.int32),np.zeros((1,1),bool)
    order=np.argsort(global_id,kind='stable');_,starts,counts=np.unique(global_id[order],return_index=True,return_counts=True)
    colors=[]
    pair_keys=pairs[:,0].astype(np.int64)*len(global_id)+pairs[:,1]
    sorting=np.argsort(pair_keys);sorted_keys=pair_keys[sorting]
    for degree in np.unique(counts[counts>1]):
        nodes=order[starts[np.flatnonzero(counts==degree),None]+np.arange(degree)[None,:]]
        even=int(degree)+(int(degree)%2);rotation=np.arange(even)
        for round_index in range(even-1):
            if round_index>=len(colors):colors.append([])
            left=rotation[:even//2];right=rotation[even//2:][::-1]
            valid=(left<degree)&(right<degree);left,right=left[valid],right[valid]
            a=np.minimum(nodes[:,left],nodes[:,right]).ravel();b=np.maximum(nodes[:,left],nodes[:,right]).ravel()
            keys=a.astype(np.int64)*len(global_id)+b
            found=np.searchsorted(sorted_keys,keys)
            if not np.array_equal(sorted_keys[found],keys):raise ValueError('Coloring lost a complete pair')
            colors[round_index].append(sorting[found])
            rotation=np.concatenate((rotation[:1],rotation[-1:],rotation[1:-1]))
    colors=[np.concatenate(pieces) for pieces in colors];width=max(map(len,colors))
    if len(colors)*width>2_000_000:raise ValueError('Colored contact slot budget exceeded')
    result=np.zeros((len(colors),width),np.int32);active=np.zeros_like(result,dtype=bool)
    for i,indices in enumerate(colors):result[i,:len(indices)]=indices;active[i,:len(indices)]=True
    if not np.array_equal(np.sort(result[active]),np.arange(len(pairs))):raise ValueError('Each complete contact pair must have one color')
    return result,active


def elastic_energy(tau, material, damage=0.):
    return jnp.sum(tau*tau, axis=(-2, -1))/(4*material.shear_modulus*material.density*jnp.maximum(1-jnp.asarray(damage),1e-12))


def damage_volume_energy(J, damage, material):
    factor=jnp.where(J>1,1-damage,1.)
    return factor*volume_energy(J,material)


def thermodynamics(state, material):
    J = jnp.linalg.det(state.F)
    eos_energy = state.internal_energy-state.cold_work-state.fracture_energy-elastic_energy(state.tau, material, state.damage)
    # The EOS pressure remains K(1-J); only tensile traction/free energy degrade.
    eos_energy=eos_energy+volume_energy(J,material)-damage_volume_energy(J,state.damage,material)
    base=eos_state(J,eos_energy,material)
    factor=jnp.where(J>1,1-state.damage,1.)
    sound=base.sound_speed*jnp.sqrt(factor)
    longitudinal=jnp.sqrt(sound**2+4*material.shear_modulus*jnp.maximum(J,1.)*(1-state.damage)/(3*material.density))
    # Fully failed tensile matter is traction-free and no longer obeys the
    # intact-solid J<=2 assumption. Positive finite J and temperature still apply.
    free=(state.damage>=1)&(J>=1)&jnp.isfinite(J)&jnp.isfinite(base.temperature)&(base.temperature>0)&jnp.isfinite(eos_energy)
    return base._replace(sound_speed=sound,longitudinal_speed=longitudinal,valid=base.valid|free)


def energy_ledger(state, grid):
    """Joules: translation, affine, physical internal, boundary, numerical reservoir."""
    translation = .5*jnp.sum(state.mass*jnp.sum(state.v**2, axis=-1))
    affine = grid.layout.dx**2/8*jnp.sum(state.mass*jnp.sum(state.C**2, axis=(-2, -1)))
    internal = jnp.sum(state.mass*state.internal_energy)
    return jnp.stack((translation, affine, internal, state.boundary_reservoir, state.numerical_reservoir))


def make_state(x, volume, body_id, material, grid, *, v=None, F=None, C=None,
               eos_energy=None, tau=None, plastic_strain=None, cold_work=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if x.shape != (n, 3) or n == 0:
        raise ValueError('x must be a nonempty (N,3) array')
    volume = np.asarray(volume, dtype=np.float64)
    ids = np.asarray(body_id)
    if volume.shape != (n,) or np.any(volume <= 0) or not np.isfinite(volume).all():
        raise ValueError('Reference volume must be finite and positive')
    if ids.shape != (n,) or ids.dtype.kind not in 'iu' or not np.array_equal(np.unique(ids), np.arange(grid.layout.fibre_count)):
        raise ValueError('Both body IDs 0 and 1 must be present')
    if grid.layout.fibre_count < 1:
        raise ValueError('At least one independent material field required')

    def array(value, shape, default=0.):
        result = np.full(shape, default) if value is None else np.asarray(value, dtype=np.float64)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f'Initial field requires finite shape {shape}')
        return jnp.asarray(result, dtype=jnp.float64)

    x = array(x, (n, 3))
    if x.dtype != jnp.float64:
        raise ValueError('Enable jax_enable_x64 before initializing impact MPM')
    deformation = array(np.broadcast_to(np.eye(3), (n, 3, 3)) if F is None else F, (n, 3, 3))
    stress = array(tau, (n, 3, 3))
    cold = array(cold_work, (n,))
    ep = array(plastic_strain, (n,))
    if np.any(np.asarray(cold) < 0) or np.any(np.asarray(ep) < 0):
        raise ValueError('Plastic history and stored cold work must be nonnegative')
    initial_J = jnp.linalg.det(deformation)
    initial_eos = volume_energy(initial_J, material) if eos_energy is None else array(eos_energy, (n,))
    internal = initial_eos+cold+elastic_energy(stress, material)
    initial_J = jnp.linalg.det(deformation)
    initial_thermo = eos_state(initial_J, internal-cold-elastic_energy(stress, material), material)
    initial_entropy = initial_thermo.entropy
    state = ImpactState(x, array(v, (n, 3)), deformation, array(C, (n, 3, 3)),
        material.density*jnp.asarray(volume), jnp.asarray(volume), jnp.asarray(ids, dtype=jnp.int32),
        internal, stress, ep, cold, jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.), jnp.zeros(3),
        jnp.asarray(0.), jnp.asarray(0.), jnp.zeros(2), jnp.zeros(2),
        initial_entropy, initial_entropy, jnp.zeros(n),
        initial_thermo.phase_extrapolated, initial_thermo.tension_extrapolated,
        initial_thermo.temperature, initial_J, initial_J, jnp.zeros(n), jnp.zeros(n), jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.), jnp.zeros(len(grid.pairs)), jnp.asarray(0.), jnp.asarray(True))
    validate_state(state, material, grid)
    return state


def _gather(field, support):
    return jnp.take(field, support.indices, axis=0, mode='fill', fill_value=0.)


def transfer_to_grid(state, support, grid):
    """Stress-free APIC transfer and its nonnegative per-particle variance loss."""
    layout = grid.layout
    weighted_mass = state.mass[:, None]*support.weights
    affine_velocity = state.v[:, None, :]+jnp.einsum('pij,pkj->pki', state.C, support.dpos)
    mass = jnp.zeros(layout.node_count, dtype=state.x.dtype).at[support.indices].add(weighted_mass, mode='drop')
    momentum = jnp.zeros((layout.node_count, 3), dtype=state.x.dtype).at[support.indices].add(
        weighted_mass[..., None]*affine_velocity, mode='drop')
    velocity = momentum/jnp.where(mass > 0, mass, 1.)[:, None]
    difference = affine_velocity-_gather(velocity, support)
    loss = .5*jnp.sum(support.weights*jnp.sum(difference*difference, axis=-1), axis=-1)
    return mass, velocity, loss


def transfer_to_particles(velocity, support, grid):
    v, C = grid_to_particle(velocity, support, grid.layout)
    fitted = v[:, None, :]+jnp.einsum('pij,pkj->pki', C, support.dpos)
    residual = _gather(velocity, support)-fitted
    loss = .5*jnp.sum(support.weights*jnp.sum(residual*residual, axis=-1), axis=-1)
    return v, C, loss


def stress_kick(velocity, mass, volume, tau, support, grid, dt):
    """Use the same MLS gradient for force and midpoint stress power."""
    gradient = 4/grid.layout.dx**2*support.weights[..., None]*support.dpos
    forces = -volume[:, None, None]*jnp.einsum('pij,pkj->pki', tau, gradient)
    force = jnp.zeros_like(velocity).at[support.indices].add(forces, mode='drop')
    updated = velocity+dt*force/jnp.where(mass > 0, mass, 1.)[:, None]
    updated = jnp.where((mass > 0)[:, None], updated, 0.)
    _, midpoint_C = grid_to_particle(.5*(velocity+updated), support, grid.layout)
    work = dt*volume*jnp.sum(tau*midpoint_C, axis=(-2, -1))
    return updated, work


def stress_work(old_velocity, final_velocity, volume, tau, support, grid, dt):
    """Particle work conjugate to the actual complete-step grid velocity path."""
    _, midpoint_C = grid_to_particle(.5*(old_velocity+final_velocity), support, grid.layout)
    return dt*volume*jnp.sum(tau*midpoint_C, axis=(-2, -1))


def clamp_grid(velocity, mass, mask):
    corrected = jnp.where(mask, 0., velocity)
    reservoir = .5*jnp.sum(mass*jnp.sum(jnp.where(mask,velocity**2,0.), axis=-1))
    impulse = jnp.sum(mass[:, None]*(corrected-velocity), axis=0)
    return corrected, reservoir, impulse


def paired_contact(velocity,mass,normals,pairs,clamped=None,iterations=1024,tolerance=.005,colors=None,color_active=None,initial_lambda=None):
    """Mass-normalized accelerated projected dual solve of the same contact QP.

    Fixed DOFs have zero inverse mass. The diagonal metric uses actual graph
    degrees: sum of normalized pair projectors divided by endpoint maximum
    degree is bounded by identity. FISTA with this fixed metric and restart
    accelerates the ill-conditioned coordinate solve without weakening KKT.
    Feasible input chooses zero impulse, avoiding arbitrary dual self-stress.
    """
    if clamped is None:clamped=jnp.zeros_like(velocity,dtype=bool)
    if not len(pairs):
        return velocity,jnp.zeros_like(mass),jnp.zeros_like(velocity),jnp.all(jnp.where(clamped,velocity,0.)==0),jnp.asarray(0.),jnp.asarray(0.),jnp.asarray(0),jnp.zeros(0,dtype=velocity.dtype)
    a,b=pairs[:,0],pairs[:,1]
    norm=jnp.linalg.norm(normals,axis=-1);normal=normals/jnp.maximum(norm,1e-30)[:,None]
    occupied=(mass[a]>0)&(mass[b]>0)
    inverse=jnp.where(clamped,0.,1/jnp.where(mass>0,mass,1.)[:,None])
    diagonal=jnp.sum(normal*normal*(inverse[a]+inverse[b]),axis=-1)
    enabled=occupied&(norm>0)&(diagonal>0)
    root=jnp.sqrt(jnp.where(enabled,diagonal,1.))
    degree=jnp.zeros_like(mass).at[a].add(enabled.astype(mass.dtype)).at[b].add(enabled.astype(mass.dtype))
    metric=jnp.maximum(jnp.maximum(degree[a],degree[b]),1.)
    def evaluate(mu):
        multiplier=jnp.where(enabled,mu/root,0.)
        internal=jnp.zeros_like(velocity).at[a].add(-multiplier[:,None]*normal).at[b].add(multiplier[:,None]*normal)
        value=velocity+inverse*internal
        closing=jnp.sum((value[a]-value[b])*normal,axis=-1)
        residual=jnp.where(mu>0,closing,jnp.maximum(closing,0.))
        error=jnp.max(jnp.where(enabled,jnp.abs(residual),0.),initial=0.)
        return value,internal,closing,error
    initial_closing=jnp.max(jnp.where(enabled,jnp.maximum(jnp.sum((velocity[a]-velocity[b])*normal,axis=-1),0.),0.),initial=0.)
    seed=jnp.zeros(len(pairs),dtype=velocity.dtype) if initial_lambda is None else jnp.maximum(initial_lambda,0.)
    seed=jnp.where(enabled&(initial_closing>tolerance),seed,0.)
    mu=seed*root;value,internal,closing,error=evaluate(mu)
    initial=(mu,mu,jnp.asarray(1.,dtype=velocity.dtype),value,internal,error,jnp.asarray(0))
    def pending(carry):return (carry[-1]<iterations)&(carry[-2]>tolerance)
    def update(carry):
        mu,y,rate,value,internal,error,count=carry
        _,_,closing_y,_=evaluate(y)
        next_mu=jnp.where(enabled,jnp.maximum(y+closing_y/(root*metric),0.),0.)
        next_value,next_internal,closing,error=evaluate(next_mu)
        direction=next_mu-mu
        restart=jnp.sum(metric*direction*(y-next_mu))>0
        next_rate=jnp.where(restart,1.,.5*(1+jnp.sqrt(1+4*rate*rate)))
        beta=jnp.where(restart,0.,(rate-1)/next_rate)
        next_y=next_mu+beta*direction
        return next_mu,next_y,next_rate,next_value,next_internal,error,count+1
    mu,y,rate,result,internal,error,count=jax.lax.while_loop(pending,update,initial)
    multiplier=jnp.where(enabled,jnp.maximum(mu,0.)/root,0.)
    closing=jnp.sum((result[a]-result[b])*normal,axis=-1)
    closure=jnp.max(jnp.where(occupied,jnp.maximum(closing,0.),0.),initial=0.)
    before=.5*jnp.sum(mass*jnp.sum(velocity*velocity,axis=-1));after=.5*jnp.sum(mass*jnp.sum(result*result,axis=-1))
    scale=jnp.maximum(jnp.sum(jnp.abs(internal)),1e-30)
    valid=(jnp.all(~occupied|(norm>0))&jnp.all(jnp.where(clamped,velocity,0.)==0)&(closure<=tolerance)&(error<=tolerance)
           &(after<=before+1e-10*jnp.maximum(before,1e-30))&(jnp.linalg.norm(internal.sum(0))<=1e-10*scale))
    loss=.5*mass*jnp.sum((result-velocity)**2,axis=-1)
    return result,loss,internal,valid,closure,error,count,multiplier


def compatible_contact_work(old_velocity,final_velocity,contact_impulses,mass,pairs,projection_loss,node_body):
    mechanical=jnp.sum(contact_impulses*.5*(old_velocity+final_velocity),axis=-1)
    # All mechanical contact work is signed numerical accounting; no fake heat.
    allocation=-mechanical
    correction=allocation-projection_loss
    aggregate=lambda values:jnp.stack([jnp.sum(jnp.where(node_body==b,values,0.)) for b in (0,1)])
    return allocation,aggregate(projection_loss),aggregate(correction),aggregate(mechanical+allocation)


def contact_normals(state, mass, support, grid):
    # Weighted offsets avoid subtracting large world coordinates near contact.
    moment = jnp.zeros((grid.layout.node_count, 3), dtype=state.x.dtype).at[support.indices].add(
        -state.mass[:, None, None]*support.weights[..., None]*support.dpos, mode='drop')
    center = moment/jnp.where(mass > 0, mass, 1.)[:, None]
    a, b = grid.pairs[:, 0], grid.pairs[:, 1]
    normal = center[b]-center[a]
    # The MLS mass-gradient difference is proportional to this centroid normal.
    # If both vanish, the interface is unresolved; do not invent a direction.
    return jnp.where((jnp.linalg.norm(normal, axis=-1) > 64*jnp.finfo(state.x.dtype).eps*grid.layout.dx)[:, None], normal, 0.)


def cayley_kinematics(F, gradient, dt):
    """Midpoint deformation update and a proper spin rotation for the return map."""
    eye = jnp.eye(3, dtype=F.dtype)
    spin = .5*(gradient-jnp.swapaxes(gradient, -1, -2))
    rate = .5*(gradient+jnp.swapaxes(gradient, -1, -2))
    rotation = jnp.linalg.solve(eye-.5*dt*spin, eye+.5*dt*spin)
    half_rotation = jnp.linalg.solve(eye-.25*dt*spin, eye+.25*dt*spin)
    strain = dt*jnp.swapaxes(half_rotation, -1, -2)@rate@half_rotation
    update = jnp.linalg.solve(eye-.5*dt*gradient, eye+.5*dt*gradient)
    return update@F, rotation, strain


def stable_timestep(state, material, grid, config):
    thermo = thermodynamics(state, material)
    compression = jnp.maximum(-jnp.trace(state.C, axis1=-2, axis2=-1), 0.)
    viscous_signal = (material.viscosity_linear*thermo.sound_speed
                      +2*material.viscosity_quadratic*grid.layout.dx*compression)
    affine_transport = 1.5*jnp.sqrt(3.)*grid.layout.dx*jnp.linalg.norm(state.C, axis=(-2, -1))
    signal = thermo.longitudinal_speed+jnp.linalg.norm(state.v, axis=-1)+affine_transport+viscous_signal
    return config.cfl*grid.layout.dx/jnp.max(signal)


def _step(state, material, grid, config, dt):
    support = stencil(state.x, state.body_id, grid.layout)
    # A swept candidate allocation is accepted only while every occupied shared
    # node pair is represented. Missing possible contact stops advancement.
    preliminary_mass,_,_=transfer_to_grid(state,support,grid)
    occupied=preliminary_mass>0
    count=jnp.zeros(grid.layout.node_count,dtype=jnp.int32).at[grid.global_node_id].add(occupied.astype(jnp.int32))
    a,b=grid.pairs[:,0],grid.pairs[:,1]
    represented=jnp.zeros(grid.layout.node_count,dtype=jnp.int32).at[a].add((occupied[a]&occupied[b]).astype(jnp.int32)).at[b].add((occupied[a]&occupied[b]).astype(jnp.int32))
    coverage=jnp.all(~occupied|(represented==count[grid.global_node_id]-1))
    old_thermo = thermodynamics(state, material)
    J = jnp.linalg.det(state.F)
    q = artificial_viscosity(old_thermo.density, old_thermo.sound_speed,
        jnp.trace(state.C, axis1=-2, axis2=-1), grid.layout.dx, material)
    effective_pressure = jnp.where(old_thermo.pressure < 0, (1-state.damage)*old_thermo.pressure, old_thermo.pressure)
    tau = state.tau-J[:, None, None]*(effective_pressure+q)[:, None, None]*jnp.eye(3)
    mass, old_velocity, p2g_loss = transfer_to_grid(state, support, grid)
    kicked, _ = stress_kick(old_velocity, mass, state.volume, tau, support, grid, dt)
    before_contact, boundary0, _ = clamp_grid(kicked, mass, grid.clamped)
    normals=contact_normals(state,mass,support,grid)
    projected, projection_loss, contact_impulses, contact_valid, maximum_closing, complementarity, contact_sweeps, contact_lambda = paired_contact(
        before_contact, mass, normals, grid.pairs, grid.clamped,config.contact_iterations,config.contact_tolerance,grid.contact_colors,grid.contact_color_active,state.contact_lambda*dt/jnp.where(state.contact_cache_dt>0,state.contact_cache_dt,dt))
    velocity, boundary1, _ = clamp_grid(projected, mass, grid.clamped)
    a,b=grid.pairs[:,0],grid.pairs[:,1]
    normal_unit=normals/jnp.maximum(jnp.linalg.norm(normals,axis=-1),1e-30)[:,None]
    closing=jnp.where((mass[a]>0)&(mass[b]>0),jnp.sum((velocity[a]-velocity[b])*normal_unit,axis=-1),0.)
    maximum_closing=jnp.maximum(jnp.max(closing,initial=0.),0.)
    contact_boundary_reaction=mass[:,None]*(projected-before_contact)-contact_impulses
    boundary_impulses = mass[:, None]*(before_contact-kicked+velocity-projected)+contact_boundary_reaction
    midpoint_velocity = .5*(old_velocity+velocity)
    boundary = -jnp.sum(boundary_impulses*midpoint_velocity)
    boundary_projection = boundary0+boundary1
    boundary_correction = boundary-boundary_projection
    impulse = boundary_impulses.sum(axis=0)
    allocation, physical_by_body, correction_by_body, body_work = compatible_contact_work(
        old_velocity, velocity, contact_impulses, mass, grid.pairs, projection_loss, grid.node_body)
    work = stress_work(old_velocity, velocity, state.volume, tau, support, grid, dt)
    specific_contact = jnp.sum(support.weights*_gather(allocation/jnp.where(mass > 0, mass, 1.), support), axis=-1)
    v, C, g2p_loss = transfer_to_particles(velocity, support, grid)
    F, rotation, strain = cayley_kinematics(state.F, C, dt)
    new_J = jnp.linalg.det(F)
    old_heat = state.internal_energy-state.cold_work-state.fracture_energy-elastic_energy(state.tau, material, state.damage)-damage_volume_energy(J, state.damage, material)
    if config.strength:
        factor=1-state.damage
        undamaged_tau=state.tau/jnp.maximum(factor,1e-12)[:,None,None]
        returned = deviatoric_return(undamaged_tau, state.plastic_strain, strain, rotation,
                                    jnp.where(factor>0,new_J,1.), dt, old_thermo.temperature, material)
        new_tau = returned.kirchhoff_stress*factor[:,None,None]
        ep=jnp.where(factor>0,returned.plastic_strain,state.plastic_strain)
        cold = state.cold_work+factor*returned.specific_cold_work
        plastic_heat = factor*returned.specific_plastic_heat
        material_valid = jnp.all(returned.valid|(factor==0))
    else:
        new_tau, ep, cold = jnp.zeros_like(state.tau), state.plastic_strain, state.cold_work
        plastic_heat = jnp.zeros_like(old_heat)
        material_valid = jnp.all(state.tau == 0) & jnp.all(state.cold_work == 0)
    # Irreversible strain-driven damage. Parameters are hypotheses, not calibration.
    proposed = jnp.clip((ep-config.damage_onset_strain)/(config.damage_failure_strain-config.damage_onset_strain), 0., 1.)
    damage = jnp.maximum(state.damage, proposed)
    retained = (1-damage)/jnp.maximum(1-state.damage, 1e-12)
    before_release = elastic_energy(new_tau, material, state.damage)
    new_tau = new_tau*retained[:, None, None]
    released = jnp.maximum(before_release-elastic_energy(new_tau, material, damage), 0.)
    released += jnp.maximum(damage_volume_energy(new_J,state.damage,material)-damage_volume_energy(new_J,damage,material),0.)
    fracture_energy = state.fracture_energy+released
    internal = (damage_volume_energy(new_J, damage, material)+old_heat+plastic_heat+cold+fracture_energy
                +elastic_energy(new_tau, material, damage))
    # Keep every noncalorimetric energy term visible as a signed numerical sink.
    compatible_internal = state.internal_energy+p2g_loss+g2p_loss+specific_contact+work/state.mass
    numerical_increment = jnp.sum(state.mass*(compatible_internal-internal))
    viscous_tau = -J[:, None, None]*q[:, None, None]*jnp.eye(3)
    viscosity_work = stress_work(old_velocity, velocity, state.volume,
                                 viscous_tau, support, grid, dt).sum()
    constitutive_defect = (work.sum()-viscosity_work
                          -jnp.sum(state.mass*(internal-state.internal_energy)))
    x = state.x+dt*v
    # x and F both use the same end-contact velocity field: first-order motion.
    current = ImpactState(x, v, F, C, state.mass, state.volume, state.body_id, internal,
        new_tau, ep, cold, state.time+dt, state.boundary_reservoir+boundary,
        state.numerical_reservoir+numerical_increment,
        state.boundary_impulse+impulse, state.boundary_projection_loss+boundary_projection,
        state.boundary_coupling_work+boundary_correction,
        state.contact_projection_loss+physical_by_body,
        state.contact_coupling_work+correction_by_body, state.initial_entropy,
        state.minimum_entropy, state.negative_entropy_increments,
        state.ever_phase_extrapolated, state.ever_tension_extrapolated,
        state.maximum_temperature, state.minimum_J, state.maximum_J, damage, fracture_energy, jnp.maximum(state.maximum_contact_closing_speed,maximum_closing), jnp.maximum(state.maximum_contact_complementarity_residual,complementarity), jnp.maximum(state.maximum_contact_iterations,contact_sweeps.astype(state.x.dtype)), contact_lambda, dt, state.valid)
    thermo = thermodynamics(current, material)
    entropy_increment = thermo.entropy-old_thermo.entropy
    current = current._replace(minimum_entropy=jnp.minimum(state.minimum_entropy, thermo.entropy),
        negative_entropy_increments=state.negative_entropy_increments+jnp.minimum(entropy_increment, 0.),
        ever_phase_extrapolated=state.ever_phase_extrapolated | thermo.phase_extrapolated,
        ever_tension_extrapolated=state.ever_tension_extrapolated | thermo.tension_extrapolated,
        maximum_temperature=jnp.maximum(state.maximum_temperature, thermo.temperature),
        minimum_J=jnp.minimum(state.minimum_J, new_J), maximum_J=jnp.maximum(state.maximum_J, new_J))
    limit = stable_timestep(state, material, grid, config)
    residual = energy_ledger(current, grid).sum()-energy_ledger(state, grid).sum()
    physical_residual = residual-numerical_increment
    momentum_residual = jnp.sum(state.mass[:, None]*(v-state.v), axis=0)-impulse
    energy_scale = jnp.maximum(jnp.abs(energy_ledger(state, grid)).sum()
                              +jnp.abs(energy_ledger(current, grid)).sum(), 1e-30)
    momentum_scale = jnp.maximum(jnp.sum(state.mass*(jnp.linalg.norm(state.v, axis=-1)
                              +jnp.linalg.norm(v, axis=-1)))+jnp.linalg.norm(impulse), 1e-30)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in current]))
    valid = (state.valid & support.valid & stencil(x, state.body_id, grid.layout).valid
             & jnp.all(old_thermo.valid) & jnp.all(thermo.valid) & contact_valid & material_valid
             & coverage & finite & (dt > 0) & (dt <= limit*(1+1e-12))
             & (jnp.abs(residual) <= 1e-10*energy_scale)
             & (jnp.linalg.norm(momentum_residual) <= 1e-10*momentum_scale))
    current = current._replace(valid=valid)
    diagnostics = ImpactDiagnostics(jnp.sum(state.mass*p2g_loss), jnp.sum(state.mass*g2p_loss),
        physical_by_body, correction_by_body, allocation.sum(), body_work,
        work.sum(), boundary, residual, physical_residual, numerical_increment, viscosity_work, constitutive_defect, momentum_residual, limit,
        q.max(), new_J.min(), thermo.temperature.min(), thermo.entropy.min(),
        entropy_increment.min(), jnp.mean(entropy_increment < 0),
        (current.minimum_entropy-current.initial_entropy).min(),
        current.negative_entropy_increments.min(), jnp.any(thermo.phase_extrapolated),
        jnp.any(thermo.tension_extrapolated), valid)
    return current, diagnostics


@partial(jax.jit, static_argnames=('config',))
def step(state, material, grid, config, dt):
    return _step(state, material, grid, config, dt)


@partial(jax.jit, static_argnames=('config', 'max_steps'))
def advance_to(state, material, grid, config, target_time, *, max_steps=10000):
    """Stop at the requested output time or first invalid state; never delete mass."""
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')

    def condition(carry):
        current, count = carry
        return current.valid & (current.time < target_time) & (count < max_steps)

    def body(carry):
        current, count = carry
        dt = jnp.minimum(stable_timestep(current, material, grid, config), target_time-current.time)
        return _step(current, material, grid, config, dt)[0], count+1

    admitted = state._replace(valid=state.valid & jnp.isfinite(target_time) & (target_time >= state.time))
    current, count = jax.lax.while_loop(condition, body, (admitted, jnp.asarray(0)))
    return current._replace(valid=current.valid & (current.time == target_time)), count


def validate_state(state, material, grid):
    if np.any(np.asarray(state.contact_lambda)<0) or float(state.contact_cache_dt)<0:
        raise FloatingPointError('Invalid transient contact cache')
    if np.any(np.asarray(state.damage)<0) or np.any(np.asarray(state.damage)>1) or np.any(np.asarray(state.fracture_energy)<0):
        raise FloatingPointError('Invalid irreversible fracture history')
    n = np.asarray(state.x).shape[0] if np.asarray(state.x).ndim == 2 else 0
    shapes = {'x': (n, 3), 'v': (n, 3), 'F': (n, 3, 3), 'C': (n, 3, 3),
        'tau': (n, 3, 3), 'mass': (n,), 'volume': (n,), 'body_id': (n,),
        'internal_energy': (n,), 'plastic_strain': (n,), 'cold_work': (n,), 'damage': (n,), 'fracture_energy': (n,),
        'initial_entropy': (n,), 'minimum_entropy': (n,), 'negative_entropy_increments': (n,),
        'ever_phase_extrapolated': (n,), 'ever_tension_extrapolated': (n,),
        'maximum_temperature': (n,), 'minimum_J': (n,), 'maximum_J': (n,),
        'time': (), 'boundary_reservoir': (), 'numerical_reservoir': (), 'boundary_impulse': (3,),
        'boundary_projection_loss': (), 'boundary_coupling_work': (), 'maximum_contact_closing_speed': (), 'maximum_contact_complementarity_residual': (), 'maximum_contact_iterations': (), 'contact_lambda': (len(grid.pairs),), 'contact_cache_dt': (),
        'contact_projection_loss': (2,), 'contact_coupling_work': (2,), 'valid': ()}
    if n == 0 or set(shapes) != set(state._fields):
        raise FloatingPointError('Impact carry requires a nonempty registered state')
    for field, shape in shapes.items():
        value = np.asarray(getattr(state, field))
        dtype_ok = (value.dtype == np.dtype(bool) if field in
                    ('valid', 'ever_phase_extrapolated', 'ever_tension_extrapolated') else
                    value.dtype.kind in 'iu' if field == 'body_id' else value.dtype == np.dtype('float64'))
        if value.shape != shape or not dtype_ok:
            raise FloatingPointError(f'Impact carry field {field} has an invalid shape or dtype')
    if not np.array_equal(np.unique(np.asarray(state.body_id)), np.arange(grid.layout.fibre_count)):
        raise FloatingPointError('Impact carry must retain both body IDs 0 and 1')
    if not bool(state.valid) or not bool(stencil(state.x, state.body_id, grid.layout).valid):
        raise FloatingPointError('Impact state is invalid or outside the allocated grid')
    thermo = thermodynamics(state, material)
    if not np.asarray(thermo.valid).all():
        raise FloatingPointError('Impact state lies outside the declared EOS domain')
    if any(not np.isfinite(np.asarray(a)).all() for a in state):
        raise FloatingPointError('Impact state contains nonfinite data')
    if np.any(np.asarray(state.mass) <= 0) or np.any(np.asarray(state.volume) <= 0):
        raise FloatingPointError('Impact particle mass and volume must remain positive')
    if not np.allclose(np.asarray(state.mass), material.density*np.asarray(state.volume),
                       rtol=32*np.finfo(np.asarray(state.mass).dtype).eps, atol=0.):
        raise FloatingPointError('Impact reference mass must equal rho0 times reference volume')
    if (np.any(np.asarray(state.plastic_strain) < 0) or np.any(np.asarray(state.cold_work) < 0)
            or float(state.boundary_projection_loss) < 0
            or np.any(np.asarray(state.contact_projection_loss) < 0) or float(state.time) < 0):
        raise FloatingPointError('Plastic history, projection losses and time must remain nonnegative')
    reservoir = float(state.boundary_projection_loss+state.boundary_coupling_work)
    ledger_scale = max(abs(reservoir)+abs(float(state.boundary_projection_loss))
                       +abs(float(state.boundary_coupling_work)), 1e-30)
    if abs(float(state.boundary_reservoir)-reservoir) > 1e-12*ledger_scale:
        raise FloatingPointError('Boundary ledger differs from its projection and coupling terms')
    if (np.any(np.asarray(state.minimum_entropy) > np.asarray(state.initial_entropy))
            or np.any(np.asarray(state.negative_entropy_increments) > 0)):
        raise FloatingPointError('Entropy-history extrema and negative increments are inconsistent')
    current_J = np.linalg.det(np.asarray(state.F))
    roundoff = 64*np.finfo(np.float64).eps
    temperature_floor = roundoff*np.maximum(np.abs(np.asarray(thermo.temperature)), 1.)
    J_floor = roundoff*np.maximum(np.abs(current_J), 1.)
    if (np.any(np.asarray(state.maximum_temperature)+temperature_floor < np.asarray(thermo.temperature))
            or np.any(np.asarray(state.minimum_J)-J_floor > current_J)
            or np.any(np.asarray(state.maximum_J)+J_floor < current_J)
            or np.any(np.asarray(state.minimum_J) < material.minimum_J-J_floor)
            or np.any((np.asarray(state.maximum_J) > material.maximum_J+J_floor)&(np.asarray(state.damage)<1))
            or np.any((np.asarray(state.maximum_temperature) >= material.melt_temperature)
                      & ~np.asarray(state.ever_phase_extrapolated))
            or np.any(np.asarray(thermo.phase_extrapolated) & ~np.asarray(state.ever_phase_extrapolated))
            or np.any(np.asarray(thermo.tension_extrapolated) & ~np.asarray(state.ever_tension_extrapolated))):
        raise FloatingPointError('EOS domain history does not contain the current state')
    tau = np.asarray(state.tau)
    scale = max(1., float(np.abs(tau).max()))
    if (np.max(np.abs(tau-np.swapaxes(tau, -1, -2))) > 1e-12*scale
            or np.max(np.abs(np.trace(tau, axis1=-2, axis2=-1))) > 1e-12*scale):
        raise FloatingPointError('Deviatoric Kirchhoff stress must be symmetric and traceless')
    return True
