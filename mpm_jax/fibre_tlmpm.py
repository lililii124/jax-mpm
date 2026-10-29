"""Total-Lagrangian quadratic MPM for one elastic-brittle fibre."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .fibre_mpm import stencil,grid_coordinates
from .materials.isotropic_brittle import initial_brittle_history,update_brittle,brittle_first_piola,brittle_energy
from .materials.isotropic_brittle import _spectrum,_elastic_response
from .fibre_tl_guides import project,make_guide_operator,sample as sample_guides,nodal_impulse


class TLState(NamedTuple):
    displacement:jax.Array
    velocity:jax.Array
    force:jax.Array
    history:object
    F:jax.Array
    time:jax.Array
    valid:jax.Array
    work:jax.Array
    impulse:jax.Array
    angular_impulse:jax.Array
    damage_work:jax.Array


def make_particle_sampler(reference,support):
    """Fuse fixed-reference interpolation; avoid velocity work for positions."""
    indices=support.indices;weights=support.weights
    def interpolate(values):
        return jnp.sum(weights[:,:,None]*jnp.take(values,indices,axis=0,mode='fill',fill_value=0),axis=1)
    @jax.jit
    def positions(displacement):return reference+interpolate(displacement)
    @jax.jit
    def velocities(velocity):return interpolate(velocity)
    def particles(state,*,positions_only=False):
        points=positions(state.displacement)
        return points if positions_only else (points,velocities(state.velocity))
    return particles


def make_solver(reference,volume,layout,material,masks,dt,protocol,*,duration,
                guide_operator=None,guide_particles=None,guide_displacement=None,guide_velocity=None,
                particle_fibre_ids=None,guide_active=None,particle_interaction=None):
    """Build a TL solver; optional guides are RATTLE centroid constraints.

    Guide callbacks return one y displacement/velocity per finite section at
    the requested time.  They never prescribe individual particles or lock a
    section's deformation modes.  An optional guide_active(time) callback
    removes released constraint rows entirely. It is evaluated at the interval
    midpoint for both RATTLE impulses, so an event-aligned final controlled
    interval completes before the next interval becomes free. Release events
    must be aligned to time-step boundaries by the process driver.
    particle_interaction(x, context), if supplied, returns potential energy,
    conservative particle forces and an admissibility flag. Its forces are
    mapped with the exact transpose of particle interpolation. The caller must
    reject unaccounted potential jumps when changing the interaction context.
    step(..., refresh_interaction_force=False) is allowed only when the cached
    total force was evaluated at this state with the same interaction context.
    Initial states and refreshed coverage require the default True.
    """
    node_count=int(layout.node_count);dx=float(layout.dx)
    reference=jnp.asarray(reference);volume=jnp.asarray(volume)
    fibre_ids=(jnp.zeros(len(reference),dtype=jnp.int32) if particle_fibre_ids is None
               else jnp.asarray(particle_fibre_ids,jnp.int32))
    if fibre_ids.shape!=(len(reference),):raise ValueError('One reference-grid fibre id per particle is required')
    support=stencil(reference,fibre_ids,layout)
    q=(reference-layout.origin)/dx;base=jnp.floor(q-.5);fraction=q-base
    from itertools import product
    offsets=jnp.array(list(product(range(3),repeat=3)),dtype=jnp.int32)
    w=jnp.stack((.5*(1.5-fraction)**2,.75-(fraction-1)**2,.5*(fraction-.5)**2),axis=1)
    dw=jnp.stack((fraction-1.5,-2*(fraction-1),fraction-.5),axis=1)/dx
    gradients=[]
    for axis in range(3):
        factors=[(dw if d==axis else w)[:,offsets[:,d],d] for d in range(3)]
        gradients.append(factors[0]*factors[1]*factors[2])
    gradient=jnp.stack(gradients,axis=2)
    mass=jnp.zeros(node_count,dtype=reference.dtype).at[support.indices].add(
        material.density*volume[:,None]*support.weights,mode='drop')
    node_reference,_=grid_coordinates(layout,jnp.arange(node_count,dtype=jnp.int32))
    mu=material.young/(2*(1+material.poisson));lam=material.young*material.poisson/((1+material.poisson)*(1-2*material.poisson))
    wave_speed=jnp.sqrt((lam+2*mu)/material.density)
    characteristic_span=jnp.maximum(jnp.max(jnp.ptp(reference,axis=0)),dx)
    momentum_scale=material.density*jnp.sum(volume)*wave_speed
    angular_momentum_scale=momentum_scale*characteristic_span
    group=jnp.full(node_count,-1,dtype=jnp.int32);valid_boundary=jnp.asarray(True)
    for i,mask in enumerate(masks):
        selected=jnp.zeros(node_count,dtype=jnp.int32).at[support.indices].max(
            jnp.broadcast_to(jnp.asarray(mask)[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
        selected=selected&(mass>0)
        valid_boundary=valid_boundary&~jnp.any(selected&(group>=0))
        group=jnp.where(selected,i,group)
    fixed=group>=0
    if guide_particles is not None:
        if guide_operator is not None:raise ValueError('Pass guide_particles or a prebuilt guide_operator, not both')
        guide_operator=make_guide_operator(reference,volume,material.density,support,mass,guide_particles)
    if guide_operator is not None:
        if guide_displacement is None or guide_velocity is None:
            raise ValueError('Guided TL dynamics require displacement and velocity target callbacks')
        guide_nodes=jnp.zeros(node_count,dtype=jnp.int32).at[guide_operator.nodes].max(1,mode='drop')>0
        valid_boundary=valid_boundary&~jnp.any(guide_nodes&fixed)
    # Estimate the initial constrained semidiscrete acoustic spectrum once.
    # This includes reference-cell quadrature and actual small nodal masses.
    @jax.jit
    def acoustic_spectrum():
        inverse_sqrt=jnp.where((mass>0)&~fixed,1/jnp.sqrt(jnp.where(mass>0,mass,1.)),0.)
        mu=material.young/(2*(1+material.poisson))
        lam=material.young*material.poisson/((1+material.poisson)*(1-2*material.poisson))
        def apply(q):
            v=q*inverse_sqrt[:,None]
            gradient_v=jnp.einsum('pki,pkj->pij',jnp.take(v,support.indices,axis=0,mode='fill',fill_value=0),gradient)
            strain=.5*(gradient_v+jnp.swapaxes(gradient_v,1,2))
            stress=2*mu*strain+lam*jnp.trace(strain,axis1=1,axis2=2)[:,None,None]*jnp.eye(3)
            force=volume[:,None,None]*jnp.einsum('pij,pkj->pki',stress,gradient)
            result=jnp.zeros_like(q).at[support.indices].add(force,mode='drop')
            return result*inverse_sqrt[:,None]
        q=jnp.sin(jnp.arange(node_count*3,dtype=reference.dtype).reshape(-1,3)+.25)*inverse_sqrt[:,None]
        q=q/jnp.linalg.norm(q)
        def body(q,_):
            result=apply(q);value=jnp.linalg.norm(result)
            return result/value,value
        _,values=jax.lax.scan(body,q,None,length=64)
        return values
    eigenvalues=acoustic_spectrum()
    critical_dt=2/jnp.sqrt(eigenvalues[-1])
    import math
    planned_steps=math.ceil(duration/min(dt,.8*float(critical_dt)))
    dt=duration/planned_steps
    def prescribe(t,order):
        if 'fixture_times' in protocol:
            times=jnp.asarray(protocol['fixture_times']);positions=jnp.asarray(protocol['fixture_positions'])
            index=jnp.clip(jnp.searchsorted(times,t,side='right')-1,0,len(times)-2)
            duration=times[index+1]-times[index];phase=jnp.clip((t-times[index])/duration,0.,1.)
            blend=.5*(1-jnp.cos(jnp.pi*phase))
            if order==0:value=positions[index]*(1-blend)+positions[index+1]*blend-positions[0]
            elif order==1:value=jnp.where((t>=times[index])&(t<times[index+1]),
                .5*jnp.pi/duration*jnp.sin(jnp.pi*phase)*(positions[index+1]-positions[index]),0.)
            else:value=jnp.where((t>=times[index])&(t<times[index+1]),
                .5*(jnp.pi/duration)**2*jnp.cos(jnp.pi*phase)*(positions[index+1]-positions[index]),0.)
            selected=jnp.where((group>=0)[:,None],value[jnp.maximum(group,0)],0.)
            return jnp.where(fixed[:,None],selected,0.)
        T=protocol['load_time_s'];stroke=protocol['stroke_m'];phase=jnp.clip(t/T,0.,1.)
        if order==0:value=.5*stroke*(1-jnp.cos(jnp.pi*phase))
        elif order==1:value=jnp.where((t>=0)&(t<T),.5*stroke*jnp.pi/T*jnp.sin(jnp.pi*phase),0.)
        else:value=jnp.where((t>=0)&(t<T),.5*stroke*(jnp.pi/T)**2*jnp.cos(jnp.pi*phase),0.)
        sign=jnp.where(group==0,-.5,jnp.where(group==1,.5,0.))
        return jnp.zeros((node_count,3),dtype=reference.dtype).at[:,0].set(sign*value)
    def sample(values):
        return jnp.sum(support.weights[:,:,None]*jnp.take(values,support.indices,axis=0,mode='fill',fill_value=0),axis=1)
    def deformation(u):
        gathered=jnp.take(u,support.indices,axis=0,mode='fill',fill_value=0)
        return jnp.eye(3)+jnp.einsum('pki,pkj->pij',gathered,gradient)
    def internal(F,history):
        P=brittle_first_piola(F,history,material)
        particle_force=-volume[:,None,None]*jnp.einsum('pij,pkj->pki',P,gradient)
        return jnp.zeros((node_count,3),dtype=reference.dtype).at[support.indices].add(particle_force,mode='drop')
    def interaction(u,context):
        if particle_interaction is None:
            return jnp.asarray(0.,reference.dtype),jnp.zeros_like(u),jnp.asarray(True)
        potential,force,valid=particle_interaction(reference+sample(u),context)
        nodal=jnp.zeros_like(u).at[support.indices].add(support.weights[:,:,None]*force[:,None,:],mode='drop')
        return potential,nodal,valid&jnp.isfinite(potential)&jnp.all(jnp.isfinite(force))
    initial_F=jnp.broadcast_to(jnp.eye(3),(len(reference),3,3))
    initial=TLState(jnp.zeros((node_count,3)),jnp.zeros((node_count,3)),jnp.zeros((node_count,3)),
        initial_brittle_history(len(reference)),initial_F,jnp.asarray(0.),support.valid&valid_boundary,jnp.asarray(0.),jnp.zeros(3),jnp.zeros(3),jnp.asarray(0.))
    def energy(s,interaction_context=None):
        potential,_,_=interaction(s.displacement,interaction_context)
        return .5*jnp.sum(mass[:,None]*s.velocity**2),jnp.sum(volume*brittle_energy(s.F,s.history,material))+potential
    @jax.jit
    def step(s,interaction_context=None,step_dt=None,refresh_interaction_force=True):
        h=dt if step_dt is None else jnp.asarray(step_dt,dtype=reference.dtype)
        step_size_valid=(h>0)&(h<=dt*(1+1e-12))
        old_contact_valid=jnp.asarray(True)
        if particle_interaction is not None:
            # A refreshed coverage context must also be used for the first
            # Verlet kick; cached forces may belong to the previous context.
            def refreshed(_):
                _,old_contact,valid=interaction(s.displacement,interaction_context)
                return internal(s.F,s.history)+old_contact,valid
            cached_force,old_contact_valid=jax.lax.cond(jnp.asarray(refresh_interaction_force),
                refreshed,lambda _:(s.force,jnp.asarray(True)),operand=None)
            s=s._replace(force=cached_force)
        inverse=1/jnp.where(mass>0,mass,1.)[:,None]
        acceleration=jnp.where(fixed[:,None],prescribe(s.time,2),s.force*inverse)
        half=s.velocity+.5*h*acceleration
        u=s.displacement+h*half
        u=jnp.where(fixed[:,None],prescribe(s.time+h,0),u)
        guide_position_multiplier=jnp.zeros((0,),dtype=reference.dtype)
        guide_position_error=jnp.zeros((0,),dtype=reference.dtype)
        guide_midpoint_speed=jnp.zeros((0,),dtype=reference.dtype)
        guide_end_speed=jnp.zeros((0,),dtype=reference.dtype)
        guide_position_before=jnp.zeros((0,),dtype=reference.dtype)
        guide_position_after=jnp.zeros((0,),dtype=reference.dtype)
        guide_impulse0=jnp.zeros_like(s.force)
        guide_impulse1=jnp.zeros_like(s.force)
        if guide_operator is not None:
            predicted=u
            guide_position_before=sample_guides(guide_operator,half[:,1])
            target=guide_displacement(s.time+h)
            active=None if guide_active is None else guide_active(s.time+.5*h)
            u,guide_position_multiplier,guide_position_error=project(guide_operator,u,target,active=active)
            guide_midpoint_speed=.5*(guide_velocity(s.time)+guide_velocity(s.time+h))
            guide_impulse0=jnp.zeros_like(s.force).at[:,1].set(
                nodal_impulse(guide_operator,guide_position_multiplier,node_count)/h)
            # RATTLE position correction contributes the matching half-step
            # velocity, so the subsequent force evaluation sees constrained F.
            half=half+(u-predicted)/h
            guide_position_after=sample_guides(guide_operator,half[:,1])
        half=jnp.where(fixed[:,None],(u-s.displacement)/h,half)
        F=deformation(u);history,_=update_brittle(s.history,F,material)
        force=internal(F,history)
        contact_energy,contact_force,contact_valid=interaction(u,interaction_context)
        force=force+contact_force
        following_acceleration=jnp.where(fixed[:,None],prescribe(s.time+h,2),force*inverse)
        velocity=half+.5*h*following_acceleration
        velocity=jnp.where(fixed[:,None],prescribe(s.time+h,1),velocity)
        guide_velocity_multiplier=jnp.zeros((0,),dtype=reference.dtype)
        guide_velocity_error=jnp.zeros((0,),dtype=reference.dtype)
        guide_velocity_before=jnp.zeros((0,),dtype=reference.dtype)
        guide_velocity_after=jnp.zeros((0,),dtype=reference.dtype)
        guide_velocity_target=jnp.zeros((0,),dtype=reference.dtype)
        guide_velocity_impulse_work=jnp.asarray(0.,dtype=reference.dtype)
        guide_velocity_kinetic_jump=jnp.asarray(0.,dtype=reference.dtype)
        if guide_operator is not None:
            velocity_pre_projection=velocity
            guide_velocity_target=guide_velocity(s.time+h)
            guide_end_speed=guide_velocity_target
            velocity,guide_velocity_multiplier,guide_velocity_error=project(
                guide_operator,velocity_pre_projection,guide_velocity_target,
                active=None if guide_active is None else guide_active(s.time+.5*h))
            guide_impulse1=jnp.zeros_like(s.force).at[:,1].set(
                nodal_impulse(guide_operator,guide_velocity_multiplier,node_count))
            guide_velocity_before=sample_guides(guide_operator,velocity_pre_projection[:,1])
            guide_velocity_after=sample_guides(guide_operator,velocity[:,1])
            guide_velocity_impulse_work=jnp.sum(guide_velocity_multiplier*.5*(guide_velocity_before+guide_velocity_after))
            guide_velocity_kinetic_jump=.5*jnp.sum(mass[:,None]*(velocity**2-velocity_pre_projection**2))
        reaction0=jnp.where(fixed[:,None],mass[:,None]*acceleration-s.force,0.)
        reaction1=jnp.where(fixed[:,None],mass[:,None]*following_acceleration-force,0.)
        end_work_node=jnp.sum(.5*(reaction0+reaction1)*(u-s.displacement),axis=1)
        end_work_group=jnp.zeros((len(masks),),dtype=reference.dtype).at[jnp.maximum(group,0)].add(
            jnp.where(fixed,end_work_node,0.))
        work=jnp.sum(end_work_group)
        guide_work_position=jnp.zeros((0,),dtype=reference.dtype)
        guide_work_velocity=jnp.zeros((0,),dtype=reference.dtype)
        if guide_operator is not None:
            # A Verlet kick applies elastic and constraint impulses together.
            # Split its kinetic-energy increment using the midpoint velocity
            # of the complete kick, not the separate projection substep.
            guide_kick0_midpoint=sample_guides(guide_operator,.5*(s.velocity[:,1]+half[:,1]))
            guide_kick1_midpoint=sample_guides(guide_operator,.5*(half[:,1]+velocity[:,1]))
            guide_work_position=guide_position_multiplier*guide_kick0_midpoint/h
            guide_work_velocity=guide_velocity_multiplier*guide_kick1_midpoint
            work=work+jnp.sum(guide_work_position)+jnp.sum(guide_work_velocity)
        impulse0=jnp.where(fixed[:,None],mass[:,None]*(half-s.velocity)-.5*h*s.force,0.)+guide_impulse0
        impulse1=jnp.where(fixed[:,None],mass[:,None]*(velocity-half)-.5*h*force,0.)+guide_impulse1
        impulse=jnp.sum(impulse0+impulse1,axis=0)
        torque=jnp.sum(jnp.cross(node_reference+s.displacement,impulse0)+jnp.cross(node_reference+u,impulse1),axis=0)
        # These are the same spectrum/response used by the history and Piola
        # calculations above. Keeping identical calls allows XLA to share the
        # decomposition, instead of separate eigvalsh and SVD kernels.
        _,undamaged_energy,_,_=_elastic_response(F,material)
        _,stretches,_,_=_spectrum(F)
        old_elastic=(1-s.history.damage)*undamaged_energy
        new_elastic=(1-history.damage)*undamaged_energy
        damage=jnp.sum(volume*(old_elastic-new_elastic))
        max_log_strain=jnp.max(abs(jnp.log(stretches)))
        guide_ok=jnp.asarray(True)
        if guide_operator is not None:
            guide_ok=jnp.all(abs(guide_position_error)<=1e-11*dx)&jnp.all(abs(guide_velocity_error)<=1e-10*jnp.sqrt(material.young/material.density))
        linear_error=jnp.linalg.norm(jnp.sum(mass[:,None]*velocity,axis=0)-(s.impulse+impulse))
        angular_error=jnp.linalg.norm(jnp.sum(jnp.cross(node_reference+u,mass[:,None]*velocity),axis=0)-(s.angular_impulse+torque))
        linear_relative=linear_error/jnp.maximum(momentum_scale,1e-300)
        angular_relative=angular_error/jnp.maximum(angular_momentum_scale,1e-300)
        valid=s.valid&step_size_valid&old_contact_valid&contact_valid&jnp.all(history.valid)&jnp.all(jnp.isfinite(u))&jnp.all(jnp.isfinite(velocity))&jnp.all(jnp.linalg.det(F)>0)&guide_ok&(max_log_strain<=.12)&(linear_relative<=1e-8)&(angular_relative<=1e-8)
        new=TLState(u,velocity,force,history,F,s.time+h,valid,s.work+work,s.impulse+impulse,s.angular_impulse+torque,s.damage_work+damage)
        K=.5*jnp.sum(mass[:,None]*new.velocity**2)
        U=jnp.sum(volume*new_elastic)+contact_energy
        row=dict(work_J=work,damage_J=damage,kinetic_J=K,elastic_J=U-contact_energy,interaction_J=contact_energy,
            interaction_nodal_force_l2_N=jnp.linalg.norm(contact_force),
            interaction_nodal_force_max_N=jnp.max(jnp.linalg.norm(contact_force,axis=1),initial=0.),
            residual_J=new.work-K-U-new.damage_work,min_J=jnp.min(jnp.linalg.det(F)),max_damage=jnp.max(history.damage),
            max_log_strain=max_log_strain,strain_guard_passed=max_log_strain<=.12,
            machine_abs_work_J=jnp.sum(abs(jnp.concatenate((end_work_group,guide_work_position,guide_work_velocity)))),
            end_constraint_work_J=jnp.sum(end_work_group),
            guide_position_constraint_work_J=jnp.sum(guide_work_position),
            guide_velocity_constraint_work_J=jnp.sum(guide_work_velocity),
            guide_position_impulse_L1_Ns=jnp.sum(abs(guide_position_multiplier/h)),
            guide_velocity_impulse_L1_Ns=jnp.sum(abs(guide_velocity_multiplier)),
            guide_position_speed_max_m_s=jnp.max(abs(guide_midpoint_speed),initial=0.),
            guide_velocity_speed_max_m_s=jnp.max(abs(guide_end_speed),initial=0.),
            guide_position_multiplier_kg_m=guide_position_multiplier,
            guide_position_target_speed_m_s=guide_midpoint_speed,
            guide_velocity_multiplier_Ns=guide_velocity_multiplier,
            guide_velocity_target_speed_m_s=guide_velocity_target,
            guide_velocity_before_projection_m_s=guide_velocity_before,
            guide_velocity_after_projection_m_s=guide_velocity_after,
            guide_velocity_impulse_midpoint_work_J=guide_velocity_impulse_work,
            guide_velocity_kinetic_jump_J=guide_velocity_kinetic_jump,
            guide_mean_force_on_material_N=(guide_position_multiplier/h+guide_velocity_multiplier)/h,
            guide_mean_force_max_N=jnp.max(abs((guide_position_multiplier/h+guide_velocity_multiplier)/h),initial=0.),
            guide_position_nodal_impulse_Ns=guide_impulse0.sum(axis=0),
            guide_velocity_nodal_impulse_Ns=guide_impulse1.sum(axis=0),
            guide_total_constraint_impulse_Ns=(guide_impulse0+guide_impulse1).sum(axis=0),
            guide_position_error=jnp.max(abs(guide_position_error),initial=0.),
            guide_velocity_error=jnp.max(abs(guide_velocity_error),initial=0.),
            fixture_work_total_J=new.work,damage_work_total_J=new.damage_work,
            tensile_load_N=.5*(jnp.sum(jnp.where((group==1)[:,None],reaction1,0.),axis=0)[0]
                -jnp.sum(jnp.where((group==0)[:,None],reaction1,0.),axis=0)[0]),
            momentum_error=jnp.linalg.norm(jnp.sum(mass[:,None]*velocity,axis=0)-new.impulse),valid=valid)
        row['angular_momentum_error']=angular_error;row['momentum_error']=linear_error
        row['momentum_relative_error']=linear_relative;row['angular_momentum_relative_error']=angular_relative
        row['momentum_guard_passed']=linear_relative<=1e-8;row['angular_momentum_guard_passed']=angular_relative<=1e-8
        row['momentum_scale_Ns']=momentum_scale;row['angular_momentum_scale_Nms']=angular_momentum_scale
        return new,row
    particles=make_particle_sampler(reference,support)
    geometry_check=dict(partition_error=jnp.max(abs(support.weights.sum(1)-1)),
        gradient_constant_error=jnp.max(abs(gradient.sum(1))),reference_mass=material.density*volume.sum(),nodal_mass=mass.sum(),
        gradient_affine_error=jnp.max(abs(jnp.einsum('pki,pkj->pij',support.dpos,gradient)-jnp.eye(3))),
        initial_largest_acoustic_eigenvalue=eigenvalues[-1],acoustic_64_iteration_relative_change=abs(eigenvalues[-1]/eigenvalues[-9]-1),
        estimated_initial_critical_dt=critical_dt,dt=jnp.asarray(dt),planned_steps=jnp.asarray(planned_steps),
        wave_speed_m_s=wave_speed,characteristic_span_m=characteristic_span,
        linear_momentum_scale_Ns=momentum_scale,angular_momentum_scale_Nms=angular_momentum_scale)
    return initial,step,particles,energy,geometry_check
