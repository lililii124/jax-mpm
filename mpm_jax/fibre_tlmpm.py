"""Total-Lagrangian quadratic MPM for one elastic-brittle fibre."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .fibre_mpm import stencil,grid_coordinates
from .materials.isotropic_brittle import initial_brittle_history,update_brittle,brittle_first_piola,brittle_energy


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


def make_solver(reference,volume,layout,material,masks,dt,protocol,*,duration):
    reference=jnp.asarray(reference);volume=jnp.asarray(volume)
    support=stencil(reference,jnp.zeros(len(reference),dtype=jnp.int32),layout)
    q=(reference-layout.origin)/layout.dx;base=jnp.floor(q-.5);fraction=q-base
    from itertools import product
    offsets=jnp.array(list(product(range(3),repeat=3)),dtype=jnp.int32)
    w=jnp.stack((.5*(1.5-fraction)**2,.75-(fraction-1)**2,.5*(fraction-.5)**2),axis=1)
    dw=jnp.stack((fraction-1.5,-2*(fraction-1),fraction-.5),axis=1)/layout.dx
    gradients=[]
    for axis in range(3):
        factors=[(dw if d==axis else w)[:,offsets[:,d],d] for d in range(3)]
        gradients.append(factors[0]*factors[1]*factors[2])
    gradient=jnp.stack(gradients,axis=2)
    mass=jnp.zeros(layout.node_count,dtype=reference.dtype).at[support.indices].add(
        material.density*volume[:,None]*support.weights,mode='drop')
    node_reference,_=grid_coordinates(layout,jnp.arange(layout.node_count,dtype=jnp.int32))
    group=jnp.full(layout.node_count,-1,dtype=jnp.int32);valid_boundary=jnp.asarray(True)
    for i,mask in enumerate(masks):
        selected=jnp.zeros(layout.node_count,dtype=jnp.int32).at[support.indices].max(
            jnp.broadcast_to(jnp.asarray(mask)[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
        selected=selected&(mass>0)
        valid_boundary=valid_boundary&~jnp.any(selected&(group>=0))
        group=jnp.where(selected,i,group)
    fixed=group>=0;sign=jnp.where(group==0,-.5,jnp.where(group==1,.5,0.))
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
        q=jnp.sin(jnp.arange(layout.node_count*3,dtype=reference.dtype).reshape(-1,3)+.25)*inverse_sqrt[:,None]
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
        T=protocol['load_time_s'];stroke=protocol['stroke_m'];phase=jnp.clip(t/T,0.,1.)
        if order==0:value=.5*stroke*(1-jnp.cos(jnp.pi*phase))
        elif order==1:value=jnp.where((t>=0)&(t<T),.5*stroke*jnp.pi/T*jnp.sin(jnp.pi*phase),0.)
        else:value=jnp.where((t>=0)&(t<T),.5*stroke*(jnp.pi/T)**2*jnp.cos(jnp.pi*phase),0.)
        return jnp.zeros((layout.node_count,3),dtype=reference.dtype).at[:,0].set(sign*value)
    def sample(values):
        return jnp.sum(support.weights[:,:,None]*jnp.take(values,support.indices,axis=0,mode='fill',fill_value=0),axis=1)
    def deformation(u):
        gathered=jnp.take(u,support.indices,axis=0,mode='fill',fill_value=0)
        return jnp.eye(3)+jnp.einsum('pki,pkj->pij',gathered,gradient)
    def internal(F,history):
        P=brittle_first_piola(F,history,material)
        particle_force=-volume[:,None,None]*jnp.einsum('pij,pkj->pki',P,gradient)
        return jnp.zeros((layout.node_count,3),dtype=reference.dtype).at[support.indices].add(particle_force,mode='drop')
    initial_F=jnp.broadcast_to(jnp.eye(3),(len(reference),3,3))
    initial=TLState(jnp.zeros((layout.node_count,3)),jnp.zeros((layout.node_count,3)),jnp.zeros((layout.node_count,3)),
        initial_brittle_history(len(reference)),initial_F,jnp.asarray(0.),support.valid&valid_boundary,jnp.asarray(0.),jnp.zeros(3),jnp.zeros(3),jnp.asarray(0.))
    def energy(s):
        return .5*jnp.sum(mass[:,None]*s.velocity**2),jnp.sum(volume*brittle_energy(s.F,s.history,material))
    @jax.jit
    def step(s):
        inverse=1/jnp.where(mass>0,mass,1.)[:,None]
        acceleration=jnp.where(fixed[:,None],prescribe(s.time,2),s.force*inverse)
        half=s.velocity+.5*dt*acceleration
        u=s.displacement+dt*half
        u=jnp.where(fixed[:,None],prescribe(s.time+dt,0),u)
        half=jnp.where(fixed[:,None],(u-s.displacement)/dt,half)
        F=deformation(u);history,_=update_brittle(s.history,F,material)
        force=internal(F,history)
        following_acceleration=jnp.where(fixed[:,None],prescribe(s.time+dt,2),force*inverse)
        velocity=half+.5*dt*following_acceleration
        velocity=jnp.where(fixed[:,None],prescribe(s.time+dt,1),velocity)
        reaction0=jnp.where(fixed[:,None],mass[:,None]*acceleration-s.force,0.)
        reaction1=jnp.where(fixed[:,None],mass[:,None]*following_acceleration-force,0.)
        work=jnp.sum(.5*(reaction0+reaction1)*(u-s.displacement))
        impulse0=jnp.where(fixed[:,None],mass[:,None]*(half-s.velocity)-.5*dt*s.force,0.)
        impulse1=jnp.where(fixed[:,None],mass[:,None]*(velocity-half)-.5*dt*force,0.)
        impulse=jnp.sum(impulse0+impulse1,axis=0)
        torque=jnp.sum(jnp.cross(node_reference+s.displacement,impulse0)+jnp.cross(node_reference+u,impulse1),axis=0)
        damage=jnp.sum(volume*(brittle_energy(F,s.history,material)-brittle_energy(F,history,material)))
        valid=s.valid&jnp.all(history.valid)&jnp.all(jnp.isfinite(u))&jnp.all(jnp.isfinite(velocity))&jnp.all(jnp.linalg.det(F)>0)
        new=TLState(u,velocity,force,history,F,s.time+dt,valid,s.work+work,s.impulse+impulse,s.angular_impulse+torque,s.damage_work+damage)
        K,U=energy(new)
        row=dict(work_J=work,damage_J=damage,kinetic_J=K,elastic_J=U,
            residual_J=new.work-K-U-new.damage_work,min_J=jnp.min(jnp.linalg.det(F)),max_damage=jnp.max(history.damage),
            fixture_work_total_J=new.work,damage_work_total_J=new.damage_work,
            tensile_load_N=.5*(jnp.sum(jnp.where((group==1)[:,None],reaction1,0.),axis=0)[0]
                -jnp.sum(jnp.where((group==0)[:,None],reaction1,0.),axis=0)[0]),
            momentum_error=jnp.linalg.norm(jnp.sum(mass[:,None]*velocity,axis=0)-new.impulse),valid=valid)
        row['angular_momentum_error']=jnp.linalg.norm(jnp.sum(jnp.cross(node_reference+u,mass[:,None]*velocity),axis=0)-new.angular_impulse)
        return new,row
    def particles(s):return reference+sample(s.displacement),sample(s.velocity)
    geometry_check=dict(partition_error=jnp.max(abs(support.weights.sum(1)-1)),
        gradient_constant_error=jnp.max(abs(gradient.sum(1))),reference_mass=material.density*volume.sum(),nodal_mass=mass.sum(),
        gradient_affine_error=jnp.max(abs(jnp.einsum('pki,pkj->pij',support.dpos,gradient)-jnp.eye(3))),
        initial_largest_acoustic_eigenvalue=eigenvalues[-1],acoustic_64_iteration_relative_change=abs(eigenvalues[-1]/eigenvalues[-9]-1),
        estimated_initial_critical_dt=critical_dt,dt=jnp.asarray(dt),planned_steps=jnp.asarray(planned_steps))
    return initial,step,particles,energy,geometry_check
