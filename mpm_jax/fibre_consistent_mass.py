"""Consistent APIC mass candidate for a single fibre with prescribed collars."""
import jax
import jax.numpy as jnp
from .fibre_mpm import stencil,particle_to_grid,grid_to_particle
from .materials.isotropic_brittle import update_brittle,brittle_energy


def make_step(material,layout,config,masks,velocity,*,max_iterations=256,relative_tolerance=1e-10):
    """Project in the particle translation/affine kinetic metric, then kick force."""
    D=layout.dx**2/4
    def kinetic(s):return .5*jnp.sum(s.mass*(jnp.sum(s.v*s.v,axis=1)+D*jnp.sum(s.C*s.C,axis=(1,2))))
    def potential(s,F=None,history=None):
        return jnp.sum(s.volume*brittle_energy(s.F if F is None else F,s.history if history is None else history,material))
    @jax.jit
    def step(s,warm):
        support=stencil(s.x,s.fibre_id,layout)
        zero=s._replace(tau=jnp.zeros_like(s.tau))
        mass,rhs=particle_to_grid(zero,layout,config,support)
        _,forced=particle_to_grid(s,layout,config,support)
        force_impulse=forced-rhs
        def adjoint(v,C):
            return particle_to_grid(zero._replace(v=v,C=C),layout,config,support)[1]
        def matrix(v):return adjoint(*grid_to_particle(v,support,layout))
        diagonal=jnp.zeros(layout.node_count,dtype=s.x.dtype).at[support.indices].add(
            s.mass[:,None]*support.weights**2*(1+jnp.sum(support.dpos**2,axis=2)/D),mode='drop')
        fixed=jnp.zeros(layout.node_count,dtype=bool);target=jnp.zeros((layout.node_count,3),dtype=s.x.dtype)
        boundary_valid=jnp.asarray(True)
        for i,mask in enumerate(masks):
            selected=jnp.zeros(layout.node_count,dtype=jnp.int32).at[support.indices].max(
                jnp.broadcast_to(mask[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
            selected=selected&(mass>0)
            boundary_valid=boundary_valid&~jnp.any(fixed&selected)
            target=jnp.where(selected[:,None],velocity(i,s.time+config.dt),target)
            fixed=fixed|selected
        free=(mass>0)&~fixed
        scale=jnp.where(free,diagonal,1.)[:,None]
        def constrained(v):return jnp.where(free[:,None],matrix(jnp.where(free[:,None],v,0.)),v)
        def solve(b,guess):
            b=jnp.where(free[:,None],b,0.);x=jnp.where(free[:,None],guess,0.)
            r=b-constrained(x);z=r/scale;p=z
            norm=jnp.linalg.norm(b);limit=relative_tolerance*jnp.maximum(norm,jnp.finfo(s.x.dtype).tiny)
            dot=lambda a,b:jnp.sum(a*b)
            def condition(c):
                i,x,r,z,p,rz,valid=c
                return (i<max_iterations)&(jnp.linalg.norm(r)>limit)&valid
            def body(c):
                i,x,r,z,p,rz,valid=c;Ap=constrained(p);denominator=dot(p,Ap)
                valid=valid&jnp.isfinite(denominator)&(denominator>0)
                alpha=rz/jnp.where(valid,denominator,1.)
                following=x+alpha*p
                recursive=r-alpha*Ap
                restart=(i+1)%16==0
                newr=jax.lax.cond(restart,lambda _:b-constrained(following),lambda _:recursive,None)
                newz=newr/scale;newrz=dot(newr,newz)
                newp=jnp.where(restart,newz,newz+(newrz/jnp.where(rz!=0,rz,1.))*p)
                return i+1,following,newr,newz,newp,newrz,valid
            count,x,r,z,p,rz,valid=jax.lax.while_loop(condition,body,
                (jnp.int32(0),x,r,z,p,dot(r,z),jnp.asarray(True)))
            true_residual=b-constrained(x)
            error=jnp.linalg.norm(true_residual)/jnp.maximum(norm,jnp.finfo(s.x.dtype).tiny)
            return x,error,count,valid&(error<=relative_tolerance)
        begin_free,e0,n0,ok0=solve(rhs-matrix(target),warm-target)
        begin=begin_free+target
        increment,e1,n1,ok1=solve(force_impulse,jnp.zeros_like(warm))
        end=begin+increment;midpoint=(begin+end)/2
        v0,C0=grid_to_particle(begin,support,layout)
        v,C=grid_to_particle(end,support,layout)
        advect,L=grid_to_particle(midpoint,support,layout)
        F=(jnp.eye(3)+config.dt*L)@s.F;x=s.x+config.dt*advect
        history,tau=update_brittle(s.history,F,material)
        projection_reaction=matrix(begin)-rhs
        force_reaction=matrix(increment)-force_impulse
        reaction=jnp.where(fixed[:,None],projection_reaction+force_reaction,0.)
        work=jnp.sum(target*reaction);impulse=reaction.sum(0)
        projection_loss=.5*jnp.sum(s.mass*(jnp.sum((v0-s.v)**2,axis=1)+D*jnp.sum((C0-s.C)**2,axis=(1,2))))
        valid=s.valid&support.valid&stencil(x,s.fibre_id,layout).valid&boundary_valid&ok0&ok1&jnp.all(history.valid)
        valid=valid&jnp.all(jnp.isfinite(F))&jnp.all(jnp.isfinite(x))&jnp.all(jnp.linalg.det(F)>0)
        new=s._replace(x=x,v=v,C=C,F=F,history=history,tau=tau,time=s.time+config.dt,valid=valid,
            boundary_impulse=s.boundary_impulse+impulse,boundary_work=s.boundary_work+work)
        K0,K1=kinetic(s),kinetic(new);U0,U1=potential(s),potential(new)
        Ufixed=potential(s,F=F);damage=Ufixed-U1
        stress_work=config.dt*jnp.sum(s.volume*jnp.sum(s.tau*L,axis=(1,2)))
        material_remainder=Ufixed-U0-stress_work
        remainder=work-(K1-K0)-(U1-U0)-damage
        row=dict(work_J=work,projection_loss_J=projection_loss,material_remainder_J=material_remainder,
            damage_J=damage,residual_J=remainder,split_identity_error_J=remainder-projection_loss+material_remainder,
            kinetic_J=K1,elastic_J=U1,projection_cg_relative_error=e0,force_cg_relative_error=e1,
            projection_cg_iterations=n0,force_cg_iterations=n1,valid=valid,
            min_J=jnp.min(jnp.linalg.det(F)),max_damage=jnp.max(history.damage),
            force_adjoint_error_J=jnp.sum(force_impulse*midpoint)+stress_work,
            free_reaction_norm=jnp.linalg.norm(jnp.where(free[:,None],projection_reaction+force_reaction,0.)),
            linear_momentum_error=jnp.linalg.norm(jnp.sum(s.mass[:,None]*(v-s.v),axis=0)-impulse))
        return new,end,row
    return step
