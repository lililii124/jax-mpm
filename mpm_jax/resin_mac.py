"""Isolated staggered-grid Newtonian/pressure operators for later resin fill.

Not connected to forming. D and D.T are an explicit adjoint pair. Pressure
projection uses actual face masses and fixed-face counter-impulses. There is
no artificial bulk modulus or energy reservoir. Full filling integration and
operator benchmarks are pending before any physical run.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def divergence(velocity,h):
    u,v,w=velocity
    return h*h*(u[1:]-u[:-1]+v[:,1:]-v[:,:-1]+w[:,:,1:]-w[:,:,:-1])


def transpose_divergence(pressure,h):
    return (h*h*(jnp.pad(pressure,((1,0),(0,0),(0,0)))-jnp.pad(pressure,((0,1),(0,0),(0,0)))),
            h*h*(jnp.pad(pressure,((0,0),(1,0),(0,0)))-jnp.pad(pressure,((0,0),(0,1),(0,0)))),
            h*h*(jnp.pad(pressure,((0,0),(0,0),(1,0)))-jnp.pad(pressure,((0,0),(0,0),(0,1)))))


def symmetric_gradient(velocity,h):
    u,v,w=velocity
    xx=(u[1:]-u[:-1])/h;yy=(v[:,1:]-v[:,:-1])/h;zz=(w[:,:,1:]-w[:,:,:-1])/h
    xy=((u[1:-1,1:]-u[1:-1,:-1])+(v[1:,1:-1]-v[:-1,1:-1]))/(2*h)
    xz=((u[1:-1,:,1:]-u[1:-1,:,:-1])+(w[1:,:,1:-1]-w[:-1,:,1:-1]))/(2*h)
    yz=((v[:,1:-1,1:]-v[:,1:-1,:-1])+(w[:,1:,1:-1]-w[:,:-1,1:-1]))/(2*h)
    return xx,yy,zz,xy,xz,yz


def viscosity_force(velocity,mass,fixed,h,density,viscosity):
    # Minimum volume fractions bound every strain coefficient by each incident
    # free-face mass. Missing air faces have zero weight; fixed walls have1.
    theta=tuple(jnp.where(f&(m==0),1.,m/(density*h**3)) for m,f in zip(mass,fixed))
    a,b,c=theta
    normal=(jnp.minimum(a[1:],a[:-1]),jnp.minimum(b[:,1:],b[:,:-1]),jnp.minimum(c[:,:,1:],c[:,:,:-1]))
    xy=jnp.minimum(jnp.minimum(a[1:-1,1:],a[1:-1,:-1]),jnp.minimum(b[1:,1:-1],b[:-1,1:-1]))
    xz=jnp.minimum(jnp.minimum(a[1:-1,:,1:],a[1:-1,:,:-1]),jnp.minimum(c[1:,:,1:-1],c[:-1,:,1:-1]))
    yz=jnp.minimum(jnp.minimum(b[:,1:-1,1:],b[:,1:-1,:-1]),jnp.minimum(c[:,1:,1:-1],c[:,:-1,1:-1]))
    ax,ay,az=(m>0 for m in mass)
    present=(ax[1:]|ax[:-1],ay[:,1:]|ay[:,:-1],az[:,:,1:]|az[:,:,:-1],ax[1:-1,1:]|ax[1:-1,:-1]|ay[1:,1:-1]|ay[:-1,1:-1],ax[1:-1,:,1:]|ax[1:-1,:,:-1]|az[1:,:,1:-1]|az[:-1,:,1:-1],ay[:,1:-1,1:]|ay[:,1:-1,:-1]|az[:,1:,1:-1]|az[:,:-1,1:-1])
    fractions=normal+(xy,xz,yz)
    weight=tuple(factor*viscosity*h**3*jnp.where(covered,t,0.) for factor,t,covered in zip((2,2,2,4,4,4),fractions,present))
    strain,pullback=jax.vjp(lambda value:symmetric_gradient(value,h),velocity)
    weighted=tuple(k*s for k,s in zip(weight,strain))
    force=tuple(-f for f in pullback(weighted)[0])
    dissipation=sum(jnp.sum(s*t) for s,t in zip(strain,weighted))
    return force,dissipation


class PressureResult(NamedTuple):
    velocity: object
    pressure_Pa: object
    fixture_impulse: object
    relative_residual: object
    maximum_divergence_per_s: object
    fixture_work_J: object
    projection_loss_J: object
    work_identity_J: object
    coverage_valid: object
    valid: object
    iterations: object
    stationarity_relative: object
    stationarity_absolute: object
    reconstruction_relative: object
    impulse_potential_base: object
    impulse_potential_correction: object
    dual_velocity_selected: object


def pressure_projection(velocity,mass,fixed,fluid,h,dt,*,tolerance=1e-8,max_iterations=400,preconditioner=None):
    inverse=tuple(jnp.where((m>0)&~f,1/jnp.maximum(m,1e-300),0.) for m,f in zip(mass,fixed))
    available=tuple((m>0)|f for m,f in zip(mass,fixed))
    a,b,c=available
    covered=a[1:]&a[:-1]&b[:,1:]&b[:,:-1]&c[:,:,1:]&c[:,:,:-1]
    coverage=jnp.all(~fluid|covered)
    a,b,c=inverse
    diagonal=h**4*(a[1:]+a[:-1]+b[:,1:]+b[:,:-1]+c[:,:,1:]+c[:,:,:-1])
    positive=jnp.all(~fluid|(diagonal>0))
    if preconditioner=='mixed-lsqr':
        from .resin_pressure_lsqr import project
        return project(velocity,mass,fixed,fluid,h,dt,inverse,coverage,positive,tolerance,max_iterations)
    # Nonfluid pressure unknowns are explicitly fixed to0 by identity rows;
    # fluid rows receive no regularization, pseudoinverse or inertia floor.
    diag=jnp.where(fluid,diagonal,1.)
    def operator(value):
        gradient=transpose_divergence(jnp.where(fluid,value,0.),h)
        correction=tuple(i*g for i,g in zip(inverse,gradient))
        return jnp.where(fluid,divergence(correction,h),value)
    rhs=jnp.where(fluid,divergence(velocity,h),0.)
    M=lambda value:value/diag
    if preconditioner=='galerkin':
        from .resin_pressure_multigrid import make_preconditioner
        M=make_preconditioner(inverse,fluid,h)
    def solve(_):
        if preconditioner!='galerkin':
            value,_=cg(operator,rhs,x0=jnp.zeros_like(rhs),M=M,tol=tolerance,atol=0.,maxiter=max_iterations)
            return value,jnp.asarray(-1,jnp.int32)
        # Standard preconditioned CG recurrence, with its actual iteration count.
        z=M(rhs);rho=jnp.vdot(rhs,z);threshold=tolerance**2*jnp.vdot(rhs,rhs)
        def condition(carry):
            k,x,r,p,rho=carry
            return (k<max_iterations)&(jnp.vdot(r,r)>threshold)&jnp.isfinite(rho)&(rho>0)
        def body(carry):
            k,x,r,p,rho=carry;Ap=operator(p);alpha=rho/jnp.vdot(p,Ap)
            x=x+alpha*p;r=r-alpha*Ap
            replace=jnp.vdot(r,r)<=threshold
            r=jax.lax.cond(replace,lambda _:rhs-operator(x),lambda _:r,None)
            z=M(r);newrho=jnp.vdot(r,z)
            direction=jnp.where(replace,z,z+(newrho/rho)*p)
            return k+1,x,r,direction,newrho
        k,value,_,_,_=jax.lax.while_loop(condition,body,(jnp.asarray(0,jnp.int32),jnp.zeros_like(rhs),rhs,z,rho))
        return value,k
    multiplier,iterations=jax.lax.cond(coverage&positive&(jnp.linalg.norm(rhs)>0),solve,lambda _:(jnp.zeros_like(rhs),jnp.asarray(0,jnp.int32)),None)
    gradient=transpose_divergence(multiplier,h)
    corrected=tuple(v-i*g for v,i,g in zip(velocity,inverse,gradient))
    fixture=tuple(jnp.where(f,g,0.) for f,g in zip(fixed,gradient))
    work=sum(jnp.sum(I*v) for I,v in zip(fixture,velocity))
    change=.5*sum(jnp.sum(m*(b*b-a*a)) for m,a,b in zip(mass,velocity,corrected))
    loss=.5*sum(jnp.sum(m*(b-a)**2) for m,a,b in zip(mass,velocity,corrected))
    residual=operator(multiplier)-rhs
    relative=jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(rhs),1e-300)
    div=jnp.where(fluid,divergence(corrected,h),0.)/h**3
    valid=coverage&positive&jnp.isfinite(relative)&(relative<=10*tolerance)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in corrected]))
    return PressureResult(corrected,-multiplier/dt,fixture,relative,jnp.max(jnp.abs(div)),work,loss,work-change-loss,coverage,valid,iterations,jnp.asarray(0.),jnp.asarray(0.),relative,multiplier,jnp.zeros_like(multiplier),jnp.asarray(True))
