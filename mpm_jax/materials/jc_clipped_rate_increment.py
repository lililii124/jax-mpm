"""Safeguarded Newton return map with a bracketed bisection fallback.

The residual is the same temperature-explicit Johnson-Cook radial-return
equation as the reference bisection path.  The thermal coefficient is held at
the old temperature during the mechanical solve; plastic heating follows it.
"""
import jax
import jax.numpy as jnp


def coefficients(ep, temperature, p):
    return 1-jnp.clip((temperature-p.reference_temperature)/
                      (p.melt_temperature-p.reference_temperature),0,1)**p.m


def flow(dep, ep, thermal, dt, p):
    return ((p.A+p.B*(ep+dep)**p.n)
            *(1+p.C*jnp.log(jnp.maximum(dep/(dt*p.reference_rate),1)))*thermal)


def _dense_increment(qtrial, ep, temperature, dt, p):
    thermal=coefficients(ep,temperature,p)
    tolerance=.25*jnp.maximum(1e-10,128*jnp.finfo(qtrial.dtype).eps)*jnp.maximum(p.A,qtrial)
    active=qtrial>(p.A+p.B*ep**p.n)*thermal+tolerance
    upper=qtrial/(3*p.mu)
    hardening_bound=jnp.maximum(
        jnp.maximum((qtrial/jnp.maximum(thermal,1e-30)-p.A)/jnp.maximum(p.B,1e-30),0)**(1/p.n)-ep,0)
    base_flow=(p.A+p.B*ep**p.n)*thermal
    rate_bound=dt*p.reference_rate*jnp.exp(jnp.clip(
        (qtrial/jnp.maximum(base_flow,1e-30)-1)/jnp.maximum(p.C,1e-30),-700,700))
    hardening_bound=jnp.where((p.B>0)&(thermal>0),hardening_bound,jnp.inf)
    rate_bound=jnp.where((p.C>0)&(thermal>0),rate_bound,jnp.inf)
    upper=jnp.minimum(upper,jnp.minimum(hardening_bound,rate_bound))

    def body(_,carry):
        lo,hi,x=carry
        rate=jnp.maximum(x/(dt*p.reference_rate),1)
        rate_factor=1+p.C*jnp.log(rate)
        strain=jnp.maximum(ep+x,1e-30)
        hardening=p.A+p.B*strain**p.n
        residual=qtrial-3*p.mu*x-hardening*rate_factor*thermal
        converged=(jnp.abs(residual)<=tolerance)|~active
        lo=jnp.where((residual>0)&~converged,x,lo)
        hi=jnp.where((residual<=0)&~converged,x,hi)
        derivative=(3*p.mu+thermal*(p.B*p.n*strain**(p.n-1)*rate_factor+
                    hardening*jnp.where(x>dt*p.reference_rate,p.C/jnp.maximum(x,1e-30),0)))
        proposed=x+residual/derivative
        inside=(proposed>=lo)&(proposed<=hi)&jnp.isfinite(proposed)
        proposed=jnp.where(inside,proposed,.5*(lo+hi))
        return lo,hi,jnp.where(converged,x,proposed)

    lo,hi,x=jax.lax.fori_loop(0,12,body,(jnp.zeros_like(upper),upper,.5*upper))
    residual=qtrial-3*p.mu*x-flow(x,ep,thermal,dt,p)
    needs=active&(jnp.abs(residual)>tolerance)
    def fallback(_):
        def bisect(_,bracket):
            a,b=bracket
            mid=.5*(a+b)
            residual=qtrial-3*p.mu*mid-flow(mid,ep,thermal,dt,p)
            return jnp.where(residual>0,mid,a),jnp.where(residual>0,b,mid)
        a,b=jax.lax.fori_loop(0,64,bisect,(lo,hi))
        return jnp.where(needs,.5*(a+b),x)
    x=jax.lax.cond(jnp.any(needs),fallback,lambda _:x,operand=None)
    return jnp.where(active,x,0)


def clipped_rate_increment(qtrial, ep, temperature, dt, p):
    scalar_names=('A','B','C','n','m','mu','reference_rate','reference_temperature','melt_temperature')
    if any(jnp.ndim(getattr(p,name))!=0 for name in scalar_names):
        return _dense_increment(qtrial,ep,temperature,dt,p)
    thermal=coefficients(ep,temperature,p)
    tolerance=.25*jnp.maximum(1e-10,128*jnp.finfo(qtrial.dtype).eps)*jnp.maximum(p.A,qtrial)
    active=qtrial>(p.A+p.B*ep**p.n)*thermal+tolerance
    count=jnp.sum(active.astype(jnp.int32))
    small_capacity=min(512,qtrial.shape[0])
    large_capacity=min(2048,qtrial.shape[0])
    def packed(capacity):
        ids=jnp.nonzero(active,size=capacity,fill_value=0)[0]
        chosen=jnp.arange(capacity)<count
        dep=_dense_increment(qtrial[ids],ep[ids],temperature[ids],dt,p)
        return jnp.zeros_like(qtrial).at[ids].add(jnp.where(chosen,dep,0.))
    def select(_):
        return jax.lax.cond(count<=small_capacity,
            lambda __:packed(small_capacity),
            lambda __:jax.lax.cond(count<=large_capacity,
                lambda ___:packed(large_capacity),
                lambda ___:_dense_increment(qtrial,ep,temperature,dt,p),None),None)
    return jax.lax.cond(count==0,lambda _:jnp.zeros_like(qtrial),select,None)
