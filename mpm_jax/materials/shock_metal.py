"""Energy-coupled linear-Hugoniot EOS and separate corotated metal strength."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ShockMetal(NamedTuple):
    density: float
    c0: float
    slope: float
    gamma0: float
    heat_capacity: float
    reference_temperature: float
    melt_temperature: float
    shear_modulus: float
    A: float
    B: float
    n: float
    C: float
    m: float
    reference_rate: float
    heat_fraction: float
    minimum_J: float
    maximum_J: float
    viscosity_linear: float
    viscosity_quadratic: float


def shock_metal(*, density=2700., c0=5350., slope=1.34, gamma0=2.,
                heat_capacity=900., reference_temperature=300., melt_temperature=933.15,
                shear_modulus=26e9, A=100e6, B=300e6, n=.34, C=.01, m=1.,
                reference_rate=1., heat_fraction=.9, minimum_J=.4, maximum_J=2.,
                viscosity_linear=1., viscosity_quadratic=2.):
    """Illustrative Al-like SI parameters, without alloy or phase calibration."""
    values=locals()
    if not all(np.ndim(v)==0 and np.isfinite(v) for v in values.values()):
        raise ValueError('shock-metal parameters must be finite scalars')
    positive=('density','c0','slope','heat_capacity','reference_temperature',
              'shear_modulus','A','n','m','reference_rate','minimum_J')
    if any(values[k]<=0 for k in positive):
        raise ValueError('positive material parameters required')
    if min(gamma0,B,C,viscosity_linear,viscosity_quadratic)<0 or not 0<=heat_fraction<=1:
        raise ValueError('invalid thermal, hardening or viscosity parameter')
    if not minimum_J<1<maximum_J or slope*(1-minimum_J)>=.9:
        raise ValueError('volume domain must bracket one and avoid the Hugoniot pole')
    if melt_temperature<=reference_temperature:
        raise ValueError('melting threshold must exceed reference temperature')
    return ShockMetal(**values)


class EOSState(NamedTuple):
    density: jax.Array
    pressure: jax.Array
    temperature: jax.Array
    entropy: jax.Array
    bulk_modulus: jax.Array
    sound_speed: jax.Array
    longitudinal_speed: jax.Array
    phase_extrapolated: jax.Array
    tension_extrapolated: jax.Array
    valid: jax.Array


_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def _reference(J, p):
    eta=1-J
    denominator=1-p.slope*eta
    safe=jnp.where(denominator>0,denominator,1.)
    K=p.density*p.c0**2
    ph=K*eta/safe**2
    dph=K*(1+p.slope*eta)/safe**3
    # Expansion follows p_I=K eta, e_I=c0^2 eta^2/2, an isentrope.
    pressure=jnp.where(eta>=0,ph,K*eta)
    energy=pressure*eta/(2*p.density)
    base=pressure-p.gamma0*p.density*energy
    derivative=jnp.where(eta>=0,dph*(1-.5*p.gamma0*eta)-.5*p.gamma0*ph,
                         K*(1-p.gamma0*eta))
    return eta,pressure,energy,base,derivative


def hugoniot(particle_velocity, p):
    """Compression shock jump state from an initially stationary p=e=0 state."""
    up=jnp.asarray(particle_velocity)
    us=p.c0+p.slope*up
    J=1-up/us
    pressure=p.density*us*up
    energy=.5*up**2
    valid=(up>=0)&jnp.isfinite(up)&(J>=p.minimum_J)&(J<=1)
    return J,pressure,energy,us,valid


def pressure(J, specific_energy, p):
    """Compression-positive Pa; specific_energy is EOS energy in J/kg."""
    _,_,_,base,_=_reference(jnp.asarray(J),p)
    return base+p.gamma0*p.density*specific_energy


def temperature(J, specific_energy, p):
    """Constant-Cv thermodynamic completion; no latent heat or phase mixture."""
    J=jnp.asarray(J,dtype=jnp.result_type(J,1.))
    eta,_,reference_energy,_,_=_reference(J,p)
    compression=jnp.maximum(eta,0.)
    nodes=jnp.asarray(_GL_X,dtype=J.dtype)
    weights=jnp.asarray(_GL_W,dtype=J.dtype)

    def integrate(i,total):
        z=.5*compression*(nodes[i]+1)
        denominator=1-p.slope*z
        safe=jnp.where(denominator>0,denominator,1.)
        return total+weights[i]*jnp.exp(-p.gamma0*z)*z*z/safe**3

    integral=.5*compression*jax.lax.fori_loop(0,32,integrate,jnp.zeros_like(J))
    reference_T=jnp.exp(p.gamma0*eta)*(p.reference_temperature+p.c0**2*p.slope*integral/p.heat_capacity)
    return reference_T+(specific_energy-reference_energy)/p.heat_capacity


def energy_from_temperature(J, T, p):
    J=jnp.asarray(J)
    _,_,reference_energy,_,_=_reference(J,p)
    return reference_energy+p.heat_capacity*(T-temperature(J,reference_energy,p))


def eos_state(J, specific_energy, p):
    J=jnp.asarray(J,dtype=jnp.result_type(J,1.))
    e=jnp.asarray(specific_energy)
    eta,_,_,_,derivative=_reference(J,p)
    rho=p.density/jnp.where(J>0,J,1.)
    value=pressure(J,e,p)
    T=temperature(J,e,p)
    # c_s^2=(dp/drho)_e+(dp/de)_rho p/rho^2, not an isothermal derivative.
    cs2=J*J*(derivative+p.gamma0*value)/p.density
    entropy=p.heat_capacity*(jnp.log(jnp.where(T>0,T/p.reference_temperature,1.))-p.gamma0*eta)
    valid=(J>=p.minimum_J)&(J<=p.maximum_J)&(1-p.slope*eta>0)&(T>0)&(cs2>0)
    valid=valid&jnp.isfinite(e)&jnp.isfinite(value)&jnp.isfinite(T)&jnp.isfinite(cs2)
    bulk=rho*cs2
    sound=jnp.sqrt(jnp.maximum(cs2,0))
    shear_speed2=p.shear_modulus*jnp.maximum(J,1.)/p.density
    longitudinal=jnp.sqrt(jnp.maximum(cs2,0)+4*shear_speed2/3)
    return EOSState(rho,value,T,entropy,bulk,sound,longitudinal,T>=p.melt_temperature,value<0,valid)


def artificial_viscosity(density, sound_speed, divergence, cell_length, p):
    """Scalar shock pressure q>=0, zero for expansion and volume-preserving shear."""
    compression=jnp.maximum(-divergence,0.)
    velocity=cell_length*compression
    q=density*(p.viscosity_linear*sound_speed*velocity+p.viscosity_quadratic*velocity**2)
    valid=(density>0)&(sound_speed>=0)&(cell_length>0)&jnp.isfinite(divergence)&jnp.isfinite(q)
    return jnp.where(valid,q,jnp.nan)


class EnergyStep(NamedTuple):
    specific_energy: jax.Array
    midpoint_pressure: jax.Array
    pressure_work: jax.Array
    viscosity_heat: jax.Array
    residual: jax.Array
    valid: jax.Array


def volume_energy_step(J_old, J_new, energy_old, q, heat, p):
    """Solve de=-(p_mid+q)dv+heat exactly for the discrete midpoint EOS.

    heat is a separately supplied specific dissipative input. Do not supply
    plastic heat again when energy already comes from a compatible total-energy
    update. The same q and work must be used by the spatial momentum scheme.
    """
    J_old,J_new,energy_old,q,heat=map(jnp.asarray,(J_old,J_new,energy_old,q,heat))
    dv=(J_new-J_old)/p.density
    midpoint_J=.5*(J_old+J_new)
    a=p.gamma0*p.density
    _,_,_,base,_=_reference(midpoint_J,p)
    denominator=1+.5*a*dv
    safe=jnp.where(denominator>0,denominator,1.)
    new=(energy_old*(1-.5*a*dv)-(base+q)*dv+heat)/safe
    midpoint_p=pressure(midpoint_J,.5*(energy_old+new),p)
    work=-midpoint_p*dv
    viscous=-q*dv
    residual=new-energy_old-work-viscous-heat
    valid=eos_state(J_old,energy_old,p).valid&eos_state(J_new,new,p).valid
    valid=valid&(denominator>0)&(q>=0)&(heat>=0)&((dv<=0)|(q==0))
    valid=valid&jnp.isfinite(q)&jnp.isfinite(heat)&jnp.isfinite(residual)
    return EnergyStep(new,midpoint_p,work,viscous,residual,valid)


def flow_stress(ep, rate, T, p):
    thermal=jnp.clip((T-p.reference_temperature)/(p.melt_temperature-p.reference_temperature),0.,1.)
    return (p.A+p.B*jnp.maximum(ep,0.)**p.n)*(1+p.C*jnp.log(jnp.maximum(rate/p.reference_rate,1.)))*(1-thermal**p.m)


class DeviatoricStep(NamedTuple):
    kirchhoff_stress: jax.Array
    plastic_strain: jax.Array
    plastic_increment: jax.Array
    specific_plastic_work: jax.Array
    specific_plastic_heat: jax.Array
    specific_cold_work: jax.Array
    specific_elastic_energy: jax.Array
    specific_return_dissipation: jax.Array
    yield_residual: jax.Array
    phase_extrapolated: jax.Array
    valid: jax.Array


def deviatoric_return(tau_old, ep_old, strain_increment, rotation, J, dt, T, p):
    """Old-frame corotated radial return, followed by the supplied proper rotation.

    Stress is Kirchhoff: spatial Cauchy stress is tau/J-p*I. Temperature is frozen
    during this substep. Return dissipation is the elastic-predictor energy loss;
    its excess over plastic work is backward-Euler numerical dissipation.
    """
    tau_old=jnp.asarray(tau_old)
    strain_increment=jnp.asarray(strain_increment)
    rotation=jnp.asarray(rotation)
    eye=jnp.eye(3,dtype=tau_old.dtype)
    strain=.5*(strain_increment+jnp.swapaxes(strain_increment,-1,-2))
    strain=strain-jnp.trace(strain,axis1=-2,axis2=-1)[...,None,None]*eye/3
    trial=tau_old+2*p.shear_modulus*strain
    qtrial=jnp.sqrt(jnp.maximum(1.5*jnp.sum(trial**2,axis=(-2,-1)),0.))
    safe_dt=jnp.where(dt>0,dt,1.)
    active=qtrial>J*flow_stress(ep_old,0.,T,p)

    def bisect(_,bracket):
        lo,hi=bracket;mid=.5*(lo+hi)
        residual=qtrial-3*p.shear_modulus*mid-J*flow_stress(ep_old+mid,mid/safe_dt,T,p)
        return jnp.where(residual>0,mid,lo),jnp.where(residual>0,hi,mid)

    lo,hi=jax.lax.fori_loop(0,64,bisect,(jnp.zeros_like(qtrial),qtrial/(3*p.shear_modulus)))
    tol=256*jnp.finfo(tau_old.dtype).eps*jnp.maximum(p.A*J,qtrial)+1e-6

    def unresolved(lo,hi):
        increment=.5*(lo+hi)
        stress=jnp.maximum(qtrial-3*p.shear_modulus*increment,0.)
        residual=stress-J*flow_stress(ep_old+increment,increment/safe_dt,T,p)
        return active&(jnp.abs(residual)>tol)

    def needs_refinement(carry):
        lo,hi,count=carry
        return (count<64)&jnp.any(unresolved(lo,hi))

    def refine(carry):
        lo,hi,count=carry
        pending=unresolved(lo,hi)
        next_lo,next_hi=bisect(0,(lo,hi))
        return jnp.where(pending,next_lo,lo),jnp.where(pending,next_hi,hi),count+1

    # Fractional hardening has a steep derivative at vanishing plastic strain.
    # Refine only roots that miss the existing stress-residual tolerance, with
    # at most 128 total bisections; the final validity gate remains unchanged.
    lo,hi,_=jax.lax.while_loop(needs_refinement,refine,(lo,hi,jnp.asarray(0)))
    dep=jnp.where(active,.5*(lo+hi),0.)
    qnew=jnp.maximum(qtrial-3*p.shear_modulus*dep,0.)
    returned=trial*(qnew/jnp.maximum(qtrial,1e-30))[...,None,None]
    rotated=rotation@returned@jnp.swapaxes(rotation,-1,-2)
    elastic=jnp.sum(returned**2,axis=(-2,-1))/(4*p.shear_modulus*p.density)
    loss=(jnp.sum(trial**2,axis=(-2,-1))-jnp.sum(returned**2,axis=(-2,-1)))/(4*p.shear_modulus*p.density)
    plastic=qnew*dep/p.density
    residual=qnew-J*flow_stress(ep_old+dep,dep/safe_dt,T,p)
    rotation_tolerance=max(1e-10,128*jnp.finfo(tau_old.dtype).eps)
    proper=jnp.max(jnp.abs(jnp.swapaxes(rotation,-1,-2)@rotation-eye),axis=(-2,-1))<rotation_tolerance
    proper=proper&(jnp.abs(jnp.linalg.det(rotation)-1)<rotation_tolerance)
    symmetric=jnp.max(jnp.abs(tau_old-jnp.swapaxes(tau_old,-1,-2)),axis=(-2,-1))<tol
    deviatoric=jnp.abs(jnp.trace(tau_old,axis1=-2,axis2=-1))<tol
    valid=proper&symmetric&deviatoric&(dt>0)&(J>=p.minimum_J)&(J<=p.maximum_J)&(T>0)&(ep_old>=0)
    valid=valid&(~active|(jnp.abs(residual)<=tol))&jnp.all(jnp.isfinite(rotated),axis=(-2,-1))
    valid=valid&jnp.isfinite(ep_old)&jnp.isfinite(T)&jnp.isfinite(J)&jnp.isfinite(loss)
    return DeviatoricStep(rotated,ep_old+dep,dep,plastic,p.heat_fraction*plastic,
                          (1-p.heat_fraction)*plastic,elastic,loss,residual,
                          T>=p.melt_temperature,valid)
