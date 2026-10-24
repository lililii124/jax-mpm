"""Illustrative linear volume elasticity with plastic-work-only calorimetry.

Compression is epsilon_v=1-J. This is not a shock EOS or a phase model.
The actual calorimetric temperature may exceed Tm; only strength softening
saturates at Tm. No heat is discarded by clipping temperature.
"""
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from .shock_metal import EOSState, artificial_viscosity, deviatoric_return, flow_stress

class LinearMetal(NamedTuple):
    density: float
    bulk_modulus: float
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

def linear_metal(*, density=2700., bulk_modulus=2700.*5350.**2,
                 heat_capacity=900., reference_temperature=300., melt_temperature=933.15,
                 shear_modulus=26e9, A=100e6, B=300e6, n=.34, C=.01, m=1.,
                 reference_rate=1., heat_fraction=.9, minimum_J=.4, maximum_J=2.,
                 viscosity_linear=1., viscosity_quadratic=2.):
    values=locals()
    if not all(np.ndim(v)==0 and np.isfinite(v) for v in values.values()):
        raise ValueError('linear-metal parameters must be finite scalars')
    positive=('density','bulk_modulus','heat_capacity','reference_temperature',
              'shear_modulus','A','n','m','reference_rate','minimum_J')
    if any(values[k]<=0 for k in positive):
        raise ValueError('positive material parameters required')
    if min(B,C,viscosity_linear,viscosity_quadratic)<0 or not 0<=heat_fraction<=1:
        raise ValueError('invalid hardening, heat fraction or viscosity')
    if not minimum_J<1<maximum_J or melt_temperature<=reference_temperature:
        raise ValueError('invalid volume domain or melting threshold')
    return LinearMetal(**values)

def pressure(J, specific_energy, p):
    """Compression-positive pressure; calorimetric energy does not affect it."""
    return p.bulk_modulus*(1-jnp.asarray(J))

def volume_energy(J, p):
    """Recoverable volumetric energy per reference mass, J/kg."""
    return p.bulk_modulus*(1-jnp.asarray(J))**2/(2*p.density)

def temperature(J, specific_energy, p):
    """EOS energy includes recoverable volume energy plus accumulated plastic heat."""
    return p.reference_temperature+(jnp.asarray(specific_energy)-volume_energy(J,p))/p.heat_capacity

def energy_from_temperature(J,T,p):
    return volume_energy(J,p)+p.heat_capacity*(jnp.asarray(T)-p.reference_temperature)

def eos_state(J,specific_energy,p):
    J=jnp.asarray(J)
    rho=p.density/jnp.where(J>0,J,1.)
    value=pressure(J,specific_energy,p)
    T=temperature(J,specific_energy,p)
    cs2=p.bulk_modulus*J**2/p.density
    sound=jnp.sqrt(jnp.maximum(cs2,0.))
    longitudinal=jnp.sqrt(jnp.maximum(cs2,0.)+4*p.shear_modulus*jnp.maximum(J,1.)/(3*p.density))
    entropy=p.heat_capacity*jnp.log(jnp.where(T>0,T/p.reference_temperature,1.))
    valid=(J>=p.minimum_J)&(J<=p.maximum_J)&(T>0)&jnp.isfinite(T)&jnp.isfinite(value)&jnp.isfinite(specific_energy)
    return EOSState(rho,value,T,entropy,rho*cs2,sound,longitudinal,T>=p.melt_temperature,value<0,valid)
