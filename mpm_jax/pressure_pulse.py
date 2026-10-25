"""Prescribed pressure on a material surface using Nanson area vectors (SI)."""
import jax.numpy as jnp


def pressure(time, peak, duration):
    """Positive compressive sin² pulse; zero outside0<t<duration."""
    phase=jnp.clip(jnp.asarray(time)/duration,0.,1.)
    value=peak*jnp.sin(jnp.pi*phase)**2
    return jnp.where((time>0)&(time<duration),value,0.)


def pressure_integral(time, peak, duration):
    """Continuous pressure impulse per undeformed area, Pa s."""
    phase=jnp.clip(jnp.asarray(time)/duration,0.,1.)
    return peak*duration*(phase/2-jnp.sin(2*jnp.pi*phase)/(4*jnp.pi))


def surface_force(deformation, reference_area, time, peak, duration):
    """Newton force on the +Y material face; pressure acts inward.

    cross(F e_z,F e_x)=cofactor(F)e_y carries both current normal and
    current/reference area ratio. reference_area is zero off the loaded face.
    """
    area_vector=jnp.cross(deformation[..., :,2],deformation[..., :,0])
    return -pressure(time,peak,duration)*reference_area[...,None]*area_vector
