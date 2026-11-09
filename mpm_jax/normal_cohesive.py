"""Irreversible Mode-I cohesive opening with compressive recontact."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .implicit_contact import discrete_pressure,opening


class Interface(NamedTuple):
    plus: jax.Array
    minus: jax.Array
    area: jax.Array
    normal: jax.Array
    valid: jax.Array
    maximum: jax.Array
    bonded: jax.Array


def initialize(line,bonded):
    bonded=jnp.asarray(bonded,dtype=bool)
    if bonded.shape!=line.area.shape:
        raise ValueError('Require one bonded flag per interface quadrature point')
    maximum=jnp.where(bonded,jnp.maximum(opening(line),0.),0.)
    return Interface(*line,maximum,bonded)


def remaining(maximum,law):
    d0,df=law.peak_opening,law.final_opening
    h=jnp.maximum(maximum,d0)
    return jnp.clip(d0*(df-h)/(h*(df-d0)),0.,1.)


def energy(interface,law):
    gap=opening(interface)
    intact=jnp.where(interface.bonded,remaining(interface.maximum,law),0.)
    stored=.5*law.stiffness*jnp.sum(interface.area*(intact*jnp.maximum(gap,0.)**2+jnp.minimum(gap,0.)**2))
    fraction=jnp.clip((interface.maximum-law.peak_opening)/(law.final_opening-law.peak_opening),0.,1.)
    dissipated=law.energy*jnp.sum(interface.area*jnp.where(interface.bonded,fraction,0.))
    return stored,dissipated


def update_history(interface):
    maximum=jnp.where(interface.bonded,jnp.maximum(interface.maximum,jnp.maximum(opening(interface),0.)),0.)
    return interface._replace(maximum=maximum)


def discrete_traction(old,new,maximum,bonded,law):
    """Return exact incremental-work traction and its new-opening derivative.

    Loading work is integrated over the piecewise linear envelope. Unloading
    uses the previous secant stiffness. Trial gaps never mutate stored history.
    """
    K,S,d0,df=law.stiffness,law.strength,law.peak_opening,law.final_opening
    compression,compression_tangent=discrete_pressure(old,new,K)
    h=jnp.maximum(maximum,jnp.maximum(old,0.));intact=remaining(h,law)
    fixed=intact*(.5*K*(old+new)-compression)
    fixed_tangent=intact*(.5*K-compression_tangent)
    loading=(new>h)&bonded
    old_positive=jnp.maximum(old,0.)
    work=.5*K*intact*(h-old_positive)*(h+old_positive)
    a=jnp.minimum(h,d0);b=jnp.clip(new,a,d0)
    work+=.5*K*(b-a)*(a+b)
    a=jnp.clip(h,d0,df);b=jnp.clip(new,a,df)
    work+=S/(df-d0)*(b-a)*(df-.5*(a+b))
    increment=jnp.where(loading,new-old,1.)
    tensile=work/increment
    envelope=jnp.where(new<=d0,K*jnp.maximum(new,0.),S*jnp.clip((df-new)/(df-d0),0.,1.))
    tangent=(envelope-tensile)/increment
    # These smooth branches avoid subtracting nearly equal envelope tractions.
    elastic_loading=loading&(new<=d0)&(old>=0)
    soft_loading=loading&(old==h)&(h>=d0)&(new<=df)
    tensile=jnp.where(elastic_loading,.5*K*(old+new),tensile)
    tangent=jnp.where(elastic_loading,.5*K,jnp.where(soft_loading,-.5*S/(df-d0),tangent))
    tensile=jnp.where(loading,tensile,fixed)
    tangent=jnp.where(loading,tangent,fixed_tangent)
    return compression+jnp.where(bonded,tensile,0.),compression_tangent+jnp.where(bonded,tangent,0.)
