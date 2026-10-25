"""Unadmitted work-compatible midpoint flow with original endpoint yield law.

No material parameters change. The plastic direction uses corotated midpoint
stress; the endpoint obeys the original Johnson-Cook equation. Plastic heat
uses midpoint plastic power, never an energy-ledger correction.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from .materials.shock_metal import flow_stress


class MidpointStrength(NamedTuple):
    kirchhoff_stress:object
    plastic_strain:object
    specific_plastic_heat:object
    specific_cold_work:object
    specific_plastic_work:object
    specific_return_dissipation:object
    midpoint_stress:object
    energy_identity:object
    yield_residual:object
    valid:object


def half_rotation(rotation):
    scalar=.5*j.sqrt(j.maximum(1+j.trace(rotation,axis1=-2,axis2=-1),0.))
    vector=j.stack((rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],rotation[...,1,0]-rotation[...,0,1]),-1)/(4*j.maximum(scalar,1e-30))[...,None]
    half_scalar=j.sqrt(.5*(1+scalar));half_vector=vector/(2*half_scalar[...,None])
    a,b,c=half_vector[...,0],half_vector[...,1],half_vector[...,2];zero=j.zeros_like(a)
    skew=j.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape((*a.shape,3,3))
    return j.eye(3)+2*half_scalar[...,None,None]*skew+2*skew@skew


def midpoint_return(tau_old,ep_old,strain_increment,rotation,J,dt,T,p):
    eye=j.eye(3);half=half_rotation(rotation)
    strain=half.swapaxes(-1,-2)@(.5*(strain_increment+strain_increment.swapaxes(-1,-2)))@half
    strain=strain-j.trace(strain,axis1=-2,axis2=-1)[...,None,None]*eye/3
    center=tau_old+p.shear_modulus*strain
    qcenter=j.sqrt(j.maximum(1.5*j.sum(center*center,axis=(-2,-1)),0.))
    trial=tau_old+2*p.shear_modulus*strain
    qtrial=j.sqrt(j.maximum(1.5*j.sum(trial*trial,axis=(-2,-1)),0.))
    active=qtrial>J*flow_stress(ep_old,0.,T,p)
    minimum_scale=j.clip(j.sum(center*tau_old,axis=(-2,-1))/(2*j.maximum(j.sum(center*center,axis=(-2,-1)),1e-60)),0.,1.)
    upper=2*qcenter/(3*p.shear_modulus)*(1-minimum_scale)
    def evaluate(dep):
        qbar=j.maximum(qcenter-1.5*p.shear_modulus*dep,0.)
        bar=center*(qbar/j.maximum(qcenter,1e-30))[...,None,None]
        end=2*bar-tau_old;qend=j.sqrt(j.maximum(1.5*j.sum(end*end,axis=(-2,-1)),0.))
        residual=qend-J*flow_stress(ep_old+dep,dep/j.where(dt>0,dt,1.),T,p)
        return residual,bar,end,qbar
    bracket=evaluate(upper)[0]<=0
    def bisect(_,bounds):
        lo,hi=bounds;mid=.5*(lo+hi);residual=evaluate(mid)[0]
        return j.where(residual>0,mid,lo),j.where(residual>0,hi,mid)
    lo,hi=jax.lax.fori_loop(0,128,bisect,(j.zeros_like(upper),upper))
    dep=j.where(active,.5*(lo+hi),0.);residual,bar,end,qbar=evaluate(dep)
    rotated=rotation@end@rotation.swapaxes(-1,-2);midpoint=half@bar@half.swapaxes(-1,-2)
    plastic=qbar*dep/p.density
    energy_identity=j.sum(bar*strain,axis=(-2,-1))/p.density-(j.sum(end*end,axis=(-2,-1))-j.sum(tau_old*tau_old,axis=(-2,-1)))/(4*p.shear_modulus*p.density)-plastic
    tolerance=256*j.finfo(tau_old.dtype).eps*j.maximum(p.A*J,qtrial)+1e-6
    proper=(j.max(j.abs(rotation.swapaxes(-1,-2)@rotation-eye),axis=(-2,-1))<max(1e-10,128*j.finfo(tau_old.dtype).eps))&(j.abs(j.linalg.det(rotation)-1)<max(1e-10,128*j.finfo(tau_old.dtype).eps))
    proper=proper&(j.max(j.abs(half.swapaxes(-1,-2)@half-eye),axis=(-2,-1))<max(1e-10,128*j.finfo(tau_old.dtype).eps))
    proper=proper&(j.max(j.abs(half@half-rotation),axis=(-2,-1))<max(1e-10,128*j.finfo(tau_old.dtype).eps))
    valid=proper&(dt>0)&(J>=p.minimum_J)&(J<=p.maximum_J)&(T>0)&(ep_old>=0)
    valid=valid&(j.max(j.abs(tau_old-tau_old.swapaxes(-1,-2)),axis=(-2,-1))<tolerance)&(j.abs(j.trace(tau_old,axis1=-2,axis2=-1))<tolerance)
    valid=valid&(~active|(bracket&(j.abs(residual)<=tolerance)))&j.all(j.isfinite(rotated),axis=(-2,-1))&j.isfinite(plastic)&(plastic>=0)
    return MidpointStrength(rotated,ep_old+dep,p.heat_fraction*plastic,(1-p.heat_fraction)*plastic,plastic,plastic,midpoint,energy_identity,residual,valid)
