"""Exact frozen RT0 cell-characteristic transport, independent of momentum G2P.

Each MAC component is linear only in its own cell coordinate. Its true
continuous divergence equals MAC D. Cell crossings use shared face fluxes;
log(J) integrates actual cell divergence, without determinant normalization.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class Transport(NamedTuple):
    x: object
    logJ: object
    projected_time: object
    crossings: object
    path_bound: object
    roundoff_distance: object
    valid: object


def represent(position,index,layout):
    origin=jnp.asarray(layout.origin,dtype=position.dtype);q=(position-origin)/layout.dx
    outside=(q<index)|(q>=index+1)
    lower=q<index;boundary=jnp.where(lower,index,index+1).astype(position.dtype);toward=index.astype(position.dtype)+.5
    distance=jnp.abs(q-boundary);allowed=64*jnp.finfo(position.dtype).eps*jnp.maximum(jnp.abs(boundary),1.)
    valid=jnp.all(~outside|(distance<=allowed))
    candidate=jnp.nextafter(boundary,toward)
    def body(_,value):
        physical=origin+value*layout.dx;back=jnp.floor((physical-origin)/layout.dx).astype(jnp.int32)
        return jnp.where(back==index,value,jnp.nextafter(value,toward))
    candidate=jax.lax.fori_loop(0,4,body,candidate);physical=origin+candidate*layout.dx
    adjusted=jnp.where(outside,physical,position);back=jnp.floor((adjusted-origin)/layout.dx).astype(jnp.int32)
    valid=valid&jnp.all(back==index)
    return adjusted,jnp.max(jnp.abs(adjusted-position),axis=1),valid


def advect(x,velocity,active,solid,projected,layout,dt,*,max_crossings=8):
    origin=jnp.asarray(layout.origin,dtype=x.dtype);shape=jnp.asarray(layout.shape);h=layout.dx
    index=jnp.floor((x-origin)/h).astype(jnp.int32)
    def cell_value(field,index):
        inside=jnp.all((index>=0)&(index<shape),axis=1);safe=jnp.clip(index,0,shape-1)
        return field[safe[:,0],safe[:,1],safe[:,2]],inside
    wall,inside=cell_value(solid,index);valid=jnp.all(~active|(inside&~wall));left=jnp.where(active,dt,0.)
    logJ=jnp.zeros(len(x),dtype=x.dtype);projected_time=jnp.zeros_like(logJ);crossings=jnp.zeros(len(x),dtype=jnp.int32);path=jnp.zeros_like(logJ);roundoff=jnp.zeros_like(logJ)
    def body(_,carry):
        position,index,left,logJ,projected_time,count,path,roundoff,ok=carry
        fraction=(position-origin)/h-index;L=[];R=[]
        for c in range(3):
            lo=index;hi=index+jax.nn.one_hot(c,3,dtype=jnp.int32);fs=jnp.asarray(velocity[c].shape);lo=jnp.clip(lo,0,fs-1);hi=jnp.clip(hi,0,fs-1)
            L.append(velocity[c][lo[:,0],lo[:,1],lo[:,2]]);R.append(velocity[c][hi[:,0],hi[:,1],hi[:,2]])
        L=jnp.stack(L,axis=1);R=jnp.stack(R,axis=1);rate=(R-L)/h;u=L+(R-L)*fraction
        sign=jnp.sign(u).astype(jnp.int32);distance=h*jnp.where(sign>0,1-fraction,-fraction);face_velocity=jnp.where(sign>0,R,L);possible=(u!=0)&(u*face_velocity>0)
        denominator=jnp.where(u!=0,u,1.);ratio=rate*distance/denominator
        logarithm=jnp.where(jnp.abs(ratio)<.1,jnp.log1p(jnp.where(possible,ratio,0.)),jnp.log(jnp.where(possible,jnp.abs(face_velocity/denominator),1.)))
        time=jnp.where(rate==0,distance/denominator,logarithm/jnp.where(rate!=0,rate,1.));time=jnp.where(possible,jnp.maximum(time,0.),jnp.inf)
        axis=jnp.argmin(time,axis=1);first=jnp.min(time,axis=1);event=(left>0)&(first<=left);segment=jnp.where(event,first,left)
        phi=jnp.where(rate==0,segment[:,None],jnp.expm1(rate*segment[:,None])/jnp.where(rate!=0,rate,1.))
        delta=u*phi;position=position+delta;left=left-segment
        end_velocity=u*jnp.exp(rate*segment[:,None]);path=path+segment*jnp.linalg.norm(jnp.maximum(jnp.abs(u),jnp.abs(end_velocity)),axis=1)
        logJ=logJ+segment*jnp.sum(rate,axis=1);is_projected,_=cell_value(projected,index);projected_time=projected_time+jnp.where(is_projected,segment,0.)
        selector=jax.nn.one_hot(axis,3,dtype=jnp.int32);direction=jnp.sum(sign*selector,axis=1);next_index=index+selector*direction[:,None];next_wall,next_inside=cell_value(solid,next_index)
        ok=ok&jnp.all(~event|(next_inside&~next_wall))
        index=jnp.where(event[:,None],next_index,index);count=count+event.astype(jnp.int32)
        position,rounding,side_valid=represent(position,index,layout);roundoff=jnp.maximum(roundoff,rounding);ok=ok&side_valid
        return position,index,left,logJ,projected_time,count,path,roundoff,ok
    position,index,left,logJ,projected_time,count,path,roundoff,valid=jax.lax.fori_loop(0,max_crossings,body,(x,index,left,logJ,projected_time,crossings,path,roundoff,valid))
    wall,inside=cell_value(solid,index);valid=valid&jnp.all(~active|(inside&~wall))&jnp.all(left<=dt*1e-12)&jnp.all(jnp.where(active,path/h,0.)<=.25)&jnp.all(jnp.isfinite(position))
    return Transport(position,logJ,projected_time,count,path,roundoff,valid)
