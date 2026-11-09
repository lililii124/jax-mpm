"""Fixed-capacity convex cut geometry and compatible bulk/boundary quadrature."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Polygon(NamedTuple):
    vertices: jax.Array
    count: jax.Array
    valid: jax.Array


def polygon(vertices,capacity=12):
    """Pad a counterclockwise convex polygon; its topology remains explicit."""
    vertices=jnp.asarray(vertices)
    if not jnp.issubdtype(vertices.dtype,jnp.floating):
        vertices=vertices.astype(jnp.asarray(0.).dtype)
    if vertices.ndim!=2 or vertices.shape[1]!=2 or not 3<=len(vertices)<=capacity:
        raise ValueError('Require 3..capacity planar vertices')
    edge=jnp.roll(vertices,-1,axis=0)-vertices
    delta=vertices[None,:,:]-vertices[:,None,:]
    cross=edge[:,None,0]*delta[:,:,1]-edge[:,None,1]*delta[:,:,0]
    scale=jnp.max(jnp.abs(edge))**2
    area=.5*jnp.sum(vertices[:,0]*jnp.roll(vertices[:,1],-1)-vertices[:,1]*jnp.roll(vertices[:,0],-1))
    valid=jnp.all(jnp.isfinite(vertices))&jnp.all(cross>=-64*jnp.finfo(vertices.dtype).eps*scale)&(area>0)
    return Polygon(jnp.pad(vertices,((0,capacity-len(vertices)),(0,0))),jnp.asarray(len(vertices)),valid)


def clip(poly,normal,offset):
    """Intersect with normal.x >= offset; overflow invalidates the result.

    Coordinates are differentiable while the active topology is fixed. The
    discrete vertex count and inside/outside decisions are not differentiated.
    """
    normal=jnp.asarray(normal,dtype=poly.vertices.dtype)
    offset=jnp.asarray(offset,dtype=poly.vertices.dtype)
    capacity=len(poly.vertices);index=jnp.arange(capacity)
    a=poly.vertices;b=poly.vertices[(index+1)%jnp.maximum(poly.count,1)]
    da=a@normal-offset;db=b@normal-offset
    active=index<poly.count;inside_a=da>=0;inside_b=db>=0
    crossing=active&(inside_a!=inside_b)&(da!=0)&(db!=0)
    ratio=da/jnp.where(crossing,da-db,1.)
    intersection=a+ratio[:,None]*(b-a)
    values=jnp.stack((a,intersection),axis=1).reshape(-1,2)
    keep=jnp.stack((active&inside_a,crossing),axis=1).reshape(-1)
    destinations=jnp.cumsum(keep.astype(jnp.int32))-1
    packed=jnp.zeros((2*capacity+1,2),dtype=values.dtype).at[jnp.where(keep,destinations,2*capacity)].add(jnp.where(keep[:,None],values,0.))
    count=jnp.sum(keep)
    valid=poly.valid&(count<=capacity)&jnp.all(jnp.isfinite(normal))&jnp.isfinite(offset)&(jnp.sum(normal**2)>0)
    return Polygon(packed[:capacity],jnp.minimum(count,capacity),valid)


def split(poly,normal,offset):
    return clip(poly,normal,offset),clip(poly,-jnp.asarray(normal),-offset)


def _gauss(dtype):
    root=jnp.sqrt(jnp.asarray(3./5.,dtype=dtype))
    return jnp.stack(((1-root)/2,jnp.asarray(.5,dtype=dtype),(1+root)/2)),jnp.asarray([5/18,4/9,5/18],dtype=dtype)


def bulk_quadrature(poly):
    """Nine-point Duffy rule per fan triangle, with zero-weight padding.

    Split at spline knots before integrating basis gradients: this rule is
    exact for cubic polynomials, not for a piece crossing a spline knot.
    """
    gauss,weight=_gauss(poly.vertices.dtype)
    u,v=jnp.meshgrid(gauss,gauss,indexing='ij');wu,wv=jnp.meshgrid(weight,weight,indexing='ij')
    u,v=u.ravel(),v.ravel();w=(wu*wv).ravel()*(1-u)
    a=poly.vertices[0];b=poly.vertices[1:-1];c=poly.vertices[2:]
    ab,ac=b-a,c-a
    determinant=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    active=jnp.arange(len(b))<jnp.maximum(poly.count-2,0)
    points=a+u[None,:,None]*ab[:,None,:]+((1-u)*v)[None,:,None]*ac[:,None,:]
    weights=jnp.where(active,determinant,0.)[:,None]*w
    return points.reshape(-1,2),weights.reshape(-1)


def boundary_quadrature(poly):
    """Three-point edge rule with outward normals for a CCW polygon."""
    gauss,weight=_gauss(poly.vertices.dtype)
    index=jnp.arange(len(poly.vertices));a=poly.vertices
    b=poly.vertices[(index+1)%jnp.maximum(poly.count,1)]
    edge=b-a;squared=jnp.sum(edge**2,axis=1)
    length=jnp.where(squared>0,jnp.sqrt(jnp.where(squared>0,squared,1.)),0.)
    active=(index<poly.count)&(poly.count>=3)
    normal=jnp.stack((edge[:,1],-edge[:,0]),axis=1)/jnp.where(length>0,length,1.)[:,None]
    points=a[:,None,:]+gauss[None,:,None]*edge[:,None,:]
    weights=jnp.where(active,length,0.)[:,None]*weight
    return points.reshape(-1,2),weights.reshape(-1),jnp.repeat(normal,3,axis=0)
