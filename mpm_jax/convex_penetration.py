"""Penetration-only distance inside an arbitrary convex CCW polygon.
For an interior point, distance to the polygon boundary equals the smallest
normalized inward halfspace distance. Exterior points have zero penetration.
This helper is solely for penetration reductions, never activation or normals.
"""
import jax.numpy as jnp

def polygon_penetration(x, polygon):
    edges=jnp.roll(polygon,-1,axis=0)-polygon
    rel=x[:,None,:]-polygon[None,:,:]
    cross=edges[None,:,0]*rel[...,1]-edges[None,:,1]*rel[...,0]
    inside=jnp.all(cross>=0.,axis=1)
    length=jnp.sqrt(jnp.sum(edges*edges,axis=1))
    depth=jnp.min(cross/length[None,:],axis=1)
    return jnp.where(inside,depth,0.)
