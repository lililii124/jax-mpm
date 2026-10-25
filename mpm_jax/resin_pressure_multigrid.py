"""SPD Galerkin V-cycle preconditioner for the unchanged physical MAC CG.

Coarse edges/Dirichlet anchors are exact sums for piecewise-constant P;
R=P.T. Symmetric damped-Jacobi pre/post smoothing gives a linear SPD map.
No physical face mass, fine operator, tolerance or iteration limit is changed.
"""
import jax
import jax.numpy as jnp
from .resin_mac import divergence,transpose_divergence


def block_sum(x):
    padded=jnp.pad(x,tuple((0,n%2) for n in x.shape));a,b,c=(n//2 for n in padded.shape)
    return padded.reshape(a,2,b,2,c,2).sum(axis=(1,3,5))


def graph(mask,edges,anchor):
    a,b,c=edges;diagonal=anchor+a[1:]+a[:-1]+b[:,1:]+b[:,:-1]+c[:,:,1:]+c[:,:,:-1]
    return mask,edges,anchor,jnp.where(mask&(diagonal>0),diagonal,1.)


def apply_graph(level,x):
    mask,edges,anchor,diagonal=level;grad=transpose_divergence(jnp.where(mask,x,0.),1.)
    return anchor*x+divergence(tuple(w*g for w,g in zip(edges,grad)),1.)


def coarsen(level):
    mask,edges,anchor,diagonal=level;out=[]
    for axis,edge in enumerate(edges):
        e=jnp.moveaxis(edge,axis,0);e=jnp.pad(e,((0,mask.shape[axis]%2),(0,e.shape[1]%2),(0,e.shape[2]%2)))[::2]
        n,a,b=e.shape;e=e.reshape(n,a//2,2,b//2,2).sum(axis=(2,4));out.append(jnp.moveaxis(e,0,axis))
    return graph(block_sum(mask.astype(jnp.int32))>0,tuple(out),block_sum(anchor))


def make_preconditioner(inverse,fluid,h,*,max_levels=6):
    edges=[];anchor=jnp.zeros(fluid.shape,dtype=inverse[0].dtype)
    for axis,inv in enumerate(inverse):
        left=jnp.pad(fluid,tuple((1,0) if k==axis else (0,0) for k in range(3)));right=jnp.pad(fluid,tuple((0,1) if k==axis else (0,0) for k in range(3)));w=h**4*inv
        edges.append(jnp.where(left&right,w,0.));onleft=jnp.where(left&~right,w,0.);onright=jnp.where(right&~left,w,0.);s1=[slice(None)]*3;s0=[slice(None)]*3;s1[axis]=slice(1,None);s0[axis]=slice(None,-1);anchor=anchor+onleft[tuple(s1)]+onright[tuple(s0)]
    levels=[graph(fluid,tuple(edges),anchor)]
    while len(levels)<max_levels and max(levels[-1][0].shape)>4:levels.append(coarsen(levels[-1]))
    omega=2./3.
    def cycle(i,rhs):
        level=levels[i];mask,edges,anchor,diagonal=level
        def smooth(x):return x+omega*jnp.where(mask,(rhs-apply_graph(level,x))/diagonal,0.)
        if i==len(levels)-1:return jax.lax.fori_loop(0,80,lambda _,x:smooth(x),jnp.zeros_like(rhs))
        x=jax.lax.fori_loop(0,4,lambda _,x:smooth(x),jnp.zeros_like(rhs));residual=jnp.where(mask,rhs-apply_graph(level,x),0.);correction=cycle(i+1,block_sum(residual));correction=jnp.repeat(jnp.repeat(jnp.repeat(correction,2,axis=0),2,axis=1),2,axis=2);correction=correction[tuple(slice(0,n) for n in rhs.shape)];x=x+jnp.where(mask,correction,0.);return jax.lax.fori_loop(0,4,lambda _,y:smooth(y),x)
    return lambda rhs:jnp.where(fluid,cycle(0,jnp.where(fluid,rhs,0.)),rhs)
