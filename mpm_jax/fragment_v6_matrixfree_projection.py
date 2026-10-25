"""Bounded JAX kernels for the unchanged consistent-mass/contact convex QP.

No Gram matrix or restricted-contact Schur matrix is formed here. Sparse Q and
physical J are applied and transposed. Dykstra's two dual corrections are part
of the nearest-point problem, not energy reservoirs or state corrections.
Geometry/preconditioner preparation and impact integration are not installed
by this module; a successful frozen-operator check is not production approval.
"""
from functools import partial
from itertools import product
import jax
import jax.numpy as jnp


def _grid_keys(body,nodes):
    body=jnp.asarray(body,jnp.int64);nodes=jnp.asarray(nodes,jnp.int64)
    width=1<<20;bias=1<<19
    return body*(1<<60)+(nodes[...,0]+bias)*(1<<40)+(nodes[...,1]+bias)*width+(nodes[...,2]+bias)


@partial(jax.jit,static_argnames=('capacity',))
def quadratic_reconstruction(x,mass,body,F,reference_root,dx,fixed_particles,ordered_node_keys,*,capacity=65536):
    """Build the original27-point Hermite reconstruction entirely in JAX.

    The supplied node roster belongs to the fixed-step preconditioner. Exact
    coverage is checked; storage overflow or a different roster cannot silently
    drop physical nodes. No Gram, COO reconstruction, or inverse is assembled.
    """
    offsets=jnp.asarray(tuple(product(range(3),repeat=3)),dtype=jnp.int64)
    base=jnp.floor(x/dx-.5).astype(jnp.int64);nodes=base[:,None,:]+offsets
    fraction=x/dx-base
    one=jnp.stack((.5*(1.5-fraction)**2,.75-(fraction-1)**2,.5*(fraction-.5)**2),axis=1)
    weights=jnp.prod(jnp.stack(tuple(one[:,offsets[:,a],a] for a in range(3)),axis=-1),axis=-1)
    keys=_grid_keys(body[:,None],nodes)
    unique,inverse=jnp.unique(keys.ravel(),size=capacity,fill_value=jnp.iinfo(jnp.int64).max,return_inverse=True)
    ids=inverse.reshape(weights.shape)
    gm=jnp.zeros(capacity,x.dtype).at[ids].add(mass[:,None]*weights,mode='drop')
    fixed=jnp.zeros(capacity,jnp.int32).at[ids].max(jnp.broadcast_to(fixed_particles[:,None],ids.shape).astype(jnp.int32),mode='drop')>0
    free=(gm>0)&~fixed
    ordered=_grid_keys(ordered_node_keys[:,0],ordered_node_keys[:,1:])
    selected=jnp.searchsorted(unique,ordered)
    selected_keys=jnp.take(unique,selected,mode='fill',fill_value=-1)
    selected_free=jnp.take(free,selected,mode='fill',fill_value=False)
    roster=jnp.full(capacity,-1,jnp.int32).at[selected].set(jnp.arange(len(ordered),dtype=jnp.int32),mode='drop')
    slots=jnp.take(roster,ids,mode='fill',fill_value=-1);mask=slots>=0;safe=jnp.maximum(slots,0)
    r=nodes*dx-x[:,None,:]
    gradient=4/dx**2*jnp.einsum('pki,pij->pkj',r,F@reference_root)
    basis=jnp.concatenate((jnp.ones(weights.shape+(1,),x.dtype),gradient),axis=-1)
    raw=jnp.sqrt(mass)[:,None]*weights
    norm2=jnp.zeros(capacity,x.dtype).at[ids].add(raw**2*jnp.sum(basis*basis,axis=-1),mode='drop')
    norm=jnp.sqrt(jnp.take(norm2,selected,mode='fill',fill_value=0))
    scale=jnp.where(mask,raw/jnp.maximum(norm[safe],jnp.finfo(x.dtype).tiny),0.)
    encoded_safe=jnp.all((body>=0)&(body<2))&jnp.all(jnp.abs(nodes)<(1<<19))
    coverage=jnp.all(jnp.take(unique,ids,mode='fill',fill_value=-1)==keys)
    valid=encoded_safe&coverage&jnp.all(selected_keys==ordered)&jnp.all(selected_free)&(jnp.sum(free)==len(ordered))&jnp.all(norm>0)
    return dict(slots=safe,scale=scale,basis=basis,column_norm=norm,valid=valid,free_node_count=jnp.sum(free),mass_error=jnp.sum(gm)-jnp.sum(mass))


def reconstructed_apply(parameters,z):
    samples=z[parameters['slots']]
    values=jnp.einsum('pk,pkb,pkc->pbc',parameters['scale'],parameters['basis'],samples)
    return values.reshape(-1,z.shape[-1])


def reconstructed_transpose(parameters,y,nodes):
    values=jnp.einsum('pk,pkb,pbc->pkc',parameters['scale'],parameters['basis'],y.reshape(-1,4,y.shape[-1]))
    return jnp.zeros((nodes,y.shape[-1]),y.dtype).at[parameters['slots']].add(values)


def _bounded_loop(body,state,limit,stopped):
    def condition(pair):return (pair[0]<limit)&(~jnp.all(stopped(pair[1])))
    def advance(pair):return pair[0]+1,body(pair[0],pair[1])
    return jax.lax.while_loop(condition,advance,(jnp.asarray(0),state))[1]


def sparse_apply(data,row,column,x,rows):
    """COO matvec/matmat, with fixed-capacity indices supplied by the caller."""
    return jnp.zeros((rows,)+x.shape[1:],x.dtype).at[row].add(data.reshape((-1,)+(1,)*(x.ndim-1))*x[column])


def consistent_project(y,apply,transpose,modes,right,right_transpose,*,iterations=256,relative_tolerance=1e-13):
    """Three simultaneous LSQR solves, preserving the exact polynomial range.

    right must be an invertible linear map; right_transpose must be its actual
    transpose. A shift used to construct that map never enters the objective.
    Final physical normal residual and moment/work gates remain caller duties.
    """
    orth=lambda x:x-modes@(modes.T@x)
    A=lambda z:orth(apply(right(z)))
    AT=lambda z:right_transpose(transpose(orth(z)))
    known=modes@(modes.T@y);b=orth(y)
    norm=lambda v:jnp.sqrt(jnp.sum(v*v,axis=0))
    safe=lambda a:jnp.where(a>0,a,1.)
    beta=norm(b);u=b/safe(beta);v=AT(u);alpha=norm(v);v=v/safe(alpha)
    x=jnp.zeros_like(v);w=v;rhobar=alpha;phibar=beta
    initial_normal=norm(AT(b));done=initial_normal==0
    # A bounded loop is compiled once; each converged RHS is frozen separately.
    def body(i,state):
        x,u,v,w,alpha,beta,rhobar,phibar,done,count=state
        unew=A(v)-u*alpha;bn=norm(unew);unew=unew/safe(bn)
        vnew=AT(unew)-v*bn;an=norm(vnew);vnew=vnew/safe(an)
        rho=jnp.hypot(rhobar,bn);c=rhobar/safe(rho);s=bn/safe(rho)
        theta=s*an;rnew=-c*an;phi=c*phibar;pnew=s*phibar
        xn=x+w*(phi/safe(rho));wn=vnew-w*(theta/safe(rho))
        residual=b-A(xn);normal=norm(AT(residual))
        newly=normal<=relative_tolerance*jnp.maximum(initial_normal,jnp.finfo(y.dtype).tiny)
        choose=lambda new,old:jnp.where(done,old,new)
        return (choose(xn,x),choose(unew,u),choose(vnew,v),choose(wn,w),choose(an,alpha),choose(bn,beta),choose(rnew,rhobar),choose(pnew,phibar),done|newly,count+(~done))
    state=_bounded_loop(body,(x,u,v,w,alpha,beta,rhobar,phibar,done,jnp.zeros(y.shape[1],jnp.int32)),iterations,lambda s:s[-2])
    x,_,_,_,_,_,_,_,done,count=state
    projected=known+A(x)
    normal=norm(transpose(y-projected));initial=norm(transpose(y))
    return projected,dict(converged=done,iterations=count,physical_normal_residual=normal,physical_normal_relative_residual=normal/jnp.maximum(initial,jnp.finfo(y.dtype).tiny),kinetic_loss=.5*jnp.sum(y*y-projected*projected),orthogonality=jnp.sum(projected*(y-projected)))


def cone_project(y,sample,adjoint,mobility,*,sweeps=512,tolerance=.005):
    """Original positive-diagonal congruence QP, in original m/s KKT units."""
    g=sample(y);diagonal=jnp.diag(mobility);positive=diagonal>0
    scale=jnp.where(positive,jax.lax.rsqrt(jnp.where(positive,diagonal,1.)),1.)
    M=scale[:,None]*mobility*scale[None,:];rhs=scale*g
    feasible=jnp.max(g,initial=0.)<=tolerance
    def sweep(_,state):
        mu,gradient,done,count=state
        def row(i,pair):
            value,grad=pair
            updated=jnp.maximum(0.,value[i]-grad[i]/jnp.where(positive[i],M[i,i],1.))
            updated=jnp.where(positive[i],updated,0.);change=jnp.where(done,0.,updated-value[i])
            return value.at[i].add(change),grad+M[:,i]*change
        next_mu,next_gradient=jax.lax.fori_loop(0,g.shape[0],row,(mu,gradient))
        lam=scale*next_mu;native=-next_gradient/scale
        kkt=jnp.max(jnp.where(lam>0,jnp.abs(native),jnp.maximum(native,0.)),initial=0.)
        return next_mu,next_gradient,done|(kkt<=tolerance),count+(~done)
    mu,_,done,count=_bounded_loop(sweep,(jnp.zeros_like(g),-rhs,feasible,jnp.asarray(0)),sweeps,lambda s:s[-2])
    lam=scale*mu;delta=-adjoint(lam);following=y+delta;final=sample(following)
    kkt=jnp.max(jnp.where(lam>0,jnp.abs(final),jnp.maximum(final,0.)),initial=0.)
    valid=(kkt<=tolerance)&jnp.all(jnp.where(positive,True,g<=tolerance))&jnp.all(diagonal>=0)
    return following,lam,dict(valid=valid,sweeps=count,complementarity_m_s=kkt,closing_m_s=jnp.max(final,initial=0.),identity_J=.5*jnp.sum(following**2-y**2+delta**2)+jnp.dot(lam,final))


def intersect_project(trial,project_range,project_cone,sample,*,iterations=128,relative_tolerance=1e-10,closing_tolerance=.005):
    """Nearest point in range(Q0) intersect physical closing cone, not MAP.

    Returned physical/reaction impulses are Dykstra dual variables. They must
    independently satisfy original moment/work/KKT gates before integration.
    """
    def body(i,state):
        z,p,q,lam,done,count=state
        a,range_report=project_range(z+p);pn=z+p-a
        zn,ln,cone_report=project_cone(a+q);qn=a+q-zn
        relative=jnp.linalg.norm(zn-z)/jnp.maximum(jnp.linalg.norm(trial),jnp.finfo(trial.dtype).tiny)
        range_gap=jnp.linalg.norm(zn-a)/jnp.maximum(jnp.linalg.norm(trial),jnp.finfo(trial.dtype).tiny)
        converged=(relative<=relative_tolerance)&(range_gap<=relative_tolerance)&cone_report['valid']&jnp.all(range_report['physical_normal_relative_residual']<=relative_tolerance)
        choose=lambda n,o:jnp.where(done,o,n)
        return choose(zn,z),choose(pn,p),choose(qn,q),choose(ln,lam),done|converged,count+(~done)
    zero=jnp.zeros_like(trial);m=sample(trial).shape[0]
    z,p,q,lam,done,count=_bounded_loop(body,(trial,zero,zero,jnp.zeros(m,trial.dtype),jnp.asarray(False),jnp.asarray(0)),iterations,lambda s:s[-2])
    admissible,range_report=project_range(z);gap=jnp.linalg.norm(admissible-z)/jnp.maximum(jnp.linalg.norm(trial),jnp.finfo(trial.dtype).tiny)
    closing=sample(z);kkt=jnp.max(jnp.where(lam>0,jnp.abs(closing),jnp.maximum(closing,0.)),initial=0.)
    identity=z+p+q-trial
    return z,dict(converged=done&(gap<=relative_tolerance)&(kkt<=closing_tolerance),iterations=count,range_relative_gap=gap,complementarity_m_s=kkt,physical_impulse=-q,boundary_reaction=-p,multiplier=lam,dual_decomposition_relative_error=jnp.linalg.norm(identity)/jnp.maximum(jnp.linalg.norm(trial),jnp.finfo(trial.dtype).tiny),boundary_virtual_work_J=jnp.sum(z*(-p)),physical_contact_power_J=jnp.sum(z*(-q)),projection_loss_J=.5*jnp.sum(trial**2-z**2))
