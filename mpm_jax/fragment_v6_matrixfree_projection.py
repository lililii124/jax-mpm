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
def quadratic_reconstruction(x,mass,body,F,reference_root,dx,fixed_particles,ordered_node_keys=None,*,capacity=65536):
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
    if ordered_node_keys is None:
        selected=jnp.arange(capacity);ordered=unique;width=1<<20;bias=1<<19
        ordered_node_keys=jnp.stack((unique//(1<<60),(unique//(1<<40))%width-bias,(unique//width)%width-bias,unique%width-bias),axis=1)
        roster_valid=jnp.asarray(True);active=free
    else:
        ordered=_grid_keys(ordered_node_keys[:,0],ordered_node_keys[:,1:]);selected=jnp.searchsorted(unique,ordered)
        selected_keys=jnp.take(unique,selected,mode='fill',fill_value=-1);selected_free=jnp.take(free,selected,mode='fill',fill_value=False)
        roster_valid=jnp.all(selected_keys==ordered)&jnp.all(selected_free)&(jnp.sum(free)==len(ordered));active=jnp.ones(len(ordered),bool)
    roster=jnp.full(capacity,-1,jnp.int32).at[selected].set(jnp.arange(len(ordered),dtype=jnp.int32),mode='drop')
    slots=jnp.take(roster,ids,mode='fill',fill_value=-1);mask=(slots>=0)&jnp.take(free,ids,mode='fill',fill_value=False);safe=jnp.maximum(slots,0)
    r=nodes*dx-x[:,None,:]
    gradient=4/dx**2*jnp.einsum('pki,pij->pkj',r,F@reference_root)
    basis=jnp.concatenate((jnp.ones(weights.shape+(1,),x.dtype),gradient),axis=-1)
    raw=jnp.sqrt(mass)[:,None]*weights
    norm2=jnp.zeros(capacity,x.dtype).at[ids].add(raw**2*jnp.sum(basis*basis,axis=-1),mode='drop')
    norm=jnp.sqrt(jnp.take(norm2,selected,mode='fill',fill_value=0));norm=jnp.where(active,norm,1.)
    scale=jnp.where(mask,raw/jnp.maximum(norm[safe],jnp.finfo(x.dtype).tiny),0.)
    encoded_safe=jnp.all((body>=0)&(body<2))&jnp.all(jnp.abs(nodes)<(1<<19))
    coverage=jnp.all(jnp.take(unique,ids,mode='fill',fill_value=-1)==keys)
    valid=encoded_safe&coverage&roster_valid&jnp.all(norm>0)
    moment=jnp.zeros((capacity,3),x.dtype).at[ids].add(-mass[:,None,None]*weights[...,None]*r,mode='drop')
    center=moment/jnp.where(gm>0,gm,1.)[:,None]
    fixed_keys=jnp.sort(jnp.where((gm>0)&fixed,unique,jnp.iinfo(jnp.int64).max))
    return dict(slots=safe,scale=scale,raw_scale=jnp.where(mask,raw,0.),basis=basis,column_norm=norm,valid=valid,node_keys=ordered_node_keys,node_active=active,active_rows=jnp.repeat(jnp.any(mask,axis=1),4),grid_mass=jnp.take(gm,selected,mode='fill',fill_value=0),center_offset=jnp.take(center,selected,axis=0,mode='fill',fill_value=0),fixed_grid_keys=fixed_keys,free_node_count=jnp.sum(free),mass_error=jnp.sum(gm)-jnp.sum(mass))


def reconstructed_apply(parameters,z):
    samples=z[parameters['slots']]
    values=jnp.einsum('pk,pkb,pkc->pbc',parameters['scale'],parameters['basis'],samples)
    return values.reshape(-1,z.shape[-1])


def reconstructed_transpose(parameters,y,nodes):
    values=jnp.einsum('pk,pkb,pbc->pkc',parameters['scale'],parameters['basis'],y.reshape(-1,4,y.shape[-1]))
    return jnp.zeros((nodes,y.shape[-1]),y.dtype).at[parameters['slots']].add(values)


def raw_reconstructed_apply(parameters,z):
    return jnp.einsum('pk,pkb,pkc->pbc',parameters['raw_scale'],parameters['basis'],z[parameters['slots']]).reshape(-1,z.shape[-1])


def raw_reconstructed_transpose(parameters,y):
    values=jnp.einsum('pk,pkb,pbc->pkc',parameters['raw_scale'],parameters['basis'],y.reshape(-1,4,y.shape[-1]))
    return jnp.zeros((len(parameters['column_norm']),y.shape[-1]),y.dtype).at[parameters['slots']].add(values)


def polynomial_modes(parameters,dx):
    """Orthonormalize the8 exact body-polynomial modes, preserving zero rows."""
    keys=parameters['node_keys'];coords=keys[:,1:]*dx;columns=[];valid=parameters['valid']
    for body in range(2):
        mask=(keys[:,0]==body)&parameters['node_active'];count=jnp.sum(mask);valid=valid&(count>0)
        center=jnp.sum(jnp.where(mask[:,None],coords,0.),axis=0)/jnp.maximum(count,1)
        columns.append(mask.astype(coords.dtype))
        columns.extend(jnp.where(mask,(coords[:,axis]-center[axis])/dx,0.) for axis in range(3))
    polynomials=jnp.stack(columns,axis=1);K=raw_reconstructed_apply(parameters,polynomials)
    active=parameters['active_rows'];count=jnp.sum(active)
    selected=jnp.nonzero(active,size=len(active),fill_value=0)[0];retained=jnp.arange(len(active))<count
    packed=jnp.where(retained[:,None],K[selected],0.)
    orthogonal,triangular=jnp.linalg.qr(packed,mode='reduced')
    T=jnp.zeros_like(K).at[selected].add(jnp.where(retained[:,None],orthogonal,0.))
    valid=valid&jnp.all(jnp.abs(jnp.diag(triangular))>0)
    error=jnp.max(jnp.abs(T.T@T-jnp.eye(8)))
    return T,polynomials,dict(valid=valid&jnp.all(jnp.isfinite(T))&(error<=1e-12),orthogonality_error=error)


@partial(jax.jit,static_argnames=('pair_capacity','particle_capacity'))
def physical_contact(parameters,T,polynomials,dx,fixed_node_keys=None,*,pair_capacity=512,particle_capacity=4096):
    """Original centroid/affine sampler and exact adjoint, prepared in JAX.

    Mobility uses contact-adjacent particles plus the8 polynomial modes. No P0
    projection or full dense J is formed. Capacities are storage guards; an
    overflow returns invalid and cannot drop an admitted contact or particle.
    """
    keys=parameters['node_keys'];encoded=_grid_keys(keys[:,0],keys[:,1:]);active=parameters['node_active'];gm=parameters['grid_mass'];n=len(gm)
    target=encoded+(1<<60);partner=jnp.searchsorted(encoded,target)
    shared=(keys[:,0]==0)&active&(jnp.take(encoded,partner,mode='fill',fill_value=-1)==target)&jnp.take(active,partner,mode='fill',fill_value=False)
    count=jnp.sum(shared);first=jnp.nonzero(shared,size=pair_capacity,fill_value=0)[0];pair_active=jnp.arange(pair_capacity)<count
    second=jnp.where(pair_active,jnp.take(partner,first),0)
    delta=parameters['center_offset'][second]-parameters['center_offset'][first];length=jnp.linalg.norm(delta,axis=1)
    normals=jnp.where((pair_active&(length>64*jnp.finfo(gm.dtype).eps*dx))[:,None],delta/jnp.where(length>0,length,1.)[:,None],0.)
    owner=jnp.full(n,-1,jnp.int32).at[jnp.where(pair_active,first,n)].set(jnp.arange(pair_capacity,dtype=jnp.int32),mode='drop')
    owner=owner.at[jnp.where(pair_active,second,n)].set(jnp.arange(pair_capacity,dtype=jnp.int32),mode='drop')
    sign=jnp.zeros(n,gm.dtype).at[jnp.where(pair_active,first,n)].set(1.,mode='drop').at[jnp.where(pair_active,second,n)].set(-1.,mode='drop')
    slot=parameters['slots'];row=owner[slot];adjacent=jnp.any((row>=0)&(parameters['raw_scale']!=0),axis=1)
    particle_count=jnp.sum(adjacent);selected=jnp.nonzero(adjacent,size=particle_capacity,fill_value=0)[0];selected_active=jnp.arange(particle_capacity)<particle_count
    pslot=slot[selected];prows=jnp.where(selected_active[:,None]&(row[selected]>=0),row[selected],pair_capacity)
    coefficient=parameters['raw_scale'][selected]*sign[pslot]/jnp.where(gm[pslot]>0,gm[pslot],1.)
    coefficient=coefficient[...,None]*parameters['basis'][selected]*selected_active[:,None,None]
    columns=4*jnp.arange(particle_capacity)[:,None]+jnp.arange(4)[None,:]
    H=jnp.zeros((pair_capacity,4*particle_capacity),gm.dtype).at[prows[...,None],columns[:,None,:]].add(coefficient,mode='drop')
    K=raw_reconstructed_apply(parameters,polynomials);mix=T.T@K;gridT=jnp.linalg.solve(mix.T,polynomials.T).T
    reconstructedT=raw_reconstructed_transpose(parameters,T)/jnp.where(gm>0,gm,1.)[:,None]
    correction=(gridT[first]-gridT[second])-(reconstructedT[first]-reconstructedT[second]);correction=jnp.where(pair_active[:,None],correction,0.)
    localT=T.reshape(-1,4,8)[selected].reshape(-1,8);G=H@localT
    gram=H@H.T+G@correction.T+correction@G.T+correction@(T.T@T)@correction.T
    mobility=gram*(normals@normals.T);mobility=.5*(mobility+mobility.T)
    fixed_hit=jnp.asarray(False)
    if fixed_node_keys is not None:
        for selected_nodes in (first,second):
            value=encoded[selected_nodes];hit=jnp.take(fixed_node_keys,jnp.searchsorted(fixed_node_keys,value),mode='fill',fill_value=-1)==value
            fixed_hit=fixed_hit|jnp.any(pair_active&hit)
    valid=parameters['valid']&(count<=pair_capacity)&(particle_count<=particle_capacity)&jnp.all(jnp.isfinite(mobility))&(~fixed_hit)&jnp.all(encoded[1:]>=encoded[:-1])
    return dict(first=first,second=second,normals=normals,correction=correction,modes=T,mobility=mobility,valid=valid,pair_count=count,adjacent_particle_count=particle_count,contact_reaches_original_fixed_node=fixed_hit,grid_mass=gm)


def physical_sample(reconstruction,contact,y):
    grid=raw_reconstructed_transpose(reconstruction,y)/jnp.where(contact['grid_mass']>0,contact['grid_mass'],1.)[:,None]
    relative=grid[contact['first']]-grid[contact['second']]+contact['correction']@(contact['modes'].T@y)
    return jnp.sum(relative*contact['normals'],axis=1)


def physical_adjoint(reconstruction,contact,multiplier):
    impulse=multiplier[:,None]*contact['normals'];n=len(contact['grid_mass'])
    grid=jnp.zeros((n,3),impulse.dtype).at[contact['first']].add(impulse).at[contact['second']].add(-impulse)
    grid=grid/jnp.where(contact['grid_mass']>0,contact['grid_mass'],1.)[:,None]
    return raw_reconstructed_apply(reconstruction,grid)+contact['modes']@(contact['correction'].T@impulse)


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


def cone_project(y,sample,adjoint,mobility,*,sweeps=512,tolerance=.005,physical_rows=None):
    """Original positive-diagonal congruence QP, in original m/s KKT units."""
    g=sample(y);diagonal=jnp.diag(mobility);positive=diagonal>0
    scale=jnp.where(positive,jax.lax.rsqrt(jnp.where(positive,diagonal,1.)),1.)
    M=scale[:,None]*mobility*scale[None,:];rhs=scale*g
    eig=jnp.linalg.eigvalsh(.5*(M+M.T));rows=mobility.shape[0] if physical_rows is None else physical_rows
    threshold=rows*jnp.finfo(y.dtype).eps*jnp.max(eig,initial=0.)
    psd=(jnp.min(eig,initial=0.)>=-threshold)&jnp.all(jnp.where(positive[:,None],True,mobility==0))
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
    valid=psd&(kkt<=tolerance)&jnp.all(jnp.where(positive,True,g<=tolerance))&jnp.all(diagonal>=0)
    return following,lam,dict(valid=valid,sweeps=count,mobility_rank=jnp.sum(eig>threshold),mobility_rank_threshold=threshold,minimum_balanced_eigenvalue=jnp.min(eig,initial=0.),complementarity_m_s=kkt,closing_m_s=jnp.max(final,initial=0.),identity_J=.5*jnp.sum(following**2-y**2+delta**2)+jnp.dot(lam,final))


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
