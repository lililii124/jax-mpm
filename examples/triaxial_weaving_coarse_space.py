"""One Galerkin yarn-geometry coarse preconditioner; full3D equations untouched.

Six cross-section rigid velocity modes on4h axial hats are only a coarse
solver space. Its matrix is assembled from the real3D quadrature tangent,
not from a rod constitutive model. All original nodal unknowns remain.
"""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh


def assemble(context,direction,radius,dx,dt,E,nu,*,coarse_capacity,free_capacity,section_count):
    started=time.perf_counter();m=np.asarray(context['mass']);q=np.asarray(context['node_reference']);fixed=np.asarray(context['clamped']);indices=np.asarray(context['indices']);B=np.asarray(context['gradients']);F0=np.asarray(context['F0']);volume=np.asarray(context['volume'])
    free=np.flatnonzero((m>0)&~fixed);assert len(free)<=free_capacity
    tangent=np.asarray(direction,dtype=float);tangent/=np.linalg.norm(tangent)
    center=np.average(np.asarray(context['reference']),axis=0,weights=np.asarray(context['particle_mass']))
    s=(q[free]-center)@tangent;origin=np.floor(s.min()/(4*dx))*(4*dx);fraction=(s-origin)/(4*dx);k0=np.floor(fraction).astype(np.int32);f=fraction-k0;knots=int(k0.max()+2);dimension=6*knots;assert dimension<=coarse_capacity
    points=np.asarray(context['reference']).reshape(section_count,-1,3);point_mass=np.asarray(context['particle_mass']).reshape(section_count,-1);centres=np.sum(points*point_mass[...,None],axis=1)/point_mass.sum(axis=1)[:,None]
    centre_s=(centres-center)@tangent;assert np.all(np.diff(centre_s)>0)
    centerline=np.column_stack([np.interp(s,centre_s,centres[:,axis]) for axis in range(3)])
    r=(q[free]-centerline)/radius;hats=np.stack((1-f,f),axis=1)
    # P[node,side,mode,component], translations and rotations of full3D nodes.
    modes=np.broadcast_to(np.eye(3),(len(free),3,3));rot=np.cross(np.eye(3)[None,:,:],r[:,None,:]);modes=np.concatenate((modes,rot),axis=1)
    P=hats[:,:,None,None]*modes[:,None,:,:]
    dofs=(6*(k0[:,None]+np.arange(2))[...,None]+np.arange(6)).reshape(len(free),12)
    P12=P.reshape(len(free),12,3);local_mass=m[free,None,None]*np.einsum('nai,nbi->nab',P12,P12)
    keys=dofs[:,:,None]*coarse_capacity+dofs[:,None,:]
    coarse_mass=np.bincount(keys.ravel(),weights=local_mass.ravel(),minlength=coarse_capacity**2).reshape(coarse_capacity,coarse_capacity)
    norm=np.diag(coarse_mass).copy();active=norm>max(float(norm.max())*1e-12,np.finfo(float).tiny);scale=np.where(active,1/np.sqrt(np.where(active,norm,1.)),0.)
    # Gather sparse local P derivatives: <=3 axial knots for a27node stencil.
    full_k=np.zeros(len(m)+1,np.int32);full_P=np.zeros((len(m)+1,2,6,3));full_free=np.zeros(len(m)+1,bool)
    full_k[free]=k0;full_P[free]=P;full_free[free]=True
    support_k=full_k[indices];live=full_free[indices];minimum=np.min(np.where(live,support_k,2**30),axis=1);minimum=np.where(np.any(live,axis=1),minimum,0)
    maximum=np.max(np.where(live,support_k,minimum[:,None]),axis=1);assert np.all(maximum-minimum<=1)
    gradient=np.zeros((len(F0),3,6,3,3));particle=np.arange(len(F0))
    for side in range(2):
        for node in range(27):
            slot=np.clip(support_k[:,node]+side-minimum,0,2)
            value=full_P[indices[:,node],side,:,:,None]*B[:,node,None,None,:]
            gradient[particle,slot]+=value
    gradient=gradient.reshape(len(F0),18,3,3);dF=np.einsum('paij,pjk->paik',gradient,F0)
    C=F0.swapaxes(-1,-2)@F0;values,Q=np.linalg.eigh(C);stretch=np.sqrt(values);R=F0@((Q*(1/stretch)[:,None,:])@Q.swapaxes(-1,-2));trace=(stretch-1).sum(1)
    U=R.swapaxes(-1,-2)@F0;inverse=np.linalg.inv(np.trace(U,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)-U)
    A=np.einsum('pji,pajk->paik',R,dF);S=A-A.swapaxes(-1,-2);axial=np.stack((S[:,:,2,1],S[:,:,0,2],S[:,:,1,0]),axis=-1);omega=np.einsum('pij,paj->pai',inverse,axial)
    a,b,c=omega[:,:,0],omega[:,:,1],omega[:,:,2];zero=np.zeros_like(a);skew=np.stack((zero,-c,b,c,zero,-a,-b,a,zero),axis=-1).reshape(len(F0),18,3,3)
    dR=np.einsum('pij,pajk->paik',R,skew);mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    dP=2*mu*dF+lam*np.trace(A,axis1=-2,axis2=-1)[:,:,None,None]*R[:,None]+(lam*trace-2*mu)[:,None,None,None]*dR
    stiffness=volume[:,None,None]*np.einsum('paij,pbij->pab',dF,dP)
    local_dofs=(6*(minimum[:,None]+np.arange(3))[...,None]+np.arange(6)).reshape(len(F0),18)
    # Some fully constrained particles have zero columns beyond the true end;
    # clip those zero entries only, never discard a nonzero contribution.
    assert not np.any((local_dofs>=coarse_capacity)&(np.max(abs(stiffness),axis=1)>1e-20))
    local_dofs=np.minimum(local_dofs,coarse_capacity-1);keys=local_dofs[:,:,None]*coarse_capacity+local_dofs[:,None,:]
    K=np.bincount(keys.ravel(),weights=stiffness.ravel(),minlength=coarse_capacity**2).reshape(coarse_capacity,coarse_capacity)
    coarse=(coarse_mass+.25*dt**2*K)*scale[:,None]*scale[None,:];coarse=.5*(coarse+coarse.T);coarse[~active,~active]=1.
    active_ids=np.flatnonzero(active);sub=coarse[np.ix_(active_ids,active_ids)];rank=len(active_ids);fallback=False;dropped=0;smallest=None
    try:
        factor=cho_factor(sub,lower=True,check_finite=False);inverse_sub=cho_solve(factor,np.eye(len(sub)),check_finite=False)
    except np.linalg.LinAlgError:
        eigen,Q=eigh(sub,check_finite=False);keep=eigen>1e-10*max(float(eigen.max()),1.);smallest=float(eigen.min());dropped=int((~keep).sum());inverse_sub=(Q[:,keep]/eigen[keep])@Q[:,keep].T;rank=int(keep.sum());fallback=True
    coarse_inverse=np.zeros_like(coarse);coarse_inverse[np.ix_(active_ids,active_ids)]=.5*(inverse_sub+inverse_sub.T)
    padded_ids=np.full(free_capacity,len(m),np.int32);padded_ids[:len(free)]=free
    padded_k=np.zeros(free_capacity,np.int32);padded_k[:len(free)]=k0
    padded_h=np.zeros((free_capacity,2));padded_h[:len(free)]=hats
    padded_r=np.zeros((free_capacity,3));padded_r[:len(free)]=r
    data=dict(node_ids=padded_ids,knot0=padded_k,hats=padded_h,radial=padded_r,scale=scale,coarse_inverse=coarse_inverse)
    report=dict(assembly_seconds=time.perf_counter()-started,coarse_spacing_over_h=4,coarse_dimension=dimension,coarse_capacity=coarse_capacity,positive_free_nodes=len(free),free_capacity=free_capacity,coarse_active_columns=len(active_ids),coarse_rank=rank,rank_fallback=fallback,coarse_dropped_modes=dropped,coarse_smallest_eigenvalue=smallest,coarse_matrix_bytes=coarse_inverse.nbytes,original_3D_dofs_unchanged=True,coarse_operator='GalerkinP^T(M+dt^2/4*K_exact(F0))*P from full3D particle quadrature')
    return data,report


def correction(residual,data):
    import jax.numpy as jnp
    ids=data['node_ids'];k=data['knot0'];hats=data['hats'];radial=data['radial'];scale=data['scale'];inverse=data['coarse_inverse']
    r=jnp.take(residual,ids,axis=0,mode='fill',fill_value=0.);modes=jnp.concatenate((r,jnp.cross(radial,r)),axis=1);dofs0=6*k[:,None]+jnp.arange(6,dtype=jnp.int32)
    coarse=jnp.zeros(len(scale)).at[dofs0].add(hats[:,0,None]*modes,mode='drop').at[dofs0+6].add(hats[:,1,None]*modes,mode='drop')
    solution=scale*(inverse@(scale*coarse));coeff0=solution[dofs0];coeff1=solution[dofs0+6];coeff=hats[:,0,None]*coeff0+hats[:,1,None]*coeff1
    value=coeff[:,:3]+jnp.cross(coeff[:,3:],radial)
    return jnp.zeros_like(residual).at[ids].set(value,mode='drop')
