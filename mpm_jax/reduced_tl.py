"""Fixed Ritz projection of the compact MPM small-strain tangent and inertia."""
from itertools import product
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_solve
import numpy as np

from .compact_tl import surface_weights,point_material


def clamped_basis(nodes, settings, order=3):
    """Bending, core zigzag shear, thickness compression and membrane modes."""
    if not 2 <= order <= 9:
        raise ValueError('Ritz order must lie between two and nine')
    x = np.asarray(nodes)
    span = np.asarray((settings.length, settings.width))-2*settings.edge
    q = (x[:, :2]-settings.edge)/span
    free = np.all((q > 0) & (q < 1), axis=1)
    q = np.clip(q, 0, 1)
    envelope = np.sin(np.pi*q)**2
    membrane_envelope = np.sin(np.pi*q)
    derivative = np.pi*np.sin(2*np.pi*q)/span
    values, derivatives, membrane_values = [], [], []
    for axis in range(2):
        va, da, ma = [], [], []
        for n in range(order):
            polynomial = np.polynomial.legendre.Legendre.basis(n)
            p = polynomial(2*q[:, axis]-1)
            dp = polynomial.deriv()(2*q[:, axis]-1)*2/span[axis]
            va.append(envelope[:, axis]*p)
            da.append(derivative[:, axis]*p+envelope[:, axis]*dp)
            ma.append(membrane_envelope[:, axis]*p)
        values.append(va);derivatives.append(da);membrane_values.append(ma)
    depth = 2*settings.skin+settings.core
    z = x[:, 2]-depth/2
    compression = np.clip((x[:, 2]-settings.skin)/settings.core, 0, 1)-.5
    modes, labels = [], []
    for a, b in product(range(order), repeat=2):
        shape = values[0][a]*values[1][b]
        dx = derivatives[0][a]*values[1][b]
        dy = values[0][a]*derivatives[1][b]
        membrane_shape = membrane_values[0][a]*membrane_values[1][b]
        bending = np.column_stack((-z*dx, -z*dy, shape))
        shear_x = np.column_stack((compression*membrane_shape,np.zeros(len(x)),np.zeros(len(x))))
        shear_y = np.column_stack((np.zeros(len(x)),compression*membrane_shape,np.zeros(len(x))))
        thickness = np.column_stack((np.zeros(len(x)),np.zeros(len(x)),compression*membrane_shape))
        membrane_x = np.column_stack((membrane_shape,np.zeros(len(x)),np.zeros(len(x))))
        membrane_y = np.column_stack((np.zeros(len(x)),membrane_shape,np.zeros(len(x))))
        for family, mode in zip(('bending','core-shear-x','core-shear-y','core-compression','membrane-x','membrane-y'),
                                (bending,shear_x,shear_y,thickness,membrane_x,membrane_y)):
            mode[~free] = 0.
            modes.append(mode);labels.append(f'{family}-{a+1}-{b+1}')
    return np.stack(modes, axis=-1), labels, ~free


def normalize_basis(basis, nodal_mass):
    flat = basis.reshape(-1, basis.shape[-1])
    weighted = flat*np.repeat(nodal_mass, 3)[:, None]
    gram = flat.T@weighted
    factor = np.linalg.cholesky(.5*(gram+gram.T))
    transform = np.linalg.solve(factor.T, np.eye(len(factor)))
    normalized = (flat@transform).reshape(basis.shape)
    return normalized, transform, float(np.linalg.cond(gram))


def strains_at_points(basis, indices, gradients):
    displacement_gradient = jnp.einsum('pkir,pkj->pijr', basis[indices], gradients)
    return .5*(displacement_gradient+jnp.swapaxes(displacement_gradient, 1, 2))


def projection(base, order=3, assembly_chunk=256, extra_basis=None):
    """Accumulate r-by-r operators in quadrature chunks, never P-by-r-by-r."""
    if assembly_chunk < 1:raise ValueError('Positive assembly chunk required')
    support = base['support']
    _, density0, _, _ = base['properties'](jnp.asarray(base['q0']))
    reference0 = base['reference_grid'](support, density0)
    raw, labels, clamped = clamped_basis(support.nodes, base['settings'], order)
    nodal_mass=np.asarray(reference0.nodal_mass)
    V, transform, condition = normalize_basis(raw,nodal_mass)
    enrichment=dict(snapshot_count=0,retained_modes=0)
    original_back=None
    if extra_basis is not None:
        original_transform=transform
        original_condition=condition
        face_ids,face_weights=surface_weights(support,base['geometry']['bottom'])
        original_back=np.einsum('pk,pkr->pr',face_weights,V[face_ids,2,:])
        snapshots=np.asarray(extra_basis,dtype=float)
        if snapshots.ndim!=3 or snapshots.shape[:2]!=raw.shape[:2] or not np.isfinite(snapshots).all():
            raise ValueError('Snapshot enrichment must have shape (nodes,3,snapshots) and finite values')
        if snapshots.shape[-1]<1 or np.max(np.abs(snapshots[clamped]))>1e-14:
            raise ValueError('Snapshot enrichment must be nonempty and satisfy the fixed clamp')
        flat=V.reshape(-1,V.shape[-1]);mass=np.repeat(nodal_mass,3)
        residual=snapshots.reshape(len(flat),-1).copy()
        # Remove the analytic Ritz space before POD, so small displacement
        # residuals containing missing thickness strains remain represented.
        residual-=flat@(flat.T@(mass[:,None]*residual))
        weighted=residual*np.sqrt(mass[:,None])
        vectors,singular,_=np.linalg.svd(weighted,full_matrices=False)
        keep=singular>singular[0]*1e-8
        if singular[0]<=0 or not keep.any():raise ValueError('Snapshots add no independent modes')
        extras=(vectors[:,keep]/np.sqrt(mass[:,None])).reshape(raw.shape[:2]+(-1,))
        extras[clamped]=0.
        labels+=['snapshot-residual-pod-'+str(i+1) for i in range(extras.shape[-1])]
        # Recondition the already mass-normalized analytic vectors and POD
        # vectors together. Reusing the raw analytic Gram matrix next to unit
        # POD vectors can lose orthogonality through scale disparity.
        combined=np.concatenate((V,extras),axis=-1)
        V,correction,condition=normalize_basis(combined,nodal_mass)
        raw_map=np.eye(combined.shape[-1]);raw_map[:len(original_transform),:len(original_transform)]=original_transform
        transform=raw_map@correction
        enrichment=dict(snapshot_count=snapshots.shape[-1],retained_modes=int(keep.sum()),
            original_analytic_gram_condition=original_condition,
            residual_mass_weighted_singular_values=singular.tolist(),relative_singular_cutoff=1e-8,
            snapshots_sha256=__import__('hashlib').sha256(snapshots.tobytes()).hexdigest())
    if np.max(np.abs(V[clamped])) != 0:raise ValueError('Reduced basis violates the clamp')
    basis = jnp.asarray(V)
    count, rank = len(support.positions), V.shape[-1]
    padded = math.ceil(count/assembly_chunk)*assembly_chunk
    conn = jnp.asarray(np.pad(support.indices, ((0,padded-count),(0,0))))
    grad = jnp.asarray(np.pad(support.gradients, ((0,padded-count),(0,0),(0,0))))
    volume = jnp.asarray(np.pad(support.volume, (0,padded-count)))

    def assemble(density,material):
        reference = base['reference_grid'](support, density)
        flat = basis.reshape(-1, rank)
        mass = flat.T@(jnp.repeat(reference.nodal_mass,3)[:,None]*flat)
        lam = jnp.pad(material.lam[:,0,0],(0,padded-count))
        mu = jnp.pad(material.mu[:,0,0],(0,padded-count))
        def chunk(K, index):
            start = index*assembly_chunk
            ids = lax.dynamic_slice_in_dim(conn,start,assembly_chunk)
            derivative = lax.dynamic_slice_in_dim(grad,start,assembly_chunk)
            vol = lax.dynamic_slice_in_dim(volume,start,assembly_chunk)
            shear = lax.dynamic_slice_in_dim(mu,start,assembly_chunk)
            first = lax.dynamic_slice_in_dim(lam,start,assembly_chunk)
            strain = strains_at_points(basis,ids,derivative)
            # The weights enter linearly; use two matmuls rather than local Hessians.
            B = strain.reshape(assembly_chunk*9,rank)
            weighted = (strain*(2*vol*shear)[:,None,None,None]).reshape(-1,rank)
            trace = jnp.trace(strain,axis1=1,axis2=2)
            following = K+B.T@weighted+trace.T@((vol*first)[:,None]*trace)
            return following,None
        stiffness,_ = lax.scan(jax.checkpoint(chunk,prevent_cse=False),
            jnp.zeros((rank,rank)),jnp.arange(padded//assembly_chunk))
        return .5*(mass+mass.T),.5*(stiffness+stiffness.T),jnp.dot(jnp.asarray(support.volume),density)

    def operators(q):
        _,density,_,material=base['properties'](q)
        return assemble(density,material)

    def operators_for_fraction(phi):
        skin=jnp.asarray(base['geometry']['skin'])
        young=jnp.where(skin,70e9,.3*70e9*phi**2)
        poisson=jnp.where(skin,.33,.25)
        return assemble(2700*phi,point_material(young,poisson))

    top_ids,top_w = surface_weights(support,base['geometry']['top'])
    bottom_ids,bottom_w = surface_weights(support,base['geometry']['bottom'])
    front = np.einsum('pk,pkr->pr',top_w,V[top_ids,2,:])
    back = np.einsum('pk,pkr->pr',bottom_w,V[bottom_ids,2,:])
    loads = np.asarray(base['load_shape']).reshape(3,-1)@V.reshape(-1,rank)
    skin_ids = np.flatnonzero(base['geometry']['skin'])
    stress_basis = np.empty((len(skin_ids),6,rank))
    lam = 70e9*.33/(1.33*.34);mu = 70e9/(2*1.33)
    for start in range(0,len(skin_ids),assembly_chunk):
        selected = skin_ids[start:start+assembly_chunk]
        G = np.einsum('pkir,pkj->pijr',V[support.indices[selected]],support.gradients[selected],optimize=True)
        strain = .5*(G+G.swapaxes(1,2))
        trace = np.trace(strain,axis1=1,axis2=2)
        stress = 2*mu*strain+lam*np.eye(3)[None,:,:,None]*trace[:,None,None,:]
        stress -= np.eye(3)[None,:,:,None]*np.trace(stress,axis1=1,axis2=2)[:,None,None,:]/3
        stress_basis[start:start+len(selected)] = np.stack((stress[:,0,0],stress[:,1,1],stress[:,2,2],
            math.sqrt(2)*stress[:,1,2],math.sqrt(2)*stress[:,0,2],math.sqrt(2)*stress[:,0,1]),axis=1)
    return dict(basis=basis, basis_numpy=V, transform=transform, mode_labels=labels, rank=rank,
                operators=operators,operators_for_fraction=operators_for_fraction,
                loads=jnp.asarray(loads),back=jnp.asarray(back),front=jnp.asarray(front),
                original_back=jnp.asarray(back if original_back is None else original_back),
                skin_stress=jnp.asarray(stress_basis),skin_indices=skin_ids,clamped=clamped,
                baseline_mass_gram_condition=condition,assembly_chunk=assembly_chunk,
                enrichment=enrichment,
                basis_sha256=__import__('hashlib').sha256(V.tobytes()).hexdigest())


def newmark(mass,stiffness,load,settings,dt):
    """Average-acceleration Newmark for the projected linear dynamic system."""
    steps=math.ceil(settings.horizon/dt)
    dt=settings.horizon/steps
    times=jnp.arange(steps+1)*dt
    pressure=jnp.where(times<settings.pulse_duration,
        settings.peak_pressure*jnp.sin(jnp.pi*times/settings.pulse_duration)**2,0.)
    factor=jnp.linalg.cholesky(mass+.25*dt**2*stiffness)
    zero=jnp.zeros(len(load))
    def body(carry,n):
        u,v,a,work=carry
        predicted=u+dt*v+.25*dt**2*a
        acceleration=cho_solve((factor,True),pressure[n+1]*load-stiffness@predicted)
        following=predicted+.25*dt**2*acceleration
        velocity=v+.5*dt*(a+acceleration)
        work=work+.5*(pressure[n]+pressure[n+1])*jnp.dot(load,following-u)
        kinetic=.5*jnp.dot(velocity,mass@velocity)
        elastic=.5*jnp.dot(following,stiffness@following)
        return (following,velocity,acceleration,work),(following,velocity,kinetic,elastic,work)
    _,values=lax.scan(body,(zero,zero,zero,jnp.asarray(0.)),jnp.arange(steps))
    displacement=jnp.concatenate((zero[None],values[0]))
    velocity=jnp.concatenate((zero[None],values[1]))
    energy=jnp.column_stack((jnp.concatenate((jnp.zeros(1),values[2])),
        jnp.concatenate((jnp.zeros(1),values[3])),jnp.concatenate((jnp.zeros(1),values[4]))))
    return times,displacement,velocity,energy


def ks_update(maximum,total,values,rho,mask):
    """Stable unnormalized log-sum-exp; rho is fixed by the total sample count."""
    shift=lax.stop_gradient(jnp.maximum(maximum,jnp.max(values)))
    total=total*jnp.exp(rho*(maximum-shift))+jnp.sum(jnp.exp(rho*(values-shift))*mask)
    return shift,total


def response(projected,displacement,settings,time_chunk=16,ks_relative_error=.005):
    """All face quadrature and all saved times, with rematerialized time chunks."""
    ntime=len(displacement);rank=projected['rank']
    padded=math.ceil(ntime/time_chunk)*time_chunk
    trajectory=jnp.pad(displacement,((0,padded-ntime),(0,0)))
    stress=projected['skin_stress'].reshape(-1,rank)
    if not 0 < ks_relative_error <= .01:raise ValueError('KS relative-limit error must be in (0,.01]')
    rho_d=math.log(projected['back'].shape[0]*ntime)/ks_relative_error
    rho_s=math.log(projected['skin_stress'].shape[0]*ntime)/ks_relative_error
    def chunk(carry,index):
        ds,dp,ss,sp,maximum_d,maximum_s,score=carry
        states=lax.dynamic_slice_in_dim(trajectory,index*time_chunk,time_chunk)
        back=projected['back']@states.T
        sigma=(stress@states.T).reshape((-1,6,time_chunk))
        vm=jnp.sqrt(1.5*jnp.sum(sigma*sigma,axis=1)+1e-30)
        mask=(index*time_chunk+jnp.arange(time_chunk))<ntime
        ds,dp=ks_update(ds,dp,jnp.abs(back)/settings.displacement_limit,rho_d,mask[None,:])
        ss,sp=ks_update(ss,sp,vm/settings.stress_limit,rho_s,mask[None,:])
        # Endpoints have trapezoid weights; padded states have zero weight.
        weight=jnp.where((index*time_chunk+jnp.arange(time_chunk)==0)
            |(index*time_chunk+jnp.arange(time_chunk)==ntime-1),.5,1.)*mask
        score+=jnp.sum(jnp.mean(back**2,axis=0)*weight)/((ntime-1)*settings.displacement_limit**2)
        return (ds,dp,ss,sp,jnp.maximum(maximum_d,jnp.max(jnp.abs(back))),jnp.maximum(maximum_s,jnp.max(vm)),score),None
    initial=tuple(jnp.asarray(x) for x in (0.,0.,0.,0.,0.,0.,0.))
    result,_=lax.scan(jax.checkpoint(chunk,prevent_cse=False),initial,jnp.arange(padded//time_chunk))
    ds,dp,ss,sp,maximum_d,maximum_s,score=result
    return jnp.asarray([score,ds+jnp.log(dp)/rho_d,ss+jnp.log(sp)/rho_s,maximum_d,maximum_s])
