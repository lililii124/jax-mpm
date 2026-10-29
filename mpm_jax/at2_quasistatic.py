"""Quasi-static reference-grid MPM AT2 with a particle-side split precrack.

Particles provide fixed reference-volume integration (2x2 Gauss points/cell).
The grid uses linear tensor-product tent functions, embedded in three spatial
coordinates for the shared AT2 weak operator. For grid supports cut by the
initial crack, particle-side Heaviside enrichment selects independent upper
and lower coefficient DOFs. The unsplit tip remains shared. This is a reference
Galerkin MPM discretization, not an advecting/PIC MPM integrator. Its equivalence
to a particular quadrature-grid discretization is not a claim of equivalence
to the existing quadratic-B-spline dynamic implementation.

Default geometry-only initialization prescribes NO damage trace: d(0)=0,
and every damage coefficient is free except the irreversible box. Explicit
legacy-unit-seed mode exists only to identify/reproduce old evidence.

JAX evaluates spectral stress/tangents and particle weak forms. Sparse CPU
linear algebra solves the assembled Newton systems. There is no history field
and no optimizer differentiation through eigenvectors with repeated spectra.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def split_reference_particles(n, crack_length):
    if n%2 or n<12 or abs(round(crack_length*n)-crack_length*n)>1e-10:
        raise ValueError('Even n>=12 and an exactly grid-aligned crack tip required')
    grid=np.array([(i/n,j/n,0.) for j in range(n+1) for i in range(n+1)])
    tip=int(round(crack_length*n));duplicate={}
    coordinates=grid.tolist()
    for i in range(tip):
        node=(n//2)*(n+1)+i;duplicate[node]=len(coordinates);coordinates.append(grid[node].tolist())
    points=[];nodes=[];weights=[];gradient=[];triangles=[]
    gauss=(.5-.5/np.sqrt(3),.5+.5/np.sqrt(3))
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i;connect=[a,a+1,a+n+2,a+n+1]
            # N_i^+(X)=N_i(X) H(y-.5), N_i^-=N_i(X)(1-H).
            # Only support traces on the existing slit are split; no new
            # damage-dependent enrichment is introduced during propagation.
            if j>=n//2:connect=[duplicate.get(v,v) for v in connect]
            triangles.extend([[connect[0],connect[1],connect[2]],[connect[0],connect[2],connect[3]]])
            for s in gauss:
                for t in gauss:
                    points.append([(i+s)/n,(j+t)/n,0.]);nodes.append(connect)
                    weights.append([(1-s)*(1-t),s*(1-t),s*t,(1-s)*t])
                    gradient.append(np.array([[-(1-t),-(1-s),0],[1-t,-s,0],[t,s,0],[-t,1-s,0]])*n)
    result=dict(coordinates=np.asarray(coordinates),particles=np.asarray(points),nodes=np.asarray(nodes,np.int32),weight=np.asarray(weights),gradient=np.asarray(gradient),volume=np.full(4*n*n,1/(4*n*n)),triangles=np.asarray(triangles,np.int32))
    assert np.max(abs(result['weight'].sum(axis=1)-1))<1e-14
    assert np.max(abs(result['gradient'].sum(axis=1)))<1e-12
    reproduced=np.einsum('pk,pki->pi',result['weight'],result['coordinates'][result['nodes']])
    assert np.max(abs(reproduced-result['particles']))<1e-12
    affine=np.einsum('pki,pkj->pij',result['coordinates'][result['nodes']],result['gradient'])
    assert np.max(abs(affine[:,:2,:2]-np.eye(2)))<1e-12
    return result


class Model:
    def __init__(self,n=64,ell=.04,young=1e6,poisson=.2,gc=.001,residual=1e-7,crack_length=.375,reference_data=None,initial_damage_mode='geometry-only'):
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        from .phase_field import Basis,material,damage_system,solve_damage,interpolate,scatter
        self.jax=jax;self.jnp=jnp;self._damage_system=damage_system;self._shared_solve_damage=solve_damage
        self.interpolate=interpolate;self.scatter=scatter
        data=split_reference_particles(n,crack_length) if reference_data is None else reference_data
        self.coordinates=data['coordinates'];self.xy=self.coordinates[:,:2];self.particles=data['particles']
        self.nodes=data['nodes'];self.tri=data['triangles'];self.volume=data['volume'];self.area=self.volume
        self.weight=data['weight'];self.grad=data['gradient']
        self.support_count=self.nodes.shape[1]
        vector_count=2*self.support_count
        self.basis=Basis(jnp.asarray(self.nodes),jnp.asarray(self.weight),jnp.asarray(self.grad),jnp.array(True))
        self.v=jnp.asarray(self.volume);self.law=material(young,poisson,gc,ell,residual)
        self.n=n;self.ell=ell;self.young=young;self.poisson=poisson;self.gc=gc;self.residual=residual;self.crack_length=crack_length
        if initial_damage_mode not in ['geometry-only','legacy-unit-seed']:
            raise ValueError('Unknown initial-damage mode')
        self.initial_damage_mode=initial_damage_mode
        self.geometric_crack_trace=(abs(self.xy[:,1]-.5)<1e-12)&(self.xy[:,0]<=crack_length+1e-12)
        self.damage_fixed=(self.geometric_crack_trace.copy() if initial_damage_mode=='legacy-unit-seed' else np.zeros(len(self.xy),bool))
        self.seed=self.damage_fixed  # Compatibility alias: NEVER the geometric trace in the new mode.
        self.lam=self.law.lam;self.mu=self.law.mu
        self.B=np.zeros((len(self.nodes),3,vector_count))
        self.B[:,0,0::2]=self.grad[:,:,0];self.B[:,1,1::2]=self.grad[:,:,1]
        self.B[:,2,0::2]=self.grad[:,:,1];self.B[:,2,1::2]=self.grad[:,:,0]
        self.dofs=(2*self.nodes[:,:,None]+np.arange(2)).reshape(-1,vector_count)
        self.ku_rows=np.broadcast_to(self.dofs[:,:,None],(len(self.nodes),vector_count,vector_count)).ravel()
        self.ku_cols=np.broadcast_to(self.dofs[:,None,:],(len(self.nodes),vector_count,vector_count)).ravel()
        self.kd_rows=np.broadcast_to(self.nodes[:,:,None],(len(self.nodes),self.support_count,self.support_count)).ravel()
        self.kd_cols=np.broadcast_to(self.nodes[:,None,:],(len(self.nodes),self.support_count,self.support_count)).ravel()
        self.mass=self.volume[:,None,None]*self.weight[:,:,None]*self.weight[:,None,:]
        grad_block=self.volume[:,None,None]*np.einsum('pki,pli->pkl',self.grad,self.grad)
        self.fracture_matrix=self.assemble_d(gc/ell*self.mass+gc*ell*grad_block)
        bottom=np.flatnonzero(self.xy[:,1]<1e-12);top=np.flatnonzero(self.xy[:,1]>1-1e-12)
        self.top=2*top+1;self.fixed=np.unique(np.r_[2*bottom+1,self.top,0]);self.free=np.setdiff1d(np.arange(2*len(self.xy)),self.fixed)
        B=jnp.asarray(self.B);dofs=jnp.asarray(self.dofs);volume=self.v
        nodes=jnp.asarray(self.nodes);weight=jnp.asarray(self.weight);lam=self.lam;mu=self.mu
        basis_e=jnp.array([[[1.,0],[0,0]],[[0.,0],[0,1]],[[0,.5],[.5,0]]])
        def particle_kernel(u,d):
            ev=jnp.einsum('pij,pj->pi',B,u[dofs])
            eps=jnp.zeros((len(ev),2,2)).at[:,0,0].set(ev[:,0]).at[:,1,1].set(ev[:,1]).at[:,0,1].set(ev[:,2]/2).at[:,1,0].set(ev[:,2]/2)
            vals,Q=jnp.linalg.eigh(eps);positive=jnp.maximum(vals,0);negative=jnp.minimum(vals,0);tr=ev[:,0]+ev[:,1]
            eye=jnp.eye(2);ep=jnp.einsum('pia,pa,pja->pij',Q,positive,Q);en=eps-ep
            plus=.5*lam*jnp.maximum(tr,0)**2+mu*jnp.sum(positive**2,axis=1)
            minus=.5*lam*jnp.minimum(tr,0)**2+mu*jnp.sum(negative**2,axis=1)
            sp=lam*jnp.maximum(tr,0)[:,None,None]*eye+2*mu*ep
            sn=lam*jnp.minimum(tr,0)[:,None,None]*eye+2*mu*en
            vec=lambda a:jnp.stack((a[:,0,0],a[:,1,1],a[:,0,1]),axis=1)
            dp=jnp.sum(weight*d[nodes],axis=1);g=residual+(1-residual)*(1-dp)**2
            stress=g[:,None]*vec(sp)+vec(sn)
            local=volume[:,None]*jnp.einsum('pij,pi->pj',B,stress)
            force=jnp.zeros_like(u).at[dofs].add(local)
            energy=jnp.dot(volume,g*plus+minus)
            # Exact Frechet derivative of the positive spectral tensor. This
            # handles repeated/zero eigenvalues without singular eigenvector AD.
            delta=vals[:,:,None]-vals[:,None,:];num=positive[:,:,None]-positive[:,None,:]
            scale=jnp.maximum(jnp.max(abs(vals),axis=1),1e-15)
            distinct=abs(delta)>scale[:,None,None]*1e-9
            ratio=num/jnp.where(distinct,delta,1.)
            repeated=jnp.where(vals[:,:,None]+vals[:,None,:]>0,1.,jnp.where(vals[:,:,None]+vals[:,None,:]<0,0.,.5))
            F=jnp.where(distinct,ratio,repeated)
            local_e=jnp.einsum('pia,kij,pjb->pkab',Q,basis_e,Q)
            dep=jnp.einsum('pia,pab,pkab,pjb->pkij',Q,F,local_e,Q)
            trace_derivative=jnp.array([1.,1.,0.]);hp=jnp.where(tr>0,1.,jnp.where(tr<0,0.,.5))
            cp=lam*hp[:,None,None,None]*trace_derivative[None,:,None,None]*eye+2*mu*dep
            full=lam*trace_derivative[:,None,None]*eye+2*mu*basis_e
            c=cp*g[:,None,None,None]+full[None]-cp
            D=jnp.stack((c[:,:,0,0],c[:,:,1,1],c[:,:,0,1]),axis=1)
            ke=volume[:,None,None]*jnp.einsum('pia,pij,pjb->pab',B,D,B)
            return energy,force,plus,eps,ke
        self.kernel=jax.jit(particle_kernel)
        # Reuse the PUBLIC weak operator; host assembly below is only a sparse
        # solve backend for exactly the same particle-integrated quadratic.
        def damage_rhs(plus):
            return scatter((2*(1-residual)*self.v*plus)[:,None]*self.basis.weight,self.basis,len(self.xy))
        self.rhs=jax.jit(damage_rhs)

    def assemble_d(self,local):
        matrix=sparse.coo_matrix((local.ravel(),(self.kd_rows,self.kd_cols)),shape=(len(self.xy),)*2).tocsr()
        if self.support_count>4:matrix.eliminate_zeros()
        return matrix

    def mechanical(self,u,d,tangent=False):
        energy,force,plus,eps,ke=self.kernel(self.jnp.asarray(u),self.jnp.asarray(d))
        result=(float(energy),np.asarray(force),np.asarray(plus),np.asarray(eps))
        if tangent:
            K=sparse.coo_matrix((np.asarray(ke).ravel(),(self.ku_rows,self.ku_cols)),shape=(len(u),)*2).tocsr()
            if self.support_count>4:K.eliminate_zeros()
            return result+(K,)
        return result

    def equilibrate(self,u,d,load,tol=2e-7,maxiter=60):
        u=u.copy();u[self.fixed]=0;u[self.top]=load
        for it in range(maxiter):
            energy,force,plus,eps,K=self.mechanical(u,d,True)
            scale=max(self.young*max(abs(load),1e-9)/self.n,1e-14)
            error=float(np.max(abs(force[self.free]))/scale)
            if error<tol:return u,error,it+1
            du=np.zeros_like(u);du[self.free]=spsolve(K[self.free][:,self.free],-force[self.free]);slope=float(force@du)
            if not np.isfinite(du).all() or slope>=0:raise RuntimeError('Non-descent MPM mechanical Newton step')
            alpha=1.
            for _ in range(35):
                trial=u+alpha*du
                if self.mechanical(trial,d)[0]<=energy+1e-4*alpha*slope+1e-18:break
                alpha*=.5
            else:raise RuntimeError('Mechanical energy line search failed')
            u=trial
        raise RuntimeError('MPM mechanical Newton did not converge: '+str(error))

    def damage(self,plus,lower,warm,tol=2e-8,maxiter=100):
        if not np.any(plus) and not np.any(lower) and not np.any(self.damage_fixed):
            # The existing geometric slit is already traction free. With no
            # seeded damage and no tensile energy, d=0 is the unique AT2 minimum.
            return np.zeros_like(lower),0.,0
        A=self.fracture_matrix+self.assemble_d(2*(1-self.residual)*plus[:,None,None]*self.mass)
        rhs=np.asarray(self.rhs(self.jnp.asarray(plus)));diag=A.diagonal();d=np.clip(warm,lower,1);d[self.seed]=1
        scale=max(np.max(abs(rhs)),self.gc/self.ell/self.n**2,1e-20)
        # At zero tensile energy the stationary precrack is a strictly convex
        # seed-Dirichlet problem. Starting PDAS with only the seed released its
        # neighbours one support ring at a time; 100 rings do not cover n192.
        # Solve globally, then retain exactly the SAME box and projected-KKT
        # acceptance test. The loaded irreversible active-set path is unchanged.
        if not np.any(plus) and np.all(lower[~self.seed]==0) and np.all(lower[self.seed]==1):
            free=~self.seed;cold=lower.copy()
            cold[free]=spsolve(A[free][:,free],rhs[free]-A[free][:,self.seed]@cold[self.seed])
            cold=np.clip(cold,lower,1);cold[self.seed]=1
            gradient=A@cold-rhs;projected=cold-np.clip(cold-gradient/diag,lower,1);projected[self.seed]=0
            error=float(np.max(abs(projected*diag))/scale)
            if error<tol:return cold,error,1
            d=cold
        for it in range(maxiter):
            grad=A@d-rhs;trial=d-grad/diag
            low=trial<=lower;high=(trial>=1)|self.seed;fixed=low|high;free=~fixed
            dn=d.copy();dn[low]=lower[low];dn[high]=1
            if np.any(free):dn[free]=spsolve(A[free][:,free],rhs[free]-A[free][:,fixed]@dn[fixed])
            dn=np.clip(dn,lower,1);dn[self.seed]=1
            grad=A@dn-rhs;proj=dn-np.clip(dn-grad/diag,lower,1);proj[self.seed]=0
            error=float(np.max(abs(proj*diag))/scale);d=dn
            if error<tol:return d,error,it+1
        raise RuntimeError('Particle-integrated damage active set did not converge: '+str(error))

    def fracture(self,d):return float(.5*d@(self.fracture_matrix@d))

    def operator_equivalence(self,plus,d):
        operator,rhs,_=self._damage_system(self.jnp.asarray(plus),self.v,self.basis,len(d),self.law)
        A=self.fracture_matrix+self.assemble_d(2*(1-self.residual)*plus[:,None,None]*self.mass)
        applied=np.asarray(operator(self.jnp.asarray(d)));expected=A@d
        return float(np.max(abs(applied-expected))/max(np.max(abs(expected)),1e-30))
