"""Sparse consistent affine projection; shifts affect preconditioning only.

All physical grid columns are retained. The original whitened reconstruction
is Q0. Column equilibration, an invertible LDL square-root right factor, and
an exact known-range decomposition change the coordinates, never the least
squares objective. This is a fixed-geometry reference, not an impact solver.
"""
import gc
import numpy as np
from scipy.sparse import coo_matrix,diags,eye
from scipy.sparse.linalg import splu,LinearOperator,lsmr
from tools.metal_frozen_transfer import support
from tools.metal_affine_resource import require_commit_headroom


class ConsistentAffineProjection:
    def __init__(self,x,mass,body,dx,F,reference_root,preconditioner_shift=1e-8,clamped_particles=None,build_solver=True):
        self.x,self.mass,self.body,self.dx=map(np.asarray,(x,mass,body,dx))
        self.F,self.reference_root=F,reference_root
        self.n=len(x)
        w,r,ids,_,gm=support(x,body,mass,float(dx))
        active=gm>0;lookup=np.full(len(gm),-1,int);lookup[active]=np.arange(active.sum())
        ids=lookup[ids];self.full_node_count=int(active.sum())
        free=np.ones(self.full_node_count,bool)
        if clamped_particles is not None:
            clamped_particles=np.asarray(clamped_particles,bool)
            if clamped_particles.shape!=(self.n,):raise ValueError('Original fixed-particle mask required')
            indices=ids[clamped_particles].ravel();free[indices[indices>=0]]=False
        self.free_nodes=np.flatnonzero(free);self.clamped_nodes=np.flatnonzero(~free)
        mapping=np.full(self.full_node_count,-1,int);mapping[free]=np.arange(free.sum())
        ids=np.where(ids>=0,mapping[np.maximum(ids,0)],-1);self.nodes=int(free.sum())
        if self.nodes==0:raise ValueError('No unconstrained grid degrees of freedom')
        rows=np.broadcast_to(np.arange(len(x))[:,None,None]*4+np.arange(4)[None,None,:],(len(x),27,4))
        cols=np.broadcast_to(ids[:,:,None],rows.shape)
        gradient=4/float(dx)**2*np.einsum('pki,pij->pkj',r,F@reference_root)
        values=np.sqrt(mass)[:,None,None]*w[:,:,None]*np.concatenate((np.ones((*w.shape,1)),gradient),axis=2)
        keep=cols>=0
        Q0=coo_matrix((values[keep],(rows[keep],cols[keep])),shape=(4*len(x),self.nodes)).tocsr()
        self.Q0=Q0
        del rows,cols,values,keep,gradient
        self.column_norm=np.sqrt(np.asarray(Q0.power(2).sum(axis=0)).ravel())
        if np.any(self.column_norm<=0):raise ValueError('Unexpected zero physical column')
        Q=Q0@diags(1/self.column_norm);self.Q=Q
        if not build_solver:
            self._build_modes(Q0, ids, r, body, x, dx)
            self.tri=self.A=self.Qp=None
            self.factor_report=dict(built=False, physical_columns_removed=0)
            return
        gram=(Q.T@Q).tocsc()
        factor_headroom=require_commit_headroom()
        if not 0<preconditioner_shift<1:raise ValueError('Positive preconditioner-only shift required')
        factor=splu(gram+preconditioner_shift*eye(self.nodes,format='csc'),permc_spec='MMD_AT_PLUS_A',
                    diag_pivot_thresh=0,options={'SymmetricMode':True})
        if not np.array_equal(factor.perm_r,factor.perm_c):raise ValueError('Symmetric factor ordering required')
        self.order=np.argsort(factor.perm_c);self.Qp=Q[:,self.order]
        U=factor.U;D=U.diagonal()
        if np.any(D<=0):raise ValueError('Positive LDL pivots required')
        self.root=np.sqrt(D)
        # Avoid constructing a full sparse subtraction with capacity for
        # twice the factor nonzeros. The symmetric-pattern difference is
        # accumulated in bounded chunks, with the identical Frobenius norm.
        transpose_L=factor.L.T.tocsc()
        error_squared=0.
        diagonal=diags(D,format='csc')
        for start in range(0,U.shape[1],64):
            stop=min(start+64,U.shape[1])
            difference=U[:,start:stop]-diagonal@transpose_L[:,start:stop]
            error_squared+=float(np.dot(difference.data,difference.data))
        del transpose_L,difference
        self.factor_report=dict(shift_preconditioner_only=preconditioner_shift,physical_columns_removed=0,
            L_nnz=int(factor.L.nnz),U_nnz=int(U.nnz),minimum_pivot=float(D.min()),
            available_commit_before_factor_bytes=factor_headroom,
            LDL_relative_residual=float(np.sqrt(error_squared)/np.linalg.norm(U.data)))
        del factor,gram
        gc.collect()
        require_commit_headroom()
        self.tri=splu(U.tocsc(),permc_spec='NATURAL',diag_pivot_thresh=0)
        del U
        self._build_modes(Q0, ids, r, body, x, dx)
        self.A=LinearOperator(Q.shape,matvec=lambda z:self.orth(self.Qp@self.right(z)),
            rmatvec=lambda z:self.right_transpose(self.Qp.T@self.orth(z)),dtype=float)

    def _build_modes(self, Q0, ids, r, body, x, dx):
        # These polynomial grid fields are explicitly in range(Q), including
        # each body's translations and rotations. Deflation is exact algebra:
        # range(Q)=span(T) orthogonal-sum range((I-T T.T)Q).
        coords=np.zeros((self.nodes,3));nodebody=np.zeros(self.nodes,int)
        valid=ids>=0
        coords[ids[valid]]=(x[:,None,:]+r)[valid]
        nodebody[ids[valid]]=np.broadcast_to(body[:,None],ids.shape)[valid]
        modes=[]
        for b in np.unique(nodebody):
            mask=nodebody==b;center=np.mean(coords[mask],axis=0)
            modes.append(mask.astype(float))
            modes.extend(mask*(coords[:,j]-center[j])/float(dx) for j in range(3))
        K=Q0@np.array(modes).T
        # Exact structural zero rows occur for particles entirely supported
        # on fixed nodes. Excluding those rows from QR keeps their prescribed
        # v=W=0 exact instead of introducing Householder roundoff there.
        active_rows=np.diff(Q0.indptr)>0
        self.T=np.zeros_like(K)
        self.T[active_rows],_=np.linalg.qr(K[active_rows],mode='reduced')

    def right(self,z):return self.tri.solve(self.root*z)
    def right_transpose(self,z):return self.root*self.tri.solve(z,trans='T')
    def orth(self,z):return z-self.T@(self.T.T@z)
    def whiten(self,v,W):
        values=np.concatenate((v[:,:,None],W@self.reference_root),axis=2).transpose(0,2,1)
        return values.reshape(4*self.n,3)*np.repeat(np.sqrt(self.mass),4)[:,None]
    def unwhiten(self,y):
        values=(y/np.repeat(np.sqrt(self.mass),4)[:,None]).reshape(self.n,4,3).transpose(0,2,1)
        return values[:,:,0],values[:,:,1:]@np.linalg.inv(self.reference_root)
    def project(self,y,maxiter=256):
        if not 1<=maxiter<=512:raise ValueError('Bounded 1..512 iterations required')
        known=self.T@(self.T.T@y);rhs=self.orth(y)
        projected=np.zeros_like(y);reports=[]
        for a in range(y.shape[1]):
            # The right factor comes from a Gram LDL factorization. Its
            # resolvable singular condition is bounded by eps^(-1/2): above
            # this, squared singular values lie at the factor's roundoff
            # level. Stop before an exact null mode becomes amplified noise.
            # This is a numerical guard, not acceptance: the caller must
            # independently audit Q0 residual, idempotence and momenta.
            answer=lsmr(self.A,rhs[:,a],atol=1e-13,btol=1e-13,maxiter=maxiter,conlim=1/np.sqrt(np.finfo(float).eps))
            projected[:,a]=known[:,a]+self.A@answer[0]
            residual=y[:,a]-projected[:,a]
            reports.append(dict(component=a,stop_flag=int(answer[1]),iterations=int(answer[2]),
                scaled_normal_residual=float(np.linalg.norm(self.Q.T@residual)),
                original_normal_residual=float(np.linalg.norm(self.Q0.T@residual)),
                estimated_condition=float(answer[6]),condition_guard=float(1/np.sqrt(np.finfo(float).eps)),
                transformed_normal_residual=float(answer[4])))
        return projected,reports

