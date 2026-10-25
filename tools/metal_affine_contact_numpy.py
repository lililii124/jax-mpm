"""Instantaneous frictionless contact in the same affine kinetic metric.

The sample is the reconstruction's lumped H-adjoint plus an exact affine-mode
correction. Its adjoint lies in the reconstruction range, so contact does not
undo the consistent projection. No grid inverse or grid-null velocity is used.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import nnls
from tools.metal_frozen_transfer import support


class CompatibleContact:
    def __init__(self,op,prescribed_normals=None):
        self.op=op;x=op.x;m=op.mass;dx=float(op.dx)
        w,r,ids,wm,gm=support(x,op.body,m,dx)
        active=gm>0;lookup=np.full(len(gm),-1,int);lookup[active]=np.arange(active.sum());ids=lookup[ids]
        gm=gm[active];coords=np.zeros((len(gm),3),int);body=np.zeros(len(gm),int)
        valid=ids>=0
        coords[ids[valid]]=np.rint((x[:,None,:]+r)[valid]/dx).astype(int)
        body[ids[valid]]=np.broadcast_to(op.body[:,None],ids.shape)[valid]
        moment=np.stack([np.bincount(ids[valid],weights=(-wm[:,:,None]*r)[:,:,c][valid],minlength=len(gm)) for c in range(3)],-1)
        center=moment/gm[:,None]
        table={tuple(coords[i]):i for i in np.flatnonzero(body==1)}
        pairs=np.array([(i,table[tuple(coords[i])]) for i in np.flatnonzero(body==0) if tuple(coords[i]) in table],int).reshape(-1,2)
        delta=center[pairs[:,1]]-center[pairs[:,0]];length=np.linalg.norm(delta,axis=1)
        self.normal=np.divide(delta,length[:,None],out=np.zeros_like(delta),where=length[:,None]>64*np.finfo(float).eps*dx)
        if prescribed_normals is not None:
            # Only analytic fixtures prescribe a resolved interface direction;
            # the impact entry point always uses the original centroid law.
            normals=np.broadcast_to(prescribed_normals,self.normal.shape).copy()
            if np.max(abs(np.linalg.norm(normals,axis=1)-1),initial=0)>1e-12:raise ValueError('Unit fixture normals required')
            self.normal=normals
        self.pairs=pairs;count=len(pairs)
        self.node_coordinates,self.node_body,self.grid_mass=coords,body,gm
        polynomials=[]
        for b in np.unique(body):
            mask=body==b;c=np.mean(coords[mask]*dx,axis=0);polynomials.append(mask.astype(float))
            polynomials.extend(mask*(coords[:,j]*dx-c[j])/dx for j in range(3))
        polynomials=np.array(polynomials).T;mix=op.T.T@(op.Q0@polynomials)
        gridT=np.linalg.solve(mix.T,polynomials.T).T
        self.affine_reconstruction_error=float(np.linalg.norm(op.Q0@gridT-op.T))
        entries=[];correction=np.zeros((count,3*op.T.shape[1]));gridmaps=[]
        for c in range(3):
            rows=np.repeat(np.arange(count),2);cols=pairs.ravel();sign=np.tile([1.,-1.],count)
            vals=np.repeat(self.normal[:,c],2)*sign
            grid=coo_matrix((vals,(rows,cols)),shape=(count,len(gm))).tocsr()
            weighted=coo_matrix((vals/gm[cols],(rows,cols)),shape=grid.shape).tocsr()
            block=(weighted@op.Q0.T).tocoo()
            entries.append((block.data,block.row,3*block.col+c))
            correction[:,c::3]=grid@gridT-block.tocsr()@op.T
        self.B0=coo_matrix((np.concatenate([e[0] for e in entries]),
            (np.concatenate([e[1] for e in entries]),np.concatenate([e[2] for e in entries]))),shape=(count,12*op.n)).tocsr()
        self.C=correction
        G=np.empty_like(correction)
        for c in range(3):G[:,c::3]=self.B0[:,c::3]@op.T
        self.mobility=np.asarray((self.B0@self.B0.T).toarray())+G@correction.T+correction@G.T+correction@correction.T
        self.mobility=.5*(self.mobility+self.mobility.T)

    def sample(self,y):return self.B0@y.ravel()+self.C@(self.op.T.T@y).ravel()
    def adjoint(self,impulse):
        return (self.B0.T@impulse).reshape(4*self.op.n,3)+self.op.T@(self.C.T@impulse).reshape(self.op.T.shape[1],3)
    def _balanced_system(self):
        if hasattr(self,'_contact_balance'):return self._contact_balance
        M=self.mobility;diag=np.diag(M);positive=diag>0;zero=diag==0
        if np.any(diag<0) or np.any(M[zero]!=0):raise ValueError('Nonpositive contact diagonal is not a true zero mobility row')
        scale=np.ones(len(diag));scale[positive]=1/np.sqrt(diag[positive])
        balanced=scale[:,None]*M*scale[None,:]
        raw_eig=np.linalg.eigvalsh(M);eig,V=np.linalg.eigh(balanced)
        raw_threshold=max(M.shape)*np.finfo(float).eps*float(raw_eig.max(initial=0))
        threshold=max(M.shape)*np.finfo(float).eps*float(eig.max(initial=0));keep=eig>threshold
        if raw_eig.min(initial=0)<-raw_threshold or eig.min(initial=0)<-threshold:raise ValueError('Indefinite contact mobility')
        condition=lambda values:float(values[-1]/values[0]) if len(values) and values[0]>0 else None
        diagnostics=dict(mobility_diagonal_min=float(diag.min()) if len(diag) else 0.,mobility_diagonal_max=float(diag.max(initial=0)),
            mobility_scale_min=float(scale.min()) if len(scale) else 1.,mobility_scale_max=float(scale.max(initial=0)),
            unscaled_eigenvalue_min=float(raw_eig[0]) if len(raw_eig) else 0.,unscaled_eigenvalue_max=float(raw_eig.max(initial=0)),
            unscaled_eigen_condition=condition(raw_eig),unscaled_mobility_rank=int(np.sum(raw_eig>raw_threshold)),unscaled_rank_threshold=raw_threshold,
            balanced_eigenvalue_min=float(eig[0]) if len(eig) else 0.,balanced_eigenvalue_max=float(eig.max(initial=0)),
            balanced_eigen_condition=condition(eig),balanced_rank_threshold=threshold,
            constraint_scaling='Positive diagonal congruence SMS, Sg; physical multiplier lambda=S*mu; no regularization',true_zero_mobility_rows=int(zero.sum()))
        self._contact_balance=(scale,zero,eig,V,keep,diagnostics)
        return self._contact_balance
    def project(self,y,tolerance=.005,return_impulse=False):
        closing=self.sample(y)
        scale,zero,eig,V,keep,diagnostics=self._balanced_system()
        feasible=bool(np.max(closing,initial=0)<=tolerance)
        zero_solvable=bool(np.max(closing[zero],initial=0)<=tolerance)
        multiplier=np.zeros(len(closing));solver_error=None
        if not feasible and zero_solvable:
            A=np.sqrt(eig[keep])[:,None]*V[:,keep].T
            target=(V[:,keep].T@(scale*closing))/np.sqrt(eig[keep])
            try:multiplier=scale*nnls(A,target,maxiter=1024)[0] if np.any(keep) else multiplier
            except RuntimeError as error:solver_error=str(error)
        delta=-self.adjoint(multiplier);following=y+delta;final=self.sample(following)
        residual=np.where(multiplier>0,abs(final),np.maximum(final,0.))
        identity=float(.5*np.sum(following**2)-.5*np.sum(y*y)+.5*np.sum(delta**2)+np.dot(multiplier,final))
        report=dict(valid=bool(zero_solvable and solver_error is None and np.max(residual,initial=0)<=tolerance),pairs=len(self.pairs),
            mobility_rank=int(keep.sum()),mobility_nullity=int((~keep).sum()),
            initial_closing_m_s=float(np.max(closing,initial=0)),final_closing_m_s=float(np.max(final,initial=0)),
            complementarity_m_s=float(np.max(residual,initial=0)),kinetic_loss_J=float(.5*np.sum(y*y)-.5*np.sum(following**2)),
            instantaneous_work_identity_J=identity,affine_reconstruction_error=self.affine_reconstruction_error,
            maximum_affine_correction=float(np.max(abs(self.C),initial=0)),feasible_fast_path=feasible,
            solver_error=solver_error,**diagnostics)
        return (following,report,multiplier) if return_impulse else (following,report)
