"""One frozen actual-input JAX QP comparison; no material or trajectory replay.

The current migration boundary is explicit: reference sparse geometry and LDL
preconditioner construction remain SciPy. Every iterative solve below is JAX;
no restricted146-column Schur construction or SciPy callback is used in JIT.
"""
import json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_dynamic_numpy import moments
from mpm_jax.fragment_v6_matrixfree_projection import sparse_apply,consistent_project,cone_project,intersect_project


def triangular_blocks(matrix,size=128):
    """Exact block triangular solve preparation; not an approximate objective."""
    from scipy.sparse import eye
    n=matrix.shape[0];blocks=(n+size-1)//size;N=blocks*size
    dense=np.zeros((blocks,size,size));off=[[] for _ in range(blocks)]
    coo=matrix.tocoo()
    for r,c,v in zip(coo.row,coo.col,coo.data):
        if r//size==c//size:dense[r//size,r%size,c%size]=v
        else:off[r//size].append((r%size,c,v))
    for r in range(n,N):dense[r//size,r%size,r%size]=1.
    cap=max(map(len,off),default=0);row=np.zeros((blocks,cap),np.int32);col=row.copy();data=np.zeros((blocks,cap))
    for b,entries in enumerate(off):
        for i,(r,c,v) in enumerate(entries):row[b,i]=r;col[b,i]=c;data[b,i]=v
    dense,row,col,data=map(jnp.asarray,(dense,row,col,data))
    lower=bool(np.any(coo.row>coo.col));assert not (lower and np.any(coo.row<coo.col))
    parameters=(dense,row,col,data)
    def solve(rhs,parameters=parameters):
        dense,row,col,data=parameters
        padded=jnp.pad(rhs,((0,N-n),(0,0)))
        def body(i,x):
            b=i if lower else blocks-1-i;offset=b*size
            correction=jnp.zeros((size,rhs.shape[1]),rhs.dtype).at[row[b]].add(data[b,:,None]*x[col[b]])
            value=jax.lax.dynamic_slice(padded,(offset,0),(size,rhs.shape[1]))-correction
            answer=solve_triangular(dense[b],value,lower=lower)
            return jax.lax.dynamic_update_slice(x,answer,(offset,0))
        return jax.lax.fori_loop(0,blocks,body,jnp.zeros_like(padded))[:n]
    return solve,dict(blocks=blocks,block_size=size,off_diagonal_capacity=cap,exact_triangular_blocks=True),parameters


def main():
    started=time.monotonic();out=Path('/work/result');out.mkdir()
    def load(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    old=load(Path('/original/result/step-22/accepted.npz'));gold=load(Path('/gold/result/midpoint-raw.npz'))
    args=(old['x'],old['mass'],old['body_id'],.00025,old['F'],old['reference_root'])
    allowed=ConsistentAffineProjection(*args,clamped_particles=old['candidate_fixed_particles'])
    config=dict(old,x=gold['geometry_x'],F=gold['geometry_F'])
    current=ConsistentAffineProjection(config['x'],config['mass'],config['body_id'],.00025,config['F'],config['reference_root'],build_solver=False)
    contact=CompatibleContact(current)
    assert np.array_equal(contact.normal,gold['normals'])
    assert np.array_equal(allowed.tri.perm_r,np.arange(allowed.nodes)) and np.array_equal(allowed.tri.perm_c,np.arange(allowed.nodes))
    upper,pc,upper_parameters=triangular_blocks(allowed.tri.U);lower,_,lower_parameters=triangular_blocks(allowed.tri.U.T)
    root=jnp.asarray(allowed.root)
    right=lambda z:upper(root[:,None]*z)
    rightT=lambda z:root[:,None]*lower(z)
    def sparse(matrix):
        m=matrix.tocoo();data,row,col=map(jnp.asarray,(m.data,m.row.astype(np.int32),m.col.astype(np.int32)))
        parameters=(data,row,col)
        return (lambda x,p=parameters:sparse_apply(p[0],p[1],p[2],x,m.shape[0]),lambda x,p=parameters:sparse_apply(p[0],p[2],p[1],x,m.shape[1]),parameters)
    Q,QT,q_parameters=sparse(allowed.Qp);B,BT,b_parameters=sparse(contact.B0)
    T=jnp.asarray(allowed.T);currentT=jnp.asarray(current.T);C=jnp.asarray(contact.C);M=jnp.asarray(contact.mobility)
    def sample(y):return B(y.ravel())+C@(currentT.T@y).ravel()
    def adjoint(lam):return BT(lam).reshape(-1,3)+currentT@(C.T@lam).reshape(currentT.shape[1],3)
    trial=jnp.asarray(gold['trial']);prepare_seconds=time.monotonic()-started
    parameters=dict(upper=upper_parameters,lower=lower_parameters,q=q_parameters,b=b_parameters,root=root,T=T,currentT=currentT,C=C,M=M)
    def kernel(y,p):
        q=lambda x:Q(x,p['q']);qt=lambda x:QT(x,p['q'])
        right=lambda x:upper(p['root'][:,None]*x,p['upper'])
        rightT=lambda x:p['root'][:,None]*lower(x,p['lower'])
        sample=lambda x:B(x.ravel(),p['b'])+p['C']@(p['currentT'].T@x).ravel()
        adjoint=lambda lam:BT(lam,p['b']).reshape(-1,3)+p['currentT']@(p['C'].T@lam).reshape(p['currentT'].shape[1],3)
        project_range=lambda x:consistent_project(x,q,qt,p['T'],right,rightT,iterations=64)
        project_cone=lambda x:cone_project(x,sample,adjoint,p['M'],sweeps=512,tolerance=1e-7)
        return intersect_project(y,project_range,project_cone,sample,iterations=64)
    solve=jax.jit(kernel)
    print(json.dumps(dict(phase='jax-compiling',prepare_seconds=prepare_seconds,particles=len(old['mass']),preconditioner=pc,scope='SciPy preparation; pure JAX iterative kernel, not all-JAX impact integration')),flush=True)
    # Large operator arrays are dynamic inputs: prevent XLA constant folding
    # from replicating the padded triangular workspaces during compilation.
    del allowed
    import gc;gc.collect()
    begun=time.monotonic();compiled=solve.lower(trial,parameters).compile();compile_seconds=time.monotonic()-begun
    begun=time.monotonic();following,report=compiled(trial,parameters);following.block_until_ready();solve_seconds=time.monotonic()-begun
    following=np.asarray(following);values={k:np.asarray(v) for k,v in report.items()};physical=values.pop('physical_impulse');reaction=values.pop('boundary_reaction');lam=values.pop('multiplier')
    report={k:v.item() for k,v in values.items()};relative=float(np.linalg.norm(following-gold['following'])/np.linalg.norm(gold['following']))
    native=contact.sample(following);kkt=float(np.max(np.where(lam>0,abs(native),np.maximum(native,0)),initial=0))
    dualphysical=-contact.adjoint(lam);dualdifference=float(np.linalg.norm(physical-dualphysical))
    P,L=moments(config,2*physical);boundarywork=float(np.sum(following*2*reaction));contactwork=float(np.sum(following*2*physical))
    tol=1.414599609375e-8;torquetol=1.5782228774332438e-14
    passed=bool(report['converged'] and relative<=1e-10 and kkt<=.005 and abs(boundarywork)<=tol and contactwork<=tol and np.linalg.norm(L)<=torquetol)
    np.savez_compressed(out/'jax-raw.npz',trial=np.asarray(trial),following=following,physical=physical,reaction=reaction,multiplier=lam)
    result=dict(status='passed' if passed else 'failed',source_input='Original accepted22 and already-solved same frozen qmid golden; no replay',particles=len(old['mass']),prepare_seconds=prepare_seconds,compile_seconds=compile_seconds,solve_seconds=solve_seconds,relative_velocity_difference=relative,original_KKT_m_s=kkt,physical_dual_difference=dualdifference,physical_momentum=P.tolist(),physical_torque=L.tolist(),boundary_work_J=boundarywork,contact_work_J=contactwork,report=report,preconditioner=pc,migration_boundary='Geometry and exact sparse LDL setup still SciPy. All LSQR, triangular application, cone and Dykstra iterations JAX64; no146-column Schur, no JIT callbacks. Not yet an all-JAX impact step.')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert passed,'Bounded frozen-input migration failed; preserve raw and diagnose, do not integrate'


if __name__=='__main__':main()
