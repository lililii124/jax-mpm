"""Bounded sparse least-squares at original 125 ns; no time advancement."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from tools.metal_frozen_transfer import support
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--state',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--maxiter',type=int,default=4096)
options=parser.parse_args()
if options.output.exists():raise FileExistsError('Preserve previous evidence')
if not 1<=options.maxiter<=4096:raise ValueError('Bounded 1..4096 iterations required')
p=options.state
if hashlib.sha256(p.read_bytes()).hexdigest()!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':
 raise ValueError('Exact original admitted initial.npz required')
s=dict(np.load(p,allow_pickle=False)); x=s['x'];m=s['mass'];C=s['C'];v=s['v'];dx=.00025;n=len(x)
w,r,ids,wm,gm=support(x,s['body_id'],m,dx)
active=gm>0;lookup=np.full(len(gm),-1);lookup[active]=np.arange(active.sum());ids=lookup[ids].astype(int)
rows=np.broadcast_to(np.arange(n)[:,None,None]*4+np.arange(4)[None,None,:],(n,27,4))
cols=np.broadcast_to(ids[:,:,None],rows.shape)
values=np.sqrt(m)[:,None,None]*w[:,:,None]*np.concatenate((np.ones((n,27,1)),2*r/dx),axis=2)
keep=cols>=0
Q=coo_matrix((values[keep],(rows[keep],cols[keep])),shape=(4*n,int(active.sum()))).tocsr()
y=np.concatenate((v[:,:,None],C*dx/2),axis=2).transpose(0,2,1).reshape(4*n,3)*np.repeat(np.sqrt(m),4)[:,None]
# Diagonal scaling only; does not change range or prune modes.
d=np.sqrt(np.asarray(Q.power(2).sum(axis=0)).ravel()); Q=Q@__import__('scipy').sparse.diags(1/d)
projected=np.zeros_like(y);solves=[]
for a in range(3):
 t=time.monotonic();ans=lsmr(Q,y[:,a],atol=1e-13,btol=1e-13,maxiter=options.maxiter,conlim=1e12);projected[:,a]=Q@ans[0]
 solves.append(dict(component=a,status=int(ans[1]),iterations=int(ans[2]),residual_norm=float(ans[3]),normal_residual_norm=float(ans[4]),condition_estimate=float(ans[6]),seconds=time.monotonic()-t))
 print(json.dumps(solves[-1]),flush=True)
raw=projected/np.repeat(np.sqrt(m),4)[:,None]
raw=raw.reshape(n,4,3).transpose(0,2,1)
pv=raw[:,:,0];pC=raw[:,:,1:]*2/dx
spin=lambda cc:dx**2/4*np.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
linear=lambda vv:np.sum(m[:,None]*vv,axis=0)
angular=lambda vv,cc:np.sum(m[:,None]*(np.cross(x,vv)+spin(cc)),axis=0)
moment_changes=dict(linear=(linear(pv)-linear(v)).tolist(),angular=(angular(pv,pC)-angular(v,C)).tolist())
stress_power_change=float(np.sum(s['volume'][:,None,None]*s['tau']*(pC-C)))
energy=.5*np.sum(y*y);loss=energy-.5*np.sum(projected*projected);variance=.5*np.sum((y-projected)**2)
report=dict(kind='fixed-original-state bounded algebra only; not impact admission',state_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),particles=n,occupied_nodes=int(active.sum()),matrix_shape=Q.shape,nnz=Q.nnz,solves=solves,initial_kinetic_J=float(energy),projection_loss_J=float(loss),variance_J=float(variance),variance_identity_residual_J=float(loss-variance),normal_residual_norm=float(np.linalg.norm(Q.T@(y-projected))),projection_orthogonality_J=float(np.sum(projected*(y-projected))),production_admitted=False,solver_converged=all(t['status'] in (1,2,4,5) for t in solves),moment_changes=moment_changes,maximum_velocity_change_m_s=float(np.max(abs(pv-v))),maximum_gradient_change_per_s=float(np.max(abs(pC-C))),stress_power_change_W=stress_power_change)
report['source_sha256']={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path('tools/metal_frozen_transfer.py'))}
options.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)



