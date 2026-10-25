"""Read-only sparse reference on an authenticated failed physical pressure input."""
import json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
p=Path('/input/rejected-pressure-input.npz');a=np.load(p);fluid=a['fluid'];h=float(a['h']);shape=fluid.shape;ids=np.full(shape,-1,dtype=np.int64);ids[fluid]=np.arange(fluid.sum());n=int(fluid.sum());diag=np.zeros(n);anchor=np.zeros(n);rows=[];cols=[];data=[];velocity=tuple(a['velocity'+str(i)] for i in range(3));masses=tuple(a['mass'+str(i)] for i in range(3));fixed=tuple(a['fixed'+str(i)] for i in range(3));inv=tuple(np.divide(1,m,out=np.zeros_like(m),where=(m>0)&~f) for m,f in zip(masses,fixed))
for axis,im in enumerate(inv):
 l=np.pad(ids,tuple((1,0) if k==axis else (0,0) for k in range(3)),constant_values=-1);r=np.pad(ids,tuple((0,1) if k==axis else (0,0) for k in range(3)),constant_values=-1);w=h**4*im
 for this,other in [(l,r),(r,l)]:
  keep=(this>=0)&(w>0);np.add.at(diag,this[keep],w[keep]);exterior=keep&(other<0);np.add.at(anchor,this[exterior],w[exterior]);off=keep&(other>=0);rows.extend(this[off]);cols.extend(other[off]);data.extend(-w[off])
rows.extend(np.arange(n));cols.extend(np.arange(n));data.extend(diag);A=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsc()
def D(v):return h*h*(v[0][1:]-v[0][:-1]+v[1][:,1:]-v[1][:,:-1]+v[2][:,:,1:]-v[2][:,:,:-1])
rhs=D(velocity)[fluid];nc,labels=connected_components(A,directed=False);anch=np.bincount(labels,weights=anchor);rhs_sum=np.bincount(labels,weights=rhs);rhs_abs=np.bincount(labels,weights=np.abs(rhs));sizes=np.bincount(labels);closed=np.where(anch==0)[0];gauges=np.array([np.flatnonzero(labels==k)[0] for k in closed],dtype=int);keep=np.ones(n,dtype=bool);keep[gauges]=False;started=time.monotonic();sol=np.zeros(n);sol[keep]=spsolve(A[keep][:,keep],rhs[keep]);elapsed=time.monotonic()-started;full=np.zeros(shape);full[fluid]=sol;gr=[]
for axis in range(3):gr.append(h*h*(np.pad(full,tuple((1,0) if k==axis else (0,0) for k in range(3)))-np.pad(full,tuple((0,1) if k==axis else (0,0) for k in range(3)))))
true=D(tuple(i*g for i,g in zip(inv,gr)))[fluid]-rhs;corrected=tuple(v-i*g for v,i,g in zip(velocity,inv,gr));positive=np.concatenate([m[(m>0)&~f] for m,f in zip(masses,fixed)]);out=dict(step=int(a['step']),physical_steps_executed=0,unknowns=n,nonzeros=int(A.nnz),source_mass_min=float(positive.min()),source_mass_max=float(positive.max()),matrix_diagonal_ratio=float(diag.max()/diag.min()),direct_seconds=elapsed,finite=bool(np.isfinite(sol).all()),assembled_relative=float(np.linalg.norm(A@sol-rhs)/np.linalg.norm(rhs)),true_flux_relative=float(np.linalg.norm(true)/np.linalg.norm(rhs)),maximum_divergence_per_s=float(np.max(np.abs(D(corrected)[fluid]))/h**3),components=int(nc),closed_components=[dict(size=int(sizes[k]),rhs_sum=float(rhs_sum[k]),rhs_abs=float(rhs_abs[k])) for k in closed],scope='Read-only SciPy reference; no production solver replacement, regularization, mass cutoff, or physical advancement');Path('pressure-reference.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
