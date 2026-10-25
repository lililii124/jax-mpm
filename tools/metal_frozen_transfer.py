"""Bounded NumPy frozen-support zero-force remap diagnostic, not a restart."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.audit_metal_onset_state import invariants


def support(x, body, mass, dx):
    base = np.floor(x/dx-.5).astype(np.int64)
    offsets = np.array(list(itertools.product(range(3), repeat=3)))
    nodes = base[:,None,:]+offsets
    f = x/dx-base
    w = np.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2),axis=1)
    weights = np.prod(np.stack([w[:,offsets[:,a],a] for a in range(3)],axis=-1),axis=-1)
    keys = np.concatenate((np.broadcast_to(body[:,None,None],(*nodes.shape[:2],1)),nodes),axis=-1)
    _, inverse = np.unique(keys.reshape(-1,4),axis=0,return_inverse=True)
    inverse = inverse.reshape(len(x),27)
    wm = mass[:,None]*weights
    gm = np.bincount(inverse.ravel(),weights=wm.ravel())
    return weights,nodes*dx-x[:,None,:],inverse,wm,gm


def remap(v, C, mass, dx, frozen_support, scheme, increment=None):
    w,r,ids,wm,gm = frozen_support
    pv = v[:,None,:]+np.einsum('pij,pkj->pki',C,r)
    gp = np.stack([np.bincount(ids.ravel(),weights=(wm*pv[:,:,a]).ravel(),minlength=len(gm)) for a in range(3)],axis=-1)
    old = gp/np.where(gm>0,gm,1)[:,None]
    final = old if increment is None else old+increment
    field = final if scheme=='APIC' else final-old
    gathered=field[ids]
    nv=np.sum(w[:,:,None]*gathered,axis=1)
    nC=4/dx**2*np.einsum('pk,pki,pkj->pij',w,gathered,r)
    if scheme=='DELTA': nv,nC=v+nv,C+nC
    kinetic=lambda vv,cc: float(.5*np.sum(mass[:,None]*vv**2)+dx**2/8*np.sum(mass[:,None,None]*cc**2))
    kp0,kp1=kinetic(v,C),kinetic(nv,nC)
    kg0=float(.5*np.sum(gm[:,None]*old**2));kg1=float(.5*np.sum(gm[:,None]*final**2))
    work=float(np.sum(gm[:,None]*(final-old)*(.5*(final+old))))
    p2g,g2p=kp0-kg0,kg1-kp1
    return nv,nC,dict(p2g_J=p2g,g2p_signed_J=g2p,grid_work_J=work,particle_delta_J=kp1-kp0,
                      balance_residual_J=(kp1-kp0)+p2g+g2p-work)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=32)
    a=p.parse_args()
    if not 1<=a.repeats<=128: raise ValueError('Require 1..128 remaps')
    if a.output.exists(): raise FileExistsError('Preserve evidence')
    admission=json.loads(a.admission.read_text())
    with np.load(a.state,allow_pickle=False) as saved: fields={k:saved[k].copy() for k in saved.files}
    digest=lambda arr: hashlib.sha256(arr.tobytes()).hexdigest()
    hashes={k:digest(v) for k,v in fields.items()}
    if hashes!=admission['initial_array_sha256']: raise ValueError('Exact admitted saved state required')
    if float(fields['time'])!=125000*1e-12 or not bool(fields['valid']): raise ValueError('Exact valid 125ns required')
    source={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in admission['numerical_source']}
    if source!=admission['numerical_source']: raise ValueError('Frozen source differs')
    dx=.00025;m=fields['mass'];x=fields['x']
    frozen=support(x,fields['body_id'],m,dx)
    w,r,_,_,gm=frozen
    support_errors=dict(partition=float(np.max(abs(w.sum(1)-1))),
                        first_moment_m=float(np.max(abs(np.sum(w[:,:,None]*r,axis=1)))),
                        mass_kg=float(gm.sum()-m.sum()))
    if support_errors['partition']>1e-12 or support_errors['first_moment_m']>1e-12*dx or abs(support_errors['mass_kg'])>1e-12*m.sum():
        raise ValueError('Frozen support identities failed')
    rows=[]
    for scheme in ('APIC','DELTA'):
        v,C=fields['v'].copy(),fields['C'].copy();total=0.
        for i in range(a.repeats):
            v,C,d=remap(v,C,m,dx,frozen,scheme);total+=d['p2g_J']+d['g2p_signed_J']
            state=SimpleNamespace(**(fields|{'v':v,'C':C}))
            rows.append(dict(scheme=scheme,remap=i+1,cumulative_signed_transfer_J=total,**d,invariants=invariants(state,dx)))
    if hashes!={k:digest(v) for k,v in fields.items()}: raise RuntimeError('Input mutated')
    if source!={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in source}: raise RuntimeError('Frozen source mutated')
    result=dict(scope='Fixed support, dt=0, zero force/contact/boundary. No production admission.',
                production_restart_allowed=False,state_sha256=hashlib.sha256(a.state.read_bytes()).hexdigest(),
                frozen_array_sha256=hashes,frozen_source_sha256=source,
                support_errors=support_errors,
                diagnostic_source_sha256={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in (__file__,'mpm_jax/fragment_v6_delta_transfer.py')},
                initial_invariants=invariants(SimpleNamespace(**fields),dx),rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
