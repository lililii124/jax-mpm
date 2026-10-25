"""Small NumPy analytical transfer benchmark, never an impact restart.

FLIP here is the zero-force round trip only, so identity is expected and cannot
admit FLIP for contact, stress work, deformation, or fragmentation.
"""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.audit_metal_onset_state import invariants


def transfer(x, v, C, mass, dx, scheme):
    base = np.floor(x/dx-.5).astype(int)
    offsets = np.array(list(itertools.product(range(3), repeat=3)))
    indices = base[:,None,:]+offsets
    f = x/dx-base
    w = np.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2),axis=1)
    weights = np.prod(np.stack([w[:,offsets[:,a],a] for a in range(3)],axis=-1),axis=-1)
    nodes, inverse = np.unique(indices.reshape(-1,3),axis=0,return_inverse=True)
    inverse = inverse.reshape(len(x),27)
    dpos = indices*dx-x[:,None,:]
    velocity = v[:,None,:]+np.einsum('pij,pkj->pki',C if scheme=='APIC' else np.zeros_like(C),dpos)
    wm = mass[:,None]*weights
    gm = np.zeros(len(nodes)); gp = np.zeros((len(nodes),3))
    np.add.at(gm,inverse,wm); np.add.at(gp,inverse,wm[:,:,None]*velocity)
    gv = gp/np.where(gm>0,gm,1)[:,None]
    out_v = np.sum(weights[:,:,None]*gv[inverse],axis=1)
    out_C = 4/dx**2*np.einsum('pk,pki,pkj->pij',weights,gv[inverse],dpos)
    if scheme=='FLIP': out_v,out_C = v.copy(),C.copy()  # zero grid increment only
    if scheme=='PIC': out_C = np.zeros_like(C)
    return out_v,out_C,dict(grid_mass_kg=float(gm.sum()), partition_error=float(np.max(abs(weights.sum(1)-1))),
                           first_moment_error_m=float(np.max(abs(np.sum(weights[:,:,None]*dpos,axis=1)))))


def benchmark():
    dx = .25
    x = np.array(list(itertools.product((-.31,-.06,.19,.44),repeat=3)))
    m = np.full(len(x),2/len(x)); translation=np.array([1.,-2.,3.])
    omega=np.array([.4,-.2,.7])
    skew=np.array([[0,-omega[2],omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]])
    cases={'translation':(np.broadcast_to(translation,x.shape),np.zeros((len(x),3,3))),
           'rigid_rotation':(np.cross(omega,x)+translation,np.broadcast_to(skew,(len(x),3,3))),
           'general_affine':(x@np.array([[.2,.3,0],[-.1,.5,.2],[.1,0,-.3]]).T+translation,
                             np.broadcast_to(np.array([[.2,.3,0],[-.1,.5,.2],[.1,0,-.3]]),(len(x),3,3)))}
    rows=[]
    for case,(v,C) in cases.items():
        for scheme in ('PIC','FLIP','APIC'):
            # PIC/FLIP do not carry affine subparticle momentum in this comparison.
            initial_C = C if scheme=='APIC' else np.zeros_like(C)
            initial=invariants(SimpleNamespace(x=x,v=v,C=initial_C,mass=m,internal_energy=np.zeros(len(x))),dx)
            ov,oc,grid=transfer(x,v,initial_C,m,dx,scheme)
            final=invariants(SimpleNamespace(x=x,v=ov,C=oc,mass=m,internal_energy=np.zeros(len(x))),dx)
            rows.append(dict(case=case,scheme=scheme,initial=initial,final=final,**grid,
                             velocity_error=float(np.max(abs(ov-v))),affine_error=float(np.max(abs(oc-initial_C))),
                             momentum_error=float(np.linalg.norm(np.array(final['linear_momentum_kg_m_s'])-initial['linear_momentum_kg_m_s'])),
                             angular_error=float(np.linalg.norm(np.array(final['angular_momentum_kg_m2_s'])-initial['angular_momentum_kg_m2_s'])),
                             delta_kinetic_J=final['translation_J']+final['affine_J']-initial['translation_J']-initial['affine_J']))
    apic=[r for r in rows if r['scheme']=='APIC']
    passed=all(max(r['velocity_error'],r['affine_error'],r['momentum_error'],r['angular_error'],abs(r['delta_kinetic_J']),abs(r['grid_mass_kg']-2))<1e-12 for r in apic)
    return dict(analytical_total_mass_kg=2.,apic_affine_reproduction_passed=passed,rows=rows,
                scope='Stress-free transfer identities only. No material/contact/fragmentation validation; FLIP identity is zero-force only.',
                production_restart_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier evidence')
    result=benchmark();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['apic_affine_reproduction_passed']:raise RuntimeError('Analytical APIC gates failed')


if __name__=='__main__':main()
