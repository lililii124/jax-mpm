"""Fixed-state/small-window compatibility proof; no real impact continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.metal_material_affine_numpy import AffineOperators,pack,unpack


def moments(op,z):return op.linear(z),op.angular(z)


def qualify():
    eps=np.finfo(float).eps;rng=np.random.default_rng(703)
    x=rng.uniform(-.4,.4,(16,3));mass=rng.uniform(.5,1.5,len(x));body=np.zeros(len(x),int);dx=1.
    F=np.broadcast_to(np.diag([1.1,.9,1.05]),(len(x),3,3)).copy()
    inverse=np.linalg.inv(F);D0=inverse@np.broadcast_to(dx**2/4*np.eye(3),F.shape)@inverse.swapaxes(-1,-2)
    op=AffineOperators(x,F,mass,body,dx,D0)
    v=rng.normal(size=x.shape);C=rng.normal(size=F.shape);W=C@F;z=pack(v,W)
    old_energy=float(.5*np.sum(mass[:,None]*v*v)+dx**2/8*np.sum(mass[:,None,None]*C*C))
    assert abs(op.energy(z)-old_energy)<64*eps*old_energy
    projected=op.project(z);twice=op.project(projected)
    kinetic_loss=op.energy(z)-op.energy(projected)
    variance=.5*np.dot(op.Hroot@(z-projected),op.Hroot@(z-projected))
    assert kinetic_loss>=-64*eps*old_energy and abs(kinetic_loss-variance)<128*eps*old_energy
    assert np.linalg.norm(op.Hroot@(twice-projected))<128*eps*np.linalg.norm(op.Hroot@z)
    p0,l0=moments(op,z);p1,l1=moments(op,projected)
    assert np.linalg.norm(p1-p0)<128*eps*max(np.linalg.norm(op.Hroot@z),1.)
    assert np.linalg.norm(l1-l0)<128*eps*max(np.linalg.norm(op.Hroot@z),1.)
    reconstructed=op.G@(op.R@projected)
    reconstruction_error=float(np.linalg.norm(op.Hroot@(reconstructed-projected)))
    assert reconstruction_error<1e-10*np.linalg.norm(op.Hroot@projected)
    tau=rng.normal(size=F.shape);tau=.5*(tau+tau.swapaxes(-1,-2));volume=np.full(len(x),.01)
    kicked,work=op.stress_kick(projected,tau,volume,.01)
    assert abs(work['kinetic_change_J']+work['physical_stress_work_J'])<128*eps*old_energy
    assert abs(work['physical_stress_work_J']-work['grid_stress_work_J'])<128*eps*old_energy
    pk,lk=moments(op,kicked)
    assert np.linalg.norm(pk-p1)<128*eps*max(np.linalg.norm(op.Hroot@z),1.)
    assert np.linalg.norm(lk-l1)<128*eps*max(np.linalg.norm(op.Hroot@z),1.)
    # A force-free affine trajectory is represented exactly at both geometries.
    # W remains constant; C_transfer changes, while physical affine energy/spin
    # stay constant. No exponential-F update or numerical heat is involved.
    L0=np.array([[.1,.2,0.],[-.1,.05,.1],[0.,0.,-.03]])
    affine_v=x@L0.T+np.array([.3,-.2,.1]);affine_W=np.broadcast_to(L0,F.shape)@F
    affine=pack(affine_v,affine_W);assert np.linalg.norm(op.Hroot@(op.project(affine)-affine))<128*eps*np.linalg.norm(op.Hroot@affine)
    T=.1;new_x=x+T*affine_v;new_F=F+T*affine_W
    moved=AffineOperators(new_x,new_F,mass,body,dx,D0)
    after=moved.project(affine)
    affine_projection_error=float(np.linalg.norm(moved.Hroot@(after-affine)))
    assert affine_projection_error<256*eps*np.linalg.norm(op.Hroot@affine)
    assert abs(moved.energy(affine)-op.energy(affine))<64*eps*max(op.energy(affine),1.)
    assert np.linalg.norm(moved.angular(affine)-op.angular(affine))<128*eps*max(op.energy(affine),1.)
    exact_L=L0@np.linalg.inv(np.eye(3)+T*L0)
    assert np.max(abs(moved.physical_gradient(affine)-exact_L))<64*eps
    # Deliberate singular grid case: first-derivative/value sampling on8corners
    # leaves null modes. The rigid-reproducing gauge must handle them explicitly.
    corners=np.array(list(__import__('itertools').product((-.25,.25),repeat=3)))
    xx=np.concatenate((corners,corners));ids=np.repeat([0,1],len(corners));m=np.repeat([.7/len(corners),1.3/len(corners)],len(corners));FF=np.broadcast_to(np.eye(3),(len(xx),3,3)).copy();DD=.25*FF
    contact_op=AffineOperators(xx,FF,m,ids,1.,DD)
    assert contact_op.nullity>0
    initial_v=np.where((ids==0)[:,None],np.array([3.,0.,0.]),np.array([-1.,0.,0.]));initial=pack(initial_v,np.zeros_like(FF));initial=contact_op.project(initial)
    rows=contact_op.matching_contact_rows(np.array([1.,0.,0.]));assert len(rows)>0
    contacted,contact=contact_op.contact(initial,rows)
    assert contact['valid'] and contact['kinetic_loss_J']>=0
    expected_loss=.5*(.7*1.3/(.7+1.3))*16
    assert abs(contact['kinetic_loss_J']-expected_loss)<256*eps*max(contact_op.energy(initial),1.)
    assert np.max(abs(unpack(contacted)[0]-np.array([.4,0.,0.])))<1e-11
    assert np.linalg.norm(contact['linear_change'])<256*eps*max(contact_op.energy(initial),1.)
    assert np.linalg.norm(contact['angular_change'])<256*eps*max(contact_op.energy(initial),1.)
    # Split contact impulses are zero-time changes: no drift/internal-energy
    # change is attached to them. The subsequent frozen-stress work is paired
    # with the same reconstructed midpoint velocity, then a second contact loss.
    stress=rng.normal(size=FF.shape);stress=.5*(stress+stress.swapaxes(-1,-2));vol=np.full(len(xx),.001)
    stressed,stress_work=contact_op.stress_kick(contacted,stress,vol,1e-3)
    final,second_contact=contact_op.contact(stressed,rows)
    assert second_contact['valid'] and second_contact['kinetic_loss_J']>=-1e-12
    balance=contact_op.energy(final)-contact_op.energy(initial)+contact['kinetic_loss_J']+stress_work['physical_stress_work_J']+second_contact['kinetic_loss_J']
    assert abs(balance)<256*eps*max(contact_op.energy(initial),1.)
    return dict(passed=True,impact_started=False,initial_old_metric_energy_J=old_energy,material_metric_initial_difference_J=op.energy(z)-old_energy,
        projection=dict(kinetic_loss_J=kinetic_loss,variance_identity_residual_J=kinetic_loss-variance,idempotence_error=float(np.linalg.norm(op.Hroot@(twice-projected))),rank=op.rank,nullity=op.nullity,rank_backward_error_tolerance=float(op.rank_tolerance),grid_reconstruction_weighted_error=reconstruction_error),
        stress=work,free_affine=dict(new_geometry_projection_error=affine_projection_error,energy_change_J=moved.energy(affine)-op.energy(affine),maximum_C_transfer_change=float(np.max(abs(moved.transfer_coefficient(affine)-op.transfer_coefficient(affine)))),physical_L_error=float(np.max(abs(moved.physical_gradient(affine)-exact_L)))),
        contact=dict(**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in contact.items()},analytic_loss_J=expected_loss,massless_grid_null_modes=contact_op.nullity),
        contact_stress_contact_work_balance_J=float(balance),
        numerical_heat_added=0,production_restart_allowed=False,public_media_allowed=False,
        limitations=['Dense small reference only; no scalable solver or full original material return integration yet.','The transported material-affine kinetic metric differs from the old fixed-grid C norm after deformation; it agrees exactly at the authenticated seed. Both must be reported, never silently substituted.','Grid null modes require an explicit rigid-reproducing velocity gauge, not an arbitrary inverse or physical eigenvalue pruning.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior proof')
    result=qualify();result['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('tools/metal_material_affine_numpy.py','tools/prove_metal_material_affine.py')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
