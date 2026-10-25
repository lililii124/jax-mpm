"""Targeted fixed-node and constrained-contact virtual-work checks."""
import argparse,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_dirichlet_numpy import DirichletContact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    cube=np.array(np.meshgrid([-.25,.25],[-.25,.25],[-.25,.25])).reshape(3,-1).T
    x=np.concatenate((cube,cube,[[3.,0,0]]));n=len(x);body=np.r_[np.zeros(8,int),np.ones(9,int)]
    mass=np.r_[np.full(8,.7),np.full(8,1.3),.2];fixed=np.zeros(n,bool);fixed[-1]=True
    F=np.broadcast_to(np.array([[1.1,.2,0],[0,.9,.1],[0,0,1.05]]),(n,3,3)).copy()
    E=np.broadcast_to(.5*np.eye(3),F.shape).copy()
    full=ConsistentAffineProjection(x,mass,body,1.,F,E)
    constrained=ConsistentAffineProjection(x,mass,body,1.,F,E,clamped_particles=fixed)
    difference=constrained.Q0-full.Q0[:,constrained.free_nodes]
    assert np.max(abs(difference.data),initial=0)<1e-14
    contact=CompatibleContact(full,np.array([1.,0,0]));restricted=DirichletContact(contact,constrained)
    rng=np.random.default_rng(41);y=rng.normal(size=(4*n,3));free,_=full.project(y);allowed,_=constrained.project(free)
    before=.5*np.sum(free**2);after=.5*np.sum(allowed**2)
    assert before>=after-1e-12
    assert abs(before-after-.5*np.sum((allowed-free)**2))<1e-11
    v,W=constrained.unwhiten(allowed)
    assert np.all(v[fixed]==0) and np.all(W[fixed]==0)
    following,row,reaction,physical=restricted.apply(allowed)
    projected,_=constrained.project(following)
    assert row['valid'] and np.linalg.norm(projected-following)<1e-10
    assert abs(row['boundary_virtual_work_J'])<1e-10
    virtual,_=constrained.project(rng.normal(size=y.shape))
    orthogonal=float(np.sum(virtual*reaction));assert abs(orthogonal)<1e-10
    report=dict(passed=True,eliminated_nodes=len(constrained.clamped_nodes),
        clamped_particle_velocity_exact_zero=True,clamped_particle_W_exact_zero=True,
        boundary_projection_loss_J=float(before-after),contact=row,
        reaction_random_admissible_virtual_work=orthogonal,production_admitted=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
