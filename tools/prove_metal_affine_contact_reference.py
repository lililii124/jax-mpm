"""Analytic contact, exact rigid modes and transported-metric contact checks."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    cube=np.array(np.meshgrid([-.25,.25],[-.25,.25],[-.25,.25])).reshape(3,-1).T
    x=np.concatenate((cube,cube));body=np.repeat([0,1],8);mass=np.repeat([.7,1.3],8)
    records=[]
    for name,mapF in [('seed',np.eye(3)),('transported',np.array([[1.1,.2,0],[0,.9,.1],[0,0,1.05]]))]:
        F=np.broadcast_to(mapF,(16,3,3)).copy();root=np.broadcast_to(.5*np.eye(3),F.shape).copy()
        op=ConsistentAffineProjection(x,mass,body,1.,F,root);contact=CompatibleContact(op,np.array([1.,0,0]))
        v=np.zeros_like(x);v[:,0]=np.where(body==0,3.,-1.);W=np.zeros_like(F);y=op.whiten(v,W)
        following,row=contact.project(y);nv,nW=op.unwhiten(following)
        analytic=.5*(8*.7)*(8*1.3)/(8*.7+8*1.3)*4**2
        assert row['valid'] and abs(row['kinetic_loss_J']-analytic)<1e-10
        assert np.max(abs(nv[:,0]-.4))<1e-10 and np.max(abs(nW))<1e-10
        L=np.array([[.1,-.3,.2],[.3,.05,-.1],[-.2,.1,.02]])
        affine=op.whiten(x@L.T+np.array([1.,2.,3.]),L@F)
        reproduction=float(np.max(abs(contact.sample(affine)),initial=0))
        assert reproduction<1e-10
        rng=np.random.default_rng(11);mu=rng.normal(size=len(contact.pairs));sample=rng.normal(size=y.shape)
        adjoint_error=float(np.dot(contact.sample(sample),mu)-np.sum(sample*contact.adjoint(mu)))
        assert abs(adjoint_error)<1e-10
        row.update(fixture=name,analytic_loss_J=analytic,affine_common_point_velocity_error=reproduction,adjoint_identity=adjoint_error)
        records.append(row)
    report=dict(passed=True,fixtures=records,production_admitted=False,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        (Path(__file__),Path('tools/metal_affine_contact_numpy.py'),Path('tools/metal_affine_projection_numpy.py'))})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
