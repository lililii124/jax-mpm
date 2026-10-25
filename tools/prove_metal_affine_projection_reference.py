"""Compare sparse projection with independent small dense SVD references."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    rng=np.random.default_rng(17);results=[]
    fixtures=[('random',rng.uniform(-.4,.4,(16,3))),
              ('singular',np.array(np.meshgrid([-.25,.25],[-.25,.25],[-.25,.25])).reshape(3,-1).T)]
    cube=fixtures[1][1]
    fixtures.extend((f'crossing-{phase}',cube+np.array([phase,.7*phase,-.3*phase])) for phase in (.13,.27,.6))
    for name,x in fixtures:
        n=len(x);mass=rng.uniform(.5,1.5,n);body=np.zeros(n,int)
        F=np.broadcast_to(np.diag([1.1,.9,1.05]),(n,3,3)).copy();E=np.linalg.inv(F)*.5
        op=ConsistentAffineProjection(x,mass,body,1.,F,E)
        y=rng.normal(size=(4*n,3));projection,solves=op.project(y)
        Q=op.Q0.toarray();U,s,_=np.linalg.svd(Q,full_matrices=False)
        threshold=max(Q.shape)*np.finfo(float).eps*s[0];keep=s>threshold
        reference=U[:,keep]@(U[:,keep].T@y)
        error=float(np.linalg.norm(projection-reference));twice,_=op.project(projection)
        identity=float(.5*np.sum(y*y)-.5*np.sum(projection*projection)-.5*np.sum((y-projection)**2))
        assert error<1e-11*np.linalg.norm(y)
        assert np.linalg.norm(twice-projection)<1e-11*np.linalg.norm(y)
        assert abs(identity)<1e-11*np.sum(y*y)
        results.append(dict(fixture=name,rank=int(keep.sum()),nullity=Q.shape[1]-int(keep.sum()),
            dense_SVD_error=error,idempotence=float(np.linalg.norm(twice-projection)),projection_identity_J=identity))
    report=dict(passed=True,fixtures=results,production_admitted=False,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),Path('tools/metal_affine_projection_numpy.py'))})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
