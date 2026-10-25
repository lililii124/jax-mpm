"""Authenticated full-state instantaneous contact; no impact time advance."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--projected',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    state_sha=hashlib.sha256(a.state.read_bytes()).hexdigest()
    if state_sha!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Original state required')
    s=dict(np.load(a.state,allow_pickle=False));m=s['mass'];h=.00025;F=s['F'];E=np.linalg.inv(F)*h/2
    op=ConsistentAffineProjection(s['x'],m,s['body_id'],h,F,E)
    y=np.load(a.projected,allow_pickle=False)['weighted_projected']
    contact=CompatibleContact(op);following,report=contact.project(y)
    v,W=op.unwhiten(y);nv,nW=op.unwhiten(following);D0=E@E.swapaxes(-1,-2)
    accurate=lambda arr:np.array([math.fsum(arr[:,j]) for j in range(3)])
    spin=lambda WW:np.stack(((WW@D0@F.swapaxes(-1,-2))[:,2,1]-(WW@D0@F.swapaxes(-1,-2))[:,1,2],
        (WW@D0@F.swapaxes(-1,-2))[:,0,2]-(WW@D0@F.swapaxes(-1,-2))[:,2,0],
        (WW@D0@F.swapaxes(-1,-2))[:,1,0]-(WW@D0@F.swapaxes(-1,-2))[:,0,1]),-1)
    report['linear_change']=accurate(m[:,None]*(nv-v)).tolist()
    report['angular_change']=accurate(m[:,None]*(np.cross(s['x'],nv-v)+spin(nW)-spin(W))).tolist()
    repeated,repeat_rows=op.project(following)
    report['range_preservation_weighted_norm']=float(np.linalg.norm(repeated-following))
    report['state_sha256']=state_sha;report['projected_npz_sha256']=hashlib.sha256(a.projected.read_bytes()).hexdigest()
    report['time_advanced_s']=0.;report['internal_heat_added_J']=0.;report['production_admitted']=False
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        (Path(__file__),Path('tools/metal_affine_contact_numpy.py'),Path('tools/metal_affine_projection_numpy.py'))}
    np.savez_compressed(a.output.with_suffix('.npz'),weighted_contact=following)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
