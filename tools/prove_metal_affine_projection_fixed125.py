"""Source-bound fixed-state projection audit; never advances the impact."""
import argparse,gc,hashlib,json,math,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shift',type=float,default=1e-8);p.add_argument('--compare',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    state_sha=hashlib.sha256(a.state.read_bytes()).hexdigest()
    if state_sha!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Original authenticated state required')
    s=dict(np.load(a.state,allow_pickle=False));dx=.00025;m=s['mass'];F=s['F']
    reference_root=np.linalg.inv(F)*(dx/2);W=s['C']@F
    start=time.monotonic();op=ConsistentAffineProjection(s['x'],m,s['body_id'],dx,F,reference_root,a.shift)
    y=op.whiten(s['v'],W);projected,solves=op.project(y);twice,second_solves=op.project(projected)
    norm=float(np.linalg.norm(projected));idempotence=float(np.linalg.norm(twice-projected))
    pv,pW=op.unwhiten(projected);pC=pW@np.linalg.inv(F)
    spin=lambda C:dx**2/4*np.stack((C[:,2,1]-C[:,1,2],C[:,0,2]-C[:,2,0],C[:,1,0]-C[:,0,1]),axis=-1)
    accurate_sum=lambda values:np.array([math.fsum(values[:,j]) for j in range(3)])
    linear=lambda v:accurate_sum(m[:,None]*v)
    angular=lambda v,C:accurate_sum(m[:,None]*(np.cross(s['x'],v)+spin(C)))
    loss=float(.5*np.sum(y*y)-.5*np.sum(projected*projected));variance=float(.5*np.sum((y-projected)**2))
    report=dict(kind='fixed 125 ns state; no impact integration or media admission',state_sha256=state_sha,
        factor=op.factor_report,solves=solves,second_solves=second_solves,seconds=time.monotonic()-start,
        initial_kinetic_J=float(.5*np.sum(y*y)),projection_loss_J=loss,variance_J=variance,
        variance_identity_residual_J=loss-variance,idempotence_weighted_norm=idempotence,
        idempotence_kinetic_perturbation_bound_J=norm*idempotence+.5*idempotence**2,
        linear_change=(linear(pv)-linear(s['v'])).tolist(),angular_change=(angular(pv,pC)-angular(s['v'],s['C'])).tolist(),
        maximum_velocity_change_m_s=float(np.max(abs(pv-s['v']))),maximum_gradient_change_per_s=float(np.max(abs(pC-s['C']))),
        original_energy_tolerance_J=1.414599609375e-8,physical_columns_removed=0,production_admitted=False)
    if a.compare:
        previous=np.load(a.compare,allow_pickle=False)['weighted_projected']
        delta=float(np.linalg.norm(projected-previous))
        report['preconditioner_comparison']=dict(path=str(a.compare),weighted_difference_norm=delta,
            kinetic_perturbation_bound_J=norm*delta+.5*delta**2)
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        (Path(__file__),Path('tools/metal_affine_projection_numpy.py'),Path('tools/metal_frozen_transfer.py'))}
    np.savez_compressed(a.output.with_suffix('.npz'),weighted_projected=projected)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
