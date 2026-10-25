"""Full original-field contact/stress/contact work, frozen geometry diagnostic."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--projected',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    a.output.mkdir(parents=True)
    sha=hashlib.sha256(a.state.read_bytes()).hexdigest()
    if sha!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Exact original initial state required')
    s=dict(np.load(a.state,allow_pickle=False));m=s['mass'];h=.00025;F=s['F'];E=np.linalg.inv(F)*h/2
    material=json.loads(a.admission.read_text())['material']
    op=ConsistentAffineProjection(s['x'],m,s['body_id'],h,F,E)
    contact=CompatibleContact(op);initial=np.load(a.projected,allow_pickle=False)['weighted_projected']
    start,first=contact.project(initial)
    if not first['valid']:raise ValueError('Original contact tolerance failed before stress')
    v,W=op.unwhiten(start);L=W@np.linalg.inv(F);J=np.linalg.det(F)
    pressure=material['bulk_modulus']*(1-J);pressure=np.where(pressure<0,(1-s['damage'])*pressure,pressure)
    rho=material['density']/J;sound=np.sqrt(material['bulk_modulus']*J*J/material['density'])
    compression=np.maximum(-np.trace(L,axis1=-2,axis2=-1),0);speed=h*compression
    q=rho*(material['viscosity_linear']*sound*speed+material['viscosity_quadratic']*speed**2)
    tau=s['tau']-J[:,None,None]*(pressure+q)[:,None,None]*np.eye(3)
    covector=-s['volume'][:,None,None]/np.sqrt(m)[:,None,None]*tau@np.linalg.inv(F).swapaxes(-1,-2)@np.linalg.inv(E).swapaxes(-1,-2)
    whiteforce=np.concatenate((np.zeros((len(m),3,1)),covector),axis=2).transpose(0,2,1).reshape(4*len(m),3)
    force,force_rows=op.project(whiteforce)
    report=dict(kind='frozen geometry and stress; no material/time advancement or accepted impact state',state_sha256=sha,
        projected_npz_sha256=hashlib.sha256(a.projected.read_bytes()).hexdigest(),initial_contact=first,
        projected_force_diagnostics=force_rows,levels=[],numerical_heat_added_J=0.,production_admitted=False)
    kinetic=lambda y:float(.5*np.sum(y*y))
    for dt in (2e-11,1e-11,5e-12):
        kicked=start+dt*force;following,last=contact.project(kicked)
        midpoint=.5*(start+kicked);_,midW=op.unwhiten(midpoint);midL=midW@np.linalg.inv(F)
        work=float(dt*np.sum(s['volume'][:,None,None]*tau*midL))
        viscous=-J[:,None,None]*q[:,None,None]*np.eye(3)
        viscosity=float(dt*np.sum(s['volume'][:,None,None]*viscous*midL))
        residual=kinetic(following)-kinetic(initial)+first['kinetic_loss_J']+last['kinetic_loss_J']+work
        row=dict(dt_s=dt,final_contact=last,stress_work_J=work,viscosity_work_J=viscosity,
            stress_virtual_work_residual_J=float(work+dt*np.sum(midpoint*force)),
            coupled_work_residual_J=residual,kinetic_change_J=kinetic(following)-kinetic(initial))
        report['levels'].append(row)
        np.savez_compressed(a.output/f'frozen-{dt:.0e}.npz',weighted_initial=initial,weighted_first_contact=start,
            weighted_force=force,weighted_kicked=kicked,weighted_final=following,F=F,x=s['x'],mass=m,
            D0=E@E.swapaxes(-1,-2),reference_root=E,stress=tau,pressure=pressure,viscosity=q)
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        (Path(__file__),Path('tools/metal_affine_contact_numpy.py'),Path('tools/metal_affine_projection_numpy.py'))}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
