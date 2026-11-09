"""Annulus-sensitivity diagnostics for saved cohesive crack-tip fields."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.crack_tip import fit_tip_fields,mts_angle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    summary=args.run/'summary.json'
    info=json.loads((summary if summary.exists() else args.run/'input.json').read_text())
    all_rows=list(csv.DictReader((args.run/'history.csv').open()))
    history=[]
    for i,row in enumerate(all_rows):
        if row['valid']!='True' or not (args.run/f'frame-{i:03d}.npz').exists():
            break
        history.append(row)
    if not history:
        raise ValueError('No valid saved states')
    growth=next((i for i,r in enumerate(history) if int(r['vertices'])>2),len(history)-1)
    selected=sorted(set([max(growth-1,0),growth,min(growth+1,len(history)-1),len(history)-1]))
    E,nu=info['young_Pa'],info['poisson']
    mu,lam=E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu))
    area=np.prod(np.array(info['upper_m'])-info['lower_m'])
    fitted=jax.jit(fit_tip_fields)
    results=[]
    for frame in selected:
        with np.load(args.run/f'frame-{frame:03d}.npz') as data:
            x,F,path=data['x'],data['F'],.5*(data['plus']+data['minus'])
        J=np.linalg.det(F)
        stress=(mu*(F@F.swapaxes(-1,-2)-np.eye(2))+lam*np.log(J)[:,None,None]*np.eye(2))/J[:,None,None]
        direction=path[-1]-path[-2];direction/=np.linalg.norm(direction)
        for radius in (.001,.0015,.002,.0025):
            offset=x-path[-1]
            r2=np.sum(offset*offset,axis=1)/radius**2
            weights=np.where((offset@direction>=0)&(r2<1),(1-r2)**2,0.)
            average=np.einsum('p,pij->ij',weights,stress)/max(weights.sum(),1e-100)
            normal=np.array([-direction[1],direction[0]])
            fit=fitted(jnp.asarray(x),jnp.asarray(stress),jnp.full(len(x),area/len(x)),
                       jnp.asarray(path[-1]),jnp.asarray(direction),radius)
            results.append(dict(frame=frame,time_s=float(history[frame]['time_s']),radius_m=radius,
                                mode_I_Pa_sqrt_m=float(fit.mode_I),mode_II_Pa_sqrt_m=float(fit.mode_II),
                                T_Pa=float(fit.T),relative_residual=float(fit.relative_residual),condition=float(fit.condition),
                                forward_average_tangent_Pa=float(direction@average@direction),
                                forward_average_normal_Pa=float(normal@average@normal),
                                forward_average_shear_Pa=float(direction@average@normal),
                                valid_fit_matrix=bool(fit.valid),diagnostic_mts_relative_angle_deg=float(np.rad2deg(float(mts_angle(fit.mode_I,fit.mode_II)))),
                                current_last_segment_angle_deg=float(np.rad2deg(np.arctan2(direction[1],direction[0])))))
    length=E/(1-nu*nu)*info['law']['energy']/info['law']['strength']**2
    report=dict(run=args.run.name,completed=summary.exists(),valid_saved_frames=len(history),
                excluded_history_rows=len(all_rows)-len(history),nominal_plane_strain_cohesive_length_m=length,
                annulus_inner_radius_fraction=.4,rows=results,
                interpretation='Elastic-field diagnostics only; no LEFM-dominant annulus or MTS steering validation established.',
                driver_source_sha256=info['source_sha256'],
                diagnostic_source_sha256={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in ('tools/diagnose_extrinsic_tip.py','mpm_jax/crack_tip.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
