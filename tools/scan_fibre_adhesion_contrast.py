"""Same-parameter local adhesion scan before any new fibre MPM comparison.

Finite-ring BEM snap thresholds require mesh convergence. Large differences at
small Tabor number do not qualify JKR as a quantitative peer comparison.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parameters(mu):
    E=.8e6;nu=.25;R=.001;W=.08;reduced=E/(2*(1-nu**2))
    interaction=(R*W*W/(reduced*reduced*mu**3))**(1/3)
    a0=(9*np.pi*W*R*R/(2*reduced))**(1/3)
    return {'young_Pa':E,'poisson':nu,'effective_modulus_Pa':reduced,'radius_m':R,'work_of_adhesion_J_per_m2':W,'interaction_range_m':interaction,'Tabor_mu':mu,'zero_load_JKR_radius_m':a0,'zero_load_a_over_R':a0/R,'outer_radius_m':.0003}


def scan(mu,rings,points,output,models=('hertz','jkr','md','self-consistent'),max_sweeps=2048):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.local_contact_models import local_contact_model,initial_local_contact_history,update_local_contact
    p=parameters(mu);a0=p['zero_load_JKR_radius_m'];R=p['radius_m'];scale=a0*a0/R
    # Identical physical indentation history, including approach, indentation,
    # hold, fine unloading through snap and a final separated interval.
    approach=np.linspace(-2*p['interaction_range_m'],.2*scale,max(12,points//4))
    unload=np.linspace(.2*scale,-.65*scale,points)
    path=np.r_[approach,np.repeat(approach[-1],4),unload,np.repeat(unload[-1],4)]
    output.mkdir(parents=True,exist_ok=True);results={}
    for model in models:
        law=local_contact_model(model,effective_modulus=p['effective_modulus_Pa'],effective_radius=R,work_of_adhesion=p['work_of_adhesion_J_per_m2'],interaction_range=p['interaction_range_m'],outer_radius=p['outer_radius_m'],rings=rings)
        history=initial_local_contact_history(law)
        @jax.jit
        def march(h):
            def one(old,indent):
                new,response=update_local_contact(old,indent,law,max_sweeps=max_sweeps,tolerance=1e-8)
                diagnostic=jnp.array([response.details.equilibrium_residual,response.details.complementarity_residual,response.details.energy_increase,response.details.sweeps,response.details.domain_clear,response.details.converged]) if model=='self-consistent' else jnp.zeros(6)
                return new,(response.normal_force,response.radius,response.potential,response.active,response.valid,diagnostic)
            return jax.lax.scan(one,h,jnp.asarray(path))
        started=time.monotonic();_,values=march(history);values=jax.device_get(values);elapsed=time.monotonic()-started
        force,radius,energy,active,valid,diagnostic=map(np.asarray,values)
        first=len(approach)+4;attached=np.flatnonzero(active[first:]);detach=first+attached[-1]+1 if len(attached) and attached[-1]+first+1<len(path) else None
        spring={}
        for stiffness in (.25,.5,1.,2.):
            load=force[first:first+points];delta=path[first:first+points];drive=-delta-load/stiffness
            # Stable displacement-control branch requires k + dP/ddelta >0.
            stable=stiffness+np.gradient(load,delta)>0
            candidates=np.flatnonzero((~active[first:first+points])|~stable)
            index=int(candidates[0]) if len(candidates) else None
            spring[str(stiffness)]={'combined_stability_exit_index':index,'combined_stability_exit_drive_m':float(drive[index]) if index is not None else None}
        record={'model':model,'all_local_solves_valid':bool(valid.all()),'first_invalid_index':int(np.flatnonzero(~valid)[0]) if not valid.all() else None,'minimum_force_N':float(force.min()),'maximum_radius_m':float(radius.max()),'maximum_a_over_R':float(radius.max()/R),'intrinsic_detachment_indentation_m':float(path[detach]) if detach is not None else None,'signed_contact_path_work_J':float(np.trapezoid(force,path)),'seconds_including_compile':elapsed,'common_series_stiffness':spring}
        if not valid.all():record['first_invalid_diagnostics']={k:float(v) for k,v in zip(('equilibrium_residual','compatibility_residual','energy_increase_J','sweeps','domain_clear','converged'),diagnostic[record['first_invalid_index']])}
        record['max_sweeps']=max_sweeps
        with (output/(model+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['index','indentation_m','force_N','radius_m','potential_J','active','valid']);writer.writerows(zip(range(len(path)),path,force,radius,energy,active,valid))
        results[model]=record;print(json.dumps(record),flush=True)
    report={'schema':'fibre-adhesion-contrast-local-scan-v1','parameters':p,'rings':rings,'points':len(path),'models':results,'JKR_asymptotic_screen':mu>=5,'all_models_valid':set(results)=={'hertz','jkr','md','self-consistent'} and all(r['all_local_solves_valid'] for r in results.values()),'selected_models_valid':all(r['all_local_solves_valid'] for r in results.values()),'public_comparison_qualified':False,'scope':'Representative local contact branches only; no fibre geometry, actuator dynamics, visible separation time or continuum BEM snap convergence is established. Hertz is a nonadhesive baseline; MD/SC use the same W/range and JKR is screened at Tabor>=5.'}
    (output/'scan.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mu',type=float,default=5.);p.add_argument('--rings',type=int,default=128);p.add_argument('--points',type=int,default=129);p.add_argument('--models',nargs='+',choices=['hertz','jkr','md','self-consistent'],default=['hertz','jkr','md','self-consistent']);p.add_argument('--max-sweeps',type=int,default=2048);p.add_argument('--output',type=Path,default=ROOT/'.local/fibre-adhesion-contrast-scan');a=p.parse_args();scan(a.mu,a.rings,a.points,a.output,a.models,a.max_sweeps)


if __name__=='__main__':main()
