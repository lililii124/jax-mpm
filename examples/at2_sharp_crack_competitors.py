"""Diagnose metastable AT2 minima at ONE frozen load, without a new load path.

Both guesses use exactly the same previous accepted damage as the irreversible
lower bound. The extended analytic crack profile is ONLY an optimizer initial
guess: no added damage trace is prescribed and every new coefficient may heal
back to the common physical lower bound during minimization. The output is
diagnostic and never replaces a checkpoint in the source load path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def solve(model, u, initial, lower, load, maxiter, callback=None):
    d=initial.copy();history=[]
    for it in range(maxiter):
        u,mechanical,_=model.equilibrate(u,d,load)
        elastic,force,plus,eps=model.mechanical(u,d)
        exact,_,_=model.damage(plus,lower,d)
        candidate=np.clip(d+1.6*(exact-d),lower,1);candidate[model.seed]=1
        old_energy=elastic+model.fracture(d)
        if model.mechanical(u,candidate)[0]+model.fracture(candidate)>old_energy+1e-15:candidate=exact
        d=candidate;u,mechanical,_=model.equilibrate(u,d,load)
        elastic,force,plus,eps=model.mechanical(u,d)
        trial,_,_=model.damage(plus,lower,d);fixed_point=float(np.max(abs(trial-d)))
        history.append({'iteration':it,'energy':elastic+model.fracture(d),'fixed_point':fixed_point,'mechanical_residual':mechanical})
        if callback is not None:callback(it,u,d,history[-1])
        if it%25==0:print(json.dumps(history[-1]),flush=True)
        if fixed_point<1e-7 and mechanical<2e-7:break
    A=model.fracture_matrix+model.assemble_d(2*(1-model.residual)*plus[:,None,None]*model.mass)
    rhs=np.asarray(model.rhs(model.jnp.asarray(plus)));diag=A.diagonal();grad=A@d-rhs
    projected=d-np.clip(d-grad/diag,lower,1);projected[model.seed]=0
    kkt=float(np.max(abs(projected*diag))/max(np.max(abs(rhs)),model.gc/model.ell/model.n**2,1e-20))
    G=np.einsum('pki,pkj->pij',u.reshape(-1,2)[model.nodes],model.grad[:,:,:2]);det=np.linalg.det(np.eye(2)+G)
    rotation=np.rad2deg(abs(np.arctan2(G[:,1,0]-G[:,0,1],2+G[:,0,0]+G[:,1,1])))
    strain=float(abs(np.linalg.eigvalsh(eps)).max())
    checks={'mechanical':bool(mechanical<2e-7),'damage':bool(kkt<=2e-6),'fixed_point':bool(fixed_point<1e-7),'irreversibility':bool((d-lower).min()>=-1e-12),'small_strain':strain<=.02,'rotation':bool(rotation.max()<=1),'det_F':bool(det.min()>.5)}
    return u,d,{'converged_and_valid':all(checks.values()),'checks':checks,'elastic':elastic,'fracture':model.fracture(d),'total_energy':elastic+model.fracture(d),'reaction':float(force[model.top].sum()),'maximum_principal_strain':strain,'damage_kkt':kkt,'iterations':history}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--load',type=float,required=True);p.add_argument('--output',required=True);p.add_argument('--maxiter',type=int,default=500);p.add_argument('--initial-plateau-cells',type=float,default=0.);p.add_argument('--initial-damage-mode',choices=['geometry-only','legacy-unit-seed'],default='geometry-only');p.add_argument('--competitor-peak',type=float,default=.9);args=p.parse_args()
    if not 0<args.competitor_peak<1:p.error('Candidate profile peak must be strictly below one')
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    parameters=json.loads((source/'parameters.json').read_text())
    from mpm_jax.at2_quasistatic import Model
    reference_data=None
    if parameters.get('ny') is not None and parameters['ny']!=parameters['n']:
        from mpm_jax.at2_cut_reference import odd_cut_reference_particles
        reference_data=odd_cut_reference_particles(parameters['n'],parameters['ny'],parameters['crack_length'])
    model=Model(initial_damage_mode=args.initial_damage_mode,reference_data=reference_data,**{key:parameters[key] for key in ['n','ell','young','poisson','gc','residual','crack_length']})
    source_mode=parameters.get('initial_damage_mode','legacy-unit-seed')
    if source_mode!=args.initial_damage_mode:raise ValueError('Cannot reinterpret a checkpoint from another physical initial-damage mode')
    frame=source/('frame-%04d.npz'%args.frame);accepted=np.load(frame)
    u=accepted['displacement'].ravel();lower=accepted['damage'];extended=np.maximum(lower,args.competitor_peak*np.exp(-np.maximum(abs(model.xy[:,1]-.5)-args.initial_plateau_cells/model.n,0)/model.ell))
    result={'source':str(source),'previous_accepted_frame':args.frame,'source_frame_sha256':hashlib.sha256(frame.read_bytes()).hexdigest(),'frozen_load':args.load,'same_physical_lower_bound':True,'initial_damage_mode':args.initial_damage_mode,'candidate_profile_peak':args.competitor_peak,'extended_profile_is_initial_guess_only':True,'initial_guess_plateau_half_width_m':args.initial_plateau_cells/model.n,'load_path_recomputed':False,'source_checkpoint_modified':False,'branches':{}}
    for label,guess in [('warm',lower),('extended',extended)]:
        un,dn,review=solve(model,u.copy(),guess,lower,args.load,args.maxiter)
        result['branches'][label]=review;np.savez_compressed(out/(label+'.npz'),displacement=un.reshape(-1,2),damage=dn,physical_lower_bound=lower)
        (out/'review.json').write_text(json.dumps(result,indent=2))
    a=result['branches']['warm'];b=result['branches']['extended']
    result['extended_minus_warm_energy']=b['total_energy']-a['total_energy']
    result['lower_competitor_found']=a['converged_and_valid'] and b['converged_and_valid'] and b['total_energy']<a['total_energy']-1e-10
    result['scope']='Frozen-load metastability diagnostic only; no global-minimum proof, no accepted continuation path and no manufactured balancing energy.'
    (out/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps({key:value for key,value in result.items() if key!='branches'},indent=2))


if __name__=='__main__':main()
