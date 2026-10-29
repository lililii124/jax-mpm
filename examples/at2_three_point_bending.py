"""Current-energy variational AT2, explicit notch, zero initial damage, 3PB."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--displacement',type=float,default=1.2e-4);ap.add_argument('--increments',type=int,default=24);ap.add_argument('--maxiter',type=int,default=400);args=ap.parse_args()
    from mpm_jax.at2_bending import BendingModel
    from examples.at2_sharp_crack_competitors import solve
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False);model=BendingModel();data=model.reference_data
    params={**vars(args),**model.boundary_description,'initial_damage_mode':'geometry-only','physical_initial_damage_max':0.,'gc':model.gc,'ell':model.ell,'young':model.young,'poisson':model.poisson,'residual':model.residual,'n':40,'ny':129,'crack_length':model.notch,'geometry_verification':model.geometry_verification,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'parameters.json').write_text(json.dumps(params,indent=2));(folder/'material-case.json').write_text(json.dumps(model.boundary_description,indent=2))
    np.savez_compressed(folder/'reference-grid.npz',coordinates=model.coordinates,particles=model.particles,nodes=model.nodes,weight=model.weight,gradient=model.grad,volume=model.volume,seed=model.damage_fixed,geometric_crack_trace=model.geometric_crack_trace,**{k:data[k] for k in ['display_coordinates','display_nodes','display_weights','display_quads']})
    np.savez_compressed(folder/'material-fields.npz',particle_inclusion=np.zeros(len(model.nodes),bool),particle_modulus_ratio=np.ones(len(model.nodes)),particle_gc_ratio=np.ones(len(model.nodes)),crack_metric=np.eye(2))
    u=np.zeros(2*len(model.xy));d=np.zeros(len(model.xy));np.savez_compressed(folder/'zero-load-state.npz',displacement=u.reshape(-1,2),damage=d)
    # One meaningful geometry/energy/tangent verification before continuation.
    probe=np.c_[2e-5*model.xy[:,0]+1e-5*model.xy[:,1],4e-5*model.xy[:,1]].ravel();pd=.2*np.exp(-abs(model.xy[:,0]-.5)/model.ell);direction=np.random.default_rng(935).normal(size=u.shape)*1e-6;h=1e-4
    E,F,plus,eps,K=model.mechanical(probe,pd,True);Ep,Fp,*_=model.mechanical(probe+h*direction,pd);Em,Fm,*_=model.mechanical(probe-h*direction,pd)
    verification={**model.geometry_verification,'energy_gradient_relative_error':float(abs((Ep-Em)/(2*h)-F@direction)/max(abs(F@direction),1e-20)),'tangent_relative_error':float(abs((Fp-Fm)/(2*h)-K@direction).max()/max(abs(K@direction).max(),1e-20)),'damage_operator_relative_error':model.operator_equivalence(plus,pd)}
    assert max(verification[k] for k in ['energy_gradient_relative_error','tangent_relative_error','damage_operator_relative_error'])<1e-6,verification
    (folder/'kernel-verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps({'verification':verification}),flush=True)
    pending=list(np.linspace(0,args.displacement,args.increments+1));records=[];rejects=[];work=0.;previous_load=0.;previous_reaction=0.;previous_energy=0.;attempt=0;start=time.time();failure=None
    minimum=args.displacement/args.increments/256
    while pending:
        load=float(pending.pop(0));step=attempt;attempt+=1;lower=d.copy();old_u=u.copy()
        try:
            wu,wd,warm=solve(model,old_u.copy(),d.copy(),lower,-load,args.maxiter)
            if not warm['converged_and_valid']:raise RuntimeError('Warm branch did not pass true simultaneous force/KKT/validity checks')
            cu,cd,chosen=wu,wd,warm;selected=False;branch={'load':load,'same_physical_lower_bound':True,'warm':warm,'candidate_guess_only':True,'candidate_profile_peak':.9,'new_damage_coefficients_free':True}
            if load>0:
                guess=np.maximum(lower,.9*np.exp(-np.maximum(abs(model.xy[:,0]-.5)-model.hx,0)/model.ell))
                try:
                    eu,ed,extended=solve(model,old_u.copy(),guess,lower,-load,args.maxiter);branch['extended']=extended
                    if extended['converged_and_valid'] and extended['total_energy']<warm['total_energy']-max(abs(warm['total_energy'])*1e-10,1e-15):cu,cd,chosen=eu,ed,extended;selected=True
                except RuntimeError as exc:branch['candidate_failure']=str(exc)
            branch['selected_competitor']=selected;(folder/f'branches-{step:04d}.json').write_text(json.dumps(branch,indent=2))
            elastic,force,plus,eps=model.mechanical(cu,cd);fracture=model.fracture(cd);total=elastic+fracture;reaction=-float(force[model.top].sum());step_work=.5*(reaction+previous_reaction)*(load-previous_load);prospective=work+step_work;gap=total-prospective
            if load>0 and abs(gap)>.05*max(abs(prospective),1e-20):raise RuntimeError('Measured external-work/energy difference requires actual load refinement')
        except RuntimeError as exc:
            rejected={'attempt':step,'attempted_downward_travel':load,'physical_lower_bound_load':previous_load,'reason':str(exc)};rejects.append(rejected);(folder/'rejected-load-attempts.json').write_text(json.dumps(rejects,indent=2));print(json.dumps({'rejected':rejected}),flush=True)
            if records and load-previous_load>minimum*(1+1e-10):pending[:0]=[.5*(load+previous_load),load];continue
            failure=str(exc);break
        u=cu;d=cd;work=prospective;grad=np.einsum('pki,pkj->pij',u.reshape(-1,2)[model.nodes],model.grad[:,:,:2]);det=np.linalg.det(np.eye(2)+grad);rotation=np.rad2deg(abs(np.arctan2(grad[:,1,0]-grad[:,0,1],2+grad[:,0,0]+grad[:,1,1])))
        dp=np.sum(model.weight*d[model.nodes],axis=1);near=(abs(model.particles[:,0]-.5)<=2*model.ell)&(dp>.8);advance=max(float(model.particles[near,1].max())-model.notch,0.) if near.any() else 0.
        row={'step':step,'displacement_m':load,'head_displacement_y_m':-load,'reaction_N_per_m':reaction,'elastic_J_per_m':elastic,'fracture_J_per_m':fracture,'total_energy_J_per_m':total,'external_work_J_per_m':work,'energy_minus_work_J_per_m':gap,'increment_energy_minus_work_J_per_m':total-previous_energy-step_work,'maximum_principal_strain':float(abs(np.linalg.eigvalsh(eps)).max()),'maximum_rotation_degrees':float(rotation.max()),'minimum_det_F':float(det.min()),'crack_advance_d08_m':advance,'maximum_damage':float(d.max()),'minimum_damage_increment':float((d-lower).min()),'mechanical_residual':chosen['iterations'][-1]['mechanical_residual'],'damage_kkt':chosen['damage_kkt'],'selected_competitor':int(selected),'alternations':len(chosen['iterations']),'wall_seconds':time.time()-start}
        records.append(row);np.savez_compressed(folder/f'frame-{step:04d}.npz',displacement=u.reshape(-1,2),damage=d)
        with (folder/'history.csv').open('w',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(records)
        np.savez_compressed(folder/'checkpoint.npz',displacement=u.reshape(-1,2),damage=d,previous_load_damage=lower,accepted_load=load,completed=False)
        previous_load=load;previous_reaction=reaction;previous_energy=total;print(json.dumps(row),flush=True)
    completed=bool(failure is None and records and abs(records[-1]['displacement_m']-args.displacement)<1e-14);reactions=np.array([r['reaction_N_per_m'] for r in records]);peak=int(reactions.argmax()) if len(reactions) else 0
    summary={'completed':completed,'failure':failure,'physical_initial_damage_max':0.,'physical_initial_AT2_energy':0.,'damage_fixed_nodes':0,'accepted_load_steps':len(records),'rejected_load_attempts':len(rejects),'energy_gap_over_external_work':abs(records[-1]['energy_minus_work_J_per_m'])/max(abs(work),1e-20) if records else None,'irreversible':all(r['minimum_damage_increment']>=-1e-12 for r in records),'mechanical_and_damage_converged':all(r['mechanical_residual']<2e-7 and r['damage_kkt']<=2e-6 for r in records),'small_strain':all(r['maximum_principal_strain']<=.02 and r['maximum_rotation_degrees']<=1 and r['minimum_det_F']>.5 for r in records),'peak_reaction_N_per_m':float(reactions.max()) if len(reactions) else None,'load_drop':bool(len(reactions)>peak+1 and reactions[-1]<.7*reactions[peak]),'crack_advance_d08_m':records[-1]['crack_advance_d08_m'] if records else 0.,'mesh_convergence_established':False,'load_increment_convergence_established':False,'scope':'Actual small-strain three-point bending with a geometric bottom notch; no preseeded damage or prescribed propagation path.'}
    summary['single_mesh_fracture_observed']=bool(completed and summary['load_drop'] and summary['crack_advance_d08_m']>.05)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(folder/'checkpoint.npz',displacement=u.reshape(-1,2),damage=d,accepted_load=previous_load,completed=completed);print(json.dumps(summary,indent=2));return 0 if completed else 2

if __name__=='__main__':raise SystemExit(main())
