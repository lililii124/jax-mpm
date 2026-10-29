"""Variational AT2 reference: sharp single-edge crack, quasi-static tension.

Independent quasi-static reference-grid MPM benchmark. Plane
strain, Miehe spectral tension/compression split, particle Gauss integration of
the quadratic degradation and crack density, and current tensile energy are
used. No history-field replacement, damage thresholding, artificial damping,
or strain-based removal enters the solver. Existing crack faces have separate
displacement DOFs; their common tip is sharp. SI units throughout.

Run: python -m examples.at2_sharp_crack_benchmark --n 64 --output outputs/at2-sharp-n64
Default d(0)=0: the split/cut geometric crack prescribes no unit damage.
Use --ny 49 with --n 48 for the true odd-grid cut geometry.
Use the SAME --ell, material, loading and crack length for mesh comparisons.
The default load schedule continues through fracture; a failed convergence or
small-strain check stops the run and never qualifies its prefix as complete.
"""
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def stationary_profile(n,ell,gc):
    """1D exact Neumann-end profile, with prescribed d(0)=1, on [-.5,.5]."""
    y=np.linspace(-.5,.5,n+1);h=1/n
    local=gc/ell*h*np.array([[2.,1.],[1.,2.]])/6+gc*ell*np.array([[1.,-1.],[-1.,1.]])/h
    rows=[];cols=[];values=[]
    for i in range(n):
        for a in range(2):
            for b in range(2):rows.append(i+a);cols.append(i+b);values.append(local[a,b])
    A=sparse.coo_matrix((values,(rows,cols)),shape=(n+1,n+1)).tocsr();free=np.arange(n+1)!=n//2
    d=np.zeros(n+1);d[n//2]=1;d[free]=spsolve(A[free][:,free],-A[free][:,~free].toarray().ravel())
    exact=np.cosh((.5-np.abs(y))/ell)/np.cosh(.5/ell)
    energy=.5*d@(A@d);analytical=gc*np.tanh(.5/ell)
    return {'n':n,'ell':ell,'h_over_ell':h/ell,'maximum_profile_error':float(abs(d-exact).max()),'energy_J_per_m':float(energy),'exact_energy_J_per_m':float(analytical),'relative_energy_error':float(abs(energy-analytical)/analytical)}


def run(args):
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise RuntimeError('Fresh output directory required')
    from mpm_jax.at2_quasistatic import Model
    reference_data=None
    if args.ny is not None and args.ny!=args.n:
        from mpm_jax.at2_cut_reference import odd_cut_reference_particles
        reference_data=odd_cut_reference_particles(args.n,args.ny,args.crack_length)
    model=Model(args.n,args.ell,args.young,args.poisson,args.gc,args.residual,args.crack_length,reference_data=reference_data,initial_damage_mode=args.initial_damage_mode)
    if args.ell*min(args.n,args.ny or args.n)<1.8:raise ValueError('Require ell/h>=1.8; qualify using fixed-ell mesh refinement')
    params=vars(args).copy();params.update(formulation='Reference-grid MPM; fixed Gauss material points; particle-side split linear tent support; plane-strain spectral AT2',damage_driving='current tensile energy',irreversibility='previous converged LOAD state <= d <= 1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mpm_module_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'mpm_jax/at2_quasistatic.py').read_bytes()).hexdigest())
    source_root=Path(__file__).resolve().parents[1]
    params['phase_field_module_sha256']=hashlib.sha256((source_root/'mpm_jax/phase_field.py').read_bytes()).hexdigest()
    params['competitor_module_sha256']=hashlib.sha256((source_root/'examples/at2_sharp_crack_competitors.py').read_bytes()).hexdigest()
    params['cut_reference_module_sha256']=(hashlib.sha256((source_root/'mpm_jax/at2_cut_reference.py').read_bytes()).hexdigest() if reference_data is not None else None)
    params['effective_ny']=args.ny or args.n
    params['initial_damage_prescribed_on_geometric_trace']=args.initial_damage_mode=='legacy-unit-seed'
    params['initial_damage_value']=0. if args.initial_damage_mode=='geometry-only' else 1.
    params['existing_geometric_crack_surface_energy_in_ledger']=False
    (folder/'parameters.json').write_text(json.dumps(params,indent=2))
    profile=(stationary_profile(args.n,args.ell,args.gc) if args.initial_damage_mode=='legacy-unit-seed' else {'physical_initial_damage':0.,'initial_AT2_energy':0.,'unit_profile_is_not_physical_initial_state':True})
    (folder/'stationary-profile.json').write_text(json.dumps(profile,indent=2))
    u=np.zeros(2*len(model.xy));lower=np.zeros(len(model.xy))
    if args.initial_damage_mode=='legacy-unit-seed':lower[model.damage_fixed]=1
    d,_,_=model.damage(np.zeros(len(model.nodes)),lower,lower)
    physical_d0_max=float(d.max());physical_initial_fracture=model.fracture(d)
    np.savez_compressed(folder/'zero-load-state.npz',displacement=np.zeros((len(model.xy),2)),damage=d,initial_damage_mode=args.initial_damage_mode)
    if reference_data is None:
        N=len(model.xy);display={'display_coordinates':model.coordinates,'display_nodes':np.c_[np.arange(N),np.zeros(N,int)],'display_weights':np.tile([1.,0.],(N,1)),'display_quads':model.nodes[::4]}
    else:display={key:reference_data[key] for key in ['display_coordinates','display_nodes','display_weights','display_quads']}
    np.savez_compressed(folder/'reference-grid.npz',coordinates=model.coordinates,particles=model.particles,nodes=model.nodes,weight=model.weight,gradient=model.grad,volume=model.volume,seed=model.damage_fixed,geometric_crack_trace=model.geometric_crack_trace,**display)
    records=[];initial_fracture=model.fracture(d);work=0.;previous_energy=initial_fracture;previous_reaction=0.;previous_load=0.;start=time.time()
    if args.restart:
        source=Path(args.restart)
        prior=json.loads((source/'parameters.json').read_text())
        for key in ['n','ell','young','poisson','gc','residual','crack_length','ny','initial_damage_mode','competitor_peak','energy_competitor','adaptive_energy']:
            if prior.get(key,False if key in ['energy_competitor','adaptive_energy'] else ('legacy-unit-seed' if key=='initial_damage_mode' else (1. if key=='competitor_peak' else None)))!=getattr(args,key):raise ValueError('Restart changes physical/discretization parameter: '+key)
        signature_keys=['source_sha256','mpm_module_sha256','phase_field_module_sha256','competitor_module_sha256','cut_reference_module_sha256']
        old_signature={key:prior.get(key) for key in signature_keys}
        if (source.parent/'source-manifest.json').exists():
            source_manifest=json.loads((source.parent/'source-manifest.json').read_text())
            for key,name in [('phase_field_module_sha256','mpm_jax/phase_field.py'),('competitor_module_sha256','examples/at2_sharp_crack_competitors.py'),('cut_reference_module_sha256','mpm_jax/at2_cut_reference.py')]:
                if key not in prior:old_signature[key]=source_manifest.get(name)
        current_signature={key:params[key] for key in signature_keys}
        changed_sources=[key for key in signature_keys if old_signature[key]!=current_signature[key]]
        if changed_sources and not (args.allow_source_migration and args.source_migration_reason):
            raise ValueError('Restart source signatures changed or are unknown: '+str(changed_sources)+'; explicit source migration and a reason are required')
        accepted=np.load(source/'checkpoint.npz')
        u=accepted['displacement'].ravel().copy();d=accepted['damage'].copy()
        if u.shape!=(2*len(model.xy),) or d.shape!=(len(model.xy),):raise ValueError('Restart topology mismatch')
        with (source/'history.csv').open(newline='') as f:
            for row in csv.DictReader(f):records.append({k:(int(v) if k in ['step','alternations'] else float(v)) for k,v in row.items()})
        for row in records:
            row.setdefault('selected_competitor',0);row.setdefault('competitor_energy_minus_warm',0.);row.setdefault('competitor_alternations',0)
        if not records or abs(float(accepted['accepted_load'])-records[-1]['displacement_m'])>1e-14:raise ValueError('Checkpoint/history accepted load mismatch')
        for row in records:
            frame=f"frame-{row['step']:04d}.npz"
            shutil.copy2(source/frame,folder/frame)
        initial_fracture=records[0]['initial_fracture_J_per_m'];work=records[-1]['external_work_J_per_m']
        previous_energy=records[-1]['total_energy_J_per_m'];previous_reaction=records[-1]['reaction_N_per_m'];previous_load=records[-1]['displacement_m']
        (folder/'restart-provenance.json').write_text(json.dumps({'source':str(source),'checkpoint_sha256':hashlib.sha256((source/'checkpoint.npz').read_bytes()).hexdigest(),'accepted_load':previous_load,'physical_lower_bound':'current accepted damage, never an unconverged trial','previous_source_signatures':old_signature,'current_source_signatures':current_signature,'changed_sources':changed_sources,'explicit_source_migration':bool(changed_sources),'source_migration_reason':args.source_migration_reason,'numerical_equivalence_claim':False,'previous_nominal_increments':prior['increments'],'new_nominal_increments':args.increments,'historical_load_schedule_unchanged':True},indent=2))
    frozen_lower=d.copy();old=d.copy();old_u=u.copy()
    if previous_load>args.displacement+1e-14:raise ValueError('Restart accepted load exceeds requested target')
    failure=None;rejections=[];pending=list(np.linspace(0,args.displacement,args.increments+1))
    if args.restart:pending=[v for v in pending if v>previous_load+1e-14]
    next_step=0 if not records else records[-1]['step']+1
    minimum_increment=args.displacement/args.increments/(2**args.max_load_refinements)
    try:
        for attempt,load in enumerate(pending):
            step=next_step;next_step+=1
            old=d.copy();old_u=u.copy();frozen_lower=old.copy();alternating=[]
            try:
                for stagger in range(args.max_alternations):
                    before=model.mechanical(u,d)[0]+model.fracture(d)
                    u,mechanical_error,newton=model.equilibrate(u,d,load)
                    elastic,force,plus,eps=model.mechanical(u,d)
                    dn,damage_error,damage_iterations=model.damage(plus,frozen_lower,d)
                    # Overrelaxed block Gauss-Seidel: for a convex quadratic
                    # damage block any omega in (0,2) descends before projection.
                    # Check the actual energy after bound projection; fall back to
                    # the exact block minimizer if roundoff/active sets spoil it.
                    candidate=np.clip(d+args.damage_relaxation*(dn-d),frozen_lower,1);candidate[model.seed]=1
                    block_before=model.mechanical(u,d)[0]+model.fracture(d)
                    candidate_energy=model.mechanical(u,candidate)[0]+model.fracture(candidate)
                    if candidate_energy<=block_before+max(abs(block_before)*1e-12,1e-18):dn=candidate
                    after=model.mechanical(u,dn)[0]+model.fracture(dn)
                    if stagger>0 and after>before+max(abs(before)*1e-8,1e-15):raise RuntimeError('Alternating energy increased at fixed loading')
                    change=float(abs(dn-d).max());d=dn
                    # Re-equilibrate on the UPDATED damage, then check the UPDATED
                    # current-energy damage KKT: both conditions must hold together.
                    u,mechanical_error,newton2=model.equilibrate(u,d,load)
                    elastic,force,plus,eps=model.mechanical(u,d)
                    trial,damage_error,_=model.damage(plus,frozen_lower,d)
                    fixed_point=float(abs(trial-d).max())
                    alternating.append({'iteration':stagger,'energy':elastic+model.fracture(d),'damage_change':change,'fixed_point':fixed_point,'mechanical_residual':mechanical_error,'trial_minimizer_kkt':damage_error,'newton_iterations':newton+newton2})
                    if stagger%25==0:
                        (folder/'current-attempt.json').write_text(json.dumps({'load':float(load),'physical_lower_bound_load':float(previous_load),**alternating[-1]},indent=2))
                        print(json.dumps({'attempted_load':float(load),**alternating[-1]}),flush=True)
                    if fixed_point<args.stagger_tolerance and mechanical_error<2e-7 and damage_error<2e-8:break
                else:
                    (folder/'failed-convergence.json').write_text(json.dumps({'attempted_load':float(load),'iterations':alternating},indent=2))
                    raise RuntimeError('Coupled alternating minimization did not converge')
                # Check true KKT at accepted d, rather than the residual of a different
                # trial minimizer. The projection uses the physical load-step bound.
                A=model.fracture_matrix+model.assemble_d(2*(1-model.residual)*plus[:,None,None]*model.mass)
                rhs=np.bincount(model.nodes.ravel(),weights=((2*(1-model.residual)*plus*model.volume)[:,None]*model.weight).ravel(),minlength=len(d))
                gradient=A@d-rhs;diag=A.diagonal();proj=d-np.clip(d-gradient/diag,frozen_lower,1);proj[model.seed]=0
                accepted_kkt=float(np.max(abs(proj*diag))/max(abs(rhs).max(),model.gc/model.ell/model.n**2,1e-20))
                if accepted_kkt>args.coupled_kkt:raise RuntimeError('Accepted coupled damage KKT too large: '+str(accepted_kkt))
            except RuntimeError as exc:
                gap=load-previous_load
                rejection={'attempt':step,'attempted_load':float(load),'previous_accepted_load':float(previous_load),'reason':str(exc),'iterations':alternating}
                rejections.append(rejection)
                (folder/'rejected-load-attempts.json').write_text(json.dumps(rejections,indent=2))
                if gap>minimum_increment*(1+1e-10) and records:
                    # Roll back BOTH fields; numerical iterates never become the
                    # physical irreversibility bound. Retry the smaller load,
                    # then the original target from the newly accepted state.
                    u=old_u;d=old
                    pending[attempt+1:attempt+1]=[previous_load+.5*gap,float(load)]
                    print(json.dumps({'rejected_load':float(load),'retry_load':previous_load+.5*gap,'reason':str(exc)}),flush=True)
                    continue
                raise
            selected_competitor=0;candidate_iterations=0;competitor_difference=0.
            if args.energy_competitor and load>0:
                from examples.at2_sharp_crack_competitors import solve as minimize_competitor
                warm_energy=elastic+model.fracture(d)
                extended=np.maximum(frozen_lower,args.competitor_peak*np.exp(-np.maximum(abs(model.xy[:,1]-.5)-1/model.n,0)/model.ell))
                branch={'load':float(load),'same_physical_lower_bound':True,'new_damage_coefficients_free':True,'plateau_initial_guess_only':True,'candidate_profile_peak':args.competitor_peak,'physical_initial_damage_mode':args.initial_damage_mode,'initial_guess_plateau_half_width_m':1/model.n,'warm_energy':warm_energy,'warm_damage_kkt':accepted_kkt}
                try:
                    cu,cd,review=minimize_competitor(model,old_u.copy(),extended,frozen_lower,load,args.max_alternations)
                    candidate_iterations=len(review['iterations']);competitor_difference=review['total_energy']-warm_energy
                    branch['competitor']=review
                    if review['converged_and_valid'] and review['total_energy']<warm_energy-max(abs(warm_energy)*1e-10,1e-15):
                        u=cu;d=cd;selected_competitor=1
                        elastic,force,plus,eps=model.mechanical(u,d)
                        accepted_kkt=review['damage_kkt'];mechanical_error=review['iterations'][-1]['mechanical_residual']
                except RuntimeError as exc:
                    branch['competitor_failure']=str(exc)
                branch['selected_competitor']=bool(selected_competitor)
                (folder/f'branches-{step:04d}.json').write_text(json.dumps(branch,indent=2))
            reaction=float(force[model.top].sum());fracture=model.fracture(d);total=elastic+fracture
            step_work=.5*(reaction+previous_reaction)*(load-previous_load)
            prospective_work=work+step_work
            energy_gap=total-initial_fracture-prospective_work
            if args.adaptive_energy and load>0 and abs(energy_gap)>args.energy_tolerance*max(abs(prospective_work),1e-20):
                gap=load-previous_load
                rejection={'attempt':step,'attempted_load':float(load),'previous_accepted_load':float(previous_load),'reason':'Refine measured external-work/total-energy difference','energy_difference':energy_gap,'prospective_work':prospective_work,'selected_competitor':bool(selected_competitor)}
                rejections.append(rejection);(folder/'rejected-load-attempts.json').write_text(json.dumps(rejections,indent=2))
                if gap>minimum_increment*(1+1e-10) and records:
                    u=old_u;d=old;pending[attempt+1:attempt+1]=[previous_load+.5*gap,float(load)]
                    print(json.dumps({'energy_refinement':rejection}),flush=True)
                    continue
                raise RuntimeError('Measured energy gap remains above tolerance at minimum load increment')
            work=prospective_work
            # No exemption of newly fractured cells from the small-strain check.
            max_strain=float(np.max(np.abs(np.linalg.eigvalsh(eps))))
            gradient=np.einsum('pki,pkj->pij',u.reshape(-1,2)[model.nodes],model.grad[:,:,:2])
            determinant=np.linalg.det(np.eye(2)+gradient)
            rotation=np.abs(np.arctan2(gradient[:,1,0]-gradient[:,0,1],2+gradient[:,0,0]+gradient[:,1,1]))
            max_rotation=float(np.rad2deg(rotation).max())
            if max_strain>args.strain_limit or max_rotation>1. or determinant.min()<=.5:raise RuntimeError('Small-strain/rotation/determinant validity limit exceeded: '+str((max_strain,max_rotation,determinant.min())))
            centroid=model.particles[:,:2];near=np.abs(centroid[:,1]-.5)<=model.ell
            high=(np.sum(model.weight*d[model.nodes],axis=1)>.8)&near
            advance=max(float(centroid[high,0].max())-model.crack_length,0.) if high.any() else 0.
            new_band=(centroid[:,0]>model.crack_length+model.ell)&(np.sum(model.weight*d[model.nodes],axis=1)>.5)
            width=float(np.sum(model.area[new_band])/max(advance-model.ell,1/model.n)) if new_band.any() else 0.
            row={'step':step,'selected_competitor':selected_competitor,'competitor_energy_minus_warm':competitor_difference,'competitor_alternations':candidate_iterations,'displacement_m':float(load),'reaction_N_per_m':reaction,'elastic_J_per_m':elastic,'fracture_J_per_m':fracture,'initial_fracture_J_per_m':initial_fracture,'total_energy_J_per_m':total,'external_work_J_per_m':work,'energy_minus_initial_minus_work_J_per_m':total-initial_fracture-work,'increment_energy_minus_work_J_per_m':total-previous_energy-step_work,'maximum_principal_strain':max_strain,'maximum_rotation_degrees':max_rotation,'minimum_det_F':float(determinant.min()),'crack_advance_d08_m':advance,'new_band_area_d05_over_advance_m':width,'maximum_damage':float(d.max()),'minimum_damage_increment':float((d-old).min()),'mechanical_residual':mechanical_error,'damage_kkt':accepted_kkt,'alternations':stagger+1,'wall_seconds':time.time()-start}
            records.append(row)
            np.savez_compressed(folder/f'frame-{step:04d}.npz',displacement=u.reshape(-1,2),damage=d)
            (folder/f'convergence-{step:04d}.json').write_text(json.dumps(alternating))
            with (folder/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(records)
            print(json.dumps(row),flush=True)
            previous_load=load;previous_reaction=reaction;previous_energy=total
            np.savez_compressed(folder/'.checkpoint-tmp.npz',displacement=u.reshape(-1,2),damage=d,previous_load_damage=frozen_lower,accepted_load=previous_load,accepted_step=step,completed=False)
            (folder/'.checkpoint-tmp.npz').replace(folder/'checkpoint.npz')
            if step==0:
                equivalence=model.operator_equivalence(plus,d)
                (folder/'shared-operator-equivalence.json').write_text(json.dumps({'relative_max_error':equivalence,'passed':equivalence<1e-12}))
                if equivalence>1e-12:raise RuntimeError('Shared MPM weak operator mismatch')
    except (RuntimeError,ValueError) as exc:
        failure=str(exc)
        np.savez_compressed(folder/'failed-attempt.npz',displacement=u.reshape(-1,2),damage=d,attempted_load=load,physical_lower_bound=frozen_lower)
        if records:
            accepted=np.load(folder/f"frame-{int(records[-1]['step']):04d}.npz");u=accepted['displacement'].ravel();d=accepted['damage']
        else:u=old_u;d=old
    np.savez_compressed(folder/'checkpoint.npz',xy=model.xy,triangles=model.tri,displacement=u.reshape(-1,2),damage=d,previous_load_damage=frozen_lower,accepted_load=previous_load,completed=failure is None)
    reactions=np.array([r['reaction_N_per_m'] for r in records]);peak=int(np.argmax(reactions)) if len(reactions) else 0
    checks={'full_load_schedule':bool(failure is None and records and abs(records[-1]['displacement_m']-args.displacement)<1e-14),'irreversible':all(r['minimum_damage_increment']>=-1e-12 for r in records),'small_strain':all(r['maximum_principal_strain']<=args.strain_limit for r in records),'load_drop':len(reactions)>peak+1 and reactions[-1]<.7*reactions[peak],'crack_advance':bool(records and records[-1]['crack_advance_d08_m']>.15),'initial_damage_protocol':(physical_d0_max==0. and physical_initial_fracture==0. if args.initial_damage_mode=='geometry-only' else profile['relative_energy_error']<.04),'mechanical_and_damage_converged':bool(records and max(r['damage_kkt'] for r in records)<=args.coupled_kkt)}
    energy_gap_fraction=(abs(records[-1]['energy_minus_initial_minus_work_J_per_m'])/max(abs(records[-1]['external_work_J_per_m']),1e-20)) if records else None
    checks['external_work_energy_balance']=bool(energy_gap_fraction is not None and energy_gap_fraction<=args.energy_tolerance)
    checks={key:bool(value) for key,value in checks.items()}
    summary={'initial_damage_mode':args.initial_damage_mode,'physical_initial_damage_max':physical_d0_max,'physical_initial_AT2_energy':physical_initial_fracture,'damage_fixed_nodes':int(model.damage_fixed.sum()),'energy_gap_over_external_work':energy_gap_fraction,'energy_tolerance':args.energy_tolerance,'completed':checks['full_load_schedule'],'checks':checks,'single_mesh_fracture_observed':all(checks.values()),'failure':failure,'accepted_load_steps':len(records),'nominal_load_steps':args.increments+1,'rejected_load_attempts':len(rejections),'minimum_allowed_load_increment':minimum_increment,'peak_reaction_N_per_m':float(reactions.max()) if len(reactions) else None,'peak_step':peak,'mesh_convergence_established':False,'load_increment_convergence_established':False,'energy_multistart':args.energy_competitor,'adaptive_energy_integration':args.adaptive_energy,'global_minimum_proven':False,'scope':'Small-strain reference-grid MPM; no equivalence claim to quadratic dynamic MPM, material calibration, load-increment or mesh convergence from one run.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=64);parser.add_argument('--ny',type=int,default=None,help='Optional odd vertical cell count with exact cut-cell/trace basis');parser.add_argument('--ell',type=float,default=.04)
    parser.add_argument('--young',type=float,default=1e6);parser.add_argument('--poisson',type=float,default=.2)
    parser.add_argument('--gc',type=float,default=.001);parser.add_argument('--residual',type=float,default=1e-7)
    parser.add_argument('--crack-length',type=float,default=.375);parser.add_argument('--displacement',type=float,default=.00006)
    parser.add_argument('--increments',type=int,default=80);parser.add_argument('--max-alternations',type=int,default=400)
    parser.add_argument('--initial-damage-mode',choices=['geometry-only','legacy-unit-seed'],default='geometry-only')
    parser.add_argument('--competitor-peak',type=float,default=.9)
    parser.add_argument('--energy-competitor',action='store_true',help='Minimize warm and freely relaxing extended initial guesses; select the lower converged energy')
    parser.add_argument('--adaptive-energy',action='store_true',help='Refine actual load increments when measured total-energy/external-work difference exceeds tolerance')
    parser.add_argument('--allow-source-migration',action='store_true')
    parser.add_argument('--source-migration-reason',default=None)
    parser.add_argument('--restart',default=None,help='Read an accepted checkpoint/history from this output directory; never write to it')
    parser.add_argument('--max-load-refinements',type=int,default=8)
    parser.add_argument('--damage-relaxation',type=float,default=1.6);parser.add_argument('--energy-tolerance',type=float,default=.05)
    parser.add_argument('--stagger-tolerance',type=float,default=1e-7);parser.add_argument('--coupled-kkt',type=float,default=2e-6)
    parser.add_argument('--strain-limit',type=float,default=.02);parser.add_argument('--output',default='outputs/at2-sharp-n64')
    args=parser.parse_args()
    if not 0<args.competitor_peak<1:parser.error('Candidate profile peak must be strictly below one')
    if not 0<args.damage_relaxation<2:parser.error('Require 0 < damage relaxation < 2')
    result=run(args);print(json.dumps(result,indent=2));return 0 if result['single_mesh_fracture_observed'] else 2


if __name__=='__main__':raise SystemExit(main())
