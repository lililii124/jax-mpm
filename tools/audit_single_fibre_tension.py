"""Read-only independent admission of saved single-fibre tension state and history."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.single_fibre_tension import descriptor,fracture_review,plan_case,save_json


def audit(folder):
    folder=Path(folder);geometry,plan=plan_case();summary=json.loads((folder/'summary.json').read_text());status=json.loads((folder/'status.json').read_text())
    if summary['definition']!=plan:raise ValueError('Source or physical plan changed')
    rows=list(csv.DictReader((folder/'observables.csv').open()));frame=status['frame']
    if len(rows)!=frame+1 or summary['steps']!=plan['frame_steps'][frame]:raise ValueError('Incomplete saved clock/history')
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp,np.load(folder/'final_state_0.npz',allow_pickle=False) as final:
        tables=json.loads(str(cp['tables']))
        if len(tables['records'])!=frame+1 or int(cp['frame'])!=frame:raise ValueError('Checkpoint saved-frame tables disagree')
        for k in final.files:
            name=k if k.startswith('history_') else 'state_'+k
            if not np.array_equal(final[k],cp[name]):raise ValueError('Final state differs from checkpoint')
        if rows != [{k:str(v) for k,v in row.items()} for row in tables['rows']]:raise ValueError('CSV differs from bound checkpoint tables')
        expected_times=np.asarray(plan['frame_steps'][:frame+1])*plan['dt']
        if not np.allclose(tables['times'],expected_times,rtol=1e-12,atol=1e-14) or float(cp['state_time'])!=tables['times'][-1]:raise ValueError('Physical saved clock changed')
        for name in ('x','damage','failed','speed','temperature'):
            array=np.load(folder/f'trajectory_{name}.npy',mmap_mode='r')
            if len(array)!=plan['saved_states']:raise ValueError('Full raw output plan changed')
            previous=None
            for i in range(frame+1):
                if descriptor(array[i])!=tables['records'][i][name] or not np.isfinite(array[i]).all():raise ValueError('Raw saved state bytes changed')
                if name in ('damage','failed') and previous is not None and np.any(array[i]<previous):raise ValueError('Irreversible history healed')
                if name in ('damage','failed'):previous=np.array(array[i])
            expected={'x':final['x'],'damage':final['history_damage'],'failed':final['history_failed'].astype(np.uint8),'speed':np.linalg.norm(final['v'],axis=1),'temperature':np.full(plan['particles'],293.15,dtype=np.float32)}[name]
            if not np.array_equal(array[frame],expected):raise ValueError('Raw endpoint differs from final state')
        if not np.array_equal(final['mass'],geometry['mass']) or not np.array_equal(final['volume'],geometry['volume']):raise ValueError('Mass/reference volume changed')
        F,D,kappa=final['F'],final['history_damage'],final['history_kappa'];E=plan['material']['young'];nu=plan['material']['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));left,s,right=np.linalg.svd(F);rotation=left@right;strain=s-1;tr=strain.sum(1)
        P=2*mu*F+(lam*tr-2*mu)[:,None,None]*rotation;tau=(1-D)[:,None,None]*P@F.transpose(0,2,1)
        stored=float(np.sum(final['volume']*(1-D)*(.5*lam*tr**2+mu*np.sum(strain**2,axis=1))))
        delta=plan['grid']['dx_m']*kappa;d0=plan['softening']['onset_displacement_m'];df=plan['softening']['failure_displacement_m'];damage=np.clip(df*(delta-d0)/(np.maximum(delta,d0)*(df-d0)),0,1)
        kinetic=.5*float(np.sum(final['mass']*np.sum(final['v']**2,axis=1)))+plan['grid']['dx_m']**2/8*float(np.sum(final['mass']*np.sum(final['C']**2,axis=(1,2))))
        impulse=np.sum(cp['progress_1'],axis=0);momentum=np.sum(final['mass'][:,None]*final['v'],axis=0)
        fracture=fracture_review(geometry,final['x'],final['history_failed'])
        stress_error=float(np.max(np.abs(tau-final['tau'])));energy_error=abs(stored-float(rows[-1]['elastic_energy_J']));momentum_error=float(np.max(np.abs(momentum-impulse)))
        checks={'positive_orientation':bool(np.all(np.linalg.det(F)>0)),'finite_state':all(bool(np.isfinite(final[n]).all()) for n in final.files),'independent_damage_envelope':bool(np.allclose(damage,D,rtol=0,atol=1e-12)),
                'independent_stress':stress_error<1e-6,'independent_stored_energy':energy_error<1e-10,
                'independent_kinetic_energy':abs(kinetic-float(rows[-1]['kinetic_energy_J']))<1e-10,'independent_momentum':momentum_error<1e-10,
                'failure_flags_match_damage':bool(np.array_equal(final['history_failed'],D>=1)),
                'fracture_matches_saved_review':fracture==summary['fracture_review']}
        if not all(checks.values()):raise ValueError('Independent material/state audit failed: '+str(checks))
    complete=frame==plan['frames']
    peak=max(float(r['tensile_load_N']) for r in rows);tail=max(abs(float(r['tensile_load_N'])) for r in rows[-5:]);scale=max(max(abs(float(r['fixture_work_J'])) for r in rows),1e-20)
    ledgers=[float(r['fixture_work_J'])-float(r['kinetic_energy_J'])-float(r['elastic_energy_J'])-float(r['damage_dissipation_J']) for r in rows]
    admission={'full_plan_completed':complete,'mass_conserved':max(abs(float(r['mass_relative_drift'])) for r in rows)<1e-12,'positive_orientation':min(float(r['min_J']) for r in rows)>0,
               'momentum_closure':max(float(r['momentum_residual_Ns']) for r in rows)<1e-10,'formal_grips':max(float(r['grip_position_error_m']) for r in rows)<1e-9,
               'nonnegative_damage_dissipation':all(float(r['damage_dissipation_J'])>=0 for r in rows),'energy_ledger_within15percent_input_work':max(map(abs,ledgers))<=.15*scale,
               'cross_section_fully_failed':bool(fracture['fully_failed_axial_sections']),'axial_path_broken':not fracture['intact_reference_path_between_grips'],
               'positive_fragment_gap':fracture['measured_intact_fragment_gap_m']>3*plan['grid']['dx_m'],'load_drop90percent':peak>0 and tail<.1*peak}
    if admission!=summary['checks']:raise ValueError('Independent fracture/force/energy gates disagree')
    passed=all(admission.values()) and all(checks.values())
    return {'schema':'single-fibre-tension-independent-review-v1','prefix_verified':True,'full_protocol_completed':complete,'numerical_fracture_accepted':passed,
            'frame':frame,'saved_states_checked':frame+1,'particles':plan['particles'],'checks':checks,'fracture_admission_checks':admission,'stress_error_Pa':stress_error,'stored_energy_error_J':energy_error,'momentum_error_Ns':momentum_error,
            'fracture':fracture,'public_visual_review_passed':False,'production_eligible':False,
            'scope':'Independent final SVD stress/stored energy/damage envelope/kinetic/momentum reconstruction; every saved raw array hash checked. Constitutive loss is accumulated online, with the declared energy residual gate; not an independent all-step reintegration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args();r=audit(a.folder);save_json(a.folder/'independent-review.json',r);print(json.dumps(r))


if __name__=='__main__':main()
