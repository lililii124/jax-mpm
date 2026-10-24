"""Independent real-prefix audit for the common Hertz/JKR fixture pair."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.fibre_adhesion_contrast import descriptor,source_record,save_json,digest


def audit(folder, producer_root):
    summary=json.loads((folder/'summary.json').read_text());plan=summary['definition'];rows=list(csv.DictReader((folder/'observables.csv').open()))
    control=plan.get('control_source')
    if control is not None and digest(producer_root/control['file'])!=control['sha256']:raise ValueError('Actual40ms physical control source differs')
    if any(digest(producer_root/n)!=h for n,h in summary['source']['files'].items()):raise ValueError('Exact producer source snapshot differs')
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp,np.load(folder/'final_state_0.npz',allow_pickle=False) as final:
        physical_time=float(final['time']);frame=int(cp['frame']);tables=json.loads(str(cp['tables']))
        if frame!=60 or len(rows)!=61 or int(cp['progress_0'])!=plan['frame_steps'][frame]:raise ValueError('Actual complete formal full61 required')
        if rows!=[{k:str(v) for k,v in row.items()} for row in tables['rows']]:raise ValueError('CSV/checkpoint disagreement')
        for name in final.files:
            field=name if name.startswith('history_') else 'state_'+name
            if not np.array_equal(final[name],cp[field]):raise ValueError('Final/continuation state mismatch')
        for name in ('x','damage','failed','temperature','speed'):
            a=np.load(folder/f'trajectory_{name}.npy',mmap_mode='r')
            for i in range(61):
                if descriptor(a[i])!=tables['records'][i][name]:raise ValueError('Changed actual saved-state raw bytes')
        F,D=final['F'],final['history_damage'];material=plan['material'];E,nu=material['young'],material['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));u,s,v=np.linalg.svd(F);rotation=u@v;strain=s-1;tr=strain.sum(1)
        expected=(1-D)[:,None,None]*(2*mu*F+(lam*tr-2*mu)[:,None,None]*rotation)@F.transpose(0,2,1)
        stress_error=float(np.max(np.abs(expected-final['tau'])));energy=float(np.sum(final['volume']*(1-D)*(.5*lam*tr**2+mu*np.sum(strain**2,axis=1))))
        momenta=[];budgets=[]
        for body in (0,1):
            mask=final['fibre_id']==body;momentum=np.sum(final['mass'][mask,None]*final['v'][mask],axis=0);boundary=cp['progress_1'][body];contact=cp['progress_4'][body];fixture=np.array([0.,cp['progress_5'][body],0.]);error=float(np.max(np.abs(momentum-boundary-contact-fixture)))
            scale=float(np.sum(np.abs(final['mass'][mask,None]*final['v'][mask]))+np.sum(np.abs(boundary))+np.sum(np.abs(contact))+np.sum(np.abs(fixture)));budget=1e-9*max(scale,1e-30);momenta.append(error);budgets.append(budget)
        checks={'actual_formal_particles':len(final['x'])==202240,'actual_f2_clock':np.isclose(float(final['time']),plan['frame_steps'][60]*plan['dt'],rtol=1e-11,atol=0),'independent_final_material_stress':stress_error<1e-6,'independent_final_stored_energy':abs(energy-float(rows[-1]['elastic_energy_J']))<1e-10,'positive_orientation':bool(np.all(np.linalg.det(F)>0)),'material_undamaged':bool(not D.any() and not final['history_failed'].any()),'per_body_momentum':all(a<=b for a,b in zip(momenta,budgets)),'free_upper_y_rigid_xz_constraints':max(float(r['grip_position_error_m']) for r in rows)<1e-9,'local_contact_valid':all(r['contact_valid']=='True' for r in rows),'mass_conserved':max(abs(float(r['mass_relative_drift'])) for r in rows)<1e-12}
    checks['complete_solver_summary']=bool(summary['completed'] and summary['steps']==summary['target_steps']==29212)
    checks['solver_numerical_admission']=bool(summary['numerical_comparison_qualified'] and all(summary['checks'].values()))
    checks={name:bool(value) for name,value in checks.items()}
    return {'schema':'fibre-pair-actual-full61-independent-audit-v1','passed':all(checks.values()),'model':plan['model'],'common_request_sha256':plan['common_request_sha256'],'checks':checks,'frame':frame,'saved_states':61,'accepted_steps':plan['frame_steps'][frame],'physical_time_s':physical_time,'stress_error_Pa':stress_error,'stored_energy_error_J':abs(energy-float(rows[-1]['elastic_energy_J'])),'per_body_momentum_error_Ns':momenta,'per_body_momentum_budget_Ns':budgets,'last':rows[-1],'complete_comparison':False,'predicted_hold_contrast_validated':False,'scope':'Actual full61 final material/impulse and every saved state. Pair hold contrast and fixed-camera admission remain separate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--producer-source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.resolve().is_relative_to(a.folder.resolve()):raise ValueError('Read-only original data: separate review output required')
    r=audit(a.folder,a.producer_source);save_json(a.output,r);print(json.dumps(r))
    if not r['passed']:raise SystemExit('Actual full61 did not pass independent audit')

if __name__=='__main__':main()
