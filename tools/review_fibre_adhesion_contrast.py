"""Fail-closed same-path, same-camera review for all4 new pull-off experiments."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.fibre_adhesion_contrast import MODELS,descriptor,source_record,save_json


def review(root,models=MODELS):
    summaries={};rows={};renders={};checks={};per_model={}
    for model in models:
        folder=Path(root)/model;summary=json.loads((folder/'summary.json').read_text());summaries[model]=summary
        if not summary['completed'] or summary['steps']!=summary['target_steps'] or summary['definition']['model']!=model:raise ValueError('All4 exact models require the complete physical plan')
        if summary['source']!=source_record():raise ValueError('Numerical source differs from current frozen experiment')
        table=list(csv.DictReader((folder/'observables.csv').open()));rows[model]=table
        with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp,np.load(folder/'final_state_0.npz',allow_pickle=False) as final:
            tables=json.loads(str(cp['tables']))
            if len(table)!=61 or len(tables['records'])!=61 or int(cp['frame'])!=60:raise ValueError('Exactly61 real saved states required')
            if table!=[{k:str(v) for k,v in r.items()} for r in tables['rows']]:raise ValueError('Saved observables differ from checkpoint')
            for field in ('x','damage','speed','failed','temperature'):
                a=np.load(folder/f'trajectory_{field}.npy',mmap_mode='r')
                for i in range(61):
                    if descriptor(a[i])!=tables['records'][i][field]:raise ValueError('Changed complete raw saved state')
            for field in final.files:
                leaf=field if field.startswith('history_') else 'state_'+field
                if not np.array_equal(final[field],cp[leaf]):raise ValueError('Final state differs from complete checkpoint')
            F=final['F'];D=final['history_damage'];p=summary['definition']['material'];E=p['young'];nu=p['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));u,s,vt=np.linalg.svd(F);rot=u@vt;strain=s-1;tr=strain.sum(1)
            tau=(1-D)[:,None,None]*(2*mu*F+(lam*tr-2*mu)[:,None,None]*rot)@F.transpose(0,2,1)
            stored=float(np.sum(final['volume']*(1-D)*(.5*lam*tr**2+mu*np.sum(strain**2,axis=1))))
            stress_error=float(np.max(np.abs(tau-final['tau'])));energy_error=abs(stored-float(table[-1]['elastic_energy_J']))
            checks[model+'_independent_material']=stress_error<1e-6 and energy_error<1e-10
            checks[model+'_no_material_failure']=not final['history_failed'].any() and not D.any()
        render=json.loads((folder/'render.json').read_text());renders[model]=render
        from PIL import Image
        with Image.open(folder/'simulation.gif') as gif:gif_frames=gif.n_frames
        checks[model+'_actual_paraview']=render['renderer']=='ParaView' and render['frames_rendered']==61 and gif_frames==61 and render['field']=='family_id' and not render['axes'] and not render['annotations'] and render.get('geometry_scale',1)==1
        checks[model+'_numerical_admission']=summary['numerical_comparison_qualified'] and all(summary['checks'].values())
        start=summary['definition']['protocol']['times_s'][2]
        detached=[float(r['time']) for i,r in enumerate(table) if i and float(r['time'])>=start and table[i-1]['contact_active']=='True' and r['contact_active']=='False']
        per_model[model]={'minimum_contact_force_N':min(float(r['contact_force_N']) for r in table),'maximum_contact_radius_m':max(float(r['contact_radius_m']) for r in table),'unloading_detachment_saved_times_s':detached,'final_actuator_work_J':float(table[-1]['actuator_work_J']),'final_actual_contact_work_J':float(table[-1]['actual_contact_work_J']),'final_damper_dissipation_J':float(table[-1]['damper_dissipation_J']),'maximum_material_damage':max(float(r['max_damage']) for r in table),'independent_final_stress_error_Pa':stress_error,'independent_final_energy_error_J':energy_error}
    common={s['common_request_sha256'] for s in summaries.values()};checks['identical_geometry_material_fixture_loading_and_camera']=len(common)==1
    reference=next(iter(rows.values()));checks['identical_saved_clock']=all([r['time'] for r in table]==[r['time'] for r in reference] for table in rows.values())
    checks['identical_actual_camera_direction']=len({json.dumps([r['camera_position'],r['camera_focal_point'],r['camera_parallel_scale']]) for r in renders.values()})==1
    checks['fixed_camera_all4']=len({json.dumps(r['camera_bounds']) for r in renders.values()})==1 and all(r['camera_bounds']==summaries[m]['definition']['camera']['focus_bounds_m'] for m,r in renders.items())
    contrasts={}
    for a,b in itertools.combinations(models,2):
        difference=[];centre_difference=[];active_difference=0;visible_hold=0
        for x,y in zip(rows[a],rows[b]):
            bx=float(x['upper_centre_displacement_y_m'])-float(x['actual_upper_collar_displacement_y_m'])-float(x['lower_centre_displacement_y_m']);by=float(y['upper_centre_displacement_y_m'])-float(y['actual_upper_collar_displacement_y_m'])-float(y['lower_centre_displacement_y_m'])
            difference.append(abs(bx-by));active_difference+=x['contact_active']!=y['contact_active']
            separation=abs((float(x['upper_centre_displacement_y_m'])-float(x['lower_centre_displacement_y_m']))-(float(y['upper_centre_displacement_y_m'])-float(y['lower_centre_displacement_y_m'])))
            centre_difference.append(separation)
            first,last=summaries[a]['definition']['protocol']['times_s'][3:5]
            if first<=float(x['time'])<=last and separation>.0003 and a=='hertz' and b=='jkr' and x['contact_active']=='False' and y['contact_active']=='True':visible_hold+=1
        contrasts[a+'/'+b]={'maximum_actual_centre_separation_difference_m':max(centre_difference),'hold_frames_with_Hertz_open_JKR_attached_and_actual_gap_difference_gt0_3mm':visible_hold,'maximum_real_bending_difference_m':max(difference),'saved_frames_with_different_attachment':active_difference,'visible_difference_gate':max(difference)>=.00005 and active_difference>=3}
    if tuple(models)==('hertz','jkr'):checks['actual_dynamic_pair_hold_contrast_gt0_3mm']=contrasts['hertz/jkr']['hold_frames_with_Hertz_open_JKR_attached_and_actual_gap_difference_gt0_3mm']>=3
    else:checks['adhesive_pairs_genuinely_visible']=all(contrasts[a+'/'+b]['visible_difference_gate'] for a,b in itertools.combinations(('jkr','md','self-consistent'),2))
    return {'schema':'fibre-adhesion-contrast-review-v1','passed':all(checks.values()),'checks':checks,'per_model':per_model,'pairwise_physical_contrast':contrasts,'production_eligible':False,'public_visual_review_passed':False,
            'scope':'Complete saved raw-state hashes, final material reconstruction, identical physical protocol/fixed ParaView camera and quantitative geometry/attachment differences. A human must still review actual frames. Finite-ring SC mesh/domain qualification is required separately; no full-time numerical reintegration is implied.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--pair',action='store_true');a=p.parse_args();r=review(a.root,('hertz','jkr') if a.pair else MODELS);save_json(a.root/'contrast-review.json',r);print(json.dumps(r))
    if not r['passed']:raise SystemExit('Four-model comparison has not passed admission')


if __name__=='__main__':main()
