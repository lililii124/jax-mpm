"""Independent40ms dynamic pair; same geometry/material/BC for Hertz and JKR.

The100ms local reference is preserved separately. Quasistatic hold predictions
are screening only; actual dynamic gap/bending/kinetic response must pass review.
"""
import argparse
import hashlib
import json
import csv
from pathlib import Path

import numpy as np
from .fibre_adhesion_contrast import plan_case,run_case,digest

ROOT=Path(__file__).resolve().parents[1]


def plan(model):
    geometry,result=plan_case(model,duration=.04)
    bounds=[[[-.016,.016],[-.0015,.0015],[-.0015,.0015]],
            [[-.0015,.0015],[.0005,.0048],[-.016,.016]]]
    # The global origin, dx, particle sampling, true coordinates, material and
    # six-cell padding are unchanged. Only unused rectangular addressing shrinks.
    result['grid']['bounds_m']=bounds
    b=np.asarray(bounds);dx=result['grid']['dx_m'];shape=np.ceil(b[:,:,1]/dx).astype(int)-np.floor(b[:,:,0]/dx).astype(int)+13
    result['grid']['planned_rectangular_nodes']=int(np.prod(shape,axis=1).sum())
    result['case']='fibre-adhesion-pair-dynamic-40ms';result['series']='fibre-adhesion-pair-dynamic-v1'
    result['physical_common']['grid_storage_bounds_m']=bounds
    result['camera'].update(direction=[1.4,.2,1.8],elevation_coefficient=.2)
    result['common_request_sha256']=hashlib.sha256(json.dumps(result['physical_common'],sort_keys=True).encode()).hexdigest()
    result['grid']['support_checked_each_step']=True
    result['contact']['maximum_a_over_fibre_radius']=.2
    result['control_source']={'file':'examples/fibre_pair_dynamic_prefix.py','sha256':digest(Path(__file__))}
    result['interpretation']='40ms explicit dynamics, not a quasistatic strength test. Analytic0.464mm separation difference is a screening prediction only. Full actual MPM pair must demonstrate>0.3mm real central separation difference, actual stable attachment versus no adhesion, measured kinetic response, clamp work, spring/damper energy and no material damage. No per-model loading shifts or camera changes.'
    return geometry,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['hertz','jkr'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-after-frame',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args();geometry,request=plan(a.model)
    result=run_case(geometry,request,a.output/a.model,resume=a.resume,stop_after_frame=a.stop_after_frame)
    rows=list(csv.DictReader((a.output/a.model/'observables.csv').open()))
    start,end=request['protocol']['times_s'][3:5]
    actual=[]
    for row in rows:
        U=float(row['elastic_energy_J']);K=float(row['kinetic_energy_J'])
        actual.append({'time_s':float(row['time']),'kinetic_energy_J':K,'elastic_energy_J':U,'K_over_U':K/U if U>1e-30 else None,'total_actuator_input_work_J':float(row['actuator_work_J']),'actual_contact_work_J':float(row['actual_contact_work_J']),'combined_measured_input_J':float(row['actuator_work_J'])+float(row['actual_contact_work_J']),'spring_energy_J':float(row['spring_energy_J']),'damper_dissipation_J':float(row['damper_dissipation_J']),'actuator_y_m':float(row['actuator_displacement_y_m']),'actual_collar_y_m':float(row['actual_upper_collar_displacement_y_m']),'actual_centre_relative_y_m':float(row['upper_centre_displacement_y_m'])-float(row['lower_centre_displacement_y_m']),'fixture_force_N':float(row['fixture_force_N']),'contact_force_N':float(row['contact_force_N']),'contact_gap_m':float(row['contact_gap_m']),'contact_active':row['contact_active']=='True'})
    hold=[r for r in actual if start<=r['time_s']<=end]
    from .single_fibre_tension import save_json
    save_json(a.output/a.model/'dynamic-response-review.json',{'case':'independent40ms-dynamic-pair','actual_saved_response':actual,'hold_samples_available':bool(hold),'actual_hold_force_displacement':hold,'quasistatic_prediction_validated':False,'scope':'Measured MPM values only. An early f2 has not reached the hold; no hold force/displacement is invented. K/U is reported for explicit dynamics, not assumed negligible.'})
    if digest(Path(__file__))!=request['control_source']['sha256']:raise ValueError('Independent40ms control source changed during integration')
    return result


if __name__=='__main__':main()
