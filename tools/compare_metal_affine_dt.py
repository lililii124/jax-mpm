"""Compare physical arrays for the fixed original 2 ps window; no replay gate."""
import argparse,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np


def load(folder):
    report=json.loads((folder/'result.json').read_text())
    if report['status']!='bounded-window-passed':raise ValueError('Only accepted windows are comparable')
    path=sorted(folder.glob('*-accepted.npz'))[-1]
    state=dict(np.load(path,allow_pickle=False))
    if report['initial_sha256']!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Different initial value problem')
    if abs(float(state['time'])-125.002e-9)>64*np.finfo(float).eps*125.002e-9:raise ValueError('Different physical endpoint')
    return state,report
def norms(values,mass):
    values=values.reshape(len(mass),-1);magnitude=np.linalg.norm(values,axis=1)
    return dict(L1=float(np.sum(mass*magnitude)/mass.sum()),L2=float(np.sqrt(np.sum(mass*magnitude*magnitude)/mass.sum())),Linf=float(np.max(magnitude)))
def channels(report):
    initial=dict(grid_projection=0.,boundary_projection=0.,contact_projection=0.,viscosity=0.)
    for stage in report['stages']:
        if stage['stage']=='initial-space-boundary':
            initial['grid_projection']=stage['projection']['grid_projection_loss_J'];initial['boundary_projection']=stage['projection']['boundary_projection_loss_J']
        elif stage['stage']=='initial-impact-contact':initial['contact_projection']=stage['derived_projection_loss_J']
    dynamic={key:float(report['ledger'][key])-initial[key] for key in initial}
    return dict(initial_zero_time_J=initial,dynamic_increment_J=dynamic,
        energy_residual_J=report['steps'][-1]['global_energy_residual_J'],
        material_work_remainder_J=sum(s['material_work_remainder_J'] for s in report['steps']),
        force_time_work_remainder_J=sum(s['force_time_work_remainder_J'] for s in report['steps']),
        contact_force_time_work_J=sum(s['contact_force_time_work_J'] for s in report['steps']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--dt1',type=Path,required=True);p.add_argument('--dt05',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior comparison')
    cells=[load(folder) for folder in (a.coarse,a.dt1,a.dt05)];mass=cells[0][0]['mass'];fields={}
    for state,_ in cells:
        if not np.array_equal(state['mass'],mass) or not np.array_equal(state['reference'],cells[0][0]['reference']):raise ValueError('Different particles/reference mass')
    selected=('x','v','W','C_transfer','L_physical','F','plastic_strain','damage','tau','internal_energy')
    for key in (*selected,'J'):
        values=[np.linalg.det(s['F']) if key=='J' else s[key] for s,_ in cells]
        differences=[norms(values[j]-values[j+1],mass) for j in (0,1)]
        sizes=[norms(v,mass) for v in values];result={}
        for metric in ('L1','L2','Linf'):
            first,second=differences[0][metric],differences[1][metric]
            roundoff=float(64*np.finfo(float).eps*max(s[metric] for s in sizes))
            result[metric]=dict(dt2_minus_dt1=first,dt1_minus_dt05=second,ratio=second/first if first else None,
                contracts=second<=first,roundoff_indistinguishable=max(first,second)<=roundoff,roundoff_scale=roundoff)
        fields[key]=result
    rows=[]
    for state,report in cells:
        contact=[stage['contact'] for stage in report['stages'] if 'contact' in stage]
        rows.append(dict(dt_ps=report['dt_s']*1e12,steps=report['accepted_steps'],time_ns=float(state['time'])*1e9,
            max_contact_closing_m_s=max(c['original_sample_closing_m_s'] for c in contact),
            max_contact_complementarity_m_s=max(c['complementarity_m_s'] for c in contact),
            minimum_CFL_limit_s=min(s['diagnostics']['minimum_cfl_limit_s'] for s in report['steps']),
            min_J=float(np.linalg.det(state['F']).min()),max_damage=float(state['damage'].max()),channels=channels(report)))
    contraction={metric:all(data[metric]['contracts'] or data[metric]['roundoff_indistinguishable'] for data in fields.values()) for metric in ('L1','L2','Linf')}
    result=dict(scope='Independent same-initial-value time-step comparison over125 to125.002ns',cells=rows,physical_field_errors=fields,
        all_selected_fields_contract_or_roundoff=contraction,
        norm_definition='Per-particle Euclidean/Frobenius differences; reference-mass weighted L1/L2; maximum particle norm for Linf',
        diagnostic_counters_excluded=True,new_fine_cell_replays_required=False,
        coarse_hardware_note='Coarse original step ran on Actions; finer cells run on assigned Xeon host. Same exact versions/dev-client and input; no cross-host bitwise claim.',
        no_fitted_solution_error_tolerance=True,production_admitted=False,public_media_allowed=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cells=rows,contraction=contraction)))


if __name__=='__main__':main()
