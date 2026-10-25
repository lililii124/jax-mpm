"""Own-method three-level temporal sensitivity, never old-APIC shape fitting."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(root):
    states=[];reports=[];diagnostics=[];origins=[]
    for divisor,cfl in ((1,.1),(2,.05),(4,.025)):
        folder=Path(root)/f'dt{divisor}'
        report=json.loads((folder/'result.json').read_text())
        if report.get('bounded_diagnostic_passed') is not True or report['cfl']!=cfl:
            raise ValueError('Every bounded physical diagnostic must pass before comparison')
        with np.load(folder/'candidate-final.npz',allow_pickle=False) as data:
            state={name:data[name].copy() for name in data.files}
        with np.load(folder/'step-diagnostics.npz',allow_pickle=False) as data:
            trace={name:data[name].copy() for name in data.files}
        if float(state['time'])!=135000*1e-12 or np.any(trace['dt_s']<=0) or abs(trace['dt_s'].sum()-1e-8)>1e-18:
            raise ValueError('Common exact physical interval and positive trace required')
        source=json.loads((folder/'source-runtime.json').read_text())
        origins.append((source['original_seed_sha256'],source['source_admission']))
        states.append(state);reports.append(report);diagnostics.append(trace)
    if not all(origin==origins[0] for origin in origins):raise ValueError('Seed/method changed between levels')
    mass=states[0]['mass'];weight=mass/mass.sum()
    if not all(np.array_equal(s['mass'],mass) for s in states):raise ValueError('Mass/population changed')
    fields={}
    for name in ('x','v','C_transfer','L_physical','F','J','plastic_strain','damage'):
        values=[np.linalg.det(s['F']) if name=='J' else s[name] for s in states]
        errors=[]
        for a,b in zip(values,values[1:]):
            difference=b-a;flat=difference.reshape(len(mass),-1)
            errors.append(dict(linf=float(abs(difference).max()),mass_weighted_L1=float(np.dot(weight,np.linalg.norm(flat,axis=1))),
                mass_weighted_L2=float(np.sqrt(np.dot(weight,np.sum(flat*flat,axis=1))))))
        scale=max(float(np.max(abs(value))) for value in values)
        floor=64*np.finfo(float).eps*max(scale,1e-30)
        contracted={norm:bool(errors[1][norm]<errors[0][norm] or max(errors[1][norm],errors[0][norm])<=floor) for norm in errors[0]}
        fields[name]=dict(successive_errors=errors,roundoff_floor=floor,contracts_or_roundoff=contracted)
    channels={}
    names=('p2g_dissipation','g2p_dissipation','contact_compatible_energy','viscosity_work','constitutive_work_defect',
        'numerical_energy_increment','geometry_grid_physical_work_mismatch_J','geometry_material_work_remainder_J')
    for name in names:
        sums=np.array([trace[name].sum() for trace in diagnostics]);changes=np.diff(sums)
        floor=64*np.finfo(float).eps*max(float(abs(sums).max()),1e-30)
        channels[name]=dict(signed_J=sums.tolist(),successive_differences_J=changes.tolist(),
            contracts_or_roundoff=bool(abs(changes[1])<abs(changes[0]) or abs(changes).max()<=floor))
    all_fields=all(all(field['contracts_or_roundoff'].values()) for field in fields.values())
    return dict(comparison_complete=True,own_CFL=[.1,.05,.025],steps=[report['steps'] for report in reports],
        all_physical_diagnostics_passed=True,fields=fields,channels=channels,all_reported_field_norms_contract_or_roundoff=all_fields,
        interpretation='Own-method temporal sensitivity only. Noncontraction remains unresolved. No Richardson order, solution accuracy budget, spatial convergence or mesh-objective fragmentation is inferred.',
        old_APIC_shape_envelope_used_as_acceptance=False,production_restart_allowed=False,public_media_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    result=compare(args.root);target=args.root/'comparison.json'
    if target.exists():raise FileExistsError('Preserve earlier comparison')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
