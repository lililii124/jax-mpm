"""Independent NumPy audit of every preserved prescribed-material substep."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    initial=dict(np.load(a.state,allow_pickle=False));report=json.loads((a.folder/'audit.json').read_text());params=report['material_parameters']
    if hashlib.sha256(a.state.read_bytes()).hexdigest()!=report['initial_state_sha256']:raise ValueError('Wrong initial state')
    m=initial['mass'];rho=params['density'];K=params['bulk_modulus'];G=params['shear_modulus']
    elastic=lambda s:np.sum(s['tau']**2,axis=(-2,-1))/(4*G*rho*np.maximum(1-s['damage'],1e-12))
    volume=lambda s:np.where(np.linalg.det(s['F'])>1,1-s['damage'],1)*K*(1-np.linalg.det(s['F']))**2/(2*rho)
    heat=lambda s:s['internal_energy']-s['cold_work']-s['fracture_energy']-elastic(s)-volume(s)
    audits=[]
    for level in report['levels']:
        previous=initial;steps=level['steps'];work=0.;max_heat=max_cold=max_geometry=0.
        W=np.load(a.folder/f'prescribed-{steps}-final.npz',allow_pickle=False)['W']
        for index in range(1,steps+1):
            path=a.folder/f'prescribed-{steps}-step-{index}.npz';s=dict(np.load(path,allow_pickle=False))
            if not bool(s['valid']) or not all(np.all(np.isfinite(v)) for v in s.values()):raise ValueError('Invalid raw material field')
            if np.any(s['plastic_strain']<previous['plastic_strain']) or np.any(s['damage']<previous['damage']):raise ValueError('Irreversibility failed')
            J=np.linalg.det(s['F'])
            if np.any((J<params['minimum_J'])|(J>params['maximum_J'])):raise ValueError('Original volume domain failed')
            heat_error=float(np.sum(m*abs(heat(s)-heat(previous)-s['plastic_heat_J_kg'])))
            cold_error=float(np.sum(m*abs(s['cold_work']-previous['cold_work']-s['plastic_heat_J_kg']*(1-params['heat_fraction'])/params['heat_fraction'])))
            geometry=float(np.max(abs(s['F']-previous['F']-float(s['dt_s'])*W)))
            if heat_error>1e-12 or cold_error>1e-12 or geometry>1e-14:raise ValueError('Independent material/geometry identity failed')
            max_heat=max(max_heat,heat_error);max_cold=max(max_cold,cold_error);max_geometry=max(max_geometry,geometry)
            work+=float(np.sum(initial['volume']*s['stress_work_density_J_m3']))
            previous=s
        change=float(np.sum(m*(previous['internal_energy']-initial['internal_energy'])))
        if abs(work-level['trace'][-1]['work_J'])>1e-14 or abs(change-level['trace'][-1]['internal_change_J'])>1e-14:raise ValueError('Raw trace differs')
        audits.append(dict(steps=steps,calorimetry_weighted_L1_max_J=max_heat,cold_work_weighted_L1_max_J=max_cold,
            affine_path_residual_max=max_geometry,raw_work_J=work,raw_internal_change_J=change,raw_material_work_difference_J=work-change))
    result=dict(passed=True,levels=audits,production_admitted=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
