"""One bounded material/drift stage, isolated so JAX memory is released."""
import argparse,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
import jax.numpy as j
from mpm_jax.materials.linear_metal import linear_metal
from tools.prove_metal_affine_material_window125 import update


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve stage')
    s=dict(np.load(a.request,allow_pickle=False));dt=float(s.pop('candidate_requested_dt_s'))
    if not 0<dt<=2e-11:raise ValueError('Bounded dt required')
    material=linear_metal(**json.loads(a.admission.read_text())['material'])
    before={k:s[k] for k in ('F','tau','plastic_strain','cold_work','fracture_energy','damage','internal_energy')}
    result,work,plastic,extra,valid,parts=update(*(j.asarray(v) for v in before.values()),j.asarray(s['W']),j.asarray(dt),material,midpoint_strength=True)
    for name,value in zip(before,result):s[name]=np.asarray(value)
    s['x']=s['x']+dt*s['v'];s['time']=s['time']+dt
    J=np.linalg.det(s['F']);factor=np.maximum(1-s['damage'],1e-12)
    elastic=np.sum(s['tau']**2,axis=(-2,-1))/(4*material.shear_modulus*material.density*factor)
    volume=np.where(J>1,1-s['damage'],1)*material.bulk_modulus*(1-J)**2/(2*material.density)
    heat=s['internal_energy']-s['cold_work']-s['fracture_energy']-elastic-volume
    temperature=material.reference_temperature+heat/material.heat_capacity
    entropy=material.heat_capacity*np.log(temperature/material.reference_temperature)
    # The caller saves the exact old entropy; no inference from evolved damage.
    previous_entropy=s.pop('candidate_previous_entropy')
    s['minimum_entropy']=np.minimum(s['minimum_entropy'],entropy)
    s['negative_entropy_increments']=s['negative_entropy_increments']+np.minimum(entropy-previous_entropy,0)
    s['maximum_temperature']=np.maximum(s['maximum_temperature'],temperature)
    s['minimum_J']=np.minimum(s['minimum_J'],J);s['maximum_J']=np.maximum(s['maximum_J'],J)
    s['ever_phase_extrapolated']=s['ever_phase_extrapolated']|(temperature>=material.melt_temperature)
    s['ever_tension_extrapolated']=s['ever_tension_extrapolated']|(J>1)
    E=s['reference_root'];D0=E@E.swapaxes(-1,-2)
    s['C']=s['W']@D0@s['F'].swapaxes(-1,-2)*(4/.00025**2);s['C_transfer']=s['C']
    s['L_physical']=s['W']@np.linalg.inv(s['F'])
    s.update(candidate_material_valid=np.asarray(valid),candidate_material_work_density=np.asarray(work),
        candidate_plastic_heat=np.asarray(plastic),candidate_return_extra_dissipation=np.asarray(extra),
        candidate_volume_work_density=np.asarray(parts[0]),candidate_volume_internal_increment=np.asarray(parts[1]),
        candidate_deviatoric_internal_increment=np.asarray(parts[2]))
    np.savez_compressed(a.output,**s)
    if not bool(valid):raise ValueError('Original material/yield/domain gate failed; raw stage preserved')
    print(json.dumps(dict(material_valid=True,dt_s=dt,min_J=float(J.min()),max_temperature=float(temperature.max()))),flush=True)


if __name__=='__main__':main()
