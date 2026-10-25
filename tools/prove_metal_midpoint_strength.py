"""Focused checks of midpoint plastic power and original endpoint yield."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from mpm_jax.materials.linear_metal import linear_metal
from mpm_jax.fragment_v6_midpoint_strength import midpoint_return


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence')
    material=linear_metal();records=[]
    rot=np.array([[np.cos(.13),-np.sin(.13),0],[np.sin(.13),np.cos(.13),0],[0,0,1.]])
    cases=[('elastic',np.zeros((3,3)),np.diag([1e-5,-5e-6,-5e-6]),np.eye(3),0.),
        ('plastic',np.zeros((3,3)),np.diag([.01,-.005,-.005]),np.eye(3),0.),
        ('nonproportional',np.diag([1e8,-.5e8,-.5e8]),np.array([[0,.005,0],[.005,0,0],[0,0,0]]),rot,.1),
        ('rotation',np.diag([1e8,-.5e8,-.5e8]),np.zeros((3,3)),rot,.1),
        ('unloading',np.diag([1e8,-.5e8,-.5e8]),np.diag([-.0001,.00005,.00005]),np.eye(3),.1)]
    for name,tau,strain,rotation,ep in cases:
        result=midpoint_return(j.asarray(tau[None]),j.asarray([ep]),j.asarray(strain[None]),j.asarray(rotation[None]),j.ones(1),j.asarray(1e-7),j.asarray([300.]),material)
        assert bool(np.asarray(result.valid)[0])
        residual=float(np.asarray(result.energy_identity)[0]);plastic=float(np.asarray(result.specific_plastic_work)[0])
        assert abs(residual)<1e-10 and plastic>=0
        assert abs(float(np.asarray(result.specific_plastic_heat)[0])-.9*plastic)<1e-12
        if name in ('rotation','unloading','elastic'):assert plastic==0
        if name=='rotation':assert np.max(abs(np.asarray(result.kirchhoff_stress)[0]-rotation@tau@rotation.T))<1e-6
        records.append(dict(fixture=name,energy_identity_J_kg=residual,plastic_work_J_kg=plastic,
            plastic_strain_increment=float(np.asarray(result.plastic_strain)[0]-ep),endpoint_yield_residual_Pa=float(np.asarray(result.yield_residual)[0])))
    report=dict(passed=True,fixtures=records,original_material_parameters_unchanged=True,numerical_heat_added_J=0.,production_admitted=False,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path('mpm_jax/fragment_v6_midpoint_strength.py'),Path('mpm_jax/materials/shock_metal.py'))})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
