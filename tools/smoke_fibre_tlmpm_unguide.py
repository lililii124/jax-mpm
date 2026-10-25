"""One-step compatibility smoke for the legacy no-guide TL-MPM API."""
import json
import sys
from pathlib import Path

import jax
import numpy as np

from mpm_jax.fibre_mpm import build_layout
from mpm_jax.fibre_tlmpm import make_solver
from mpm_jax.materials.isotropic_brittle import isotropic_brittle


def main(output):
    x=np.array([[.002,.002,.002],[.008,.002,.002]],dtype=np.float64)
    volume=np.full(2,1e-9);ids=np.zeros(2,np.int32)
    layout=build_layout(x,ids,dx=.001,padding_cells=3)
    masks=(np.array([True,False]),np.array([False,True]))
    material=isotropic_brittle(young=1e7,poisson=.25,density=1500.,tensile_strength=2e6,
        fracture_energy=1000.,characteristic_length=.001)
    initial,step,_,_,_=make_solver(x,volume,layout,material,masks,1e-8,
        dict(load_time_s=1e-3,stroke_m=1e-5),duration=1e-8)
    state,row=step(initial);jax.block_until_ready(row['valid'])
    if not bool(state.valid) or not bool(row['strain_guard_passed']):
        raise RuntimeError('Legacy unguide TL step failed validity/strain guard')
    result=dict(state='passed',physical_steps=1,guide_operator_present=False,valid=bool(state.valid),
        momentum_error=float(row['momentum_error']),angular_momentum_error=float(row['angular_momentum_error']),
        kinetic_J=float(row['kinetic_J']),elastic_J=float(row['elastic_J']),work_J=float(row['work_J']),
        callbacks_absent=True,scope='single-step API compatibility only; no F1 long-run calculation')
    p=Path(output);p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/work/no-guide-smoke.json')
