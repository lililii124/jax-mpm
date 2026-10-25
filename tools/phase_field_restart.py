"""Source-bound restart of the exported AT2 state and its cumulative ledgers."""
import hashlib
import json
from pathlib import Path
import shutil

import jax.numpy as jnp
import numpy as np

from mpm_jax.phase_field import State

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ('mpm_jax/phase_field.py', 'mpm_jax/phase_field_diagnostics.py',
           'mpm_jax/fibre_mpm.py', 'mpm_jax/materials/isotropic_brittle.py')


def load(parent, destination, x, basis, parameters, coordinates):
    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    if parent == destination or any(destination.iterdir()):
        raise RuntimeError('Restart requires a fresh directory distinct from its parent')
    summary = json.loads((parent/'summary.json').read_text())
    provenance = json.loads((parent/'source-runtime.json').read_text())
    for name in BACKEND:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != provenance['files'][name]:
            raise RuntimeError('Restart backend differs from source-bound parent: '+name)
    for key, value in parameters.items():
        if summary[key] != value:
            raise RuntimeError('Restart parameter differs: '+key)
    with np.load(parent/'particles.npz') as particles, np.load(parent/'nodal_state.npz') as nodes:
        for file in (particles, nodes):
            if not all(np.isfinite(file[key]).all() for key in file.files):
                raise RuntimeError('Nonfinite restart state')
        for key, current in [('basis_nodes',basis.nodes), ('basis_weight',basis.weight),
                             ('basis_gradient',basis.gradient), ('coordinates',coordinates)]:
            if not np.array_equal(nodes[key], np.asarray(current)):
                raise RuntimeError('Restart reference basis differs: '+key)
        if not np.array_equal(particles['reference'],x):
            raise RuntimeError('Restart reference geometry differs')
        reconstructed = 'displacement' not in particles.files
        displacement = particles['position']-x if reconstructed else particles['displacement']
        roundtrip = float(np.max(np.abs(x+displacement-particles['position'])))
        bound = float(np.max(np.spacing(np.abs(particles['position']))+np.spacing(np.abs(x))))
        state = State(jnp.asarray(displacement), jnp.asarray(particles['velocity']),
            jnp.asarray(particles['strain']), jnp.asarray(particles['history']),jnp.asarray(nodes['damage']))
    for frame in parent.glob('frame_*.npz'):
        shutil.copyfile(frame,destination/frame.name)
    shutil.copyfile(parent/'source-runtime.json',destination/'source-runtime.json')
    receipt = dict(parent=str(parent), parent_step=summary['steps'],
        parent_commit=provenance['commit'], reconstructed_displacement=reconstructed,
        position_roundtrip_max_abs=roundtrip, conservative_displacement_roundoff_bound=bound,
        limitation='Original displacement was not exported; reconstruction is not bitwise original restart.' if reconstructed else 'Exact exported state reload.',
        parent_files={name:hashlib.sha256((parent/name).read_bytes()).hexdigest()
            for name in ('summary.json','particles.npz','nodal_state.npz','history.csv','source-runtime.json')})
    (destination/'restart-provenance.json').write_text(json.dumps(receipt,indent=2))
    return state, summary, np.atleast_2d(np.loadtxt(parent/'history.csv',delimiter=',',skiprows=1)).tolist()
