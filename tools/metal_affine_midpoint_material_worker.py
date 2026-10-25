"""Isolated material evaluation for a common-time-level force solve."""
import argparse
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
import numpy as np

from mpm_jax.fragment_v6_discrete_work import update
from mpm_jax.materials.linear_metal import linear_metal
from tools.metal_affine_dynamic_numpy import thermal


def evaluate(state, parameters):
    dt = float(state.pop('candidate_requested_dt_s'))
    if not 0 < dt <= 2e-10:
        raise ValueError('Current continuation steps are bounded to 200 ps')
    material = linear_metal(**parameters)
    fields = ('F', 'tau', 'plastic_strain', 'cold_work', 'fracture_energy', 'damage', 'internal_energy')
    result = update(*(j.asarray(state[k]) for k in fields), j.asarray(state['W']), j.asarray(dt), material)
    for key, value in zip(fields, result.state):
        state[key] = np.asarray(value)
    state['x'] = state['x'] + dt * state['v']
    state['time'] = state['time'] + dt
    J, temperature, entropy = thermal(state, parameters)
    previous_entropy = state.pop('candidate_previous_entropy')
    state['minimum_entropy'] = np.minimum(state['minimum_entropy'], entropy)
    state['negative_entropy_increments'] += np.minimum(entropy - previous_entropy, 0)
    state['maximum_temperature'] = np.maximum(state['maximum_temperature'], temperature)
    state['minimum_J'] = np.minimum(state['minimum_J'], J)
    state['maximum_J'] = np.maximum(state['maximum_J'], J)
    state['ever_phase_extrapolated'] |= temperature >= material.melt_temperature
    state['ever_tension_extrapolated'] |= J > 1
    state.update(candidate_material_valid=np.asarray(result.valid),
                 candidate_material_work_density=np.asarray(result.work_density),
                 candidate_plastic_heat=np.asarray(result.plastic_heat),
                 candidate_work_piola=np.asarray(result.piola),
                 candidate_volume_identity=np.asarray(result.volume_identity),
                 candidate_deviatoric_identity=np.asarray(result.deviatoric_identity))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve material evaluations')
    with np.load(args.request) as data:
        state = {k: data[k] for k in data.files}
    parameters = json.loads(args.admission.read_text())['material']
    state = evaluate(state, parameters)
    np.savez_compressed(args.output, **state)
    if not bool(state['candidate_material_valid']):
        raise ValueError('Material update failed; returned state preserved')
    print(json.dumps({'valid': True, 'max_temperature': float(thermal(state, parameters)[1].max())}), flush=True)


if __name__ == '__main__':
    main()
