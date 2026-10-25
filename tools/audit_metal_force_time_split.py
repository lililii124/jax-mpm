"""Decompose saved kick/material work without advancing the state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def affine_velocity(white, mass, root):
    values = (white / np.repeat(np.sqrt(mass), 4)[:, None]).reshape(-1, 4, 3)
    return values[:, 1:, :].swapaxes(-1, -2) @ np.linalg.inv(root)


def audit(directory, step):
    report = json.loads((directory / 'result.json').read_text())
    dt = report['dt_s']
    files = {
        'first': next(directory.glob(f'*-step-{step}-half-kick.npz')),
        'last': next(directory.glob(f'*-step-{step}-final-half-kick.npz')),
        'request': directory / f'step-{step}-material-request.npz',
        'response': directory / f'step-{step}-material-response.npz',
    }
    terms = dict(kick_velocity=0., endpoint_geometry=0., endpoint_stress_work=0.)
    velocity_parts = {}
    with np.load(files['request']) as old, np.load(files['response']) as new:
        volume = old['volume']
        W = old['W']
        midpoint_F = .5 * (old['F'] + new['F'])
        L = W @ np.linalg.inv(midpoint_F)
        work = float(np.sum(volume * new['candidate_material_work_density']))
        old_damage = old['damage']
        new_damage = new['damage']
        end_tau = new['tau']
    for label in ('first', 'last'):
        with np.load(files[label]) as state:
            tau = state['candidate_stress'] - state['candidate_viscous_stress']
            white_mid = .5 * (state['candidate_white_before'] + state['candidate_white'])
            Wkick = affine_velocity(white_mid, state['mass'], state['reference_root'])
            Wbefore = affine_velocity(state['candidate_white_before'], state['mass'], state['reference_root'])
            Wafter = affine_velocity(state['candidate_white'], state['mass'], state['reference_root'])
            invF = np.linalg.inv(state['F'])
            terms['kick_velocity'] += .5 * dt * float(np.sum(volume[:, None, None] * tau * ((Wkick - W) @ invF)))
            terms['endpoint_geometry'] += .5 * dt * float(np.sum(volume[:, None, None] * tau * (W @ invF - L)))
            terms['endpoint_stress_work'] += .5 * dt * float(np.sum(volume[:, None, None] * tau * L))
            if label == 'first':
                velocity_parts['first_kick'] = .5 * dt * float(np.sum(volume[:, None, None] * tau * ((Wkick - W) @ invF)))
            else:
                velocity_parts['last_kick'] = .25 * dt * float(np.sum(volume[:, None, None] * tau * ((Wafter - Wbefore) @ invF)))
                velocity_parts['between_kicks_projection_contact'] = .5 * dt * float(np.sum(volume[:, None, None] * tau * ((Wbefore - W) @ invF)))
    terms['stress_quadrature'] = terms.pop('endpoint_stress_work') - work
    if np.any(new_damage >= 1):
        raise ValueError('Cannot reconstruct pre-damage stress at complete failure')
    undamaged_end = end_tau * ((1 - old_damage) / (1 - new_damage))[:, None, None]
    damage_term = .5 * dt * float(np.sum(volume[:, None, None] * (end_tau - undamaged_end) * L))
    terms['damage_endpoint_part'] = damage_term
    terms['stress_quadrature_except_endpoint_damage'] = terms['stress_quadrature'] - damage_term
    reconstructed = sum(terms[k] for k in ('kick_velocity', 'endpoint_geometry', 'stress_quadrature'))
    original = report['steps'][step]['force_time_work_remainder_J']
    if abs(reconstructed - original) > 1e-12:
        raise ValueError('Decomposition does not reconstruct the saved work remainder')
    return {
        'scope': 'Read-only decomposition of saved physical states; no new accepted time step',
        'step': step, 'dt_s': dt, 'terms_J': terms,
        'kick_velocity_parts_J': velocity_parts,
        'reconstructed_remainder_J': reconstructed, 'saved_remainder_J': original,
        'reconstruction_error_J': reconstructed - original,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.step)
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))
