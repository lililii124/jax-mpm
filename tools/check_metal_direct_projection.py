"""Compare a direct constrained projection with a saved two-stage result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from tools.metal_affine_dynamic_numpy import DynamicFrame, kinetic, moments, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.state) as data:
        state = {k: data[k] for k in data.files}
    with np.load(args.projection) as data:
        reference = {k: data[k] for k in data.files}
    prior = json.loads(args.result.read_text())
    tol = prior['energy_roundoff_J']
    before, expected = reference['white_before'], reference['white']
    fixed = state['candidate_fixed_particles']
    started = time.monotonic()
    phases = []
    frame = DynamicFrame(state, fixed, before,
        lambda phase: phases.append({'phase': phase, 'seconds': time.monotonic()-started}),
        direct_projection=True)
    got, rows, reaction = frame.project(before)
    elapsed = time.monotonic()-started
    delta = got-expected
    energy_bound = np.linalg.norm(delta)*max(np.linalg.norm(got), np.linalg.norm(expected), 1.)
    saved_row = next(row['projection'] for row in prior['stages'] if row['stage']=='end-projection')
    saved_loss = saved_row['grid_projection_loss_J']+saved_row['boundary_projection_loss_J']
    dp, dl = moments(state, reaction-reference['reaction'])
    v0, _ = unpack(state, before)
    v1, W1 = unpack(state, got)
    momentum_scale = float(np.sum(state['mass']*(np.linalg.norm(v0,axis=1)+np.linalg.norm(v1,axis=1))))
    angular_tol = prior['angular_tolerance']
    original_normal = np.linalg.norm(frame.allowed.Q0.T@(before-expected))
    direct_normal = np.linalg.norm(frame.allowed.Q0.T@(before-got))
    normal_tol = original_normal + 64*np.finfo(float).eps*np.linalg.norm(frame.allowed.Q0.data)*np.linalg.norm(before)
    gates = {
        'velocity_work_bound': bool(energy_bound <= tol),
        'combined_loss': bool(abs(rows['combined_projection_loss_J']-saved_loss) <= tol),
        'projection_work_identity': bool(abs(rows['combined_projection_identity_J']) <= tol),
        'fixed_particles': bool(np.all(v1[fixed] == 0) and np.all(W1[fixed] == 0)),
        'momentum': bool(np.linalg.norm(dp) <= 1e-10*max(momentum_scale, 1e-30)),
        'angular_momentum': bool(np.linalg.norm(dl) <= angular_tol),
        'original_normal_residual': bool(direct_normal <= normal_tol),
        'contact_sample_difference': bool(np.max(abs(frame.contact.sample(delta)),initial=0) <= .005),
    }
    report = {'passed': all(gates.values()), 'gates': gates, 'seconds': elapsed,
        'phases': phases, 'geometry': frame.report, 'projection': rows,
        'velocity_l2_difference': float(np.linalg.norm(delta)),
        'velocity_max_difference': float(np.max(abs(delta))),
        'velocity_work_bound_J': float(energy_bound),
        'kinetic_difference_J': kinetic(got)-kinetic(expected),
        'saved_combined_loss_J': saved_loss, 'impulse_difference': dp.tolist(),
        'angular_impulse_difference': dl.tolist(), 'angular_tolerance': angular_tol,
        'normal_residual': float(direct_normal), 'saved_normal_residual': float(original_normal),
        'normal_tolerance': float(normal_tol), 'energy_tolerance_J': tol,
        'scope': 'One saved full-state projection; no trajectory or full reference rerun',
        'inputs': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.state, args.projection, args.result)}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    frame.release()
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
