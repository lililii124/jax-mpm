"""Inspect current quadratic-grid coupling across a tensile specimen split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def inspect_state(x, failed, *, dx, n_grid, split_plane, left_grip, right_grip):
    x, failed = np.asarray(x), np.asarray(failed)
    left_grip, right_grip = np.asarray(left_grip), np.asarray(right_grip)
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x) or x.dtype != np.float64
            or not np.isfinite(x).all()):
        raise ValueError('Finite binary64 three-dimensional particle positions are required')
    for name, mask in (('failed', failed), ('left_grip', left_grip), ('right_grip', right_grip)):
        if mask.shape != (len(x),) or mask.dtype != np.bool_:
            raise ValueError(f'{name} must be one Boolean value per particle')
    if not left_grip.any() or not right_grip.any() or np.any(left_grip & right_grip):
        raise ValueError('Two nonempty disjoint reference grip masks are required')
    if (type(n_grid) is not int or n_grid < 2 or (n_grid + 1)**3 > 16_000_000
            or not np.isfinite(dx) or dx <= 0 or not np.isfinite(split_plane)):
        raise ValueError('Invalid or oversized grid or split plane')
    active = ~failed
    side = x[:, 0] > split_plane
    width = n_grid + 1
    owners = np.zeros(width**3, dtype=np.uint8)
    offsets = np.stack(np.meshgrid(*([np.arange(3)] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    indices = np.flatnonzero(active)
    for start in range(0, len(indices), 8192):
        batch = indices[start:start + 8192]
        q = x[batch] / dx
        base = np.floor(q - .5).astype(np.int64)
        if np.any(base < 0) or np.any(base + 2 > n_grid):
            raise ValueError('An active particle lacks complete in-domain quadratic support')
        f = q - base
        weights = np.stack((.5 * (1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2), axis=-1)
        positive = np.ones((len(batch), 27), dtype=bool)
        for axis in range(3):
            positive &= weights[:, axis, offsets[:, axis]] > 0
        nodes = base[:, None, :] + offsets
        encoded = (nodes[:, :, 0] * width + nodes[:, :, 1]) * width + nodes[:, :, 2]
        labels = np.broadcast_to(np.where(side[batch], 2, 1).astype(np.uint8)[:, None], encoded.shape)
        np.bitwise_or.at(owners, encoded[positive], labels[positive])
    both = bool(np.any(active & ~side) and np.any(active & side))
    shared = int(np.count_nonzero(owners == 3))
    axial = np.bitwise_or.reduce(owners.reshape(width, width, width), axis=(1, 2))
    occupied = np.flatnonzero(axial)
    empty = np.flatnonzero(axial == 0)
    if len(occupied):
        empty = empty[(empty > occupied[0]) & (empty < occupied[-1])]
    else:
        empty = np.empty(0, dtype=int)
    partitions = []
    for value in (False, True):
        mask = active & (side == value)
        partitions.append({'side': 'right' if value else 'left', 'particles': int(mask.sum()),
            'positive_grid_nodes': int(np.count_nonzero(owners & (2 if value else 1))),
            'x_bounds_m': [float(x[mask, 0].min()), float(x[mask, 0].max())] if mask.any() else None,
            'reference_left_grip_particles': int(np.count_nonzero(mask & left_grip)),
            'reference_right_grip_particles': int(np.count_nonzero(mask & right_grip))})
    grips_on_sides = bool(np.all(active[left_grip] & ~side[left_grip])
                         and np.all(active[right_grip] & side[right_grip]))
    return {'split_plane_m': float(split_plane), 'active_particles': int(active.sum()),
        'failed_particles': int(failed.sum()), 'both_current_sides_occupied': both,
        'shared_positive_3d_grid_nodes': shared,
        'no_shared_grid_support_at_split': both and shared == 0,
        'reference_grips_unfailed_and_on_own_sides': grips_on_sides,
        'reference_grip_sets_grid_decoupled': both and shared == 0 and grips_on_sides,
        'empty_interior_axial_node_planes': empty.tolist(), 'partitions': partitions}


def audit(folder):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    if len(summary['configs']) != 1:
        raise ValueError('This diagnostic requires one JC specimen')
    config, definition = summary['configs'][0], summary['definition']
    if (config['dim'] != 3 or config['material'] != 'johnson-cook'
            or config['failure_mode'] != 'ballistic'
            or definition.get('series') != 'jc-necking-sphinxsys-pr1127-v1'):
        raise ValueError('Expected the ballistic JC necking case')
    frame, steps = status['frame'], status['steps_completed']
    if (type(frame) is not int or frame < 1 or status['frames'] != frame
            or type(steps) is not int or steps < 1 or steps % frame
            or summary['steps'] != steps or status.get('completed') is not True
            or summary.get('completed') is not True or status.get('failed', False) is not False
            or status.get('phase') not in (None, 'completed')):
        raise ValueError('The folder diagnostic requires a completed final endpoint')
    dt = config['dt']
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('A finite positive timestep is required')
    nominal = steps * dt
    clock_tolerance = 32 * np.finfo(np.float64).eps * steps * max(nominal, dt)
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as archive:
        times = archive['time']
    if (times.shape != (frame + 1,) or times.dtype != np.float64 or not np.isfinite(times).all()
            or times[0] != 0 or not np.all(np.diff(times) > 0)
            or np.max(np.abs(times - np.arange(frame + 1) * (steps // frame) * dt)) > clock_tolerance
            or not np.isfinite(status['time']) or not np.isfinite(summary['simulation_time'])
            or abs(status['time'] - nominal) > clock_tolerance
            or abs(summary['simulation_time'] - nominal) > clock_tolerance):
        raise ValueError('Saved time index, status and complete plan differ')
    trajectory = np.load(folder / 'trajectory_x.npy', mmap_mode='r', allow_pickle=False)
    try:
        if (trajectory.ndim != 3 or trajectory.shape[2] != 3 or len(trajectory) != frame + 1
                or trajectory.dtype != np.float64):
            raise ValueError('Saved current/reference coordinates are incomplete')
        reference = np.array(trajectory[0])
        with np.load(folder / 'final_state_0.npz', allow_pickle=False) as state:
            x, failed = state['x'], state['history_failed']
            if (x.ndim != 2 or x.shape != (trajectory.shape[1], 3)
                    or summary['particles'] != [len(x)] or status['particles'] != len(x)):
                raise ValueError('Particle counts or coordinate shapes differ')
            if (state['valid'].shape != () or state['valid'].dtype != np.bool_ or not bool(state['valid'])
                    or state['history_valid'].shape != (len(x),) or state['history_valid'].dtype != np.bool_
                    or not np.all(state['history_valid'])
                    or not np.array_equal(x, trajectory[frame])):
                raise ValueError('Final state is invalid or differs from its saved frame')
            clock = state['time']
            if (clock.shape != () or clock.dtype != np.float64 or not np.isfinite(clock)
                    or abs(float(clock) - nominal) > clock_tolerance):
                raise ValueError('Final state clock differs from the completed endpoint')
        failed_history = np.load(folder / 'trajectory_failed.npy', mmap_mode='r', allow_pickle=False)
        try:
            if (failed_history.shape != (frame + 1, len(x))
                    or failed_history.dtype not in (np.dtype(bool), np.dtype(np.uint8))
                    or not np.array_equal(failed_history[frame], failed)):
                raise ValueError('Final failure labels differ from the saved endpoint')
        finally:
            failed_history._mmap.close()
        if reference.shape != x.shape or not np.isfinite(reference).all():
            raise ValueError('Reference coordinates do not match the current particles')
        left_face, right_face = definition['grip_inner_faces_x_m']
        left, right = reference[:, 0] <= left_face, reference[:, 0] >= right_face
        split = .5 * sum(definition['bar_ends_x_m'])
        result = inspect_state(x, failed, dx=config['domain_size'] / config['n_grid'],
            n_grid=config['n_grid'], split_plane=split, left_grip=left, right_grip=right)
    finally:
        trajectory._mmap.close()
    inputs = ('summary.json', 'status.json', 'final_state_0.npz', 'trajectory_x.npy',
              'trajectory_failed.npy', 'trajectory.npz')
    return {'schema': 'jc-current-grid-separation-v1', 'diagnostic_complete': True,
        'production_eligible': False, 'full_history_checked': False,
        'original_acceptance_unchanged': True, 'frame': frame, 'state': result,
        'input_sha256': {name: digest(folder / name) for name in inputs},
        'review_source_sha256': digest(Path(__file__)),
        'interpretation': 'No shared positive grid nodes is a sufficient current-state absence of grid-mediated coupling across this split for the declared ballistic local-material single-body solver, provided there are no additional interparticle or global feedback forces. It does not require all particles of an original material layer to reach D=1. This diagnostic does not verify the failure cause, separation history, resolved crack surfaces, fragment contact or experimental accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    report = audit(args.folder)
    output = args.folder / 'current-grid-separation.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'diagnostic_complete': True, 'state': report['state']}))


if __name__ == '__main__':
    main()
