"""Independent saved-state pilot review; no JAX, renderer, or publication."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def descriptor(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def projected_pixels(distance, camera):
    direction = np.asarray(camera['direction'], dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross([0., 1., 0.], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    bounds = camera['focus_bounds_m']
    corner = np.array(list(itertools.product(*[(bounds[2*i], bounds[2*i+1]) for i in range(3)])))
    scale = camera.get('parallel_scale', .56*max(np.ptp(corner@right), np.ptp(corner@up)))
    return float(abs(distance*up[1])*camera['resolution']/(2*scale))


def review(folder, output):
    plan = json.loads((folder/'plan.json').read_text())
    table = list(csv.DictReader((folder/'observables.csv').open()))
    with np.load(folder/'checkpoint.npz') as cp:
        frame = int(cp['frame'])
        tables = json.loads(str(cp['tables']))
        assert table == [{k:str(v) for k,v in row.items()} for row in tables['rows']]
        assert frame >= 1 and int(cp['progress_0']) == plan['frame_steps'][frame]
        with np.load(folder/'final_state_0.npz') as final:
            for name in final.files:
                leaf = name if name.startswith('history_') else 'state_'+name
                assert np.array_equal(final[name], cp[leaf]), 'final/checkpoint state differs'
        for name in ('x', 'damage', 'failed', 'speed', 'temperature'):
            array = np.load(folder/f'trajectory_{name}.npy', mmap_mode='r')
            for index in range(frame+1):
                assert descriptor(array[index]) == tables['records'][index][name], 'raw-state hash differs'
        F, D = cp['state_F'], cp['history_damage']
        material = plan['material']
        E, nu = material['young'], material['poisson']
        mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
        u, singular, vh = np.linalg.svd(F)
        rotation = u@vh
        strain = singular-1
        trace = strain.sum(1)
        tau = (1-D)[:, None, None]*(2*mu*F+(lam*trace-2*mu)[:, None, None]*rotation)@F.transpose(0,2,1)
        stress_error = float(np.max(np.abs(tau-cp['state_tau'])))
        energy = float(np.sum(cp['state_volume']*(1-D)*(.5*lam*trace**2+mu*np.sum(strain**2, axis=1))))
        momentum_errors = []
        for body in (0, 1):
            selected = cp['state_fibre_id'] == body
            momentum = np.sum(cp['state_mass'][selected, None]*cp['state_v'][selected], axis=0)
            residual = momentum-cp['progress_1'][body]-cp['progress_4'][body]-np.array([0.,cp['progress_5'][body],0.])
            momentum_errors.append(float(np.max(np.abs(residual))))
        checks = dict(particles=len(cp['state_x']) == 202240,
            prefix_or_later=frame >= 1, state_valid=bool(cp['state_valid']),
            independent_stress=stress_error < 1e-6,
            independent_energy=abs(energy-float(table[-1]['elastic_energy_J'])) < 1e-10,
            positive_J=bool(np.all(np.linalg.det(F) > 0)),
            undamaged=not D.any() and not cp['history_failed'].any(),
            momentum=max(momentum_errors) < 1e-10,
            contact=all(row['contact_valid'] == 'True' for row in table),
            radius=max(float(row['contact_radius_m']) for row in table)/plan['geometry']['radius_m'] < .2,
            mass=max(abs(float(row['mass_relative_drift'])) for row in table) < 1e-12,
            grips=max(float(row['grip_position_error_m']) for row in table) < 1e-9)
        work_scale = max(max(abs(float(row['actuator_work_J']))+abs(float(row['actual_contact_work_J']))
                             for row in table), 1e-20)
        checks['work_ledger'] = max(abs(float(row['energy_ledger_residual_J'])) for row in table) < .15*work_scale
        raw = np.load(folder/'trajectory_x.npy', mmap_mode='r')
        initial = raw[0]
        centre = []
        for body, fibre in enumerate(plan['geometry']['fibres']):
            lo, hi = fibre['particle_slice']
            centre.append(np.flatnonzero(np.abs(initial[lo:hi, 0 if body == 0 else 2])
                                         <= plan['geometry']['particle_spacing_m'])+lo)
        measured = []
        with np.load(folder/'particle_fields.npz') as labels:
            upper_grip = labels['grip_id'] == 2
        mass = cp['state_mass']
        for index, row in enumerate(table):
            displacement = raw[index]-initial
            relative = float(displacement[centre[1], 1].mean()-displacement[centre[0], 1].mean())
            csv_relative = float(row['upper_centre_displacement_y_m'])-float(row['lower_centre_displacement_y_m'])
            assert abs(relative-csv_relative) < 1e-12
            collar = float(np.sum(displacement[upper_grip, 1]*mass[upper_grip])/mass[upper_grip].sum())
            assert abs(collar-float(row['actual_upper_collar_displacement_y_m'])) < 1e-12
            measured.append(dict(time_s=float(row['time']), raw_relative_centre_displacement_m=relative,
                raw_upper_collar_displacement_m=collar, raw_relative_beam_bending_m=relative-collar,
                contact_gap_m=float(row['contact_gap_m']), contact_active=row['contact_active'] == 'True'))
        hold_start, hold_end = plan['protocol']['times_s'][3:5]
        hold = [row for row in measured if hold_start-1e-9 <= row['time_s'] <= hold_end+1e-9]
        report = dict(passed=all(checks.values()), checks=checks, accepted_steps=int(cp['progress_0']),
            saved_frame=frame, model=plan['model'], physical_time_s=float(cp['state_time']),
            stress_error_Pa=stress_error, energy_error_J=abs(energy-float(table[-1]['elastic_energy_J'])),
            per_body_momentum_errors_Ns=momentum_errors, raw_measured_response=measured,
            actual_hold_available=len(hold) >= 3, hold_samples=len(hold),
            hold_attached_samples=sum(row['contact_active'] for row in hold),
            maximum_hold_relative_beam_bending_m=max((abs(row['raw_relative_beam_bending_m']) for row in hold), default=None),
            visible_rendered_pixel_difference=None,
            scope='Independent actual saved-state hashes/material/impulse/centre displacement; rendered pixel contrast unmeasured.',
            production_eligible=False, public_visual_review_passed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Independent numerical prefix review failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.folder, args.output), indent=2))
