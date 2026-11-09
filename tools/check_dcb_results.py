"""Verify stored DCB fields, localization, boundary motion and energy accounting."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check(directory):
    info = json.loads((directory/'summary.json').read_text())
    rows = list(csv.DictReader((directory/'history.csv').open()))
    assert info['all_states_valid'] and len(rows) == info['frames']
    assert all(r['valid'] == 'True' for r in rows)
    columns = {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0] if k != 'valid'}
    assert all(np.isfinite(values).all() for values in columns.values())
    lo, hi = np.asarray(info['lower_m']), np.asarray(info['upper_m'])
    center, tip0 = .5*(lo[1]+hi[1]), info['initial_tip_x_m']
    qx, areas = [], []
    for left, right in ((lo[0], tip0), (tip0, hi[0])):
        count = int(np.ceil((right-left)/(info['dx_m']/2)))
        qx.extend(left+(np.arange(count)+.5)*(right-left)/count)
        areas.extend(np.full(count, (right-left)/count))
    qx, areas = np.asarray(qx), np.asarray(areas)
    bonded = qx >= tip0
    law = info['law']
    d0, df = law['strength']/law['stiffness'], 2*law['energy']/law['strength']
    E, nu = info['young_Pa'], info['poisson']
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    with np.load(directory/'frame-000.npz', allow_pickle=False) as data:
        ny = len(np.unique(data['x'][:, 1]))
    symmetry = 0.
    for i, row in enumerate(rows):
        with np.load(directory/f'frame-{i:03d}.npz', allow_pickle=False) as data:
            x, F, d = data['x'], data['F'], data['interface_damage']
            plus, minus = data['interface_plus'], data['interface_minus']
            assert np.isfinite(x).all() and np.isfinite(F).all()
            assert np.isfinite(d).all() and np.all((d>=0) & (d<=1))
            J = np.linalg.det(F)
            assert J.min() > 0
            logJ = np.log(J)
            density = .5*mu*(np.sum(F*F, axis=(1, 2))-2)-mu*logJ+.5*lam*logJ**2
            bulk = np.prod(hi-lo)*density.mean()
            tau = mu*(F @ F.swapaxes(-1, -2)-np.eye(2))+lam*logJ[:, None, None]*np.eye(2)
            stress = tau/J[:, None, None]
            np.testing.assert_allclose(data['principal_stress'], np.linalg.eigvalsh(stress)[:, -1], atol=1e-6, rtol=2e-9)
            jump = plus-minus
            stored = .5*law['stiffness']*np.sum(areas*((1-d)*bonded*(jump[:, 0]**2+np.maximum(jump[:, 1], 0)**2)+np.minimum(jump[:, 1], 0)**2))
            maximum = df*d0/(df-d*(df-d0))
            dissipation = np.sum(areas*bonded*law['energy']*np.clip((maximum-d0)/(df-d0), 0, 1))
            prefix = np.cumprod(~bonded | (d>=1-1e-8))
            tip = lo[0]+np.sum(areas*prefix)
            failed = np.sum(areas*bonded*(d>=1-1e-8))
            np.testing.assert_allclose([bulk, stored, dissipation, tip, failed],
                                       [float(row[k]) for k in ('elastic_J_per_m', 'cohesive_stored_J_per_m', 'fracture_J_per_m', 'tip_x_m', 'failed_ligament_m')], atol=2e-9, rtol=2e-8)
            reflected = x.reshape(-1, ny, 2)[:, ::-1, :].reshape(-1, 2).copy()
            reflected[:, 1] = 2*center-reflected[:, 1]
            symmetry = max(symmetry, float(np.max(np.abs(reflected-x))))
    energy = sum(columns[k] for k in ('kinetic_J_per_m', 'elastic_J_per_m', 'cohesive_stored_J_per_m',
                                     'fracture_J_per_m', 'transfer_loss_J_per_m', 'boundary_projection_loss_J_per_m'))
    remainder = columns['external_J_per_m']-energy
    scale = max(float(np.abs(columns['external_J_per_m']).max()), 1e-8)
    assert np.max(np.abs(remainder)) < 1e-7*scale
    assert columns['max_iteration_residual'].max() < 1e-8
    assert columns['transfer_loss_J_per_m'].min() > -1e-9
    assert columns['boundary_projection_loss_J_per_m'].min() > -1e-9
    assert np.diff(columns['fracture_J_per_m']).min() > -1e-9
    extension = columns['tip_x_m']-tip0
    growth = np.diff(extension)>1e-10
    detached = np.abs(columns['failed_ligament_m']-extension).max()
    result = dict(grid=info['grid'], spacing_m=info['dx_m'], frames=len(rows), final_extension_m=float(extension[-1]),
                  growing_saved_intervals=int(growth.sum()), max_disconnected_failed_m=float(detached),
                  localized_progressive_failure=bool(growth.sum()>=3 and detached<1e-10),
                  first_saved_growth_s=next((float(t) for t, e in zip(columns['time_s'], extension) if e>1e-10), None),
                  max_sampled_front_speed_m_s=float(np.max(np.diff(extension)/np.diff(columns['time_s']))),
                  final_numerical_fraction=float(columns['numerical_fraction'][-1]),
                  max_unexplained_energy_J_per_m=float(np.abs(remainder).max()),
                  max_relative_energy_remainder=float(np.abs(remainder).max()/scale),
                  max_grip_opening_error_m=float(np.abs(columns['observed_grip_opening_m']-columns['prescribed_opening_m']).max()),
                  max_position_symmetry_error_m=symmetry,
                  max_momentum_error=float(columns['momentum_error'].max()),
                  max_sampled_strain=float(columns['max_strain'].max()),
                  min_face_gap_m=float(columns['min_face_gap_m'].min()),
                  final_fracture_J_per_m=float(columns['fracture_J_per_m'][-1]),
                  static_beam_validation=False, mesh_convergence_established=False,
                  history_sha256=hashlib.sha256((directory/'history.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    return result, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [check(path) for path in args.inputs]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {p.name: result for p, (result, _) in zip(args.inputs, cases)}
    (args.output/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)
    keys = (('tip_x_m', 'Connected extension (mm)'), ('upper_force_N_per_m', 'Mean upper-grip force (N/m)'),
            ('fracture_J_per_m', 'Fracture dissipation (J/m)'), ('numerical_fraction', 'Numerical dissipation / work (%)'))
    for result, data in cases:
        label = f"h = {1000*result['spacing_m']:.3f} mm"
        for ax, (key, ylabel) in zip(axes.ravel(), keys):
            values = data[key]
            if key == 'tip_x_m':
                values = (values-values[0])*1000
            if key == 'numerical_fraction':
                values = values*100
            ax.plot(data['time_s']*1000, values, label=label)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output/'dcb-history.png', dpi=180)
    plt.close(fig)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
