"""Check saved cohesive-plate fields and plot their fracture histories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check(directory):
    summary = json.loads((directory/'summary.json').read_text())
    rows = list(csv.DictReader((directory/'history.csv').open()))
    assert summary['all_states_valid'] and len(rows) == summary['frames']
    assert all(row['valid'] == 'True' for row in rows)
    columns = {k: np.asarray([float(row[k]) for row in rows]) for k in rows[0] if k != 'valid'}
    assert all(np.isfinite(v).all() for v in columns.values())
    external = columns['external_J_per_m']
    scale = max(1., float(np.abs(external).max()))
    total = sum(columns[k] for k in ('kinetic_J_per_m', 'elastic_J_per_m',
                                    'cohesive_stored_J_per_m', 'fracture_J_per_m', 'transfer_loss_J_per_m'))
    remainder = external-total
    assert np.max(np.abs(remainder)) < 1e-7*scale
    assert columns['max_iteration_residual'].max() < 1e-8
    assert columns['transfer_loss_J_per_m'].min() > -1e-9*scale
    assert np.diff(columns['fracture_J_per_m']).min() > -1e-9*scale
    law = summary['law']
    qx, area = [], []
    for left, right in ((.01, .022), (.022, .05)):
        n = int(np.ceil((right-left)/(summary['dx_m']/2)))
        qx.extend(left+(np.arange(n)+.5)*(right-left)/n)
        area.extend(np.full(n, (right-left)/n))
    area, bonded = np.asarray(area), np.asarray(qx) >= .022
    young, nu = summary['young_Pa'], summary['poisson']
    mu, lam = young/(2*(1+nu)), young*nu/((1+nu)*(1-2*nu))
    d0, df = law['strength']/law['stiffness'], 2*law['energy']/law['strength']
    for i, row in enumerate(rows):
        with np.load(directory/f'frame-{i:03d}.npz', allow_pickle=False) as data:
            F, damage = data['F'], data['interface_damage']
            jump = data['interface_plus']-data['interface_minus']
            J = np.linalg.det(F)
            assert np.isfinite(F).all() and J.min() > 0
            assert np.all((damage >= 0) & (damage <= 1))
            W = .5*mu*(np.sum(F*F, axis=(1, 2))-2)-mu*np.log(J)+.5*lam*np.log(J)**2
            bulk = .04*.02*W.mean()
            maximum = df*d0/(df-damage*(df-d0))
            dissipated = np.sum(area*law['energy']*np.clip((maximum-d0)/(df-d0), 0, 1)*bonded)
            stored = .5*law['stiffness']*np.sum(area*((1-damage)*bonded*(jump[:, 0]**2+np.maximum(jump[:, 1], 0)**2)+np.minimum(jump[:, 1], 0)**2))
            np.testing.assert_allclose([bulk, dissipated, stored, J.min()],
                                       [float(row[k]) for k in ('elastic_J_per_m', 'fracture_J_per_m', 'cohesive_stored_J_per_m', 'min_J')], rtol=2e-9, atol=2e-8)
    full = columns['failed_ligament_m'] >= .028-1e-9
    if full[-1]:
        np.testing.assert_allclose(columns['fracture_J_per_m'][-1], .028*law['energy'], rtol=1e-11)
    result = dict(grid=summary['grid'], dx_m=summary['dx_m'], law=law,
                  frames=len(rows), first_saved_full_separation_s=float(columns['time_s'][np.flatnonzero(full)[0]]) if full.any() else None,
                  final_opening_m=float(columns['opening_m'][-1]),
                  final_fracture_J_per_m=float(columns['fracture_J_per_m'][-1]),
                  final_transfer_loss_J_per_m=float(columns['transfer_loss_J_per_m'][-1]),
                  final_transfer_fraction=float(columns['transfer_loss_J_per_m'][-1]/external[-1]),
                  max_energy_remainder_J_per_m=float(np.abs(remainder).max()),
                  max_relative_energy_remainder=float(np.abs(remainder).max()/scale),
                  max_sampled_strain=summary['max_sampled_strain'],
                  saved_bulk_and_cohesive_fields_checked=True,
                  history_sha256=hashlib.sha256((directory/'history.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    return result, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    checked = [check(path) for path in args.inputs]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {path.name: entry[0] for path, entry in zip(args.inputs, checked)}
    (args.output/'validation.json').write_text(json.dumps(dict(cases=results, general_fracture_validation=False,
        scope='Straight cohesive interface; transfer loss is measured, not removed from the trajectory'), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for i, (result, data) in enumerate(checked):
        label = f"Gc={result['law']['energy']:g} J/m², h={result['dx_m']*1000:g} mm"
        style = '--' if i == 0 else '-'
        axes[0].plot(data['time_s']*1e6, (data['tip_x_m']-.022)*1000, style, label=label)
        axes[1].plot(data['time_s']*1e6, data['fracture_J_per_m'], style, label=label)
    for ax in axes:
        ax.set_xlabel('Time (µs)')
        ax.grid(alpha=.2)
        ax.set_xlim(0, 200)
    axes[0].set_ylabel('Connected debonded length (mm)')
    axes[1].set_ylabel('Fracture dissipation (J/m)')
    axes[1].legend(fontsize=7, loc='lower right')
    fig.savefig(args.output/'fracture-history.png', dpi=180)
    plt.close(fig)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
