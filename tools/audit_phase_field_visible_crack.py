"""Fail-closed crack admission. NumPy/SciPy only; never imports JAX or renders."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label


def audit(root):
    summary = json.loads((root/'summary.json').read_text())
    history = np.genfromtxt(root/'history.csv', delimiter=',', names=True)
    with np.load(root/'particles.npz') as data:
        x, d = data['reference'], data['damage']
        gradient = data['damage_gradient']
        if not all(np.isfinite(data[k]).all() for k in data.files):
            raise RuntimeError('Nonfinite final state')
    spacing = summary['dx']/2
    axes = [np.unique(x[:, i]) for i in range(3)]
    indices = tuple(np.searchsorted(axes[i], x[:, i]) for i in range(3))
    occupied = np.zeros(tuple(map(len, axes)), dtype=bool)
    # Face-connected reference quadrature adjacency; no point deletion.
    occupied[indices] = (d >= .99) & (np.abs(x[:, 1]-.5) <= .13)
    components, count = label(occupied)
    labels = components[indices]
    candidates = []
    for number in range(1, count+1):
        points = x[labels == number]
        starts_at_notch = points[:, 0].min() <= .25+spacing
        extension = float(points[:, 0].max()-.25)
        thickness = len(np.unique(points[:, 2]))/len(axes[2])
        if starts_at_notch and extension >= .1 and thickness >= .8:
            candidates.append(dict(points=len(points), extension=extension, thickness_fraction=thickness))
    reaction = history['top_reaction']
    peak = float(reaction.max())
    load_drop = (peak-float(reaction[-1]))/peak if peak > 0 else 0.
    with np.load(root/'nodal_state.npz') as nodes:
        for key in nodes.files:
            if not np.isfinite(nodes[key]).all():
                raise RuntimeError('Nonfinite nodal field: '+key)
        nodal_bounds = bool(nodes['damage'].min() >= -1e-12 and nodes['damage'].max() <= 1+1e-12)
        exact_d = np.sum(nodes['basis_weight']*nodes['damage'][nodes['basis_nodes']], axis=1)
        exact_gradient = np.einsum('pk,pkj->pj', nodes['damage'][nodes['basis_nodes']], nodes['basis_gradient'])
        fracture = float(np.dot(nodes['particle_volume'], nodes['particle_fracture_density']))
        fields_match = (np.allclose(exact_d, d, rtol=0, atol=1e-12) and
            np.allclose(exact_gradient, gradient, rtol=0, atol=1e-12) and
            np.isclose(fracture, summary['energy'][2], rtol=1e-10, atol=1e-12))
    audit_values = summary['energy_audit']
    step_audit = summary['maximum_abs_step_audit']
    energy_scale = max(1., abs(summary['boundary_work']), sum(abs(v) for v in summary['energy']))
    # These identities check accounting only. Constitutive defect is explicit,
    # bounded separately, and never called physical dissipation.
    numerical_identities = all(abs(audit_values[k]) <= 1e-8*energy_scale and
        step_audit[k] <= 1e-9*energy_scale for k in ('kinetic_balance_error', 'total_balance_error'))
    defect_ratio = abs(audit_values['constitutive_work_defect'])/max(abs(summary['boundary_work']), 1e-12)
    transfer_ok = all(audit_values[k] >= -1e-10*energy_scale for k in ('p2g_loss', 'g2p_loss', 'damping_loss'))
    criteria = dict(contiguous_crack=bool(candidates), load_drop=load_drop >= .5,
        irreversible=summary['minimum_nodal_damage_increment'] >= -1e-12,
        kkt=summary['maximum_kkt'] <= 1e-7, nodal_quadrature_fields_match=bool(fields_match),
        energy_identities=bool(numerical_identities), transfer_losses_nonnegative=bool(transfer_ok),
        constitutive_defect_below_5_percent=defect_ratio <= .05,
        nodal_damage_bounds=nodal_bounds,
        damage_bounds=bool(d.min() >= -1e-12 and d.max() <= 1+1e-12))
    result = dict(passed=all(criteria.values()), criteria=criteria, threshold=.99,
        connectivity='six face neighbors on reference particle lattice, |y-.5|<=.13',
        minimum_extension=.1, minimum_thickness_fraction=.8, components=candidates,
        peak_reaction=peak, final_reaction=float(reaction[-1]), load_drop_fraction=load_drop,
        constitutive_defect_fraction=defect_ratio,
        limitation='Dynamic small-strain PIC experiment; accounting closure is not physical energy convergence. Recovered nodal gradient/fracture density are projections. No exact restart claimed.')
    (root/'crack-checks.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise RuntimeError('Crack/media admission failed: '+str(criteria))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output)
