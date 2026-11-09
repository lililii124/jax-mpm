"""Independently reconstruct prescribed-path cohesive tension histories (NumPy only)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from check_domain_contact import polygon_moments


def read_history(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty history: {path}')
    return rows


def reconstruct_frame(frame, info, area, covariance, mass, law, bonded, quadrature_area, normal):
    x, v, F, C = (frame[k] for k in ('x', 'v', 'F', 'C'))
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError('Nonpositive deformation determinant in saved frame')
    dx = info['dx_m']
    moment = .25*dx**2*np.eye(2) + F @ covariance @ F.swapaxes(-1, -2)
    kinetic = .5*np.sum(mass*(np.sum(v*v, axis=1) + np.trace(C @ moment @ C.swapaxes(-1, -2), axis1=-2, axis2=-1)))
    E, nu = info['young_Pa'], info['poisson']
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    logJ = np.log(J)
    density = .5*mu*(np.sum(F*F, axis=(1, 2))-2) - mu*logJ + .5*lam*logJ**2
    elastic = np.sum(area*density)

    gap = (frame['plus'] - frame['minus']) @ normal
    maximum = frame['maximum']
    d0, df = law['peak_opening'], law['final_opening']
    h = np.maximum(maximum, d0)
    remaining = np.clip(d0*(df-h)/(h*(df-d0)), 0., 1.)
    intact = np.where(bonded, remaining, 0.)
    stored = .5*law['stiffness']*np.sum(quadrature_area*(intact*np.maximum(gap, 0.)**2 + np.minimum(gap, 0.)**2))
    damage = np.clip((maximum-d0)/(df-d0), 0., 1.)
    dissipated = law['energy']*np.sum(quadrature_area*np.where(bonded, damage, 0.))
    return kinetic, elastic, stored, dissipated, gap, maximum, h


def trace_drift(frame, reference_vertices, reference_count, reference_centroid,
                fields, x0, F0, face_mask, plus, minus):
    vertices0 = x0[:, None, :] + np.einsum('pij,pkj->pki', F0,
                                             reference_vertices-reference_centroid[:, None, :])
    vertices = frame['x'][:, None, :] + np.einsum('pij,pkj->pki', frame['F'],
                                                    reference_vertices-reference_centroid[:, None, :])
    errors = []
    for field, trace in enumerate((plus, minus)):
        face = vertices[face_mask & (fields[:, None] == field)]
        order = np.argsort(face[:, 0], kind='stable')
        actual_y = np.interp(trace[:, 0], face[order, 0], face[order, 1])
        errors.append(np.abs(actual_y-trace[:, 1]))
    return float(max(np.max(values) for values in errors)), vertices


def check_run(run, source_commit=None, verified_revision=None):
    info = json.loads((run/'input.json').read_text())
    if source_commit:
        for relative, expected_hash in info['source_sha256'].items():
            blob = subprocess.run(['git', 'cat-file', 'blob', f'{verified_revision}:{relative}'],
                                  check=True, capture_output=True).stdout.replace(b'\r\n', b'\n')
            actual_hash = hashlib.sha256(blob).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f'{run}: {relative} hash differs from verified source revision {verified_revision}')
    summary = json.loads((run/'summary.json').read_text())
    rows = read_history(run/'history.csv')
    if not summary.get('completed') or len(rows) != 41 or any(row['valid'] != 'True' for row in rows):
        raise ValueError(f'{run}: expected 41 valid frames and completed summary')
    history = {key: np.asarray([float(row[key]) for row in rows])
               for key in rows[0] if key != 'valid'}
    if not all(np.isfinite(values).all() for values in history.values()):
        raise ValueError(f'{run}: nonfinite numeric history value')
    if np.any(np.diff(history['time_s']) <= 0) or not np.isclose(history['time_s'][-1], info['duration_s'], atol=1e-12, rtol=0):
        raise ValueError(f'{run}: time history is not increasing to requested duration')
    if int(history['steps'][-1]) != int(info['steps']):
        raise ValueError(f'{run}: completed step count does not match input')

    with np.load(run/'final.npz', allow_pickle=False) as data:
        final = {key: data[key].copy() for key in data.files}
    if not all(np.isfinite(values).all() for values in final.values()):
        raise ValueError(f'{run}: nonfinite final state array')
    vertices = final['reference_vertices']
    counts = final['reference_count']
    area, centroid, covariance = polygon_moments(vertices, counts)
    rho = info['density_kg_m3']
    mass = rho*area
    np.testing.assert_allclose(centroid, final['reference_centroid'], atol=1e-16, rtol=0)
    np.testing.assert_allclose(area, final['reference_area'], atol=1e-18, rtol=1e-12)
    np.testing.assert_allclose(covariance, final['reference_covariance'], atol=1e-18, rtol=1e-11)
    np.testing.assert_allclose(mass, final['mass'], atol=1e-15, rtol=1e-12)
    if not np.isclose(area.sum(), .04*.02, atol=1e-14, rtol=0):
        raise ValueError(f'{run}: reference polygon area does not sum to 0.04×0.02 m²')

    normal = final['normal']
    quadrature_area = final['area']
    if (not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)
            or not np.isfinite(quadrature_area).all() or np.any(quadrature_area <= 0)):
        raise ValueError(f'{run}: cohesive interface normal or quadrature areas are invalid')
    bonded = final['bonded'].astype(bool)
    maximum_final = final['maximum']
    law = info['law']
    # Run inputs serialize CohesiveLaw's primitive strength/energy/stiffness fields;
    # derive its bilinear openings independently instead of importing the JAX class.
    d0 = law['strength']/law['stiffness']
    Gc = law['energy']
    df = 2*Gc/law['strength']
    law = dict(law, peak_opening=d0, final_opening=df)
    if np.any(bonded & (maximum_final < -1e-14)) or np.any((~bonded) & (maximum_final != 0)):
        raise ValueError(f'{run}: invalid initial/terminal cohesive history for bonded or notch points')
    if not np.isclose(history['fracture_J_per_m'][0], 0., atol=1e-14, rtol=0):
        raise ValueError(f'{run}: initial dissipated fracture energy is nonzero')
    if np.any(np.diff(history['fracture_J_per_m']) < -1e-12):
        raise ValueError(f'{run}: dissipated fracture energy is not monotone')
    max_bonded_path_area = float(np.sum(quadrature_area[bonded]))
    expected_bonded_length = info['prescribed_path_length_m']-info['initial_crack_length_m']
    if not np.isclose(max_bonded_path_area, expected_bonded_length, atol=1e-12, rtol=0):
        raise ValueError(f'{run}: bonded prescribed path length does not match metadata')
    if np.any(history['fracture_J_per_m'] > Gc*expected_bonded_length+1e-10):
        raise ValueError(f'{run}: dissipated fracture energy exceeds Gc times bonded path')

    frames = sorted(run.glob('frame-*.npz'))
    if len(frames) != len(rows):
        raise ValueError(f'{run}: frame count does not match history')
    with np.load(frames[0], allow_pickle=False) as initial:
        x0, F0 = initial['x'].copy(), initial['F'].copy()
        initial_trace_x = initial['plus'][:, 0].copy()
    spatial_order = np.argsort(initial_trace_x)
    double_notch = info.get('notch', 'edge') == 'double'
    if bonded[spatial_order[0]] or (double_notch and bonded[spatial_order[-1]]):
        raise ValueError(f'{run}: initial edge notches must be unbonded')
    if 'initial_crack_segments_m' in info:
        initially_cracked = np.zeros(len(bonded), dtype=bool)
        for left, right in info['initial_crack_segments_m']:
            initially_cracked |= (initial_trace_x >= left) & (initial_trace_x <= right)
        if not np.array_equal(initially_cracked, ~bonded):
            raise ValueError(f'{run}: initial bonded flags differ from declared notch geometry')

    def connected_from_tips(event):
        connected = np.zeros(len(bonded), dtype=bool)
        side_lengths = []
        directions = [spatial_order, spatial_order[::-1]] if double_notch else [spatial_order]
        for indices in directions:
            reached_length = 0.
            for q in indices:
                if not bonded[q]:
                    continue
                if not event[q]:
                    break
                connected[q] = True
                reached_length += quadrature_area[q]
            side_lengths.append(reached_length)
        return connected, side_lengths
    ref_count = counts
    initial_vertices = x0[:, None, :] + np.einsum('pij,pkj->pki', F0, vertices-centroid[:, None, :])
    vertex_id = np.arange(vertices.shape[1])[None, :]
    face_mask = (np.abs(initial_vertices[:, :, 1]-.03) < 1e-12) & (vertex_id < ref_count[:, None])
    max_errors = {key: 0. for key in ('kinetic_J_per_m', 'elastic_J_per_m', 'interface_stored_J_per_m',
                                       'fracture_J_per_m', 'balance_error_J_per_m', 'connected_failed_m',
                                       'detached_failed_m', 'connected_damaged_m', 'minimum_trace_gap_m',
                                       'maximum_trace_geometry_drift_m')}
    total_momentum = []
    reconstructed = {key: [] for key in ('time_s', 'kinetic_J_per_m', 'elastic_J_per_m',
                                          'interface_stored_J_per_m', 'fracture_J_per_m',
                                          'connected_failed_m', 'detached_failed_m', 'connected_damaged_m',
                                          'numerical_loss_fraction', 'balance_error_J_per_m',
                                          'maximum_trace_geometry_drift_m')}
    max_frame_final_difference = 0.
    max_contact_power = float(np.max(history['max_interface_power_error_J_per_m']))
    max_contact_residual = float(np.max(history['max_interface_residual']))
    max_bulk_residual = float(np.max(history['max_bulk_residual']))
    max_net_moment = float(np.max(history['max_interface_net_moment_N']))
    for idx, path in enumerate(frames):
        with np.load(path, allow_pickle=False) as archive:
            frame = {key: archive[key].copy() for key in archive.files}
        if not all(np.isfinite(values).all() for values in frame.values()):
            raise ValueError(f'{run}: nonfinite frame array in {path.name}')
        kinetic, elastic, stored, dissipated, gap, maximum, h = reconstruct_frame(
            frame, info, area, covariance, mass, law, bonded, quadrature_area, normal)
        positive_gap = np.maximum(gap, 0.)
        if np.any(bonded & (h+1e-13 < positive_gap)):
            raise ValueError(f'{run}: bonded maximum opening is smaller than current tensile opening')
        if idx == 0 and np.any(bonded & (maximum > 1e-14)):
            raise ValueError(f'{run}: initial bonded damage history is nonzero')
        if np.any((~bonded) & (maximum != 0)):
            raise ValueError(f'{run}: initially unbonded notch points acquired cohesive damage history')
        if idx and np.any(maximum+1e-14 < previous_maximum):
            raise ValueError(f'{run}: maximum opening history decreased')
        previous_maximum = maximum.copy()
        damaged = maximum > d0*(1+1e-8)
        failed = maximum >= df*(1-1e-8)
        connected_failed, failed_sides = connected_from_tips(failed)
        connected_damaged, damaged_sides = connected_from_tips(damaged)
        if double_notch:
            for side, failure_length, damage_length in zip(('left', 'right'), failed_sides, damaged_sides):
                np.testing.assert_allclose(history[f'{side}_connected_failed_m'][idx], failure_length, atol=1e-12, rtol=0)
                np.testing.assert_allclose(history[f'{side}_connected_damaged_m'][idx], damage_length, atol=1e-12, rtol=0)
        conn_fail_m = float(np.sum(quadrature_area*bonded*connected_failed))
        detached_fail_m = float(np.sum(quadrature_area*bonded*failed*(~connected_failed)))
        conn_damage_m = float(np.sum(quadrature_area*bonded*connected_damaged))
        contact_gap = float(gap.min())
        drift, _ = trace_drift(frame, vertices, counts, centroid, final['fields'], x0, F0,
                               face_mask, frame['plus'], frame['minus'])
        balance = (kinetic+elastic+stored+dissipated-info['initial_energy_J_per_m']
                   +history['transfer_loss_J_per_m'][idx]+history['boundary_projection_loss_J_per_m'][idx]
                   -history['geometry_energy_J_per_m'][idx]-history['external_J_per_m'][idx])
        comparisons = {
            'kinetic_J_per_m': kinetic,
            'elastic_J_per_m': elastic,
            'interface_stored_J_per_m': stored,
            'fracture_J_per_m': dissipated,
            'balance_error_J_per_m': balance,
            'connected_failed_m': conn_fail_m,
            'detached_failed_m': detached_fail_m,
            'connected_damaged_m': conn_damage_m,
            'minimum_trace_gap_m': contact_gap,
            'maximum_trace_geometry_drift_m': drift,
        }
        for key, value in comparisons.items():
            error = abs(value-history[key][idx])
            max_errors[key] = max(max_errors[key], error)
        if idx and (dissipated+1e-12 < previous_dissipated or conn_fail_m+1e-12 < previous_connected_failed
                    or conn_damage_m+1e-12 < previous_connected_damaged):
            raise ValueError(f'{run}: fracture dissipation or connected damage history is not monotone')
        previous_dissipated, previous_connected_failed, previous_connected_damaged = dissipated, conn_fail_m, conn_damage_m
        mom = np.sum(mass[:, None]*frame['v'], axis=0)
        total_momentum.append(mom)
        for key, value in (('time_s', float(frame['time_s'])), ('kinetic_J_per_m', kinetic),
                           ('elastic_J_per_m', elastic), ('interface_stored_J_per_m', stored),
                           ('fracture_J_per_m', dissipated), ('connected_failed_m', conn_fail_m),
                           ('detached_failed_m', detached_fail_m), ('connected_damaged_m', conn_damage_m),
                           ('numerical_loss_fraction', history['numerical_loss_fraction'][idx]),
                           ('balance_error_J_per_m', balance), ('maximum_trace_geometry_drift_m', drift)):
            reconstructed[key].append(value)
        if idx == len(frames)-1:
            for key in ('x', 'F', 'v', 'C', 'plus', 'minus', 'maximum'):
                max_frame_final_difference = max(max_frame_final_difference,
                                                 float(np.max(np.abs(frame[key]-final[key]))))

    for key, error in max_errors.items():
        tolerance = 1e-9 if key in ('minimum_trace_gap_m', 'maximum_trace_geometry_drift_m') else 1e-8
        if error > tolerance:
            raise ValueError(f'{run}: reconstructed {key} differs from history by {error:g}')
    if max_frame_final_difference > 1e-14:
        raise ValueError(f'{run}: final.npz does not agree with last frame')
    if not np.isclose(reconstructed['kinetic_J_per_m'][0]+reconstructed['elastic_J_per_m'][0]
                      +reconstructed['interface_stored_J_per_m'][0]+reconstructed['fracture_J_per_m'][0],
                      info['initial_energy_J_per_m'], atol=1e-10, rtol=1e-9):
        raise ValueError(f'{run}: initial reconstructed energy differs from input')
    max_balance = float(np.max(np.abs(reconstructed['balance_error_J_per_m'])))
    if max_balance > 1e-8 or max_contact_power > 1e-8 or max_contact_residual > 1e-8 or max_bulk_residual > 1e-7 or max_net_moment > 1e-8:
        raise ValueError(f'{run}: energy or solver diagnostics exceed audit tolerances')

    ledger = final['ledger']
    last = len(rows)-1
    ledger_mismatch = {'external': float(ledger[:3].sum()-history['external_J_per_m'][last]),
                       'transfer': float(ledger[3]-history['transfer_loss_J_per_m'][last]),
                       'projection': float(ledger[4]-history['boundary_projection_loss_J_per_m'][last]),
                       'geometry': float(ledger[5]-history['geometry_energy_J_per_m'][last]),
                       'balance': float(ledger[7]-history['balance_error_J_per_m'][last])}
    if max(abs(v) for v in ledger_mismatch.values()) > 1e-8:
        raise ValueError(f'{run}: final cumulative ledger differs from final history row')
    initial_momentum = np.asarray(total_momentum)[0]
    max_momentum_change = float(np.max(np.abs(np.asarray(total_momentum)-initial_momentum)))

    frame_digest = hashlib.sha256()
    for path in frames:
        frame_digest.update(path.read_bytes())
    report = dict(run=run.name, completed=True, grid=info['grid'], dt_s=info['dt_s'],
                  actual_duration_s=float(history['time_s'][-1]), requested_duration_s=info['duration_s'],
                  final_step_count=int(history['steps'][-1]), reference_polygon_area_m2=float(area.sum()),
                  bonded_path_length_m=max_bonded_path_area, maximum_fracture_energy_J_per_m=float(history['fracture_J_per_m'][-1]),
                  fracture_energy_limit_J_per_m=float(Gc*expected_bonded_length),
                  max_frame_history_errors=max_errors, max_final_frame_npz_error=max_frame_final_difference,
                  final_ledger_mismatches_J_per_m=ledger_mismatch,
                  max_balance_error_J_per_m=max_balance,
                  max_interface_power_error_J_per_m=max_contact_power,
                  max_interface_residual=max_contact_residual, max_bulk_residual=max_bulk_residual,
                  max_interface_net_moment_N=max_net_moment,
                  initial_momentum_kg_m_s=initial_momentum.tolist(),
                  max_momentum_change_kg_m_s=max_momentum_change,
                  maximum_trace_geometry_drift_m=float(np.max(history['maximum_trace_geometry_drift_m'])),
                  final_connected_failed_m=float(history['connected_failed_m'][-1]),
                  final_detached_failed_m=float(history['detached_failed_m'][-1]),
                  final_connected_damaged_m=float(history['connected_damaged_m'][-1]),
                  final_numerical_loss_fraction=float(history['numerical_loss_fraction'][-1]),
                  initial_damage_fraction=0., initial_unbonded_quadrature_count=int(np.sum(~bonded)),
                  initial_crack_unbonded_but_not_damaged=True,
                  mesh_convergence_established=False,
                  predictive_fracture_validation=False,
                  source_commit=verified_revision,
                  source_provenance=('all input source hashes verified against git revision' if verified_revision
                                     else 'input source hashes recorded; no commit asserted'),
                  source_sha256_from_run_input=info['source_sha256'],
                  final_snapshot_sha256=hashlib.sha256((run/'final.npz').read_bytes()).hexdigest(),
                  concatenated_frames_sha256=frame_digest.hexdigest(),
                  history_sha256=hashlib.sha256((run/'history.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    profile = {key: values for key, values in reconstructed.items()}
    return report, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', default=None,
                        help='Verify every input source_sha256 against this Git revision')
    args = parser.parse_args()
    verified_revision = None
    if args.source_commit:
        verified_revision = subprocess.run(
            ['git', 'rev-parse', '--verify', f'{args.source_commit}^{{commit}}'],
            check=True, capture_output=True, text=True).stdout.strip()
    reports, profiles = {}, {}
    for run in args.runs:
        reports[run.name], profiles[run.name] = check_run(run, args.source_commit, verified_revision)

    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, values in profiles.items():
        t = np.asarray(values['time_s'])*1e6
        axes[0].plot(t, np.asarray(values['connected_failed_m'])*1e3, label=name)
        axes[0].plot(t, np.asarray(values['connected_damaged_m'])*1e3, ls=':', label=f'{name} damaged')
        axes[1].plot(t, 100*np.asarray(values['numerical_loss_fraction']), label=name)
        axes[2].plot(t, np.asarray(values['fracture_J_per_m']), label=name)
    axes[0].set(xlabel='Time (µs)', ylabel='Connected path (mm)', title='Connected failed and damaged length')
    axes[1].set(xlabel='Time (µs)', ylabel='Numerical loss (%)', title='Transfer and projection loss')
    axes[2].set(xlabel='Time (µs)', ylabel='Fracture dissipation (J/m)', title='Cohesive fracture energy')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(args.output/'domain-cohesive-history.png', dpi=150)

    result = dict(runs=reports, source_commit=verified_revision,
                  source_provenance=('all input source hashes verified against git revision' if verified_revision
                                     else 'input source hashes recorded; no commit asserted'),
                  mesh_convergence_established=False, predictive_fracture_validation=False,
                  validation_scope='Independent NumPy reconstruction of stored frames, cohesive state, energies, and ledgers; not predictive fracture validation')
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
