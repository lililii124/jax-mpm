"""Independently reconstruct planar domain-contact collision histories (NumPy only)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_history(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty history: {path}')
    return rows


def polygon_moments(vertices, count):
    """Return positive shoelace area, centroid and centroidal covariance."""
    areas = np.zeros(len(count))
    centroids = np.zeros((len(count), 2))
    covariance = np.zeros((len(count), 2, 2))
    for i, n in enumerate(count):
        if n < 3:
            continue
        p = vertices[i, :n]
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        area = .5 * cross.sum()
        if area <= 0:
            raise ValueError(f'Reference polygon {i} has nonpositive area {area}')
        center = np.sum((p + q) * cross[:, None], axis=0) / (6 * area)
        xx = np.sum((p[:, 0]**2 + p[:, 0] * q[:, 0] + q[:, 0]**2) * cross) / (12 * area)
        yy = np.sum((p[:, 1]**2 + p[:, 1] * q[:, 1] + q[:, 1]**2) * cross) / (12 * area)
        xy = np.sum((2*p[:, 0]*p[:, 1] + p[:, 0]*q[:, 1] + q[:, 0]*p[:, 1] + 2*q[:, 0]*q[:, 1]) * cross) / (24 * area)
        areas[i] = area
        centroids[i] = center
        covariance[i] = np.array([[xx, xy], [xy, yy]]) - np.outer(center, center)
    return areas, centroids, covariance


def state_metrics(frame, info, reference, counts, centroids, covariance, mass):
    x, v, F, C = (frame[k] for k in ('x', 'v', 'F', 'C'))
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError('Nonpositive deformation determinant in saved frame')
    dx = info['dx_m']
    moment = .25 * dx**2 * np.eye(2) + F @ covariance @ F.swapaxes(-1, -2)
    kinetic = .5 * np.sum(mass * (np.sum(v*v, axis=1) + np.trace(C @ moment @ C.swapaxes(-1, -2), axis1=-2, axis2=-1)))
    E, nu = info['young_Pa'], info['poisson']
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    logJ = np.log(J)
    density = .5*mu*(np.sum(F*F, axis=(1, 2))-2) - mu*logJ + .5*lam*logJ**2
    elastic = np.sum(reference * density)
    sigma = (mu*(F @ F.swapaxes(-1, -2)-np.eye(2)) + lam*logJ[:, None, None]*np.eye(2)) / J[:, None, None]
    return kinetic, elastic, sigma, (mu, lam)


def boundary_drift(frame, interface, ref_vertices, ref_count, ref_centroid, fields,
                   x_initial, F_initial, mass, cut):
    current_vertices = frame['x'][:, None, :] + np.einsum(
        'pij,pkj->pki', frame['F'], ref_vertices - ref_centroid[:, None, :])
    initial_vertices = x_initial[:, None, :] + np.einsum(
        'pij,pkj->pki', F_initial, ref_vertices - ref_centroid[:, None, :])
    vertex_ids = np.arange(ref_vertices.shape[1])[None, :]
    face_mask = (np.abs(initial_vertices[:, :, 0] - cut) < 1e-12)
    face_mask &= vertex_ids < ref_count[:, None]
    face_mask &= mass[:, None] > 0
    faces = []
    for field in (0, 1):
        pts = current_vertices[face_mask & (fields[:, None] == field)]
        if len(pts) < 2:
            raise ValueError(f'Missing material boundary for field {field}')
        order = np.argsort(pts[:, 1], kind='stable')
        faces.append(pts[order])
    plus, minus = interface['plus'], interface['minus']
    # Same field-to-trace convention as the producer: field 0 / plus, field 1 / minus.
    errs = []
    for face, trace in zip(faces, (plus, minus)):
        x_interp = np.interp(trace[:, 1], face[:, 1], face[:, 0])
        errs.append(np.abs(x_interp - trace[:, 0]))
    drift = float(max(np.max(err) for err in errs))
    geometry_gap = float(np.min(faces[0][:, 0]) - np.max(faces[1][:, 0]))
    normal = interface['normal']
    gaps = (plus - minus) @ normal
    return drift, geometry_gap, float(np.min(gaps)), current_vertices


def check_run(run):
    info = json.loads((run / 'input.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    rows = read_history(run / 'history.csv')
    if not summary.get('completed') or len(rows) != 21 or any(row['valid'] != 'True' for row in rows):
        raise ValueError(f'{run}: expected a complete 21-frame valid run')
    steps = np.asarray([float(row['steps']) for row in rows])
    numeric_history = {key: np.asarray([float(row[key]) for row in rows])
                       for key in rows[0] if key != 'valid'}
    if not all(np.isfinite(values).all() for values in numeric_history.values()):
        raise ValueError(f'{run}: nonfinite numeric history value')
    if int(steps[-1]) != int(info['steps']):
        raise ValueError(f'{run}: final accepted step count does not match input')
    if not np.isclose(numeric_history['time_s'][-1], info['duration_s'], atol=1e-12, rtol=0):
        raise ValueError(f'{run}: actual duration does not match requested duration')
    if numeric_history['time_s'][-1] >= info['minimum_end_reflection_time_s']:
        raise ValueError(f'{run}: duration reaches the first end-reflection time')
    E, nu, rho = info['young_Pa'], info['poisson'], info['density_kg_m3']
    acoustic_speed = np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    acoustic_pressure = rho*acoustic_speed*info['initial_speed_m_s']
    if not np.isclose(acoustic_speed, info['linear_wave_speed_m_s'], rtol=1e-12, atol=0):
        raise ValueError(f'{run}: input acoustic wave speed does not match material properties')
    if not np.isclose(acoustic_pressure, info['linear_impact_pressure_Pa'], rtol=1e-12, atol=0):
        raise ValueError(f'{run}: input acoustic pressure does not match rho*c*v0')
    with np.load(run / 'final.npz', allow_pickle=False) as data:
        final = {key: data[key].copy() for key in data.files}
    if not all(np.isfinite(values).all() for values in final.values()):
        raise ValueError(f'{run}: nonfinite array in final.npz')
    interface_areas = final['area']
    normal = final['normal']
    if (not np.isfinite(interface_areas).all() or np.any(interface_areas <= 0)
            or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)):
        raise ValueError(f'{run}: contact interface areas or normal are invalid')
    ref_vertices = final['reference_vertices']
    counts = final['reference_count']
    fields = final['fields']
    areas, centroids, covariance = polygon_moments(ref_vertices, counts)
    mass = info['density_kg_m3'] * areas
    np.testing.assert_allclose(centroids, final['reference_centroid'], atol=1e-16, rtol=0)
    np.testing.assert_allclose(areas, final['reference_area'], atol=1e-18, rtol=1e-12)
    np.testing.assert_allclose(covariance, final['reference_covariance'], atol=1e-18, rtol=1e-11)
    np.testing.assert_allclose(mass, final['mass'], atol=1e-15, rtol=1e-12)
    if not np.isclose(areas.sum(), .04*.01, atol=1e-14, rtol=0):
        raise ValueError(f'{run}: total reference area does not equal 0.04×0.01 m²')

    reference = areas
    row_values = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0] if key != 'valid'}
    frame_values = {key: [] for key in ('time_s', 'kinetic_J_per_m', 'elastic_J_per_m', 'contact_energy_J_per_m',
                                        'momentum_x_kg_m_s', 'momentum_y_kg_m_s', 'plateau_pressure_Pa',
                                        'boundary_drift_m', 'geometry_gap_m', 'trace_gap_m')}
    expected_frames = sorted(run.glob('frame-*.npz'))
    if len(expected_frames) != len(rows):
        raise ValueError(f'{run}: saved frame count does not match history')
    max_final_match = 0.
    max_history_error = {k: 0. for k in ('kinetic_J_per_m', 'elastic_J_per_m', 'contact_energy_J_per_m',
                                           'balance_error_J_per_m', 'minimum_trace_gap_m', 'minimum_geometry_gap_m',
                                           'maximum_trace_geometry_drift_m', 'plateau_pressure_Pa')}
    max_contact_power_error = float(np.max(row_values['max_contact_power_error_J_per_m']))
    max_contact_residual = float(np.max(row_values['max_contact_residual']))
    max_net_moment = float(np.max(row_values['max_contact_net_moment_N']))
    K_series, U_series, V_series, balance_series, plateau_series = [], [], [], [], []
    max_drift, max_abs_momentum_change = 0., 0.
    with np.load(expected_frames[0], allow_pickle=False) as initial_frame:
        frame_x0 = initial_frame['x'].copy()
        frame_F0 = initial_frame['F'].copy()
    initial_energy = info['initial_energy_J_per_m']
    cut = info['cut_m']
    c = np.sqrt((info['young_Pa']*(1-info['poisson']) /
                 ((1+info['poisson'])*(1-2*info['poisson']))) / info['density_kg_m3'])
    p_ref = rho*c*info['initial_speed_m_s']
    p_history = row_values['contact_pressure_Pa']
    # The saved pressure is sampled once per frame and misses sub-frame pulses.
    # The producer's mean pressure integrates the accepted per-step resultants.
    empirical_impulse = float(row_values['mean_contact_pressure_Pa'][-1] * .01 * row_values['time_s'][-1])
    expected_impulse = p_ref * .01 * info['duration_s']
    for idx, path in enumerate(expected_frames):
        with np.load(path, allow_pickle=False) as frame_file:
            frame = {key: frame_file[key].copy() for key in frame_file.files}
        if not all(np.isfinite(values).all() for values in frame.values()):
            raise ValueError(f'{run}: nonfinite array in {path.name}')
        kinetic, elastic, sigma, (mu, lam) = state_metrics(frame, info, reference, counts, centroids, covariance, mass)
        gaps = (frame['plus'] - frame['minus']) @ final['normal']
        contact_energy = .5 * info['contact_stiffness_Pa_per_m'] * np.sum(final['area'] * np.minimum(gaps, 0.)**2)
        time_s = float(frame['time_s'])
        total = kinetic + elastic + contact_energy
        balance = (total - initial_energy + row_values['transfer_loss_J_per_m'][idx]
                   + row_values['boundary_projection_loss_J_per_m'][idx]
                   - row_values['geometry_energy_J_per_m'][idx] - row_values['external_J_per_m'][idx])
        max_history_error['kinetic_J_per_m'] = max(max_history_error['kinetic_J_per_m'], abs(kinetic-row_values['kinetic_J_per_m'][idx]))
        max_history_error['elastic_J_per_m'] = max(max_history_error['elastic_J_per_m'], abs(elastic-row_values['elastic_J_per_m'][idx]))
        max_history_error['contact_energy_J_per_m'] = max(max_history_error['contact_energy_J_per_m'], abs(contact_energy-row_values['contact_energy_J_per_m'][idx]))
        max_history_error['balance_error_J_per_m'] = max(max_history_error['balance_error_J_per_m'], abs(balance-row_values['balance_error_J_per_m'][idx]))
        max_history_error['minimum_trace_gap_m'] = max(max_history_error['minimum_trace_gap_m'], abs(float(gaps.min())-row_values['minimum_trace_gap_m'][idx]))
        drift, geometry_gap, trace_gap, current_vertices = boundary_drift(
            frame, {'plus': frame['plus'], 'minus': frame['minus'], 'normal': final['normal']},
            ref_vertices, counts, centroids, fields, frame_x0, frame_F0,
            mass, cut)
        max_drift = max(max_drift, drift)
        max_history_error['minimum_geometry_gap_m'] = max(max_history_error['minimum_geometry_gap_m'], abs(geometry_gap-row_values['minimum_geometry_gap_m'][idx]))
        max_history_error['maximum_trace_geometry_drift_m'] = max(max_history_error['maximum_trace_geometry_drift_m'], abs(drift-row_values['maximum_trace_geometry_drift_m'][idx]))
        pressure_mask = (np.abs(frame_x0[:, 0] - cut) > .25*c*time_s) & (np.abs(frame_x0[:, 0] - cut) < .6*c*time_s)
        pressure_mask &= (frame_x0[:, 1] > .037) & (frame_x0[:, 1] < .043)
        plateau = float(np.mean(-sigma[pressure_mask, 0, 0])) if pressure_mask.any() else 0.
        max_history_error['plateau_pressure_Pa'] = max(max_history_error['plateau_pressure_Pa'], abs(plateau-row_values['plateau_pressure_Pa'][idx]))
        momentum = np.sum(mass[:, None] * frame['v'], axis=0)
        for key, value in (('time_s', time_s), ('kinetic_J_per_m', kinetic), ('elastic_J_per_m', elastic),
                           ('contact_energy_J_per_m', contact_energy), ('momentum_x_kg_m_s', momentum[0]),
                           ('momentum_y_kg_m_s', momentum[1]), ('plateau_pressure_Pa', plateau),
                           ('boundary_drift_m', drift), ('geometry_gap_m', geometry_gap), ('trace_gap_m', trace_gap)):
            frame_values[key].append(value)
        K_series.append(kinetic); U_series.append(elastic); V_series.append(contact_energy)
        balance_series.append(balance); plateau_series.append(plateau)
        max_abs_momentum_change = max(max_abs_momentum_change,
                                      float(np.max(np.abs(momentum - (frame_values['momentum_x_kg_m_s'][0], frame_values['momentum_y_kg_m_s'][0])))))
        if idx == len(expected_frames)-1:
            for key in ('x', 'v', 'F', 'C', 'plus', 'minus'):
                max_final_match = max(max_final_match, float(np.max(np.abs(frame[key]-final[key]))))

    # Verify all accepted history points have physical frames and initial energy.
    np.testing.assert_allclose(row_values['time_s'], frame_values['time_s'], atol=1e-15, rtol=0)
    np.testing.assert_allclose(K_series[0]+U_series[0]+V_series[0], initial_energy, atol=1e-11, rtol=1e-10)
    for metric, error in max_history_error.items():
        if error > (1e-8 if metric not in ('minimum_trace_gap_m', 'minimum_geometry_gap_m', 'maximum_trace_geometry_drift_m') else 1e-10):
            raise ValueError(f'{run}: reconstructed {metric} disagrees with history by {error:g}')
    if max_final_match > 1e-14:
        raise ValueError(f'{run}: final.npz differs from final frame by {max_final_match:g}')

    final_state = expected_frames[-1]
    with np.load(final_state, allow_pickle=False) as f:
        x_final, F_final = f['x'].copy(), f['F'].copy()
    _, _, sigma_final, _ = state_metrics({'x': x_final, 'v': np.zeros_like(x_final), 'F': F_final,
                                          'C': np.zeros_like(F_final)}, info, reference, counts, centroids, covariance, mass)
    t_end = row_values['time_s'][-1]
    expected_profile = np.abs(x_final[:, 0]-cut) <= c*t_end
    # Bin the deformed longitudinal position; retain every point, including transient overshoot.
    edges = np.linspace(0., .08, info['grid']+1)
    profile = -sigma_final[:, 0, 0]
    bins = np.clip(np.digitize(x_final[:, 0], edges)-1, 0, len(edges)-2)
    profile_means = np.array([profile[bins == i].mean() if np.any(bins == i) else np.nan for i in range(len(edges)-1)])
    profile_x = .5*(edges[:-1]+edges[1:])
    profile_reference = p_ref * (np.abs(profile_x-cut) <= c*t_end)

    ledger = final['ledger']
    final_row = len(rows)-1
    ledger_mismatches = dict(external_J_per_m=float(ledger[:3].sum()-row_values['external_J_per_m'][final_row]),
                             transfer_loss_J_per_m=float(ledger[3]-row_values['transfer_loss_J_per_m'][final_row]),
                             boundary_projection_loss_J_per_m=float(ledger[4]-row_values['boundary_projection_loss_J_per_m'][final_row]),
                             geometry_energy_J_per_m=float(ledger[5]-row_values['geometry_energy_J_per_m'][final_row]),
                             ledger_balance_J_per_m=float(ledger[7]-row_values['balance_error_J_per_m'][final_row]))
    if max(abs(value) for value in ledger_mismatches.values()) > 1e-9:
        raise ValueError(f'{run}: stored final energy ledger does not match history')
    max_energy_balance = float(np.max(np.abs(balance_series)))
    max_momentum_drift = max_abs_momentum_change
    if max_energy_balance > 1e-9 or max_momentum_drift > 1e-10:
        raise ValueError(f'{run}: energy balance or total momentum drift exceeds tolerance')
    if (max_contact_power_error > 1e-9 or max_contact_residual > 1e-9 or max_net_moment > 1e-9
            or np.max(row_values['max_bulk_residual']) > 1e-8):
        raise ValueError(f'{run}: contact/bulk solver diagnostics exceed tolerance')
    report = dict(run=run.name, grid=info['grid'], dt_s=info['dt_s'], stiffness_Pa_per_m=info['contact_stiffness_Pa_per_m'],
                  completed=True, duration_s=float(t_end), requested_duration_s=info['duration_s'],
                  reference_area_m2=float(areas.sum()), reference_area_error_m2=float(areas.sum()-.04*.01),
                  reference_centroid_max_error_m=float(np.max(np.abs(centroids-final['reference_centroid']))),
                  reference_covariance_max_error_m2=float(np.max(np.abs(covariance-final['reference_covariance']))),
                  max_frame_history_errors=max_history_error, max_final_frame_npz_error=max_final_match,
                  final_stored_ledger_mismatches=ledger_mismatches,
                  max_energy_balance_J_per_m=max_energy_balance,
                  final_numerical_loss_fraction=float(row_values['numerical_loss_fraction'][-1]),
                  max_contact_power_error_J_per_m=max_contact_power_error,
                  max_contact_residual=max_contact_residual, max_contact_net_moment_N=max_net_moment,
                  initial_momentum_kg_m_s=frame_values['momentum_x_kg_m_s'][0],
                  max_momentum_drift_kg_m_s=max_momentum_drift,
                  linear_wave_speed_m_s=float(c), linear_pressure_reference_Pa=float(p_ref),
                  measured_contact_impulse_Ns_per_m=empirical_impulse,
                  acoustic_impulse_reference_Ns_per_m=float(expected_impulse),
                  contact_impulse_relative_error=float(empirical_impulse/expected_impulse-1),
                  contact_impulse_reconstruction='Final per-step accumulated mean_contact_pressure × contact height × elapsed time; frame-sampled pressure can miss intra-frame transients',
                  maximum_sampled_contact_pressure_Pa=float(np.max(p_history)),
                  maximum_sampled_contact_pressure_ratio_to_acoustic_reference=float(np.max(p_history)/p_ref),
                  final_mean_contact_pressure_Pa=float(row_values['mean_contact_pressure_Pa'][-1]),
                  final_mean_contact_pressure_error_Pa=float(row_values['mean_contact_pressure_Pa'][-1]-p_ref),
                  final_mean_contact_pressure_relative_error=float(row_values['mean_contact_pressure_Pa'][-1]/p_ref-1),
                  final_reconstructed_plateau_pressure_Pa=float(plateau_series[-1]),
                  final_plateau_pressure_error_Pa=float(plateau_series[-1]-p_ref),
                  final_plateau_pressure_relative_error=float(plateau_series[-1]/p_ref-1),
                  max_nonempty_plateau_pressure_error_Pa=float(max(
                      (abs(p-p_ref) for p, t in zip(plateau_series, frame_values['time_s'])
                       if np.any((np.abs(frame_x0[:, 0]-cut) > .25*c*t) &
                                 (np.abs(frame_x0[:, 0]-cut) < .6*c*t) &
                                 (frame_x0[:, 1] > .037) & (frame_x0[:, 1] < .043))), default=0.)),
                  maximum_boundary_trace_drift_m=max_drift,
                  final_minimum_geometry_gap_m=float(frame_values['geometry_gap_m'][-1]),
                  final_minimum_trace_gap_m=float(frame_values['trace_gap_m'][-1]),
                  expected_acoustic_stress_profile_points=int(np.sum(expected_profile)),
                  mesh_convergence_established=False,
                  accuracy_scope='Finite-duration acoustic/contact comparison; excludes reflected waves and does not establish convergence')
    report['source_commit'] = '1b749b1'
    report['source_sha256_from_run_input'] = info['source_sha256']
    report['final_snapshot_sha256'] = hashlib.sha256((run/'final.npz').read_bytes()).hexdigest()
    frame_digest = hashlib.sha256()
    for path in expected_frames:
        frame_digest.update(path.read_bytes())
    report['concatenated_frames_sha256'] = frame_digest.hexdigest()
    profile_data = dict(x=profile_x.tolist(), pressure_Pa=profile_means.tolist(), acoustic_reference_Pa=profile_reference.tolist(),
                        transient_time_s=frame_values['time_s'], plateau_pressure_Pa=frame_values['plateau_pressure_Pa'],
                        numerical_loss_fraction=row_values['numerical_loss_fraction'], energy_balance_J_per_m=balance_series,
                        reconstructed_kinetic_J_per_m=K_series, reconstructed_elastic_J_per_m=U_series,
                        reconstructed_contact_energy_J_per_m=V_series, measured_contact_pressure_Pa=p_history.tolist(),
                        mean_contact_pressure_Pa=row_values['mean_contact_pressure_Pa'])
    report['history_sha256'] = hashlib.sha256((run/'history.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return report, profile_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, profiles = {}, {}
    for run in args.runs:
        reports[run.name], profiles[run.name] = check_run(run)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(runs=reports, mesh_convergence_established=False,
                  validation_scope='Independent NumPy reconstruction of stored frames, reference polygon moments, contact potential, energy, momentum, and acoustic comparisons; not a new trajectory solve')
    (args.output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name, p in profiles.items():
        t = np.asarray(p['transient_time_s'])*1e6
        label = name.replace('domain-contact-collision-', '')
        axes[0].plot(t, p['measured_contact_pressure_Pa'], label=label)
        axes[1].plot(t, 100*np.asarray(p['numerical_loss_fraction']), label=label)
        axes[2].plot(p['x'], np.asarray(p['pressure_Pa'])/1e3, marker='.', label=f'{label} computed')
    reference = next(iter(profiles.values()))
    p_reference = reports[next(iter(reports))]['linear_pressure_reference_Pa']
    final_time = np.asarray(reference['transient_time_s'])[-1]
    c_reference = reports[next(iter(reports))]['linear_wave_speed_m_s']
    cut_reference = json.loads((args.runs[0]/'input.json').read_text())['cut_m']
    acoustic_x = np.linspace(0., .08, 801)
    acoustic_profile = p_reference*(np.abs(acoustic_x-cut_reference) <= c_reference*final_time)
    axes[0].axhline(p_reference, color='0.3', ls='--', lw=1., label='acoustic reference')
    axes[2].plot(acoustic_x, acoustic_profile/1e3, color='0.2', ls='--', lw=1.2, label='acoustic reference')
    axes[0].set(xlabel='Time (µs)', ylabel='Contact pressure (Pa)', title='Contact pressure transient')
    axes[1].set(xlabel='Time (µs)', ylabel='Numerical loss (%)', title='Transfer loss fraction')
    axes[2].set(xlabel='Current x (m)', ylabel='Compression stress (kPa)', title='Final longitudinal profile')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(args.output/'domain-contact-collision.png', dpi=150)
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
