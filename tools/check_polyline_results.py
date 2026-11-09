"""Check advected crack fields, reflected loading and resolution comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load(directory):
    summary = json.loads((directory/'summary.json').read_text())
    rows = list(csv.DictReader((directory/'history.csv').open()))
    assert summary['all_states_valid'] and len(rows) == summary['frames']
    assert all(row['valid'] == 'True' for row in rows)
    columns = {key:np.array([float(r[key]) for r in rows]) for key in rows[0] if key != 'valid'}
    assert all(np.isfinite(v).all() for v in columns.values())
    assert np.diff(columns['extension_m']).min() >= 0
    assert columns['pending_extension_m'].min() >= 0
    assert columns['pending_extension_m'].max() < summary['insertion_spacing_m']
    assert columns['min_face_gap_m'].min() > -1e-10
    E, nu = summary['young_Pa'], summary['poisson']
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    frames = []
    for i, row in enumerate(rows):
        with np.load(directory/f'frame-{i:03d}.npz', allow_pickle=False) as data:
            frame = {key:data[key] for key in data.files}
        x, F, path = frame['x'], frame['F'], frame['crack']
        assert all(np.isfinite(v).all() for v in frame.values())
        assert len(path) == int(row['vertices'])
        np.testing.assert_allclose(path, .5*(frame['crack_plus']+frame['crack_minus']), atol=1e-15)
        np.testing.assert_allclose(path[-1], [float(row['tip_x_m']), float(row['tip_y_m'])], atol=1e-15)
        J = np.linalg.det(F)
        assert J.min() > 0
        density = .5*mu*(np.sum(F*F, axis=(1,2))-2)-mu*np.log(J)+.5*lam*np.log(J)**2
        np.testing.assert_allclose(.04*.02*density.mean(), float(row['elastic_J_per_m']), atol=1e-8, rtol=1e-10)
        stress = (mu*(F @ F.swapaxes(-1,-2)-np.eye(2))+lam*np.log(J)[:,None,None]*np.eye(2))/J[:,None,None]
        np.testing.assert_allclose(np.linalg.eigvalsh(stress)[:,-1], frame['principal_stress'], atol=1e-5, rtol=1e-9)
        edges = np.diff(path, axis=0)
        assert np.linalg.norm(edges, axis=-1).min() > .1*summary['insertion_spacing_m']
        for a in range(len(edges)):
            for b in range(a+2, len(edges)):
                cross = lambda u,v:u[0]*v[1]-u[1]*v[0]
                den = cross(edges[a], edges[b])
                if abs(den) < 1e-20:
                    continue
                offset = path[b]-path[a]
                t, u = cross(offset, edges[b])/den, cross(offset, edges[a])/den
                assert not (0 <= t <= 1 and 0 <= u <= 1), 'Self-intersecting saved path'
        frames.append(frame)
    return summary, columns, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--fine', type=Path, required=True)
    parser.add_argument('--mirror', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directories = dict(coarse=args.coarse, fine=args.fine, mirror=args.mirror, held=args.held)
    runs = {key:load(directory) for key,directory in directories.items()}
    result = {}
    for key,(info, rows, frames) in runs.items():
        path = frames[-1]['crack']
        angle = np.rad2deg(np.arctan2(np.diff(path, axis=0)[:,1], np.diff(path, axis=0)[:,0]))
        result[key] = dict(grid=info['grid'], frames=info['frames'],
                           final_extension_m=float(rows['extension_m'][-1]),
                           final_pending_extension_m=float(rows['pending_extension_m'][-1]),
                           initial_segment_final_angle_deg=float(angle[0]),
                           first_inserted_segment_final_angle_deg=float(angle[1]) if len(angle)>1 else None,
                           final_mouth_gap_m=float(np.linalg.norm(frames[-1]['crack_plus'][0]-frames[-1]['crack_minus'][0])),
                           maximum_sampled_strain=float(rows['max_strain'].max()),
                           final_vertices=len(path),
                           history_sha256=hashlib.sha256((directories[key]/'history.csv').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    assert np.all(runs['held'][1]['vertices'] == 2) and np.all(runs['held'][1]['extension_m'] == 0)
    mirror_error, path_error = 0., 0.
    reflection = np.diag([1., -1.])
    ny = len(np.unique(runs['coarse'][2][0]['x'][:,1]))
    for a,b in zip(runs['coarse'][2], runs['mirror'][2]):
        reflected = a['x'].reshape(-1,ny,2)[:,::-1].reshape(-1,2).copy()
        reflected[:,1] = .06-reflected[:,1]
        mirror_error = max(mirror_error, float(np.max(np.abs(reflected-b['x']))))
        reflected_F = a['F'].reshape(-1,ny,2,2)[:,::-1].reshape(-1,2,2)
        np.testing.assert_allclose(reflection @ reflected_F @ reflection, b['F'], atol=1e-9, rtol=1e-9)
        path = a['crack'].copy(); path[:,1] = .06-path[:,1]
        assert path.shape == b['crack'].shape
        path_error = max(path_error, float(np.max(np.abs(path-b['crack']))))
    assert mirror_error < 1e-10 and path_error < 1e-10
    result['checks'] = dict(max_reflected_particle_error_m=mirror_error, max_reflected_path_error_m=path_error,
                            held_crack_did_not_grow=True, all_paths_simple=True,
                            final_extension_difference_m=abs(result['fine']['final_extension_m']-result['coarse']['final_extension_m']),
                            mesh_convergence_established=False, fracture_energy_validation=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for key,label in [('coarse','h = 0.625 mm'),('fine','h = 0.469 mm'),('held','Held crack')]:
        info, rows, frames = runs[key]
        axes[0].plot(rows['time_s']*1e6,rows['extension_m']*1e3,label=label)
        path = frames[-1]['crack']
        axes[1].plot(path[:,0]*1e3,path[:,1]*1e3,label=label)
        axes[2].plot(rows['time_s']*1e6,rows['max_face_gap_m']*1e3,label=label)
    axes[0].set(xlabel='Time (microseconds)',ylabel='Inserted extension (mm)')
    axes[1].set(xlabel='x (mm)',ylabel='y (mm)',title='Computed crack centreline')
    axes[1].set_aspect('equal',adjustable='datalim')
    axes[2].set(xlabel='Time (microseconds)',ylabel='Maximum face opening (mm)')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(args.output/'polyline-history.png',dpi=140)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
