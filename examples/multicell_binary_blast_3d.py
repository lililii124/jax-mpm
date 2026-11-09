"""Replay a connected binary shared-cell blast candidate with full-domain TL-MPM."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpm_jax.compact_tl import (elastic_energy, face_stress, initial_state,
    point_material, reference_grid, rectilinear_support, step, surface_weights)
from mpm_jax.total_lagrangian import momentum, sample_nodal
from tools.projected_pressure import allocate_projected_pressure
from tools.reconstruct_multicell_blast import mask_digest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int)
    p.add_argument('--frame-interval', type=int, default=200)
    p.add_argument('--empty-column-load', choices=('omit','redistribute'), default='omit')
    p.add_argument('--split-disconnected-support', action='store_true')
    p.add_argument('--track-subframe-peak', action='store_true')
    p.add_argument('--track-bottom-point-peaks', action='store_true')
    p.add_argument('--reject-if-peak-ge', type=float)
    p.add_argument('--allow-disconnected-diagnostic', action='store_true')
    args = p.parse_args()
    if args.track_bottom_point_peaks and not args.track_subframe_peak:
        p.error('--track-bottom-point-peaks requires --track-subframe-peak')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory must be empty')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    report = json.loads(args.geometry.with_suffix('.json').read_text(encoding='utf-8'))
    if report.get('geometry_sha256') != digest(args.geometry):
        raise ValueError('Geometry does not match its reconstruction report')
    if report['inputs_sha256'] != digest(args.inputs) or report['optimization_signature'] != inputs['signature']:
        raise ValueError('Geometry metadata and optimization settings differ')
    if (not report['faces_connected'] or report['components'] != 1) and not args.allow_disconnected_diagnostic:
        raise ValueError('Binary geometry is disconnected; use only for an explicitly labelled diagnostic')
    s = inputs['binding']['settings']
    nsteps = args.steps if args.steps is not None else int(s['steps'])
    if nsteps < 1 or args.frame_interval < 1:
        raise ValueError('Steps and frame interval must be positive')
    if args.reject_if_peak_ge is not None and (not np.isfinite(args.reject_if_peak_ge) or args.reject_if_peak_ge <= 0):
        raise ValueError('Rejection peak must be finite and positive')
    with np.load(args.geometry, allow_pickle=False) as data:
        pos = np.asarray(data['positions'])
        volume = float(data['particle_volume'])
        solid = np.asarray(data['solid_mask'], dtype=bool)
    if report['solid_mask_sha256'] != mask_digest(solid):
        raise ValueError('Solid mask differs from its reconstruction report')
    nx, ny, nz = map(int, s['cell'])
    gx, gy, gz = nx * 3, ny * 3, nz + 2 * int(s['skin_layers'])
    h = np.asarray(s['cell_size']) / (nx, ny, nz)
    if solid.shape != (gz, gy, gx) or len(pos) != int(solid.sum()) * 8:
        raise ValueError('Geometry dimensions or particle count are inconsistent')
    if not np.isclose(len(pos) * volume * s['solid_density'], report['binary_mass_kg'], rtol=1e-12):
        raise ValueError('Geometry mass differs from its reconstruction report')
    support = rectilinear_support(pos, volume, h)
    grid_support = support
    split = None
    if args.split_disconnected_support:
        from mpm_jax.multifield_support import split_disconnected_node_support
        split = split_disconnected_node_support(support)
        support = split.support
    ref = reference_grid(support, float(s['solid_density']))
    skin = (pos[:, 2] < s['skin']) | (pos[:, 2] > gz * h[2] - s['skin'])
    mat = point_material(jnp.asarray(np.where(skin, s['skin_young'], s['core_young'])), .28)
    xyz = support.nodes
    edge = np.asarray(s.get('fixed_edge_width', [h[0] * 1.01, h[1] * 1.01]), dtype=float)
    if edge.shape != (2,) or not np.isfinite(edge).all() or np.any(edge <= 0) or np.any(edge >= (gx*h[0]/2, gy*h[1]/2)):
        raise ValueError('Fixed edge widths must be positive and less than half the domain')
    fixed = ((xyz[:, 0] < edge[0]) | (xyz[:, 0] > gx*h[0]-edge[0]) |
             (xyz[:, 1] < edge[1]) | (xyz[:, 1] > gy*h[1]-edge[1]))[:, None]
    fixed = jnp.asarray(np.broadcast_to(fixed, xyz.shape))
    xy = np.stack(np.meshgrid((np.arange(gx)+.5)*h[0], (np.arange(gy)+.5)*h[1], indexing='ij'), -1).reshape(-1, 2)
    area = np.full(len(xy), h[0]*h[1])
    center = np.array((gx*h[0]/2, gy*h[1]/2))
    footprint = np.exp(-.5*np.sum(((xy-center)/s['blast_radius'])**2, axis=1))
    footprint /= np.sum(footprint*area)/np.sum(area)
    occupied = np.any(solid, axis=0).T.reshape(-1)
    footprint = allocate_projected_pressure(footprint.reshape(gx,gy),occupied.reshape(gx,gy),
                                            args.empty_column_load).reshape(-1)
    top_voxel = (gz-1-np.argmax(solid[::-1], axis=0)).T.reshape(-1)
    bottom_voxel = np.argmax(solid, axis=0).T.reshape(-1)
    top = (top_voxel+1)*h[2]
    bottom = bottom_voxel*h[2]
    owner_x = np.repeat(np.arange(gx), gy)[occupied]
    owner_y = np.tile(np.arange(gy), gx)[occupied]
    xy = xy[occupied]
    area, footprint = area[occupied], footprint[occupied]
    fi, fw = surface_weights(grid_support, np.column_stack((xy, top[occupied])))
    bi, bw = surface_weights(grid_support, np.column_stack((xy, bottom[occupied])))
    if split is not None:
        from mpm_jax.multifield_support import surface_fields
        fi = surface_fields(split, fi, np.column_stack((owner_x, owner_y, top_voxel[occupied])), fw)
        bi = surface_fields(split, bi, np.column_stack((owner_x, owner_y, bottom_voxel[occupied])), bw)
    load = jnp.zeros((len(xyz), 3), dtype=jnp.float64).at[jnp.asarray(fi.reshape(-1)), 2].add(
        -s['pressure']*jnp.asarray((fw*area[:, None]*footprint[:, None]).reshape(-1)))
    dt = float(s['dt'])
    pulse = float(s['pulse'])
    wave = math.sqrt(max(s['skin_young'],s['core_young'])*(1-.28)/(s['solid_density']*(1+.28)*(1-2*.28)))
    cfl = dt*wave*math.sqrt(float(np.sum(1/h**2)))
    if cfl > s['cfl_limit']:
        raise ValueError('Time step exceeds the configured CFL limit')
    state = initial_state(ref, mat, jnp.zeros_like(load))

    def advance(st, length):
        def advance_one(current):
            t0 = current.time
            t1 = t0 + dt
            p0 = jnp.sin(jnp.pi*jnp.minimum(t0/pulse, 1.))**2*(t0 < pulse)
            p1 = jnp.sin(jnp.pi*jnp.minimum(t1/pulse, 1.))**2*(t1 < pulse)
            return step(current, ref, mat, dt, load*p0, load*p1, fixed)[0]
        if args.track_subframe_peak:
            def tracked(carry, _):
                current, peak = carry
                following = advance_one(current)
                back = jnp.sum(following.displacement[jnp.asarray(bi), 2]*jnp.asarray(bw), axis=1)
                value = jnp.abs(back) if args.track_bottom_point_peaks else jnp.max(jnp.abs(back))
                return (following, jnp.maximum(peak, value)), None
            initial_peak = (jnp.zeros(len(xy), dtype=load.dtype) if args.track_bottom_point_peaks
                            else jnp.asarray(0., dtype=load.dtype))
            (following, peak), _ = jax.lax.scan(tracked, (st, initial_peak), None, length=length)
            return following, peak
        def body(current, _):
            return advance_one(current), None
        return jax.lax.scan(body, st, None, length=length)[0], jnp.asarray(0., dtype=load.dtype)
    advance = jax.jit(advance, static_argnums=(1,))
    @jax.jit
    def observe(st):
        displacement = sample_nodal(st.displacement, ref)
        stress = face_stress(st.F, mat)
        back = jnp.sum(st.displacement[jnp.asarray(bi), 2]*jnp.asarray(bw), axis=1)
        kinetic = .5*jnp.sum(ref.nodal_mass[:, None]*st.velocity**2)
        energy = elastic_energy(st.F, ref, mat)
        linear, angular = momentum(st, ref)
        return (displacement, stress, back, kinetic, energy,
                jnp.min(jnp.linalg.det(st.F)), jnp.linalg.norm(linear-st.impulse),
                jnp.linalg.norm(angular-st.angular_impulse))

    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source_files = ('examples/multicell_binary_blast_3d.py', 'tools/reconstruct_multicell_blast.py', 'tools/projected_pressure.py', 'mpm_jax/compact_tl.py',
                    'mpm_jax/total_lagrangian.py', 'mpm_jax/materials/elastic.py')
    if split is not None:
        source_files += ('mpm_jax/multifield_support.py',)
    meta = dict(geometry_sha256=digest(args.geometry), solid_mask_sha256=mask_digest(solid), inputs_sha256=digest(args.inputs),
                source_sha256={name: digest(root/name) for name in source_files},
                jax_version=jax.__version__, numpy_version=np.__version__,
                settings=s, steps=nsteps, frame_interval=args.frame_interval, cfl=cfl,
                empty_column_load=args.empty_column_load,
                support_mode='split-disconnected' if split is not None else 'single',
                peak_sampling='every-step' if args.track_subframe_peak else 'saved-frames',
                bottom_point_peak_sampling='every-step' if args.track_bottom_point_peaks else 'not-recorded',
                rejection_peak_m=args.reject_if_peak_ge,
                back_sampling='column-underside',
                bottom_plane_sample_fraction=float(np.mean(bottom_voxel[occupied] == 0)),
                binary_mass_kg=report['binary_mass_kg'], particles=len(pos),
                connected_faces=report['faces_connected'], detached_voxels=report['detached_solid_voxels'],
                connected_components=report['components'], populated_column_fraction=float(np.mean(occupied)),
                diagnostic_only=args.allow_disconnected_diagnostic,
                limitation='Reference-grid elastic model without self-contact, plasticity or failure; projected pressure acts on the uppermost solid voxel in each populated column and does not resolve pressure on cavity sides.')
    (args.output/'inputs.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    reference_data = dict(nodes=xyz, spacing=h, shape=np.asarray(grid_support.shape),
                          support_mode=np.asarray('split-disconnected' if split is not None else 'single'))
    if args.track_bottom_point_peaks:
        reference_data['bottom_sample_columns'] = np.column_stack((owner_x, owner_y))
    if split is not None:
        reference_data.update(grid_nodes=grid_support.nodes, voxel_coordinates=split.voxel_coordinates,
                              pair_keys=split.pair_keys, pair_fields=split.pair_fields)
    np.savez_compressed(args.output/'reference.npz', **reference_data)
    rows = []
    completed = 0
    interval_peak = np.zeros(len(xy)) if args.track_bottom_point_peaks else 0.
    on_bottom_plane = bottom_voxel[occupied] == 0
    while True:
        displacement, stress, back, kinetic, energy, min_j, pe, ae = observe(state)
        back = np.abs(np.asarray(back))
        back_peak = float(back.max())
        row = dict(step=completed, time_s=float(state.time), valid=bool(state.valid),
                   back_displacement_max_m=back_peak, back_displacement_p90_m=float(np.percentile(back, 90)),
                   back_displacement_median_m=float(np.median(back)), kinetic_J=float(kinetic), elastic_J=float(energy),
                   pressure_work_J=float(state.body_work), boundary_work_J=float(state.boundary_work),
                   energy_residual_J=float(kinetic+energy-state.body_work-state.boundary_work),
                   min_J=float(min_j), momentum_residual=float(pe), angular_momentum_residual=float(ae))
        if args.track_subframe_peak:
            row['back_displacement_interval_peak_m'] = max(back_peak, float(np.max(interval_peak)))
        if np.any(on_bottom_plane):
            row['bottom_plane_displacement_max_m'] = float(back[on_bottom_plane].max())
        if np.any(~on_bottom_plane):
            row['internal_underside_displacement_max_m'] = float(back[~on_bottom_plane].max())
        if not row['valid'] or not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError(f'Invalid state at step {completed}')
        frame = dict(position=pos+np.asarray(displacement), displacement=np.asarray(displacement),
                     von_mises=np.asarray(stress), nodal_displacement=np.asarray(state.displacement))
        if args.track_bottom_point_peaks:
            frame['bottom_interval_peak_m'] = np.maximum(back, np.asarray(interval_peak))
        np.savez_compressed(args.output/f'frame-{completed:07d}.npz', **frame)
        rows.append(row)
        history_temp = args.output/'history.json.tmp'
        history_temp.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        history_temp.replace(args.output/'history.json')
        print(json.dumps(row, allow_nan=False), flush=True)
        if completed == nsteps:
            break
        peak_field = 'back_displacement_interval_peak_m' if args.track_subframe_peak else 'back_displacement_max_m'
        observed_peak = max(item[peak_field] for item in rows)
        if args.reject_if_peak_ge is not None and observed_peak >= args.reject_if_peak_ge:
            outcome = dict(reason='peak-threshold', threshold_m=args.reject_if_peak_ge,
                           observed_peak_m=observed_peak, step=completed)
            temp = args.output/'outcome.json.tmp'
            temp.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            temp.replace(args.output/'outcome.json')
            break
        count = min(args.frame_interval, nsteps-completed)
        state, interval_peak = advance(state, count)
        completed += count


if __name__ == '__main__':
    main()
