"""Independent higher-grid forward confirmation and particle-frame export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.materials import elastic
from mpm_jax.total_lagrangian import initial_state, reference_grid, sample_nodal, step
from examples.combined_bracket_design import (
    DT, HORIZON, RHO, SIGMA_ALLOW, STATIONS, VOL_FRACTION_MAX, WIDTH_MIN,
    X0, X1, Y0, Y1, VOID_FLOOR, cloud, occupancy, rail_occupancy, setup,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=48)
    parser.add_argument('--horizon', type=float, default=HORIZON)
    parser.add_argument('--dt', type=float, default=DT)
    parser.add_argument('--controls', type=float, nargs=10, required=True,
                        help='the ten saved optimized latent controls in upper-then-lower order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    jax.config.update('jax_enable_x64', True)
    args.output.mkdir(parents=True, exist_ok=False)
    config, xyz, dv, loads, masks, counts, metrics, _, bc, cfl = setup(
        args.grid, args.horizon, args.dt)
    q = jnp.asarray(args.controls, dtype=jnp.float64)
    q0 = jnp.zeros_like(q)
    eval_forward = jax.jit(metrics)

    def audit(q_value):
        merit, (response, vf, ks, per_case_tuple) = eval_forward(q_value)
        per_case = np.column_stack([np.asarray(col) for col in per_case_tuple])
        phi = np.asarray(occupancy(xyz, q_value), dtype=np.float64)
        mass = RHO * dv * phi
        patch_mass = np.asarray([mass[np.asarray(masks[k])].sum() for k in range(3)])
        rail_phi = np.asarray(rail_occupancy(xyz, q_value))
        solid_rail = rail_phi >= .5
        points = np.asarray(xyz)
        pitch = .5 / args.grid
        widths = []
        for rail in range(2):
            for station in STATIONS:
                near = np.abs(points[:, 0] - station) <= pitch
                section = solid_rail & near & (
                    points[:, 1] > .44 if rail == 0 else points[:, 1] < .44)
                if not section.any():
                    raise RuntimeError(f'empty sampled ligament section at x={station}')
                widths.append(float(Y1 - points[section, 1].min()) if rail == 0
                              else float(points[section, 1].max() - Y0))
        ref = reference_grid(xyz, dv*jnp.asarray(phi), RHO, config)
        valid = bool(np.all(per_case[:, 5] == 1) and bool(ref.valid))
        result = {
            'grid': args.grid, 'particles': int(len(xyz)), 'steps': int(round(args.horizon/args.dt)),
            'dt_s': args.dt, 'horizon_s': args.horizon, 'cfl': cfl,
            'response_normalized_squared': float(response), 'merit': float(merit),
            'volume_fraction': float(vf), 'volume_fraction_limit': VOL_FRACTION_MAX,
            'total_mass_kg': float(mass.sum()), 'load_patch_mass_kg': patch_mass.tolist(),
            'p24_stress_upper_bound_pa': float(ks), 'stress_limit_pa': SIGMA_ALLOW,
            'peak_solid_stress_pa': float(np.max(per_case[:, 2])),
            'peak_gray_stress_pa': float(np.max(per_case[:, 3])),
            'peak_floor_stress_pa': float(np.max(per_case[:, 4])),
            'solid_particle_count': int(np.count_nonzero(phi >= .5)),
            'gray_particle_count': int(np.count_nonzero((phi >= .1)&(phi < .5))),
            'floor_particle_count': int(np.count_nonzero(phi < .1)),
            'sampled_min_width_m_by_rail_station': widths,
            'sampled_width_pitch_tolerance_m': pitch,
            'load_cases': [
                {'name': name, 'response_m2': float(row[0]), 'p24_stress_bound_pa': float(row[1]),
                 'peak_solid_stress_pa': float(row[2]), 'peak_gray_stress_pa': float(row[3]),
                 'peak_floor_stress_pa': float(row[4]), 'all_steps_valid': bool(row[5])}
                for name, row in zip(('axial_tension_100N','transverse_bending_100N','pure_torsion_15Nm'),per_case)
            ],
            'reference_grid_valid': bool(ref.valid), 'all_steps_valid': valid,
            'constraints_pass': bool(valid and vf <= VOL_FRACTION_MAX+1e-9 and ks <= SIGMA_ALLOW+1e-6
                                     and min(widths)+pitch >= WIDTH_MIN-1e-12),
        }
        return result, phi

    initial_result, phi0 = audit(q0)
    optimized_result, phi = audit(q)
    for label, result in (('initial', initial_result), ('optimized', optimized_result)):
        print(f"FORWARD_{label.upper()} grid={args.grid} particles={result['particles']} "
              f"response={result['response_normalized_squared']:.9g} vf={result['volume_fraction']:.8g} "
              f"stress={result['p24_stress_upper_bound_pa']:.8g} valid={result['all_steps_valid']}", flush=True)

    # Save the representative bending history at 15-step intervals using the
    # same fixed point cloud, load pulse, boundary conditions, and TL-MPM step.
    nsteps = int(round(args.horizon/args.dt))
    frame_stride = 15
    if nsteps % frame_stride:
        raise ValueError('forward movie requires horizon/dt divisible by 15 steps')
    mat = elastic(jnp.asarray(2.0e5), jnp.asarray(.25))
    t = jnp.arange(nsteps+1, dtype=xyz.dtype) * args.dt
    ramp = min(.001, args.horizon/8)
    rise = jnp.where(t < ramp, .5*(1-jnp.cos(jnp.pi*jnp.clip(t/ramp,0,1))),1.)
    fall_t = jnp.clip((t-(args.horizon-ramp))/ramp,0,1)
    pulse = rise*jnp.where(t > args.horizon-ramp,.5*(1+jnp.cos(jnp.pi*fall_t)),1.)
    pairs = jnp.stack((pulse[:-1],pulse[1:]),axis=1)
    blocks = pairs.reshape((-1,frame_stride,2))
    zero = jnp.zeros_like(xyz)
    load_case = 1

    def capture(q_value):
        phi_value = occupancy(xyz,q_value)
        ref_value = reference_grid(xyz,dv*phi_value,RHO,config)
        state0 = initial_state(ref_value,mat,config,zero)
        acceleration = loads[load_case] / ref_value.particle_mass[:,None]
        def advance(state,pair):
            return step(state,ref_value,mat,config,pair[0]*acceleration,
                        pair[1]*acceleration,bc,bc)[0], None
        def capture_chunk(state,chunk):
            state_next,_ = jax.lax.scan(advance,state,chunk)
            position = xyz + sample_nodal(state_next.displacement,ref_value)
            return state_next,position
        _, positions = jax.lax.scan(capture_chunk,state0,blocks)
        return jnp.concatenate((xyz[None,:,:],positions),axis=0)

    capture_jit = jax.jit(capture)
    p0 = np.asarray(capture_jit(q0), dtype=np.float32)
    p1 = np.asarray(capture_jit(q), dtype=np.float32)
    if p0.shape != (nsteps//frame_stride+1,len(xyz),3) or not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise RuntimeError('saved particle trajectory has invalid shape or nonfinite coordinates')
    np.savez_compressed(args.output/'trajectory.npz',
        points=np.asarray(xyz,dtype=np.float32),
        initial_volume_ratio=np.asarray(phi0,dtype=np.float32),
        optimized_volume_ratio=np.asarray(phi,dtype=np.float32),
        initial_bending_positions=p0, optimized_bending_positions=p1,
        frame_stride_steps=np.asarray(frame_stride), dt_s=np.asarray(args.dt))
    summary = {
        'model':'3D total-Lagrangian MPM, Neo-Hookean elastic solid with p=1 Voigt density/stiffness mixture',
        'source_design_grid':24, 'verification_grid':args.grid, 'solver_precision':'float64; frame export float32',
        'geometry_m':{'outer_envelope':[X1-X0,Y1-Y0,.24], 'hole_center':[.45,.44], 'hole_radius':.065,
                      'boss_radius':.17, 'minimum_member_width_m':WIDTH_MIN},
        'loads_fixed_across_designs':[
            {'name':'axial tension','resultant_N':[100,0,0]},
            {'name':'transverse bending','resultant_N':[0,100,0]},
            {'name':'pure torsion','moment_x_Nm':15.0}],
        'materials':{'young_pa':2.0e5,'poisson':.25,'base_density_kg_m3':RHO,
                     'illustrative_stress_allow_pa':SIGMA_ALLOW,'allowable_basis':'E times 10% nominal uniaxial strain; illustrative, not calibrated strength'},
        'objective':'mean over three load cases of full-horizon time-averaged squared displacement of the load-end tip region, normalized by 1e-4 m^2',
        'constraints':{'solid_volume_fraction_max':VOL_FRACTION_MAX,
                       'p24_unormalized_stress_upper_bound_max_pa':SIGMA_ALLOW,
                       'local_width_bounds_m':[WIDTH_MIN,.12]},
        'initial_controls':[0.0]*10, 'optimized_controls':args.controls,
        'initial':initial_result, 'optimized':optimized_result,
        'trajectory':{'case':'transverse bending','frames':int(p0.shape[0]),
                      'frame_stride_steps':frame_stride,'times_s':(np.arange(p0.shape[0])*frame_stride*args.dt).tolist(),
                      'deformation_scale':1.0,'particle_count':int(len(xyz)),
                      'coordinates_exported_float32_after_float64_solution':True},
    }
    (args.output/'forward-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':str(args.output/'forward-summary.json'),
                      'trajectory':str(args.output/'trajectory.npz'),
                      'initial_constraints_pass':initial_result['constraints_pass'],
                      'optimized_constraints_pass':optimized_result['constraints_pass'],
                      'frames':int(p0.shape[0])},indent=2),flush=True)
    if not initial_result['constraints_pass'] or not optimized_result['constraints_pass']:
        raise RuntimeError('independent refined forward failed design constraints')


if __name__=='__main__':
    main()
