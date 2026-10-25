"""Run a fresh-t0 C2 metal TL midpoint window with atomic accepted steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path, **values):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, path)


def make_reference(geometry, dx):
    """Build two disjoint bodywise reference lattices and compact their IDs."""
    import jax
    import jax.numpy as jnp
    from mpm_jax.core import Config
    from mpm_jax.total_lagrangian import ReferenceGrid, reference_grid

    refs = []
    offsets = []
    node_offset = 0
    for body in (0, 1):
        take = geometry['body_id'] == body
        x = np.asarray(geometry['x'][take], dtype=np.float64)
        lower = np.floor(x.min(axis=0)/dx)*dx - 2*dx
        upper = np.ceil(x.max(axis=0)/dx)*dx + 2*dx
        ngrid = np.maximum(8, np.rint((upper-lower)/dx).astype(int))
        domain = float(ngrid.max()*dx)
        # Config is cubic; extend only the storage envelope. Particle and
        # shape locations remain their exact reference coordinates.
        counts = np.ceil((upper-lower)/dx).astype(int)
        n = int(counts.max())
        cfg = Config(n_grid=max(8, n), dt=1e-12, domain_size=max(8, n)*dx,
                     dim=3, boundary_cells=0, material='elastic', failure_mode='retain')
        local = reference_grid(jnp.asarray(x-lower), jnp.asarray(geometry['volume'][take]),
                               2700., cfg)
        indices = np.asarray(local.indices) + node_offset
        nodes = np.asarray(local.nodes) + lower
        refs.append(dict(positions=x, indices=indices, weights=np.asarray(local.weights),
            gradients=np.asarray(local.gradients), volume=np.asarray(local.volume),
            particle_mass=np.asarray(local.particle_mass), nodal_mass=np.asarray(local.nodal_mass),
            nodes=nodes, valid=bool(np.asarray(local.valid)), body=np.full(len(x), body, np.int32),
            lower=lower.tolist(), n_grid=cfg.n_grid))
        offsets.append((node_offset, node_offset+len(nodes)))
        node_offset += len(nodes)
    if not all(r['valid'] for r in refs):
        raise FloatingPointError('Initial reference shape/gradient, mass, or domain admission failed')
    positions = np.concatenate([r['positions'] for r in refs])
    indices = np.concatenate([r['indices'] for r in refs])
    weights = np.concatenate([r['weights'] for r in refs])
    gradients = np.concatenate([r['gradients'] for r in refs])
    volume = np.concatenate([r['volume'] for r in refs])
    particle_mass = np.concatenate([r['particle_mass'] for r in refs])
    nodal_mass = np.concatenate([r['nodal_mass'] for r in refs])
    nodes = np.concatenate([r['nodes'] for r in refs])
    body = np.concatenate([r['body'] for r in refs])
    reference = ReferenceGrid(jnp.asarray(positions), jnp.asarray(indices), jnp.asarray(weights),
        jnp.asarray(gradients), jnp.asarray(volume), jnp.asarray(particle_mass),
        jnp.asarray(nodal_mass), jnp.asarray(nodes), jnp.asarray(True))
    fixed = np.zeros((len(nodes), 3), dtype=bool)
    plate_start = offsets[1][0]
    plate_nodes = nodes[plate_start:]
    fixed[plate_start:] = ((np.abs(plate_nodes[:, 0]) >= .009) |
                           (np.abs(plate_nodes[:, 2]) >= .009))[:, None]
    nodal_velocity = np.zeros((len(nodes), 3), dtype=np.float64)
    sphere_start, sphere_end = offsets[0]
    sphere_mass = np.asarray(nodal_mass[sphere_start:sphere_end])
    nodal_velocity[sphere_start:sphere_end, 1] = np.where(sphere_mass > 0, -5000., 0.)
    return reference, body, fixed, nodal_velocity, dict(
        bodies=[dict(body=b, lower=refs[b]['lower'], n_grid=refs[b]['n_grid'],
                     nodes=offsets[b][1]-offsets[b][0]) for b in (0, 1)],
        particle_mass_kg=float(particle_mass.sum()), nodal_mass_kg=float(nodal_mass.sum()),
        fixed_nodes=int(np.sum(fixed[:, 0])), node_count=len(nodes))


def verify_adjoint_power_identity():
    """Check finite-dimensional A/A^T power identity for identity and full E0."""
    import jax.numpy as jnp
    from mpm_jax.total_lagrangian import ReferenceGrid
    from mpm_jax.metal_tl_midpoint import tl_velocity_map, tl_velocity_map_transpose
    rng = np.random.default_rng(20261026)
    particles, nodes = 7, 11
    indices = rng.integers(0, nodes, size=(particles, 4), dtype=np.int32)
    weights = rng.normal(size=(particles, 4))
    gradients = rng.normal(size=(particles, 4, 3))
    mass = rng.uniform(.2, 2., size=particles)
    reference = ReferenceGrid(jnp.zeros((particles, 3)), jnp.asarray(indices),
        jnp.asarray(weights), jnp.asarray(gradients), jnp.ones((particles,)),
        jnp.asarray(mass), jnp.ones((nodes,)), jnp.zeros((nodes, 3)), jnp.asarray(True))
    velocity = jnp.asarray(rng.normal(size=(nodes, 3)))
    phase_impulse = jnp.asarray(rng.normal(size=(particles, 4, 3)))
    cases = {'identity': jnp.eye(3), 'nonsymmetric': jnp.asarray(
        [[1., .2, -.1], [.3, .8, .05], [-.2, .4, 1.1]])}
    receipt = {}
    for name, E0 in cases.items():
        Av = tl_velocity_map(velocity, reference, jnp.asarray(mass), E0)
        Aty = tl_velocity_map_transpose(phase_impulse, reference,
                                        jnp.asarray(mass), E0, nodes)
        lhs, rhs = float(jnp.vdot(Av, phase_impulse)), float(jnp.vdot(velocity, Aty))
        error = abs(lhs-rhs)
        receipt[name] = {'dot_Av_y': lhs, 'dot_v_Aty': rhs,
                         'absolute_error': error,
                         'relative_error': error/max(abs(lhs), abs(rhs), 1e-30)}
        if error > 2e-12*max(1., abs(lhs), abs(rhs)):
            raise FloatingPointError(f'{name} A/A^T power identity failed: {error:g}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--end-time', type=float, default=160e-9)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-bridge', type=Path)
    args = parser.parse_args()
    if not 150e-9 <= args.end_time <= 200e-9 or not 1 <= args.max_seconds <= 1800:
        parser.error('The first-contact window is limited to 150–200 ns and 1800 s')
    if bool(args.resume_checkpoint) != bool(args.resume_bridge):
        parser.error('--resume-checkpoint and --resume-bridge must be supplied together')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Fresh-t0 TL output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    adjoint_power_check = verify_adjoint_power_identity()
    from examples.metal_sphere_impact_fracture import plan_case
    from mpm_jax.fragment_v6_discrete_work import update as material_update
    from mpm_jax.materials.linear_metal import linear_metal, volume_energy, eos_state
    from mpm_jax.metal_tl_midpoint import (
        initial_state, step, actual_q1_contact_jump, endpoint_e0_jump)

    geometry, plan = plan_case(speed=5000., sphere_diameter=.002, plate_size=(.02, .02),
        plate_thickness=.0005, gap=.00075, clamp_width=.001, dx=.00025,
        spacing=(.00025, .000125, .00025), end_time=1e-6, frames=60)
    if len(geometry['x']) != 26336 or plan['material']['density'] != 2700.:
        raise ValueError('The original C2 geometry or density changed')
    reference, body_id, fixed, v0, reference_record = make_reference(geometry, .00025)
    material = linear_metal(**plan['material'])
    particle_mass = np.asarray(reference.particle_mass)
    eye = jnp.broadcast_to(jnp.eye(3), (len(particle_mass), 3, 3))
    zeros = jnp.zeros((len(particle_mass),))
    material_fields = (eye, jnp.zeros_like(eye), zeros, zeros, zeros, zeros, zeros)
    state = initial_state(reference, material_fields, jnp.asarray(v0))
    p0 = np.asarray(state.initial_momentum)
    l0 = np.asarray(state.initial_angular_momentum)
    k0 = float(state.initial_kinetic)
    if not k0 > 0 or not np.linalg.norm(p0) > 0:
        raise FloatingPointError('Fresh t0 nonzero nodal P/L/K baselines were not bound')
    control_velocity = jnp.asarray(((1., 0., 0.), (-1., 0., 0.)))
    control_mass = jnp.ones((2,))
    control_pairs = jnp.asarray(((0, 1),), dtype=jnp.int32)
    control_normals = jnp.asarray(((1., 0., 0.),))
    control_fixed = jnp.zeros((2, 3), dtype=bool)
    (control_post, _, control_work, control_loss, control_valid,
     control_closure, control_kkt, _) = endpoint_e0_jump(
        control_velocity, control_mass, control_pairs, control_normals,
        control_fixed, tolerance=.005)
    control_record = dict(state='passed' if bool(np.asarray(control_valid)) else 'failed',
        pre_relative_closing_m_s=2., post_velocity_m_s=np.asarray(control_post).tolist(),
        post_closing_m_s=float(control_closure), kkt_m_s=float(control_kkt),
        contact_work_J=float(control_work), kinetic_loss_J=float(control_loss),
        scope='Algebraic independent diagonal-mass e0 control; no physical geometry',
        passed=(bool(np.asarray(control_valid)) and
            np.max(np.abs(np.asarray(control_post))) <= 1e-12 and
            abs(float(control_work)+1.) <= 1e-12 and abs(float(control_loss)-1.) <= 1e-12))
    if not control_record['passed']:
        raise FloatingPointError('Independent newly-active e0 control failed')
    atomic_json(args.output/'e0-startup-control.json', control_record)
    source_root = Path(__file__).resolve().parents[0]
    source_files = ['mpm_jax/metal_tl_midpoint.py', 'mpm_jax/total_lagrangian.py',
                    'mpm_jax/fragment_v6_discrete_work.py', 'mpm_jax/fragment_v6_midpoint_strength.py',
                    'mpm_jax/fragment_multifield_v6_mpm.py', 'mpm_jax/fragment_v6_matrixfree_projection.py',
                    'mpm_jax/materials/linear_metal.py', 'mpm_jax/materials/shock_metal.py',
                    'mpm_jax/fibre_mpm.py', 'mpm_jax/contact.py', 'mpm_jax/core.py',
                    'mpm_jax/fracture_impact_mpm.py',
                    'examples/metal_sphere_impact_fracture.py', 'examples/metal_sphere_geometry.py',
                    'examples/metal_plate_geometry.py', 'examples/woven_geometry.py',
                    'tools/audit_metal_fragments.py', 'run_metal_tl_fresh.py']
    hashes = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_files}
    resume_record = None
    resumed_steps = 0
    resumed_checked = False
    resumed_metrics = {}
    if args.resume_checkpoint is not None:
        bridge = json.loads(args.resume_bridge.read_text(encoding='utf-8'))
        checkpoint_hash = hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest()
        if bridge.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('Resume checkpoint differs from the source bridge')
        if bridge.get('continuation_source_sha256') != hashes:
            raise ValueError('Continuation source differs from the source bridge')
        if bridge.get('case_geometry_particles') != len(geometry['x']):
            raise ValueError('Resume bridge geometry differs from the fresh formal case')
        with np.load(args.resume_checkpoint, allow_pickle=False) as saved:
            required = {'displacement', 'velocity', 'F', 'tau', 'ep', 'cold', 'fracture',
                        'damage', 'internal', 'time_s', 'step', 'initial_P', 'initial_L',
                        'initial_K', 'viscous_loss', 'contact_work'}
            if set(saved.files) != required:
                raise ValueError('Resume checkpoint schema is incomplete or unexpected')
            resumed_time = float(saved['time_s'])
            resumed_steps = int(saved['step'])
            fields_resume = tuple(jnp.asarray(saved[k]) for k in
                ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'))
            state = type(state)(jnp.asarray(saved['displacement']), jnp.asarray(saved['velocity']),
                fields_resume, jnp.asarray(resumed_time), jnp.asarray(saved['initial_P']),
                jnp.asarray(saved['initial_L']), jnp.asarray(saved['initial_K']),
                jnp.asarray(saved['contact_work']), jnp.asarray(saved['viscous_loss']))
        if not 0 < resumed_time < args.end_time or resumed_steps < 1:
            raise ValueError('Resume checkpoint is outside the requested continuation window')
        p0 = np.asarray(state.initial_momentum)
        l0 = np.asarray(state.initial_angular_momentum)
        k0 = float(state.initial_kinetic)
        resumed_checked = bool(bridge.get('newly_active_e0_checked', False))
        resumed_metrics = bridge.get('prior_metrics', {})
        resume_record = dict(checkpoint=args.resume_checkpoint.name,
            checkpoint_sha256=checkpoint_hash,
            prior_source_sha256=bridge.get('prior_source_sha256'),
            bridge_reason=bridge.get('bridge_reason'))
        if resumed_checked:
            prior_e0 = dict(schema='metal-tl-e0-check-receipt-v1',
                status='passed-attested-by-source-bound-checkpoint',
                interpretation='Actual-q1 finite-support phase sampler closing-row e0 projection; not a geometric free-surface contact timestamp.',
                newly_active_e0_checked=True, prior_step=bridge.get('prior_step'),
                prior_time_s=bridge.get('prior_time_s'),
                accepted_checkpoint_sha256=checkpoint_hash,
                prior_run_json_sha256=bridge.get('prior_run_json_sha256'),
                prior_runner_sha256=bridge.get('prior_runner_sha256'),
                prior_capsule_sha256=bridge.get('prior_capsule_sha256'),
                detailed_event_metrics_persisted=False)
            atomic_json(args.output/'e0-prior-activation-attestation.json', prior_e0)
    case = dict(status='running', protocol='C2 fresh-t0 metal TL common-midpoint; endpoint e0; no fracture remap',
        geometry_particles=len(geometry['x']), plan_particles=plan['particles'],
        source_sha256=hashes, reference=reference_record, dt_cfl=.1,
        adjoint_power_check=adjoint_power_check,
        initial_momentum_kg_m_s=p0.tolist(), initial_angular_momentum_kg_m2_s=l0.tolist(),
        initial_kinetic_J=k0, energy_tolerance_J=1.414599609375e-8,
        requested_end_time_s=args.end_time, max_seconds=args.max_seconds)
    if resume_record is not None:
        case['resume'] = resume_record
        case['newly_active_e0_checked'] = resumed_checked
    case['startup_e0_control'] = control_record
    atomic_json(args.output/'run.json', case)
    initial_energy = k0 + float(jnp.sum(jnp.asarray(particle_mass) * material_fields[-1]))
    started = time.monotonic()
    steps = resumed_steps
    dt_last = None
    newly_active_checked = resumed_checked
    max_momentum_error = float(resumed_metrics.get('max_linear_relative', 0.))
    max_angular_error = float(resumed_metrics.get('max_angular_relative', 0.))
    max_energy_error = float(resumed_metrics.get('max_energy_error_J', 0.))
    angular_scale = max(float(np.sum(particle_mass))*.00025*5000., 1e-30)
    try:
        while float(state.time) < args.end_time:
            if time.monotonic()-started > args.max_seconds:
                case.update(status='timeout-preserved', steps=steps, elapsed_s=time.monotonic()-started)
                atomic_json(args.output/'run.json', case)
                break
            F, tau, ep, cold, fracture, damage, internal = state.material
            J = jnp.linalg.det(F)
            elastic = jnp.sum(tau*tau, axis=(-2, -1))/(4*material.shear_modulus*material.density*jnp.maximum(1-damage, 1e-12))
            thermo = eos_state(J, internal-cold-fracture-elastic, material)
            W = jnp.einsum('pki,pkj->pij', state.velocity[reference.indices], reference.gradients)
            L = W @ jnp.linalg.inv(F)
            C = jnp.sqrt(jnp.sum(L*L, axis=(-2, -1)))
            vpart = jnp.linalg.norm(jnp.sum(reference.weights[..., None]*state.velocity[reference.indices], axis=1), axis=-1)
            compression = jnp.maximum(-jnp.trace(L, axis1=-2, axis2=-1), 0.)
            visc_signal = material.viscosity_linear*thermo.sound_speed + 2*material.viscosity_quadratic*.00025*compression
            signal = thermo.longitudinal_speed + vpart + 1.5*np.sqrt(3)*.00025*C + visc_signal
            dt_cfl = .1*.00025/float(jnp.max(signal))
            dt = min(dt_cfl, args.end_time-float(state.time))
            if not np.isfinite(dt) or dt <= 0:
                raise FloatingPointError('Original CFL admitted a nonpositive/nonfinite step')

            def contact(vpre, x1, F1):
                # E0 is identity for the fresh undeformed reference; both the
                # phase sampler and its transpose use the same exact A map.
                return actual_q1_contact_jump(vpre, reference, particle_mass,
                    jnp.asarray(body_id), x1, F1, jnp.eye(3), .00025, fixed,
                    fixed_particles=jnp.asarray((body_id == 1) &
                        ((np.abs(geometry['x'][:, 0]) >= .009) |
                         (np.abs(geometry['x'][:, 2]) >= .009))))

            accepted = step(state, reference, material, dt, jnp.asarray(fixed), contact,
                            cell_length=.00025)
            if not bool(np.asarray(accepted.valid)):
                atomic_npz(args.output/'rejected.npz', displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), material=np.asarray(state.material[0]),
                    time_s=np.asarray(state.time), force_residual=np.asarray(accepted.force_residual),
                    contact_work_J=np.asarray(accepted.contact_work))
                atomic_json(args.output/'rejected.json', dict(
                    step=steps+1, time_s=float(state.time), dt_s=dt,
                    step_valid=bool(np.asarray(accepted.valid)),
                    force_residual=float(np.asarray(accepted.force_residual)),
                    contact=accepted.contact_report, source_sha256=hashes))
                raise FloatingPointError('Midpoint force/material/contact admission rejected')
            report = accepted.contact_report
            contact_row = dict(step=steps+1, time_before_s=float(state.time),
                time_after_s=float(state.time)+dt, dt_s=dt,
                candidate_pairs=int(report.get('pairs', 0)),
                resolved_normal_pairs=int(report.get('resolved_normal_pairs', 0)),
                positive_multiplier_pairs=int(report.get('positive_multiplier_pairs', 0)),
                pre_closing_m_s=float(report.get('initial_closing_m_s', 0.)),
                post_closing_m_s=float(report.get('final_closing_m_s', 0.)),
                physical_route_pre_closing_m_s=float(report.get('physical_route_pre_closing_m_s', 0.)),
                physical_route_post_closing_m_s=float(report.get('physical_route_post_closing_m_s', 0.)),
                physical_route_complementarity_m_s=float(report.get('physical_route_complementarity_m_s', 0.)),
                JphaseA_vs_Jnode_pre_max_abs_m_s=float(report.get('JphaseA_vs_Jnode_pre_max_abs_m_s', 0.)),
                JphaseA_vs_Jnode_post_max_abs_m_s=float(report.get('JphaseA_vs_Jnode_post_max_abs_m_s', 0.)),
                complementarity_m_s=float(report.get('complementarity_m_s', 0.)),
                maximum_multiplier=float(report.get('maximum_multiplier', 0.)),
                nodal_impulse_norm_Ns=float(report.get('nodal_impulse_norm_Ns', 0.)),
                physical_contact_work_J=float(report.get('jump_work_J', 0.)),
                kinetic_loss_J=float(report.get('kinetic_loss_J', 0.)),
                candidate_e0_pass=bool(report.get('newly_active_e0', False)))
            with (args.output/'contact-history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(contact_row, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            if (report['pairs'] and report['initial_closing_m_s'] > .005
                    and not newly_active_checked):
                closure = float(report['final_closing_m_s'])
                work = float(report['jump_work_J'])
                loss = float(report['kinetic_loss_J'])
                if (not report['newly_active_e0'] or closure > .005
                        or abs(work+loss) > 1e-10*max(k0, 1e-30)):
                    raise FloatingPointError('New-activation e0/jump-work check failed')
                newly_active_checked = True
                activation = dict(step=steps+1, time_s=float(state.time)+dt,
                    dt_s=dt, contact=report, source_sha256=hashes,
                    state='passed-new-activation-e0')
                atomic_json(args.output/'e0-activation.json', activation)
                case['new_activation_receipt'] = activation
            state = accepted.state
            steps += 1
            dt_last = dt
            P, Lm, K = map(np.asarray, (accepted.geometry['momentum'],
                                       accepted.geometry['angular_momentum'],
                                       accepted.geometry['kinetic']))
            pscale=max(np.linalg.norm(p0), 1e-30)
            max_momentum_error=max(max_momentum_error, float(np.linalg.norm(P-p0)/pscale))
            max_angular_error=max(max_angular_error, float(np.linalg.norm(Lm-l0)/angular_scale))
            total = float(K + jnp.sum(jnp.asarray(particle_mass)*state.material[-1])
                          + state.viscous_loss - state.contact_work)
            max_energy_error=max(max_energy_error, abs(total-initial_energy))
            if (max_momentum_error > 1e-10 or max_angular_error > 1e-10
                    or max_energy_error > 1.414599609375e-8):
                atomic_npz(args.output/'rejected.npz', displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), F=np.asarray(state.material[0]),
                    tau=np.asarray(state.material[1]), ep=np.asarray(state.material[2]),
                    cold=np.asarray(state.material[3]), fracture=np.asarray(state.material[4]),
                    damage=np.asarray(state.material[5]), internal=np.asarray(state.material[6]),
                    time_s=np.asarray(state.time), momentum_relative=np.asarray(max_momentum_error),
                    angular_relative=np.asarray(max_angular_error), energy_error_J=np.asarray(max_energy_error))
                raise FloatingPointError('Original global energy/linear/angular guard failed')
            # Complete atomic state every accepted step is the CP on timeout.
            atomic_npz(args.output/'accepted.npz', displacement=np.asarray(state.displacement),
                velocity=np.asarray(state.velocity), F=np.asarray(state.material[0]),
                tau=np.asarray(state.material[1]), ep=np.asarray(state.material[2]),
                cold=np.asarray(state.material[3]), fracture=np.asarray(state.material[4]),
                damage=np.asarray(state.material[5]), internal=np.asarray(state.material[6]),
                time_s=np.asarray(state.time), step=np.asarray(steps), initial_P= p0,
                initial_L=l0, initial_K=np.asarray(k0), viscous_loss=np.asarray(state.viscous_loss),
                contact_work=np.asarray(state.contact_work))
            case.update(status='running', steps=steps, time_s=float(state.time), dt_last_s=dt_last,
                elapsed_s=time.monotonic()-started, newly_active_e0_checked=newly_active_checked,
                max_linear_relative=max_momentum_error, max_angular_relative=max_angular_error,
                max_energy_error_J=max_energy_error)
            atomic_json(args.output/'run.json', case)
            print(json.dumps(dict(step=steps, time_s=float(state.time), dt_s=dt,
                force_residual=float(accepted.force_residual), energy_error_J=max_energy_error,
                momentum_relative=max_momentum_error, angular_relative=max_angular_error),
                separators=(',', ':')), flush=True)
        else:
            case.update(status='completed-short-window', steps=steps, time_s=float(state.time),
                dt_last_s=dt_last, elapsed_s=time.monotonic()-started,
                newly_active_e0_checked=newly_active_checked,
                max_linear_relative=max_momentum_error, max_angular_relative=max_angular_error,
                max_energy_error_J=max_energy_error)
            atomic_json(args.output/'run.json', case)
    except BaseException as exc:
        case.update(status='failed-preserved', steps=steps, time_s=float(state.time),
                    elapsed_s=time.monotonic()-started, error=repr(exc), traceback=traceback.format_exc())
        atomic_json(args.output/'run.json', case)
        raise


if __name__ == '__main__':
    main()
