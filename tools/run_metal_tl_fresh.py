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
    parser.add_argument('--audit-only', action='store_true',
                        help='Write read-only physical/material metrics for a source-bound checkpoint')
    args = parser.parse_args()
    if not 150e-9 <= args.end_time <= 250e-9 or not 1 <= args.max_seconds <= 1800:
        parser.error('The bounded contact/fracture-review window is limited to 150–250 ns and 1800 s')
    if bool(args.resume_checkpoint) != bool(args.resume_bridge):
        parser.error('--resume-checkpoint and --resume-bridge must be supplied together')
    if args.audit_only and args.resume_checkpoint is None:
        parser.error('--audit-only requires a source-bound accepted checkpoint')
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
        initial_state, step, endpoint_e0_jump)
    from mpm_jax.material_domain_contact import (
        material_domain_contact_jump, minimum_material_domain_gap,
        _world_corners)

    geometry, plan = plan_case(speed=5000., sphere_diameter=.002, plate_size=(.02, .02),
        plate_thickness=.0005, gap=.00075, clamp_width=.001, dx=.00025,
        spacing=(.00025, .000125, .00025), end_time=1e-6, frames=60)
    if len(geometry['x']) != 26336 or plan['material']['density'] != 2700.:
        raise ValueError('The original C2 geometry or density changed')
    reference, body_id, fixed, v0, reference_record = make_reference(geometry, .00025)
    particle_spacing = np.asarray(plan['particle_spacing_m'], dtype=np.float64)
    domain_gap = minimum_material_domain_gap(
        np.asarray(geometry['x']), np.broadcast_to(np.eye(3),
            (len(geometry['x']), 3, 3)), np.asarray(geometry['body_id']), particle_spacing)
    expected_first_contact_time = domain_gap['minimum_sat_gap_m']/5000.
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
    # All hash-bound implementation sources live under /work/source; keeping
    # the runner itself there makes the capsule tree a single immutable root.
    source_root = Path(__file__).resolve().parent
    source_files = ['mpm_jax/metal_tl_midpoint.py', 'mpm_jax/material_domain_contact.py',
                    'mpm_jax/total_lagrangian.py',
                    'mpm_jax/fragment_v6_discrete_work.py', 'mpm_jax/fragment_v6_midpoint_strength.py',
                    'mpm_jax/fragment_multifield_v6_mpm.py', 'mpm_jax/fragment_v6_matrixfree_projection.py',
                    'mpm_jax/materials/linear_metal.py', 'mpm_jax/materials/shock_metal.py',
                    'mpm_jax/fibre_mpm.py', 'mpm_jax/contact.py', 'mpm_jax/core.py',
                    'mpm_jax/fracture_impact_mpm.py',
                    'examples/metal_sphere_impact_fracture.py', 'examples/metal_sphere_geometry.py',
                    'examples/metal_plate_geometry.py', 'examples/woven_geometry.py',
                    'tools/audit_metal_fragments.py', 'tools/verify_material_domain_contact.py',
                    'tools/verify_tl_viscous_power.py',
                    'run_metal_tl_fresh.py']
    # Importing mpm_jax executes package initializers, so bind the complete
    # runtime Python trees as well as the focused physics sources above.
    source_files = sorted(set(source_files + [
        path.relative_to(source_root).as_posix()
        for package in ('mpm_jax', 'examples')
        for path in (source_root/package).rglob('*.py')]))
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
        continuation_dir = args.resume_bridge.resolve().parent
        prior_history = continuation_dir/'resume-contact-history.jsonl'
        prior_event = continuation_dir/'resume-first-domain-intersection.json'
        prior_activation = continuation_dir/'resume-e0-activation.json'
        if (not prior_history.is_file() or not prior_event.is_file()
                or not prior_activation.is_file()):
            raise ValueError('Resume capsule omits the source-bound contact/activation receipts')
        if (bridge.get('prior_contact_history_sha256') != hashlib.sha256(prior_history.read_bytes()).hexdigest()
                or bridge.get('prior_first_event_sha256') != hashlib.sha256(prior_event.read_bytes()).hexdigest()
                or bridge.get('prior_e0_activation_sha256') != hashlib.sha256(prior_activation.read_bytes()).hexdigest()):
            raise ValueError('Resume capsule receipts differ from their source bridge hashes')
        (args.output/'contact-history.jsonl').write_bytes(prior_history.read_bytes())
        atomic_json(args.output/'first-domain-intersection.json',
                    json.loads(prior_event.read_text(encoding='utf-8')))
        atomic_json(args.output/'e0-activation.json',
                    json.loads(prior_activation.read_text(encoding='utf-8')))
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
                interpretation='Actual-q1 intersection of advected initial material cuboids; shared-world contact points and independent physical postclosing route.',
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
        contact_geometry='advected initial material cuboids; 15-axis SAT, clipped shared-world contact points; no background-support candidate search',
        geometry_interpretation='The initial coarse particle quadrature defines a staircase material boundary; this is not an analytic sphere surface.',
        particle_spacing_m=particle_spacing.tolist(),
        initial_material_domain_gap=domain_gap,
        nominal_domain_first_contact_time_s=expected_first_contact_time,
        adjoint_power_check=adjoint_power_check,
        initial_momentum_kg_m_s=p0.tolist(), initial_angular_momentum_kg_m2_s=l0.tolist(),
        initial_kinetic_J=k0, energy_tolerance_J=1.414599609375e-8,
        dt_rule='dt=min(0.1*0.00025/max(longitudinal_speed+particle_speed+1.5*sqrt(3)*0.00025*|L|+viscous_signal), end_time-time); dt is a dynamic scalar input to the compiled midpoint kernel.',
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

    def write_endpoint_audit(status):
        u = np.asarray(state.displacement)
        v = np.asarray(state.velocity)
        Fend, tauend, ep_end, cold_end, fracture_end, damage_end, internal_end = map(np.asarray,
            (state.material[0], state.material[1], state.material[2], state.material[3],
             state.material[4], state.material[5], state.material[6]))
        p_x = np.asarray(reference.positions) + np.sum(
            np.asarray(reference.weights)[..., None]*u[np.asarray(reference.indices)], axis=1)
        half = particle_spacing*.5
        local_corners = _world_corners(np.zeros(3), np.eye(3), half)
        corners = p_x[:, None, :] + np.einsum(
            'pij,kj->pki', Fend, local_corners)
        sphere_mask, plate_mask = body_id == 0, body_id == 1
        local_y = corners[sphere_mask, :, 1]
        flat = int(np.argmin(local_y))
        local_particle, local_corner = np.unravel_index(flat, local_y.shape)
        sphere_ids = np.flatnonzero(sphere_mask)
        nose = corners[sphere_ids[local_particle], local_corner]
        plate_corners = corners[plate_mask].reshape(-1, 3)
        nearby = np.linalg.norm(plate_corners[:, (0, 2)]-nose[[0, 2]], axis=1) <= .0025
        if not np.any(nearby):
            raise FloatingPointError('No deformed plate top support near the sphere nose')
        surface_y = float(np.max(plate_corners[nearby, 1]))
        body_node = {}
        start = 0
        mass_nodes = np.asarray(reference.nodal_mass)
        for item in reference_record['bodies']:
            b, end = item['body'], start+item['nodes']
            mb, vb = mass_nodes[start:end], v[start:end]
            initial_vb = np.asarray(v0)[start:end]
            body_node[str(b)] = dict(
                initial_P_kg_m_s=(mb[:, None]*initial_vb).sum(axis=0).tolist(),
                final_P_kg_m_s=(mb[:, None]*vb).sum(axis=0).tolist(),
                delta_P_kg_m_s=(mb[:, None]*(vb-initial_vb)).sum(axis=0).tolist(),
                initial_K_J=float(.5*np.sum(mb[:, None]*initial_vb**2)),
                final_K_J=float(.5*np.sum(mb[:, None]*vb**2)))
            start = end
        particle_internal = [float(np.sum(particle_mass[mask]*internal_end[mask]))
                             for mask in (sphere_mask, plate_mask)]
        plate_nodes = np.arange(reference_record['bodies'][0]['nodes'], len(mass_nodes))
        sym_sigma = .5*(tauend+tauend.swapaxes(-1, -2))/np.maximum(
            np.linalg.det(Fend)[:, None, None], 1e-30)
        dev = sym_sigma-.5*np.trace(sym_sigma, axis1=-2, axis2=-1)[:, None, None]*np.eye(3)
        vm = np.sqrt(np.maximum(1.5*np.sum(dev*dev, axis=(-2, -1)), 0.))
        elastic_end = np.sum(tauend*tauend, axis=(-2, -1))/(
            4*material.shear_modulus*material.density*np.maximum(1-damage_end, 1e-12))
        thermo_end = eos_state(jnp.asarray(np.linalg.det(Fend)),
            jnp.asarray(internal_end-cold_end-fracture_end-elastic_end), material)
        pressure_end = np.asarray(thermo_end.pressure)
        temperature_end = np.asarray(thermo_end.temperature)
        particle_velocity = np.sum(np.asarray(reference.weights)[..., None]*
                                   v[np.asarray(reference.indices)], axis=1)
        particle_speed = np.linalg.norm(particle_velocity, axis=1)
        node_speed = np.linalg.norm(v, axis=1)
        positive_nodes = mass_nodes > 0
        fastest_node = int(np.argmax(np.where(positive_nodes, node_speed, -np.inf)))
        fastest_particle = int(np.argmax(particle_speed))

        def weighted_quantile(values, weights, q):
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            index = min(int(np.searchsorted(cumulative, q*cumulative[-1])), len(order)-1)
            return float(values[order[index]])

        speed_sampling = dict(
            nodal_max_m_s=float(node_speed[fastest_node]), nodal_max_node=fastest_node,
            nodal_max_node_mass_kg=float(mass_nodes[fastest_node]),
            nodal_max_node_body=int(0 if fastest_node < reference_record['bodies'][0]['nodes'] else 1),
            nodal_mass_weighted_rms_m_s=float(np.sqrt(np.sum(mass_nodes[:, None]*v*v)/np.sum(mass_nodes))),
            particle_sampled_max_m_s=float(particle_speed[fastest_particle]),
            particle_sampled_max_id=fastest_particle,
            particle_sampled_max_mass_kg=float(particle_mass[fastest_particle]),
            particle_sampled_max_body=int(body_id[fastest_particle]),
            particle_mass_weighted_rms_m_s=float(np.sqrt(np.sum(particle_mass*particle_speed**2)/np.sum(particle_mass))),
            particle_mass_weighted_p95_m_s=weighted_quantile(particle_speed, particle_mass, .95),
            particle_mass_weighted_p99_m_s=weighted_quantile(particle_speed, particle_mass, .99))
        material_body = {}
        for b, mask in ((0, sphere_mask), (1, plate_mask)):
            material_body[str(b)] = dict(
                pressure_min_Pa=float(np.min(pressure_end[mask])),
                pressure_max_Pa=float(np.max(pressure_end[mask])),
                temperature_min_K=float(np.min(temperature_end[mask])),
                temperature_max_K=float(np.max(temperature_end[mask])),
                plastic_strain_max=float(np.max(ep_end[mask])),
                damage_max=float(np.max(damage_end[mask])),
                damaged_particle_count=int(np.count_nonzero(damage_end[mask] > 0.)),
                fully_failed_particle_count=int(np.count_nonzero(damage_end[mask] >= 1.)))
        report = dict(status=status, step=steps, time_s=float(state.time),
            sphere_nose_world_m=nose.tolist(), nearby_deformed_plate_surface_y_m=surface_y,
            sphere_nose_to_plate_surface_gap_m=float(nose[1]-surface_y),
            positive_value_means_gap_negative_means_vertical_intrusion=True,
            plate_surface_deflection_near_nose_m=surface_y,
            plate_max_nodal_displacement_m=float(np.max(np.linalg.norm(u[plate_nodes], axis=1))),
            plate_max_particle_von_mises_Pa=float(np.max(vm[plate_mask])),
            plate_min_F_determinant=float(np.min(np.linalg.det(Fend[plate_mask]))),
            sphere_min_F_determinant=float(np.min(np.linalg.det(Fend[sphere_mask]))),
            sphere_internal_energy_J=particle_internal[0], plate_internal_energy_J=particle_internal[1],
            pressure_sign='positive means compression; EOS pressure',
            material_by_body=material_body, speed_sampling=speed_sampling,
            support_topology='Reference supports remain attached; no conservative split/remap or fragment claim.',
            body_nodal_transfer=body_node,
            geometry='deformed quadrature material cuboid corners; local plate surface within 2.5 mm of sphere nose',
            dt_last_s=dt_last, dt_cfl_limit=.1)
        atomic_json(args.output/'endpoint-physical-audit.json', report)

    def damage_monitor_row(record_step, record_time, record_dt):
        Fm, taum, epm, coldm, fracturem, damagem, internalm = map(np.asarray, state.material)
        detF = np.linalg.det(Fm)
        elastic_m = np.sum(taum*taum, axis=(-2, -1))/(
            4*material.shear_modulus*material.density*np.maximum(1-damagem, 1e-12))
        thermo_m = eos_state(jnp.asarray(detF),
            jnp.asarray(internalm-coldm-fracturem-elastic_m), material)
        pressure_m = np.asarray(thermo_m.pressure)
        temperature_m = np.asarray(thermo_m.temperature)
        fully_failed = np.flatnonzero(damagem >= 1.)
        failed_details = [dict(particle_id=int(i), body=int(body_id[i]),
            damage=float(damagem[i]), plastic_strain=float(epm[i]),
            detF=float(detF[i]), pressure_Pa=float(pressure_m[i]),
            temperature_K=float(temperature_m[i])) for i in fully_failed[:128]]
        body_summary = {}
        for b in (0, 1):
            mask = body_id == b
            body_summary[str(b)] = dict(
                pressure_min_Pa=float(np.min(pressure_m[mask])),
                pressure_max_Pa=float(np.max(pressure_m[mask])),
                temperature_min_K=float(np.min(temperature_m[mask])),
                temperature_max_K=float(np.max(temperature_m[mask])),
                plastic_strain_max=float(np.max(epm[mask])),
                damage_max=float(np.max(damagem[mask])),
                damaged_particle_count=int(np.count_nonzero(damagem[mask] > 0.)),
                fully_failed_particle_count=int(np.count_nonzero(damagem[mask] >= 1.)))
        return dict(step=int(record_step), time_s=float(record_time),
            dt_s=None if record_dt is None else float(record_dt),
            max_damage=float(damagem.max()), max_plastic_strain=float(epm.max()),
            fully_failed_particle_count=int(len(fully_failed)),
            fully_failed_particle_ids=fully_failed[:128].tolist(),
            fully_failed_point_details=failed_details,
            pressure_min_Pa=float(pressure_m.min()), pressure_max_Pa=float(pressure_m.max()),
            temperature_min_K=float(temperature_m.min()), temperature_max_K=float(temperature_m.max()),
            pressure_sign='positive means compression; EOS pressure', material_by_body=body_summary,
            support_topology='Reference supports remain unchanged; damage is not a conservative split.')

    def append_damage_history(record_step, record_time, record_dt):
        row = damage_monitor_row(record_step, record_time, record_dt)
        with (args.output/'damage-history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            stream.flush()
        return row

    if resumed_steps and not (args.output/'damage-history.jsonl').exists():
        append_damage_history(resumed_steps, float(state.time), None)

    if args.audit_only:
        write_endpoint_audit('accepted-checkpoint-read-only')
        case.update(status='read-only-checkpoint-audit-complete', steps=steps,
                    time_s=float(state.time), newly_active_e0_checked=newly_active_checked,
                    max_linear_relative=max_momentum_error,
                    max_angular_relative=max_angular_error,
                    max_energy_error_J=max_energy_error)
        atomic_json(args.output/'run.json', case)
        atomic_json(args.output/'audit-only-receipt.json', dict(
            schema='c2-tl-read-only-checkpoint-audit-v1',
            checkpoint_sha256=resume_record['checkpoint_sha256'],
            step=steps, time_s=float(state.time), source_sha256=hashes,
            interpretation='No midpoint step was run; all endpoint fields are sampled from the accepted checkpoint.'))
        return

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
                return material_domain_contact_jump(vpre, reference, particle_mass,
                    body_id, x1, F1, particle_spacing, fixed)

            accepted = step(state, reference, material, dt, jnp.asarray(fixed), contact,
                            cell_length=.00025)
            if not bool(np.asarray(accepted.valid)):
                atomic_npz(args.output/'rejected.npz', displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), material=np.asarray(state.material[0]),
                    time_s=np.asarray(state.time), force_residual=np.asarray(accepted.force_residual),
                    contact_work_J=np.asarray(accepted.contact_work), viscous_loss_J=np.asarray(accepted.viscous_loss),
                    kinetic_J=np.asarray(accepted.geometry['kinetic']),
                    internal_energy_J=np.asarray(jnp.sum(jnp.asarray(particle_mass)*state.material[-1])))
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
                dot_v_Atlambda_vs_Av_lambda_pre_abs_error=float(report.get('dot_v_Atlambda_vs_Av_lambda_pre_abs_error', 0.)),
                dot_v_Atlambda_vs_Av_lambda_pre_roundoff_bound=float(report.get('dot_v_Atlambda_vs_Av_lambda_pre_roundoff_bound', 0.)),
                dot_v_Atlambda_vs_Av_lambda_post_abs_error=float(report.get('dot_v_Atlambda_vs_Av_lambda_post_abs_error', 0.)),
                dot_v_Atlambda_vs_Av_lambda_post_roundoff_bound=float(report.get('dot_v_Atlambda_vs_Av_lambda_post_roundoff_bound', 0.)),
                maximum_material_domain_penetration_m=float(report.get('maximum_penetration_m', 0.)),
                body_contact_impulse_Ns=report.get('body_contact_impulse_Ns', [[0., 0., 0.], [0., 0., 0.]]),
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
            if report.get('pairs', 0) and not (args.output/'first-domain-intersection.json').exists():
                event_time = float(state.time)+dt
                event = dict(step=steps+1, time_s=event_time, dt_s=dt,
                    initial_domain_gap_m=domain_gap['minimum_sat_gap_m'],
                    expected_initial_gap_over_speed_time_s=expected_first_contact_time,
                    activation_time_error_s=event_time-expected_first_contact_time,
                    contact=report, source_sha256=hashes,
                    interpretation='First intersection of actual affine initial material cuboids at q1; coarse particle quadrature produces a staircase domain.')
                atomic_json(args.output/'first-domain-intersection.json', event)
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
                    angular_relative=np.asarray(max_angular_error), energy_error_J=np.asarray(max_energy_error),
                    viscous_loss_J=np.asarray(state.viscous_loss), contact_work_J=np.asarray(state.contact_work),
                    kinetic_J=np.asarray(K), internal_energy_J=np.asarray(jnp.sum(
                        jnp.asarray(particle_mass)*state.material[-1])))
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
            monitor = append_damage_history(steps, float(state.time), dt)
            case.update(status='running', steps=steps, time_s=float(state.time), dt_last_s=dt_last,
                elapsed_s=time.monotonic()-started, newly_active_e0_checked=newly_active_checked,
                max_linear_relative=max_momentum_error, max_angular_relative=max_angular_error,
                max_energy_error_J=max_energy_error)
            if monitor['fully_failed_particle_count']:
                failed_ids = np.flatnonzero(np.asarray(state.material[5]) >= 1.)
                first = int(failed_ids[0])
                Fm = np.asarray(state.material[0])
                epm = np.asarray(state.material[2])
                receipt = dict(schema='c2-first-fully-failed-point-stop-v1',
                    status='stopped-for-conservative-support-review', step=steps,
                    time_s=float(state.time), dt_s=dt,
                    accepted_checkpoint_sha256=hashlib.sha256((args.output/'accepted.npz').read_bytes()).hexdigest(),
                    fully_failed_particle_count=int(len(failed_ids)),
                    fully_failed_particle_ids=failed_ids[:128].tolist(),
                    fully_failed_point_details=monitor['fully_failed_point_details'],
                    first_point=dict(particle_id=first, body=int(body_id[first]),
                        position_m=np.asarray(accepted.geometry['position'])[first].tolist(),
                        damage=float(np.asarray(state.material[5])[first]),
                        plastic_strain=float(epm[first]), detF=float(np.linalg.det(Fm[first]))),
                    interpretation='Stop at the first fully failed material point. Its fixed reference support remains attached; this is not a detached fragment or a penetration qualification until a conservative split is designed and audited.',
                    source_sha256=hashes)
                atomic_json(args.output/'first-fully-failed-point.json', receipt)
                case.update(status='stopped-for-fracture-audit',
                    first_fully_failed_point=receipt)
                atomic_json(args.output/'run.json', case)
                break
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
        write_endpoint_audit(case['status'])
    except BaseException as exc:
        rejected_attempt = dict(step=int(steps), time_s=float(state.time),
            max_linear_relative=float(max_momentum_error),
            max_angular_relative=float(max_angular_error),
            max_energy_error_J=float(max_energy_error), error=repr(exc))
        accepted_path = args.output/'accepted.npz'
        if accepted_path.is_file():
            # A candidate can pass the material/contact solver and still fail
            # the global cumulative guards. Keep endpoint reporting on the last
            # checkpoint that passed every guard, never on that rejected state.
            with np.load(accepted_path, allow_pickle=False) as saved:
                state = state._replace(
                    displacement=jnp.asarray(saved['displacement']),
                    velocity=jnp.asarray(saved['velocity']),
                    material=tuple(jnp.asarray(saved[name]) for name in
                        ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal')),
                    time=jnp.asarray(saved['time_s']),
                    contact_work=jnp.asarray(saved['contact_work']),
                    viscous_loss=jnp.asarray(saved['viscous_loss']))
                steps = int(saved['step'])
            rejected_attempt['accepted_checkpoint_sha256'] = hashlib.sha256(
                accepted_path.read_bytes()).hexdigest()
            rejected_attempt['last_accepted_step'] = steps
            rejected_attempt['last_accepted_time_s'] = float(state.time)
        try:
            write_endpoint_audit('failed-at-last-accepted-state')
        except BaseException as audit_exc:
            case['endpoint_audit_error'] = repr(audit_exc)
        case.update(status='failed-preserved', steps=steps, time_s=float(state.time),
                    elapsed_s=time.monotonic()-started, error=repr(exc),
                    rejected_attempt=rejected_attempt, traceback=traceback.format_exc())
        atomic_json(args.output/'run.json', case)
        raise


if __name__ == '__main__':
    main()
