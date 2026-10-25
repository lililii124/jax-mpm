"""Bounded common-time-level solve from the audited 125 ns contact state."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from tools.metal_affine_dynamic_numpy import DynamicFrame, kinetic, moments, synchronize, thermal, unpack, whiten, cfl_limit, midpoint_contact_velocity
from tools.metal_affine_resource import require_commit_headroom


def save_coupled_iteration(output, index, old, midpoint, dt, entropy, candidate, force_payload):
    request_state = dict(old)
    synchronize(request_state, midpoint)
    request_state['candidate_requested_dt_s'] = np.asarray(dt)
    request_state['candidate_previous_entropy'] = entropy
    np.savez_compressed(output/f'iteration-{index}-request.npz', **request_state)
    if candidate is not None:
        np.savez_compressed(output/f'iteration-{index}-response.npz', **candidate)
    if force_payload is not None:
        np.savez_compressed(output/f'iteration-{index}-force.npz', **force_payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, default=2e-12)
    parser.add_argument('--state-sha256', default='cb3da5037c6060772b501bd668d9ee3b946e93d0ecd69da1c1b9b54f9c44e523')
    parser.add_argument('--inprocess-material', action='store_true')
    parser.add_argument('--contact-time-level', choices=('endpoint', 'midpoint'), default='endpoint')
    parser.add_argument('--direct-projection', action='store_true')
    parser.add_argument('--end-time-limit', type=float, default=126.01e-9)
    parser.add_argument('--angular-relative-tolerance', type=float, default=1e-10)
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--iteration-records', choices=('all', 'final'), default='all')
    args = parser.parse_args()
    if args.iteration_records == 'final' and not args.inprocess_material:
        parser.error('Final-only iteration records require the in-process material evaluator')
    if not re.fullmatch('[0-9a-f]{64}', args.state_sha256) or hashlib.sha256(args.state.read_bytes()).hexdigest() != args.state_sha256:
        raise ValueError('Audited post-contact seed required')
    if hashlib.sha256(args.admission.read_bytes()).hexdigest() != '02f7f83880ea9e5afb4faadfb2cad17106a1c2a9aea3a6ec52b51482ef83b28d':
        raise ValueError('Original material and tolerance record required')
    if not 0 < args.dt <= 1e-9:
        raise ValueError('Continuation steps are bounded to 1 ns and the original CFL limit')
    if not np.isfinite(args.angular_relative_tolerance) or not 0 < args.angular_relative_tolerance <= 1e-8:
        raise ValueError('A finite angular relative tolerance no greater than1e-8 is required')
    if not 1 <= args.max_iterations <= 16:
        raise ValueError('Use at most16 nonlinear evaluations')
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.state) as data:
        state = {k: data[k] for k in data.files}
    if not np.isfinite(args.end_time_limit) or not 125e-9 < args.end_time_limit <= 20e-6:
        raise ValueError('A finite continuation horizon between125 ns and20 us is required')
    if not 125e-9-1e-20 <= float(state['time']) < float(state['time'])+args.dt <= args.end_time_limit+1e-20:
        raise ValueError('Step exceeds the declared continuation horizon')
    admission = json.loads(args.admission.read_text())
    p, tol = admission['material'], admission['energy_roundoff_J']
    fixed = state['candidate_fixed_particles']
    y = whiten(state)
    initial_energy = kinetic(y) + float(np.sum(state['mass'] * state['internal_energy']))
    reference_total_energy = float(state.get('candidate_reference_total_energy',
        initial_energy+float(state['numerical_reservoir'])+float(state['boundary_reservoir'])))
    initial_momentum, initial_angular = moments(state, y)
    origin_momentum = state.get('candidate_origin_momentum', initial_momentum).copy()
    origin_angular = state.get('candidate_origin_angular', initial_angular).copy()
    prior_boundary_impulse = state.get('candidate_cumulative_boundary_impulse', np.zeros(3)).copy()
    prior_boundary_torque = state.get('candidate_cumulative_boundary_torque', np.zeros(3)).copy()
    ledger = dict(grid=0., boundary=0., contact=0., viscosity=0., dynamic_contact=0., combined_projection=0.)
    boundary_impulse = np.zeros(3)
    boundary_torque = np.zeros(3)
    report = dict(schema='metal-common-time-level125-v1', dt_s=args.dt, status='running',
                  material_backend='jax-in-process' if args.inprocess_material else 'isolated-jax-worker',
                  contact_time_level=args.contact_time_level,
                  direct_projection=args.direct_projection, end_time_limit_s=args.end_time_limit,
                  angular_relative_tolerance=args.angular_relative_tolerance,
                  nonlinear_iteration_cap=args.max_iterations,
                  iteration_records=args.iteration_records,
                  input_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
                  energy_roundoff_J=tol, iterations=[], stages=[], production_admitted=False)
    report['nonlinear_contract'] = dict(
        endpoint='Material-consistent Picard guess; proposal remains a force-balance diagnostic',
        velocity_relative_tolerance=1e-10,
        stopping_work='Actual coupled work defect <= original total energy tolerance',
        old_work_bound_budget_J=.025*tol, old_norm_work_bound_diagnostic_only=True,
        original_64eps_kinematic_gates_unchanged=True)
    frame = None
    started = time.monotonic()

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))

    def record():
        report.update(ledger=ledger, boundary_impulse=boundary_impulse, boundary_torque=boundary_torque)
        (args.output/'result.json').write_text(json.dumps(report, indent=2, default=encode)+'\n')

    def gate(condition, message):
        if not bool(condition):
            raise ValueError(message)

    def reaction_at(configuration, reaction):
        nonlocal boundary_impulse, boundary_torque
        dp, dl = moments(configuration, reaction)
        boundary_impulse += dp
        boundary_torque += dl

    def contact_diagnostics(configuration, row, duration=0.):
        configuration['maximum_contact_closing_speed'] = np.asarray(max(
            float(configuration['maximum_contact_closing_speed']), row['original_sample_closing_m_s'], row['final_closing_m_s']))
        configuration['maximum_contact_complementarity_residual'] = np.asarray(max(
            float(configuration['maximum_contact_complementarity_residual']),
            row['complementarity_m_s']+row['restricted_sample_difference_m_s']))
        configuration['candidate_contact_iteration_cap'] = np.asarray(1024)
        configuration['candidate_contact_iterations_measured'] = np.asarray(False)
        configuration['candidate_legacy_contact_cache_ignored'] = np.asarray(True)
        configuration['candidate_contact_multiplier'] = frame.restricted.last_multiplier.copy()
        configuration['candidate_contact_pair_coordinates'] = frame.contact.node_coordinates[frame.contact.pairs]
        configuration['candidate_contact_impulse_duration_s'] = np.asarray(duration)

    def project(label):
        nonlocal y
        before = y.copy()
        y, row, reaction = frame.project(y)
        if args.direct_projection:
            ledger['combined_projection'] += row['combined_projection_loss_J']
            gate(abs(row['combined_projection_identity_J']) <= tol, 'Combined projection work identity failed')
        else:
            ledger['grid'] += row['grid_projection_loss_J']
            ledger['boundary'] += row['boundary_projection_loss_J']
            gate(abs(row['grid_projection_identity_J']) <= tol and abs(row['boundary_projection_identity_J']) <= tol,
                 'Projection work identity failed')
        reaction_at(state, reaction)
        report['stages'].append(dict(stage=label, projection=row, geometry=frame.report))
        np.savez_compressed(args.output/(label+'.npz'), white_before=before, white=y, reaction=reaction)

    def contact(label):
        nonlocal y
        before = y.copy()
        y, row, reaction, physical = frame.restricted.apply(y)
        contact_diagnostics(state, row)
        loss = .5 * float(np.sum((y-before)**2))
        ledger['contact'] += loss
        reaction_at(state, reaction)
        report['stages'].append(dict(stage=label, contact=row, loss_J=loss,
            physical_impulse=moments(state, physical)[0], physical_torque=moments(state, physical)[1],
            boundary_impulse=moments(state, reaction)[0], boundary_torque=moments(state, reaction)[1]))
        np.savez_compressed(args.output/(label+'.npz'), white_before=before, white=y,
                            physical=physical, reaction=reaction, multiplier=frame.restricted.last_multiplier,
                            mobility=frame.restricted.mobility, constraint_scale=frame.restricted._balanced_system()[0],
                            closing_before=frame.restricted.sample(before), closing_after=frame.restricted.sample(y),
                            full_closing_after=frame.restricted.full_contact.sample(y),
                            x=state['x'], F=state['F'])
        record()
        gate(row['valid'] and row['original_sample_closing_m_s'] <= .005 and row['restricted_sample_difference_m_s'] <= .005,
             'Original full contact sampler tolerance failed')
        gate(abs(kinetic(y)-kinetic(before)+loss) <= tol, 'Contact projection energy identity failed')
        per_particle_loss = .5*np.sum((y-before).reshape(-1, 4, 3)**2, axis=(1, 2))
        per_particle_change = .5*np.sum((y*y-before*before).reshape(-1, 4, 3), axis=(1, 2))
        for body in (0, 1):
            selected = state['body_id'] == body
            part = float(np.sum(per_particle_loss[selected]))
            state['contact_projection_loss'][body] += part
            state['contact_coupling_work'][body] += -float(np.sum(per_particle_change[selected]))-part

    def progress(phase):
        print(json.dumps({'phase': phase, 'elapsed_s': time.monotonic()-started}), flush=True)

    try:
        require_commit_headroom()
        gate(args.dt <= cfl_limit(state, p), 'Original CFL gate failed at the start of the step')
        frame = DynamicFrame(state, fixed, y, progress, direct_projection=args.direct_projection)
        project('initial-projection')
        contact('initial-contact')
        synchronize(state, y)
        old = {k: value.copy() for k, value in state.items()}
        y0, guess = y.copy(), y.copy()
        old_entropy = thermal(old, p)[2]
        converged = False
        for iteration in range(args.max_iterations):
            candidate = None
            last_force_payload = None
            midpoint_y = .5 * (y0 + guess)
            request_state = {k: value.copy() for k, value in old.items()}
            synchronize(request_state, midpoint_y)
            request_state['candidate_requested_dt_s'] = np.asarray(args.dt)
            request_state['candidate_previous_entropy'] = old_entropy
            request = args.output/f'iteration-{iteration}-request.npz'
            response = args.output/f'iteration-{iteration}-response.npz'
            if args.iteration_records == 'all':
                np.savez_compressed(request, **request_state)
            if args.inprocess_material:
                from tools.metal_affine_midpoint_material_worker import evaluate
                candidate = evaluate(request_state, p)
                if args.iteration_records == 'all':
                    np.savez_compressed(response, **candidate)
                gate(candidate['candidate_material_valid'], 'Material update rejected; raw evaluation preserved')
            else:
                child = subprocess.run([sys.executable, '-m', 'tools.metal_affine_midpoint_material_worker',
                    '--request', str(request.resolve()), '--admission', str(args.admission.resolve()),
                    '--output', str(response.resolve())], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=120, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                (args.output/f'iteration-{iteration}-worker.log').write_text(child.stdout+child.stderr)
                gate(child.returncode == 0, 'Material worker rejected; raw evaluation preserved')
                with np.load(response) as data:
                    candidate = {k: data[k] for k in data.files}
            Fmid = .5 * (old['F'] + candidate['F'])
            inverse = np.linalg.inv(Fmid)
            _, Wmid = unpack(old, midpoint_y)
            Lmid = Wmid @ inverse
            Jmid = np.linalg.det(Fmid)
            speed = .00025 * np.maximum(-np.trace(Lmid, axis1=-2, axis2=-1), 0)
            q = p['density']/Jmid * (p['viscosity_linear'] * np.sqrt(p['bulk_modulus']*Jmid**2/p['density'])*speed
                                      + p['viscosity_quadratic']*speed**2)
            viscous = -Jmid[:, None, None] * q[:, None, None] * inverse.swapaxes(-1, -2)
            piola = candidate['candidate_work_piola'] + viscous
            covector = (-old['volume'][:, None, None]/np.sqrt(old['mass'])[:, None, None] *
                        piola @ np.linalg.inv(old['reference_root']).swapaxes(-1, -2))
            raw = np.concatenate((np.zeros((len(fixed), 3, 1)), covector), axis=2).transpose(0, 2, 1).reshape(-1, 3)
            force, reaction_force, force_rows = frame.force(raw)
            if args.contact_time_level == 'midpoint':
                proposed, contact_row, reaction_contact, physical_contact = midpoint_contact_velocity(y0, force, args.dt, frame.restricted)
                multiplier_scale = 2.
            else:
                proposed, contact_row, reaction_contact, physical_contact = frame.restricted.apply(y0 + args.dt*force)
                multiplier_scale = 1.
            residual = proposed - guess
            residual_work_bound = np.linalg.norm(residual) * max(np.linalg.norm(midpoint_y), 1.)
            actual_vmid, actual_Wmid = unpack(old, .5*(y0+proposed))
            position_error = float(np.max(abs(candidate['x']-old['x']-args.dt*actual_vmid)))
            deformation_error = float(np.max(abs(candidate['F']-old['F']-args.dt*actual_Wmid)))
            position_tolerance = 64*np.finfo(float).eps*max(float(np.max(abs(old['x']))), 1e-30)
            deformation_tolerance = 64*np.finfo(float).eps*max(float(np.max(abs(old['F']))), 1.)
            row = dict(iteration=iteration, velocity_residual_norm=float(np.linalg.norm(residual)),
                       residual_work_bound_J=float(residual_work_bound), contact=contact_row,
                       position_error=position_error, deformation_error=deformation_error,
                       force_projection=force_rows)
            # The material solve used midpoint_y=(y0+guess)/2. Retain that
            # endpoint; replacing it by the next Picard proposal breaks the
            # common-time kinematics once projection roundoff dominates.
            consistent_v, consistent_W = unpack(old, midpoint_y)
            consistent_x_error = float(np.max(abs(candidate['x']-old['x']-args.dt*consistent_v)))
            consistent_F_error = float(np.max(abs(candidate['F']-old['F']-args.dt*consistent_W)))
            candidate_reaction = args.dt*reaction_force+reaction_contact
            actual_contact_work = float(np.sum(midpoint_y*physical_contact))
            actual_boundary_work = float(np.sum(midpoint_y*candidate_reaction))
            actual_viscosity_work = args.dt*float(np.sum(old['volume'][:, None, None]*viscous*Wmid))
            actual_material_work = float(np.sum(old['volume']*candidate['candidate_material_work_density']))
            actual_material_change = float(np.sum(old['mass']*(candidate['internal_energy']-old['internal_energy'])))
            actual_coupled = (kinetic(guess)-kinetic(y0)+actual_material_work+actual_viscosity_work
                              -actual_contact_work-actual_boundary_work)
            velocity_scale = max(np.linalg.norm(y0), np.linalg.norm(guess), 1.)
            checked_velocity = midpoint_y if args.contact_time_level == 'midpoint' else guess
            original_closing = frame.contact.sample(checked_velocity)
            restricted_closing = frame.restricted.sample(checked_velocity)
            multiplier = frame.restricted.last_multiplier
            complementarity = float(np.max(np.where(multiplier>0, abs(restricted_closing),
                                                    np.maximum(restricted_closing, 0)), initial=0))
            sampler_difference = float(np.max(abs(restricted_closing-original_closing), initial=0))
            original_closing_max = float(np.max(original_closing, initial=0))
            consistent_contact_valid = (contact_row['valid'] and original_closing_max<=.005 and
                                        sampler_difference<=.005 and complementarity<=.005)
            row['consistent_endpoint'] = dict(position_error=consistent_x_error, deformation_error=consistent_F_error,
                position_tolerance=position_tolerance, deformation_tolerance=deformation_tolerance,
                velocity_relative_residual=float(np.linalg.norm(residual)/velocity_scale),
                coupled_defect_J=actual_coupled, material_identity_J=actual_material_work-actual_material_change,
                contact_work_J=actual_contact_work, boundary_work_J=actual_boundary_work,
                viscosity_work_J=actual_viscosity_work, original_sample_closing_m_s=original_closing_max,
                restricted_sample_difference_m_s=sampler_difference, complementarity_m_s=complementarity,
                original_contact_gates_passed=bool(consistent_contact_valid))
            report['iterations'].append(row)
            last_force_payload = dict(midpoint_white=midpoint_y,
                guessed_white=guess, proposed_white=proposed, raw_force=raw, projected_force=force,
                reaction_force=reaction_force, reaction_contact=reaction_contact, physical_contact=physical_contact,
                viscosity_piola=viscous, multiplier=multiplier_scale*frame.restricted.last_multiplier)
            if args.iteration_records == 'all':
                np.savez_compressed(args.output/f'iteration-{iteration}-force.npz', **last_force_payload)
            record()
            gate(contact_row['valid'] and contact_row['original_sample_closing_m_s'] <= .005 and
                 contact_row['restricted_sample_difference_m_s'] <= .005, 'Original contact tolerance failed in coupled iteration')
            if (np.linalg.norm(residual)<=1e-10*velocity_scale and abs(actual_coupled)<=tol and
                    abs(actual_material_work-actual_material_change)<=tol and abs(actual_boundary_work)<=tol and
                    actual_contact_work<=tol and actual_viscosity_work>=-tol and consistent_contact_valid and
                    consistent_x_error<=position_tolerance and consistent_F_error<=deformation_tolerance):
                converged = True
                accepted_endpoint = guess.copy()
                report['accepted_endpoint_contract'] = row['consistent_endpoint']
                contact_row = dict(contact_row, valid=True, original_sample_closing_m_s=original_closing_max,
                    final_closing_m_s=float(np.max(restricted_closing, initial=0)),
                    restricted_sample_difference_m_s=sampler_difference, complementarity_m_s=complementarity,
                    diagnostic_scope='QP proposal diagnostics with original sampler/KKT rechecked on the material-consistent endpoint')
                break
            guess = proposed
            progress('iteration-'+str(iteration))
        gate(converged, 'Common-time-level nonlinear solve reached its declared evaluation limit')
        if args.iteration_records == 'final':
            io_started = time.monotonic()
            save_coupled_iteration(args.output, iteration, old, midpoint_y, args.dt, old_entropy, candidate, last_force_payload)
            report['retained_iteration_io_seconds'] = time.monotonic()-io_started
            report['retained_iteration'] = iteration
        y = accepted_endpoint
        midpoint_actual = .5 * (y0 + y)
        midstate = {**old, 'x': .5*(old['x']+candidate['x']), 'F': Fmid}
        reaction = args.dt*reaction_force + reaction_contact
        reaction_at(midstate, reaction)
        contact_work = float(np.sum(midpoint_actual * physical_contact))
        boundary_work = float(np.sum(midpoint_actual * reaction))
        viscosity_work = args.dt * float(np.sum(old['volume'][:, None, None] * viscous * Wmid))
        material_work = float(np.sum(old['volume'] * candidate['candidate_material_work_density']))
        material_change = float(np.sum(old['mass'] * (candidate['internal_energy'] - old['internal_energy'])))
        ledger['dynamic_contact'] = -contact_work
        ledger['viscosity'] = viscosity_work
        coupled_defect = kinetic(y)-kinetic(y0)+material_work+viscosity_work-contact_work-boundary_work
        report['coupled'] = dict(material_work_J=material_work, material_change_J=material_change,
            material_identity_J=material_work-material_change, viscosity_work_J=viscosity_work,
            contact_work_J=contact_work, boundary_work_J=boundary_work, coupled_defect_J=coupled_defect,
            contact_torque=moments(midstate, physical_contact)[1],
            midpoint_closing_speed=float(np.max(frame.contact.sample(midpoint_actual), initial=0)),
            endpoint_closing_before_final_projection=float(np.max(frame.contact.sample(y), initial=0)))
        gate(abs(coupled_defect) <= tol and abs(material_work-material_change) <= tol,
             'Common-time-level work identity failed')
        gate(abs(boundary_work) <= tol and contact_work <= tol and viscosity_work >= -tol,
             'Constraint or viscosity power failed')
        state = candidate
        contact_diagnostics(state, contact_row, args.dt)
        state['candidate_contact_multiplier'] = multiplier_scale*frame.restricted.last_multiplier
        state['candidate_dynamic_contact_work_by_body'] = np.asarray([
            float(np.sum((midpoint_actual*physical_contact).reshape(-1, 4, 3)[old['body_id'] == b])) for b in (0, 1)])
        synchronize(state, y)
        state['candidate_white'] = y
        np.savez_compressed(args.output/'coupled-before-remap.npz', **state, midpoint_final_white=midpoint_actual)
        if args.contact_time_level == 'midpoint':
            contact('end-old-cone-contact')
            synchronize(state, y)
            state['candidate_white'] = y
            report['contact_split'] = 'Fixed-cone midpoint motion, old-cone inelastic impact, remap, new-cone inelastic impact'
        frame.release()
        frame = None
        gc.collect()
        frame = DynamicFrame(state, fixed, y, progress, direct_projection=args.direct_projection)
        project('end-projection')
        contact('end-contact')
        synchronize(state, y)
        J, temperature, _ = thermal(state, p)
        gate(all(np.all(np.isfinite(v)) for v in state.values()), 'Nonfinite final state')
        gate(np.array_equal(state['mass'], old['mass']) and np.array_equal(state['volume'], old['volume']), 'Mass or reference volume changed')
        gate(np.all((J >= p['minimum_J']) & (J <= p['maximum_J'])) and np.all(temperature > 0), 'Original material domain failed')
        gate(np.all(state['damage'] >= old['damage']) and np.all(state['damage'] <= 1) and
             np.all(state['plastic_strain'] >= old['plastic_strain']), 'Irreversible history failed')
        gate(np.all(state['cold_work'] >= 0) and np.all(state['fracture_energy'] >= 0), 'Negative stored work')
        stress_scale = max(1., float(np.max(abs(state['tau']))))
        gate(np.max(abs(state['tau']-state['tau'].swapaxes(-1, -2))) <= 1e-12*stress_scale and
             np.max(abs(np.trace(state['tau'], axis1=-2, axis2=-1))) <= 1e-12*stress_scale,
             'Original symmetric deviatoric stress gate failed')
        gate(np.all(state['v'][fixed] == 0) and np.all(state['W'][fixed] == 0), 'Fixed particle kinematics failed')
        gate(args.dt <= cfl_limit(state, p), 'Original CFL gate failed')
        dp, dl = moments(state, y)
        dp -= initial_momentum + boundary_impulse
        dl -= initial_angular + boundary_torque
        momentum_scale = float(np.sum(state['mass']*(np.linalg.norm(old['v'], axis=1)+np.linalg.norm(state['v'], axis=1)))) + np.linalg.norm(boundary_impulse)
        angular_scale = float(np.sum(state['mass']*np.linalg.norm(state['x'], axis=1)*
                              (np.linalg.norm(old['v'], axis=1)+np.linalg.norm(state['v'], axis=1)))) + np.linalg.norm(boundary_torque)
        energy_residual = kinetic(y)+float(np.sum(state['mass']*state['internal_energy']))+sum(ledger.values())-initial_energy
        report.update(energy_residual_J=energy_residual, momentum_residual=dp, angular_residual=dl,
                      angular_scale=angular_scale,
                      angular_roundoff_tolerance=64*np.finfo(float).eps*max(angular_scale, 1e-30),
                      angular_tolerance=max(64*np.finfo(float).eps,args.angular_relative_tolerance)*max(angular_scale,1e-30),
                      angular_relative_error=float(np.linalg.norm(dl)/max(angular_scale,1e-30)),
                      angular_roundoff_passed=bool(np.linalg.norm(dl)<=64*np.finfo(float).eps*max(angular_scale,1e-30)),
                      min_J=float(J.min()), max_temperature=float(temperature.max()), max_damage=float(state['damage'].max()))
        gate(abs(energy_residual) <= tol, 'Original full energy accounting tolerance failed')
        gate(np.linalg.norm(dp) <= 1e-10*max(momentum_scale, 1e-30), 'Original momentum gate failed')
        gate(np.linalg.norm(dl) <= report['angular_tolerance'], 'Original angular momentum gate failed')
        state['numerical_reservoir'] += (ledger['grid']+ledger['contact']+ledger['viscosity']+
                                        ledger['dynamic_contact']+ledger['combined_projection'])
        state['candidate_combined_projection_loss'] = np.asarray(
            float(old.get('candidate_combined_projection_loss',0.))+ledger['combined_projection'])
        state['boundary_reservoir'] += ledger['boundary']
        state['boundary_projection_loss'] += ledger['boundary']
        state['boundary_impulse'] += boundary_impulse
        state['candidate_white'] = y
        state['candidate_accepted_steps'] = old['candidate_accepted_steps']+1
        state['candidate_angular_relative_tolerance'] = np.asarray(args.angular_relative_tolerance)
        state['candidate_reference_total_energy'] = np.asarray(reference_total_energy)
        total_energy_residual = (kinetic(y)+float(np.sum(state['mass']*state['internal_energy']))+
            float(state['numerical_reservoir'])+float(state['boundary_reservoir'])-reference_total_energy)
        report['cumulative_energy_residual_J'] = total_energy_residual
        gate(abs(total_energy_residual) <= tol, 'Cumulative original energy tolerance failed')
        state['candidate_origin_momentum'] = origin_momentum
        state['candidate_origin_angular'] = origin_angular
        state['candidate_cumulative_boundary_impulse'] = prior_boundary_impulse+boundary_impulse
        state['candidate_cumulative_boundary_torque'] = prior_boundary_torque+boundary_torque
        final_momentum, final_angular = moments(state, y)
        cumulative_dp = final_momentum-origin_momentum-state['candidate_cumulative_boundary_impulse']
        cumulative_dl = final_angular-origin_angular-state['candidate_cumulative_boundary_torque']
        report.update(cumulative_momentum_residual=cumulative_dp, cumulative_angular_residual=cumulative_dl)
        report.update(cumulative_angular_relative_error=float(np.linalg.norm(cumulative_dl)/max(angular_scale,1e-30)),
            cumulative_angular_roundoff_passed=bool(np.linalg.norm(cumulative_dl)<=report['angular_roundoff_tolerance']))
        gate(np.linalg.norm(cumulative_dp) <= 1e-10*max(momentum_scale, 1e-30), 'Cumulative momentum gate failed')
        gate(np.linalg.norm(cumulative_dl) <= report['angular_tolerance'], 'Cumulative angular momentum gate failed')
        np.savez_compressed(args.output/'accepted.npz', **state)
        report['status'] = 'bounded-window-passed'
    except Exception as error:
        report.update(status='rejected', first_error={'type': type(error).__name__, 'message': str(error)})
        if 'response' in locals():
            if args.iteration_records == 'final' and not response.is_file():
                save_coupled_iteration(args.output, iteration, old, midpoint_y, args.dt, old_entropy, candidate, last_force_payload)
            if response.is_file():
                report['last_material_evaluation'] = response.name
            force_path = args.output/f'iteration-{iteration}-force.npz'
            if force_path.is_file():
                report['last_force_evaluation'] = force_path.name
        report['rejected_context_is_not_a_complete_physical_checkpoint'] = True
        state['valid'] = np.asarray(False)
        np.savez_compressed(args.output/'rejected.npz', **state, rejected_white=y)
    finally:
        if frame is not None:
            frame.release()
        report['total_elapsed_s'] = time.monotonic()-started
        record()
    print(json.dumps({'status': report['status'], 'error': report.get('first_error')}), flush=True)
    if report['status'] != 'bounded-window-passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
