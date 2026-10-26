"""AT2 notched-beam bending with tangentially free support shoes.

Particles supply reference quadrature. Mechanics is lumped-mass reference-grid
Galerkin dynamics, not the original particle-velocity PIC time integrator.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import time as wall_time
import shutil
from mpm_jax.phase_field_bending_diagnostics import step_telemetry
from examples.phase_field_bending_geometry import reference_cells, preflight


def run(dx=.025, length=.05, density=1., gc=.005, speed=.12,
        displacement=.035, steps=None, output='outputs/at2-bending-grid', cfl=.04,
        restart=None, telemetry=True, sparse_gradient=False):
    if length < 2*dx or density <= 0 or gc <= 0 or speed <= 0 or displacement <= 0 or cfl not in (.04,.02,.01):
        raise ValueError('Positive parameters and ell/h >=2 required')
    import jax
    import jaxlib
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_layout, grid_coordinates
    from mpm_jax.phase_field import (State, material, reference_basis, scatter,
        interpolate, solve_damage, spectral_split, energies,
        assemble_gradient_stiffness)
    x, reference_volume, cell_bounds = reference_cells(dx)
    layout = build_layout(x, np.zeros(len(x), np.int32), dx=dx, padding_cells=2)
    basis = reference_basis(jnp.asarray(x), layout)
    if not bool(basis.valid):
        raise RuntimeError('Invalid reference support')
    n = layout.node_count
    q, _ = grid_coordinates(layout, jnp.arange(n))
    volume = jnp.asarray(reference_volume)
    law = material(1000., .2, gc, length)
    gradient_stiffness = (assemble_gradient_stiffness(volume, basis, n, law)
                          if sparse_gradient else None)
    base_dt = cfl*dx/np.sqrt((law.lam+2*law.mu)/density)
    dt = base_dt
    ramp = .012
    duration = (np.sqrt(2*ramp*displacement/speed) if displacement < speed*ramp/2
                else displacement/speed+ramp/2)
    required = int(np.ceil(duration/base_dt))
    count = required if steps is None else steps
    if not 1 <= count <= 25000:
        raise ValueError('Qualification limited to 25000 steps')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError('Fresh output directory required')
    mass = scatter((density*volume)[:, None]*basis.weight, basis, n)
    active = mass > 0
    denominator = jnp.where(active, mass, 1)
    left = active & jnp.isclose(q[:, 1], 0, atol=1e-10) & (jnp.abs(q[:, 0]-.2) <= .06)
    right = active & jnp.isclose(q[:, 1], 0, atol=1e-10) & (jnp.abs(q[:, 0]-1.4) <= .06)
    punch = active & jnp.isclose(q[:, 1], .4, atol=1e-10) & (jnp.abs(q[:, 0]-.8) <= .06)
    # Flat reference-grid shoes leave both tangential directions free.
    anchor = jnp.zeros(n, bool)
    if not all(bool(jnp.any(m)) for m in (left, right, punch)):
        raise RuntimeError('Missing left/right/punch boundary rows')
    zeros = jnp.zeros(n)
    no_fixed_damage = jnp.zeros(n, bool)
    def strain_of(u):
        g = jnp.einsum('pki,pkj->pij', u[basis.nodes], basis.gradient)
        return .5*(g+jnp.swapaxes(g, -1, -2))
    def particle_state(u, v, history, damage):
        return State(jnp.sum(basis.weight[..., None]*u[basis.nodes], axis=1),
            jnp.sum(basis.weight[..., None]*v[basis.nodes], axis=1), strain_of(u), history, damage)
    def potential(u, h, d):
        return jnp.sum(energies(particle_state(u, jnp.zeros_like(u), h, d), volume, basis, law, density)[1:])
    ke = lambda v: .5*jnp.sum(mass[:, None]*v*v)
    u, v, history, damage = jnp.zeros((n, 3)), jnp.zeros((n, 3)), jnp.zeros(len(x)), zeros
    total_mass = jnp.sum(mass)
    wave_speed = jnp.sqrt((law.lam+2*law.mu)/density)
    lateral_momentum_scale = total_mass*wave_speed
    reference_centroid = jnp.sum(mass[:, None]*q, axis=0)/total_mass
    initial_momentum = jnp.sum(mass[:, None]*v, axis=0)
    maximum_lateral_momentum_relative = 0.
    maximum_xz_centroid_drift_m = 0.
    absolute_step_work, absolute_nodal_work, signed_window_work = 0., 0., 0.
    damping = 0.

    @jax.jit
    def advance(u, v, history, damage, punch_position):
        eps = strain_of(u)
        _, sp, sn = spectral_split(eps, law)
        dp, _ = interpolate(damage, basis)
        stress = ((1-law.residual)*(1-dp)**2+law.residual)[:, None, None]*sp+sn
        force = scatter(-volume[:, None, None]*jnp.einsum('pij,pkj->pki', stress, basis.gradient), basis, n)
        kicked = jnp.where(active[:, None], v+dt*force/denominator[:, None], 0)
        damped = kicked/(1+dt*damping)
        lower = (left | right) & (u[:, 1]+dt*damped[:, 1] < 0)
        upper = punch & (u[:, 1]+dt*damped[:, 1] > punch_position)
        mask = jnp.zeros((n, 3), bool).at[:, 1].set(lower | upper)
        mask = mask.at[:, 0].set(anchor).at[:, 2].set(anchor)
        values = jnp.zeros((n, 3)).at[:, 1].set(jnp.where(upper,
            (punch_position-u[:, 1])/dt, -u[:, 1]/dt))
        vc = jnp.where(mask, values, damped)
        un = u+dt*vc
        impulse = mass[:, None]*(vc-damped)
        reaction = impulse/dt
        epsn = strain_of(un)
        tensile, _, _ = spectral_split(epsn, law)
        hn = jnp.maximum(history, tensile)
        solved = solve_damage(hn, volume, basis, damage, law, no_fixed_damage, zeros,
                              gradient_stiffness=gradient_stiffness)
        dn = solved.damage
        old_p = potential(u, history, damage)
        frozen_p = potential(un, hn, damage)
        new_p = potential(un, hn, dn)
        kick_work = dt*jnp.sum(force*.5*(v+kicked))
        drift_work = dt*jnp.sum(force*vc)
        actuator_work = jnp.sum(impulse*vc)
        punch_work = jnp.sum(jnp.where(punch[:, None], impulse*vc, 0))
        support_work = jnp.sum(jnp.where((left | right)[:, None], impulse*vc, 0))
        force_work_identity = drift_work+jnp.sum(volume[:, None, None]*stress*(epsn-eps))
        contact_loss = .5*jnp.sum(mass[:, None]*(vc-damped)**2)
        damping_loss = ke(kicked)-ke(damped)
        mechanical_defect = frozen_p-old_p+drift_work
        kick_drift_difference = kick_work-drift_work
        damage_split_change = new_p-frozen_p
        closure = ke(vc)+new_p-ke(v)-old_p+damping_loss+contact_loss-actuator_work-mechanical_defect-kick_drift_difference-damage_split_change
        sg = jnp.min(jnp.where(left | right, un[:, 1], jnp.inf))
        pg = jnp.min(jnp.where(punch, punch_position-un[:, 1], jnp.inf))
        wrong = jnp.maximum(jnp.max(jnp.where(left | right, -reaction[:, 1], 0)),
                            jnp.max(jnp.where(punch, reaction[:, 1], 0)))
        audit = jnp.array([actuator_work, contact_loss, damping_loss,
            mechanical_defect, kick_drift_difference, damage_split_change, closure,
            punch_work, support_work, force_work_identity])
        return un, vc, hn, dn, solved, reaction, audit, sg, pg, wrong

    fields = ['actuator_work','contact_projection_loss','damping_loss',
        'frozen_damage_strain_integration_defect','kick_drift_work_difference',
        'damage_split_potential_change','algebraic_closure','punch_work','support_anchor_work',
        'force_drift_weak_identity_error']
    source_root = Path(__file__).resolve().parents[1]
    driver_path = source_root/'examples/phase_field_bending_free_anchor.py'
    source_files = [driver_path, source_root/'examples/phase_field_bending_geometry.py',
                    *sorted((source_root/'mpm_jax').rglob('*.py'))]
    source_hashes = {str(path.relative_to(source_root)).replace('\\', '/'):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    driver_sha256 = source_hashes[str(driver_path.relative_to(source_root)).replace('\\', '/')]
    binding = dict(schema=2, boundary_variant='free-anchor-xz-v1',
        removed_support_dofs=['left_anchor_x', 'left_anchor_z'],
        dx=dx, length=length, density=density, gc=gc,
        speed=speed, displacement=displacement, cfl=cfl, dt=base_dt, ramp=ramp,
        duration=duration, required_steps=required, particles=len(x), nodes=n,
        driver_sha256=driver_sha256, source_hashes=source_hashes,
        gradient_operator=('preassembled_fixed_gradient_csr' if sparse_gradient else 'matrix_free'),
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__, numpy_version=np.__version__,
        x64=bool(jax.config.jax_enable_x64), audit_fields=fields)
    binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    rows, cumulative = [], np.zeros(len(fields))
    max_kkt, max_strain, start_step = 0., 0., 0
    restart_checkpoint_sha256 = None
    parent_binding_sha256 = None
    observation_only_transition = False
    if restart is not None:
        with np.load(restart, allow_pickle=False) as saved:
            saved_binding = json.loads(str(saved['binding']))
            saved_binding_sha256 = hashlib.sha256(
                json.dumps(saved_binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if saved_binding != binding:
                raise RuntimeError('Restart source/runtime/physical binding mismatch')
            elif 'parent_binding_sha256' in saved.files:
                parent_binding_sha256 = str(saved['parent_binding_sha256']) or None
                observation_only_transition = bool(saved['observation_only_transition'])
            if not all(np.array_equal(saved[key], np.asarray(value)) for key, value in
                       [('reference', x), ('volume', volume), ('grid_coordinates', q), ('nodal_mass', mass)]):
                raise RuntimeError('Restart reference layout/quadrature mismatch')
            start_step = int(saved['step'])
            saved_time=float(saved['checkpoint_time_s'])
            expected_time=start_step*dt
            time_error=abs(saved_time-expected_time)
            time_bound=2*np.finfo(np.float64).eps*max(abs(saved_time),abs(expected_time),dt)
            if time_error>time_bound:
                raise RuntimeError(f'Restart time mismatch: {time_error} > {time_bound}')
            restart_checkpoint_sha256=hashlib.sha256(Path(restart).read_bytes()).hexdigest()
            if not 0 <= start_step < count:
                raise RuntimeError('Restart step must precede absolute requested stop step')
            u, v = jnp.asarray(saved['nodal_displacement']), jnp.asarray(saved['nodal_velocity'])
            history, damage = jnp.asarray(saved['history']), jnp.asarray(saved['nodal_damage'])
            cumulative = saved['cumulative_audit'].copy()
            rows = saved['history_rows'].tolist()
            max_kkt, max_strain = float(saved['maximum_kkt']), float(saved['maximum_strain'])
            maximum_lateral_momentum_relative=float(saved['maximum_lateral_momentum_relative'])
            maximum_xz_centroid_drift_m=float(saved['maximum_xz_centroid_drift_m'])
            absolute_step_work=float(saved['absolute_step_work'])
            absolute_nodal_work=float(saved['absolute_nodal_work'])
            signed_window_work=float(saved['signed_window_work'])
            if not all(np.isfinite(np.asarray(value)).all() for value in
                       (u, v, history, damage, cumulative, rows, max_kkt, max_strain,
                        maximum_lateral_momentum_relative, maximum_xz_centroid_drift_m,
                        absolute_step_work, absolute_nodal_work, signed_window_work)):
                raise RuntimeError('Nonfinite restart state')
            if u.shape != (n,3) or v.shape != (n,3) or history.shape != (len(x),) or damage.shape != (n,):
                raise RuntimeError('Restart state shape mismatch')
            if (float(jnp.min(damage)) < -1e-12 or float(jnp.max(damage)) > 1+1e-12
                    or float(jnp.min(history)) < 0 or max_strain > .1 or max_kkt > 1e-7):
                raise RuntimeError('Restart state outside qualification bounds')
            if cumulative.shape != (len(fields),):
                raise RuntimeError('Restart audit shape mismatch')
    if not (((restart is None and start_step == 0) or (restart is not None and start_step > 0))
            and start_step % 25 == 0 and (count % 25 == 0 or count == required)
            and 0 <= start_step < count <= required
            and count-start_step <= 1000 and cfl == .04):
        raise RuntimeError('Use the original CFL and segments of at most 1000 steps; endpoints must be multiples of 25 or the final required step')
    if restart is not None:
        parent_diagnostics=Path(restart).parent/'step-diagnostics.jsonl'
        if not parent_diagnostics.is_file():
            raise RuntimeError('Restart parent lacks cumulative lateral P/centroid diagnostics')
        shutil.copyfile(parent_diagnostics,destination/'step-diagnostics.jsonl')
    def physical_time(step):
        return step*dt
    def save(step):
        state = particle_state(u, v, history, damage)
        d, grad = interpolate(damage, basis)
        tensile, _, _ = spectral_split(state.strain, law)
        total = .5*law.lam*jnp.trace(state.strain, axis1=-2, axis2=-1)**2+law.mu*jnp.sum(state.strain**2, axis=(-2,-1))
        elastic_density = ((1-law.residual)*(1-d)**2+law.residual)*tensile+total-tensile
        local_fracture = .5*law.gc*d*d/law.length
        gradient_fracture = .5*law.gc*law.length*jnp.sum(grad*grad, axis=1)
        np.savez_compressed(destination/f'frame_{step:06d}.npz', reference=x,
            position=x+np.asarray(state.displacement), velocity=np.asarray(state.velocity),
            strain=np.asarray(state.strain), history=np.asarray(history), damage=np.asarray(d),
            damage_gradient=np.asarray(grad), nodal_damage=np.asarray(damage),
            nodal_displacement=np.asarray(u), nodal_velocity=np.asarray(v),
            grid_coordinates=np.asarray(q), nodal_mass=np.asarray(mass), volume=np.asarray(volume),
            elastic_energy_density=np.asarray(elastic_density),
            fracture_local_energy_density=np.asarray(local_fracture),
            fracture_gradient_energy_density=np.asarray(gradient_fracture),
            fracture_energy_density=np.asarray(local_fracture+gradient_fracture),
            reference_cell_bounds=cell_bounds,
            time=physical_time(step), gc=gc, length=length, density=density)
    def checkpoint(step):
        # Contact is an instantaneous projection; there is no independent
        # friction/plastic/contact multiplier history to carry between steps.
        path = destination/f'checkpoint_{step:06d}.npz'
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, binding=json.dumps(binding, sort_keys=True),
            step=step, nodal_displacement=np.asarray(u), nodal_velocity=np.asarray(v),
            history=np.asarray(history), nodal_damage=np.asarray(damage),
            parent_binding_sha256=parent_binding_sha256 or '',
            observation_only_transition=observation_only_transition,
            cumulative_audit=cumulative, history_rows=np.asarray(rows),
            maximum_kkt=max_kkt, maximum_strain=max_strain,
            reference=x, volume=np.asarray(volume), grid_coordinates=np.asarray(q),
            nodal_mass=np.asarray(mass), checkpoint_time_s=physical_time(step),
            maximum_lateral_momentum_relative=maximum_lateral_momentum_relative,
            maximum_xz_centroid_drift_m=maximum_xz_centroid_drift_m,
            absolute_step_work=absolute_step_work, absolute_nodal_work=absolute_nodal_work,
            signed_window_work=signed_window_work)
        temporary.replace(path)
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_momentum=jnp.sum(mass[:,None]*v,axis=0)
        checkpoint_com=jnp.sum(mass[:,None]*(q+u),axis=0)/total_mass
        pointer=dict(valid=True,step=step,time_s=physical_time(step),checkpoint_file=path.name,
            checkpoint_sha256=checkpoint_sha256,boundary_variant='free-anchor-xz-v1',
            driver_sha256=driver_sha256,binding_sha256=binding_sha256,
            parent_binding_sha256=parent_binding_sha256,
            observation_only_transition=observation_only_transition,
            max_kkt=max_kkt,maximum_strain=max_strain,
            linear_momentum_kg_m_s=np.asarray(checkpoint_momentum).tolist(),
            center_of_mass_xz_m=np.asarray(checkpoint_com[jnp.array([0,2])]).tolist(),
            maximum_lateral_momentum_relative=maximum_lateral_momentum_relative,
            maximum_xz_centroid_drift_m=maximum_xz_centroid_drift_m,
            absolute_step_work=absolute_step_work,absolute_nodal_work=absolute_nodal_work,
            signed_window_work=signed_window_work,cumulative_audit=cumulative.tolist())
        pointer_tmp=destination/'checkpoint_latest.partial.json'
        pointer_tmp.write_text(json.dumps(pointer,sort_keys=True,allow_nan=False))
        pointer_tmp.replace(destination/'checkpoint_latest.json')
    checkpoint(start_step)
    save(start_step)
    for step in range(start_step+1, count+1):
        step_wall_start=wall_time.perf_counter()
        time = min(physical_time(step), duration)
        # Continuous prescribed displacement with a linearly ramped velocity.
        travel = speed*(time*time/(2*ramp) if time < ramp else time-ramp/2)
        previous_d, previous_h = damage, history
        previous_u, previous_v = u, v
        u, v, history, damage, solved, reaction, audit, sg, pg, wrong = advance(u, v, history, damage, -travel)
        if not bool(solved.converged) or not np.isfinite(np.asarray(audit)).all():
            raise RuntimeError(f'Solver/audit failure at {step}')
        total_momentum=jnp.sum(mass[:,None]*v,axis=0)
        lateral_momentum=jnp.array([total_momentum[0],total_momentum[2]])
        lateral_momentum_relative=float(jnp.linalg.norm(lateral_momentum)/jnp.maximum(lateral_momentum_scale,1e-30))
        center_of_mass=jnp.sum(mass[:,None]*(q+u),axis=0)/total_mass
        xz_centroid_drift=float(jnp.linalg.norm(center_of_mass[jnp.array([0,2])]-reference_centroid[jnp.array([0,2])]))
        maximum_lateral_momentum_relative=max(maximum_lateral_momentum_relative,lateral_momentum_relative)
        maximum_xz_centroid_drift_m=max(maximum_xz_centroid_drift_m,xz_centroid_drift)
        if lateral_momentum_relative>1e-8:
            raise RuntimeError(f'Free X/Z total-momentum guard failed at {step}: {lateral_momentum_relative}')
        if abs(float(audit[-1])) > 1e-10*max(1., abs(float(audit[0]))):
            raise RuntimeError('Discrete weak force/actual strain-work identity failed')
        if float(jnp.min(damage-previous_d)) < -1e-12 or float(jnp.min(history-previous_h)) < -1e-12:
            raise RuntimeError('Irreversibility failure')
        if min(float(sg),float(pg)) < -1e-10 or float(wrong) > 1e-8:
            raise RuntimeError('Contact feasibility/sign failure')
        max_kkt = max(max_kkt, float(solved.relative_kkt))
        max_strain = max(max_strain,float(jnp.max(jnp.abs(strain_of(u)))))
        if max_strain > .1:
            save(step)
            raise RuntimeError('Small-strain qualification ceiling .1 exceeded')
        cumulative += np.asarray(audit)
        if telemetry:
            previous_time=physical_time(step-1)
            old_travel=speed*(previous_time*previous_time/(2*ramp) if previous_time<ramp else previous_time-ramp/2)
            row=step_telemetry(step,time,dt,dx,np.sqrt((law.lam+2*law.mu)/density),
                mass,q,previous_u,previous_v,u,v,reaction,audit,-old_travel,
                float(solved.relative_kkt),max_strain)
            absolute_step_work+=row['absolute_step_actuator_work']
            absolute_nodal_work+=row['absolute_nodal_actuator_work']
            signed_window_work+=row['actuator_work']
            row.update(window_absolute_step_work=absolute_step_work,
                window_absolute_nodal_work=absolute_nodal_work,window_signed_work=signed_window_work,
                horizontal_momentum_x_kg_m_s=float(total_momentum[0]),
                horizontal_momentum_z_kg_m_s=float(total_momentum[2]),
                horizontal_momentum_relative=lateral_momentum_relative,
                center_of_mass_x_m=float(center_of_mass[0]),center_of_mass_z_m=float(center_of_mass[2]),
                xz_centroid_drift_m=xz_centroid_drift,
                damage_solver_iterations=int(solved.iterations),
                wall_step_seconds=wall_time.perf_counter()-step_wall_start)
            with (destination/'step-diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if step == 1 or step % 25 == 0 or step == count:
            state = particle_state(u,v,history,damage)
            energy = np.asarray(energies(state,volume,basis,law,density)).copy()
            energy[0] = float(ke(v))  # authoritative kinetic energy is nodal
            pd, _ = interpolate(damage,basis)
            rows.append([step,physical_time(step),travel,float(jnp.max(pd)),float(jnp.sum(reaction[punch,1])),*energy,*cumulative])
            checkpoint(step)  # Atomic carry before CSV/frame export
            np.savetxt(destination/'history.csv',rows,delimiter=',',comments='',header=','.join([
                'step','time','punch_displacement','maximum_damage','punch_reaction',
                'nodal_kinetic','elastic','fracture',*fields]))
            print(f'step={step} damage={float(jnp.max(pd)):.6f} kkt={float(solved.relative_kkt):.3e}',flush=True)
        if step % 25 == 0 or step == count:
            save(step)
            if float(jnp.max(pd)) >= .95:
                count=step
                break
    summary = dict(status='observed_production_segment_completed_unvalidated', particles=len(x),nodes=n,
        diagnostic_window=False,observed_production_segment=True,actual_operator_cfl=cfl,actual_operator_dt=dt,
        boundary_variant='free-anchor-xz-v1',
        observation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        driver_sha256=driver_sha256,binding_sha256=binding_sha256,
        restart_checkpoint_sha256=restart_checkpoint_sha256,
        initial_momentum_kg_m_s=np.asarray(initial_momentum).tolist(),
        maximum_lateral_momentum_relative=maximum_lateral_momentum_relative,
        lateral_momentum_guard_maximum=1e-8,
        maximum_xz_centroid_drift_m=maximum_xz_centroid_drift_m,
        final_momentum_kg_m_s=np.asarray(total_momentum).tolist(),
        final_center_of_mass_xz_m=np.asarray(center_of_mass[jnp.array([0,2])]).tolist(),
        high_damage_stop_threshold=.95,
        physical_start_time=physical_time(start_step),physical_end_time=physical_time(count),
        original_dt_interval_count=(count-start_step)*(dt/base_dt),
        absolute_work_recorded=telemetry,absolute_work_scope=f'steps1 through{count} cumulative from fresh zero; checkpoint carries all windows',
        window_absolute_step_work=absolute_step_work,window_absolute_nodal_work=absolute_nodal_work,
        window_signed_work=signed_window_work,
        dx=dx,length=length,density=density,gc=gc,speed=speed,dt=dt,steps=count,
        requested_displacement=displacement,reached_requested_displacement=count>=required,cfl=cfl,
        maximum_kkt=max_kkt,maximum_strain=max_strain,audit=dict(zip(fields,cumulative.tolist())),
        resumed_from_step=start_step, restart_source=None if restart is None else str(restart),
        checkpoint_time_consistency_bound_s=(None if restart is None else time_bound),
        checkpoint_time_consistency_error_s=(None if restart is None else time_error),
        checkpoint=f'checkpoint_{count:06d}.npz', checkpoint_sha256=hashlib.sha256((destination/f'checkpoint_{count:06d}.npz').read_bytes()).hexdigest(), checkpoint_binding=binding,
        maximum_particle_damage=float(jnp.max(pd)),particles_above_point9=int(jnp.sum(pd>.9)),
        final_energy=dict(zip(['nodal_kinetic','elastic','fracture'],energy.tolist())),
        kinetic_energy_definition='lumped nodal; particle interpolated velocity is export only',
        transfer='persistent nodal unknowns; zero repeated P2G/G2P velocity reset',
        method='reference-grid lumped-mass Galerkin/MPM quadrature; symplectic Euler',
        damage_gradient_operator=('preassembled_fixed_gradient_csr' if sparse_gradient else 'matrix_free'),
        validated_crack_propagation=False, energy_accuracy_validated=False,
        reference_geometry=preflight(dx))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name, default in [('dx',.025),('length',.05),('density',1.),('gc',.005),
                          ('speed',.12),('displacement',.035)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--steps',type=int)
    p.add_argument('--cfl',type=float,choices=(.04,.02,.01),default=.04)
    p.add_argument('--output',default='outputs/at2-bending-grid')
    p.add_argument('--restart',help='Exact free-anchor checkpoint; --steps remains an absolute global stop step')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--sparse-gradient',action='store_true',
                   help='preassemble the fixed gradient stiffness; reaction term stays matrix-free')
    a = p.parse_args()
    if a.preflight:
        dt=a.cfl*a.dx/np.sqrt(1111.111111111111/a.density)
        duration = (np.sqrt(.024*a.displacement/a.speed) if a.displacement < .006*a.speed
                    else a.displacement/a.speed+.006)
        print(json.dumps(dict(**preflight(a.dx),dt=dt,
            target_steps=int(np.ceil(duration/dt)))))
    else:
        run(a.dx,a.length,a.density,a.gc,a.speed,a.displacement,a.steps,a.output,a.cfl,a.restart,True,a.sparse_gradient)
