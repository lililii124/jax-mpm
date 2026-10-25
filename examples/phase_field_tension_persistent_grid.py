"""Discrete single-edge precrack tension; persistent grid, no PIC velocity reset.

Particles supply reference quadrature. Mechanics is lumped-mass reference-grid
Galerkin dynamics, not the original particle-velocity PIC time integrator.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from examples.phase_field_tension_geometry import reference_cells, preflight
from examples.phase_field_tension_qualification import POLICY, immutable_unit_seed, inspect


def run(dx=.0125, length=.0375, density=1., gc=.0002, speed=.02,
        displacement=.0018, steps=None, output='outputs/at2-tension-grid', cfl=.04,
        restart=None):
    if length < 3*dx-1e-14 or density <= 0 or gc <= 0 or speed <= 0 or displacement <= 0 or cfl not in (.04,.02,.01):
        raise ValueError('Positive parameters and ell/h >=3 required')
    import jax
    import jaxlib
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_layout, grid_coordinates
    from mpm_jax.phase_field import (State, material, reference_basis, scatter,
        interpolate, solve_damage, spectral_split, energies)
    x, reference_volume, cell_bounds = reference_cells(dx)
    layout = build_layout(x, np.zeros(len(x), np.int32), dx=dx, padding_cells=2)
    basis = reference_basis(jnp.asarray(x), layout)
    if not bool(basis.valid):
        raise RuntimeError('Invalid reference support')
    n = layout.node_count
    q, _ = grid_coordinates(layout, jnp.arange(n))
    volume = jnp.asarray(reference_volume)
    law = material(1000., .2, gc, length)
    dt = cfl*dx/np.sqrt((law.lam+2*law.mu)/density)
    ramp = .06
    duration = (np.sqrt(2*ramp*displacement/speed) if displacement < speed*ramp/2
                else displacement/speed+ramp/2)
    required = int(np.ceil(duration/dt))
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
    bottom = active & (q[:,1] <= 1e-10)
    top = active & (q[:,1] >= .6-1e-10)
    # Minimal tangential rigid-body pins: Ax, Az and Bz. All remaining
    # transverse DOFs are free. Normal end strips contain outer support nodes.
    qa = np.asarray(q)
    candidates = np.flatnonzero(np.asarray(bottom))
    ai = candidates[np.argmin(np.sum((qa[candidates]-[0,0,0])**2,axis=1))]
    bi = candidates[np.argmin(np.sum((qa[candidates]-[.6,0,0])**2,axis=1))]
    if ai == bi or not bool(jnp.any(top)):
        raise RuntimeError('Missing distinct rigid-body pins / end rows')
    pin_a = jnp.arange(n) == int(ai)
    pin_b = jnp.arange(n) == int(bi)
    fixed_mask = jnp.zeros((n,3),bool).at[:,1].set(bottom|top)
    fixed_mask = fixed_mask.at[:,0].set(pin_a).at[:,2].set(pin_a|pin_b)
    zeros = jnp.zeros(n)
    # Three coefficient rows are necessary to obtain an actual d=1 trace
    # with this quadratic basis; this finite seed is measured, never hidden.
    seed = (q[:,0] <= .3+1e-10) & (jnp.abs(q[:,1]-.3) <= dx+1e-10)
    no_fixed_damage = seed
    damage_values = jnp.where(seed,1.,0.)
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
    initial = solve_damage(history, volume, basis, damage, law, seed, damage_values)
    if not bool(initial.converged):
        raise RuntimeError('Precrack minimum-energy initial solve failed')
    damage = initial.damage
    initial_fracture_energy = float(potential(u,history,damage))
    initial_unit_seed = immutable_unit_seed(x,np.asarray(interpolate(damage,basis)[0]),dx)
    damping = 0.  # expose undamped dynamics; do not mask energy with damping

    @jax.jit
    def advance(u, v, history, damage, end_position):
        eps = strain_of(u)
        _, sp, sn = spectral_split(eps, law)
        dp, _ = interpolate(damage, basis)
        stress = ((1-law.residual)*(1-dp)**2+law.residual)[:, None, None]*sp+sn
        force = scatter(-volume[:, None, None]*jnp.einsum('pij,pkj->pki', stress, basis.gradient), basis, n)
        kicked = jnp.where(active[:, None], v+dt*force/denominator[:, None], 0)
        damped = kicked/(1+dt*damping)
        mask = fixed_mask
        values = jnp.zeros((n,3)).at[:,1].set(jnp.where(top,
            (end_position-u[:,1])/dt, -u[:,1]/dt))
        vc = jnp.where(mask, values, damped)
        un = u+dt*vc
        impulse = mass[:, None]*(vc-damped)
        reaction = impulse/dt
        epsn = strain_of(un)
        tensile, _, _ = spectral_split(epsn, law)
        hn = jnp.maximum(history, tensile)
        solved = solve_damage(hn, volume, basis, damage, law, no_fixed_damage, damage_values)
        dn = solved.damage
        old_p = potential(u, history, damage)
        frozen_p = potential(un, hn, damage)
        new_p = potential(un, hn, dn)
        kick_work = dt*jnp.sum(force*.5*(v+kicked))
        drift_work = dt*jnp.sum(force*vc)
        actuator_work = jnp.sum(impulse*vc)
        top_work = jnp.sum(jnp.where(top[:, None], impulse*vc, 0))
        support_work = jnp.sum(jnp.where(bottom[:, None], impulse*vc, 0))
        force_work_identity = drift_work+jnp.sum(volume[:, None, None]*stress*(epsn-eps))
        contact_loss = .5*jnp.sum(mass[:, None]*(vc-damped)**2)
        damping_loss = ke(kicked)-ke(damped)
        mechanical_defect = frozen_p-old_p+drift_work
        kick_drift_difference = kick_work-drift_work
        damage_split_change = new_p-frozen_p
        closure = ke(vc)+new_p-ke(v)-old_p+damping_loss+contact_loss-actuator_work-mechanical_defect-kick_drift_difference-damage_split_change
        sg = jnp.max(jnp.abs(jnp.where(bottom,un[:,1],0)))
        pg = jnp.max(jnp.abs(jnp.where(top,un[:,1]-end_position,0)))
        wrong = jnp.asarray(0.)
        audit = jnp.array([actuator_work, contact_loss, damping_loss,
            mechanical_defect, kick_drift_difference, damage_split_change, closure,
            top_work, support_work, force_work_identity, jnp.abs(actuator_work), jnp.sum(jnp.abs(impulse*vc))])
        return un, vc, hn, dn, solved, reaction, audit, sg, pg, wrong

    fields = ['actuator_work','contact_projection_loss','damping_loss',
        'frozen_damage_strain_integration_defect','kick_drift_work_difference',
        'damage_split_potential_change','algebraic_closure','top_work','bottom_pin_work',
        'force_drift_weak_identity_error','absolute_step_work','absolute_nodal_work']
    source_root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), source_root/'examples/phase_field_tension_geometry.py', source_root/'examples/phase_field_tension_qualification.py',
                    *sorted((source_root/'mpm_jax').rglob('*.py'))]
    source_hashes = {str(path.relative_to(source_root)).replace('\\', '/'):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    binding = dict(schema=1, strain_qualification_policy=POLICY, dx=dx, length=length, density=density, gc=gc,
        speed=speed, displacement=displacement, cfl=cfl, dt=dt, ramp=ramp,
        duration=duration, required_steps=required, particles=len(x), nodes=n,
        source_hashes=source_hashes, jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__, numpy_version=np.__version__,
        x64=bool(jax.config.jax_enable_x64), audit_fields=fields)
    rows, cumulative = [], np.zeros(len(fields))
    max_kkt, max_strain, start_step = 0., 0., 0
    if restart is not None:
        with np.load(restart, allow_pickle=False) as saved:
            if json.loads(str(saved['binding'])) != binding:
                raise RuntimeError('Restart source/runtime/physical binding mismatch')
            if not all(np.array_equal(saved[key], np.asarray(value)) for key, value in
                       [('reference', x), ('volume', volume), ('grid_coordinates', q), ('nodal_mass', mass)]):
                raise RuntimeError('Restart reference layout/quadrature mismatch')
            start_step = int(saved['step'])
            if not 0 <= start_step < count:
                raise RuntimeError('Restart step must precede absolute requested stop step')
            u, v = jnp.asarray(saved['nodal_displacement']), jnp.asarray(saved['nodal_velocity'])
            history, damage = jnp.asarray(saved['history']), jnp.asarray(saved['nodal_damage'])
            cumulative = saved['cumulative_audit'].copy()
            rows = saved['history_rows'].tolist()
            max_kkt, max_strain = float(saved['maximum_kkt']), float(saved['maximum_strain'])
            if not all(np.isfinite(np.asarray(value)).all() for value in
                       (u, v, history, damage, cumulative, rows, max_kkt, max_strain)):
                raise RuntimeError('Nonfinite restart state')
            if u.shape != (n,3) or v.shape != (n,3) or history.shape != (len(x),) or damage.shape != (n,):
                raise RuntimeError('Restart state shape mismatch')
            if (float(jnp.min(damage)) < -1e-12 or float(jnp.max(damage)) > 1+1e-12
                    or float(jnp.min(history)) < 0 or max_kkt > 1e-7):
                raise RuntimeError('Restart state outside qualification bounds')
            if cumulative.shape != (len(fields),):
                raise RuntimeError('Restart audit shape mismatch')
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
            reference_cell_bounds=cell_bounds, boundary_mask=np.asarray(fixed_mask), seed_mask=np.asarray(seed), initial_fracture_energy=initial_fracture_energy,
            time=step*dt, gc=gc, length=length, density=density)
    def checkpoint(step):
        # Contact is an instantaneous projection; there is no independent
        # friction/plastic/contact multiplier history to carry between steps.
        path = destination/f'checkpoint_{step:06d}.npz'
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, binding=json.dumps(binding, sort_keys=True),
            step=step, nodal_displacement=np.asarray(u), nodal_velocity=np.asarray(v),
            history=np.asarray(history), nodal_damage=np.asarray(damage),
            cumulative_audit=cumulative, history_rows=np.asarray(rows),
            maximum_kkt=max_kkt, maximum_strain=max_strain,
            reference=x, volume=np.asarray(volume), grid_coordinates=np.asarray(q),
            nodal_mass=np.asarray(mass))
        temporary.replace(path)
    save(start_step)
    checkpoint(start_step)
    for step in range(start_step+1, count+1):
        time = min(step*dt, duration)
        # Continuous prescribed displacement with a linearly ramped velocity.
        travel = speed*(time*time/(2*ramp) if time < ramp else time-ramp/2)
        previous_d, previous_h = damage, history
        u, v, history, damage, solved, reaction, audit, sg, pg, wrong = advance(u, v, history, damage, travel)
        if not bool(solved.converged) or not np.isfinite(np.asarray(audit)).all():
            raise RuntimeError(f'Solver/audit failure at {step}')
        if abs(float(audit[9])) > 1e-10*max(1., abs(float(audit[0]))):
            raise RuntimeError('Discrete weak force/actual strain-work identity failed')
        if float(jnp.min(damage-previous_d)) < -1e-12 or float(jnp.min(history-previous_h)) < -1e-12:
            raise RuntimeError('Irreversibility failure')
        if max(float(sg),float(pg)) > 1e-10:
            raise RuntimeError('Prescribed normal displacement failure')
        max_kkt = max(max_kkt, float(solved.relative_kkt))
        max_strain = max(max_strain,float(jnp.max(jnp.abs(strain_of(u)))))
        diagnostic_gradient = np.asarray(jnp.einsum('pki,pkj->pij',u[basis.nodes],basis.gradient))
        diagnostic_displacement = np.asarray(jnp.sum(basis.weight[...,None]*u[basis.nodes],axis=1))
        diagnostic_damage = np.asarray(interpolate(damage,basis)[0])
        qualification = inspect(diagnostic_gradient,diagnostic_displacement,diagnostic_damage,
                                reference_volume,initial_unit_seed)
        with (destination/'strain-qualification.jsonl').open('a') as evidence:
            evidence.write(json.dumps(dict(step=step,**qualification),allow_nan=False)+'\n')
        if not qualification['passed']:
            save(step)
            raise RuntimeError('Partitioned small-strain/geometry qualification failed')
        cumulative += np.asarray(audit)
        # Read-only every-step stability and absolute-work evidence.
        observed = dict(step=step,time=step*dt,nominal_cfl=cfl,
            wave_plus_velocity_cfl=float(dt*(np.sqrt((law.lam+2*law.mu)/density)+jnp.max(jnp.linalg.norm(v,axis=1)))/dx),
            kkt=float(solved.relative_kkt),maximum_strain=max_strain,
            audit=dict(zip(fields,np.asarray(audit).tolist())))
        with (destination/'step-telemetry.jsonl').open('a') as telemetry:
            telemetry.write(json.dumps(observed,allow_nan=False)+'\n')
        if step == 1 or step % 25 == 0 or step == count:
            state = particle_state(u,v,history,damage)
            energy = np.asarray(energies(state,volume,basis,law,density)).copy()
            energy[0] = float(ke(v))  # authoritative kinetic energy is nodal
            pd, _ = interpolate(damage,basis)
            rows.append([step,step*dt,travel,float(jnp.max(pd)),float(jnp.sum(reaction[top,1])),*energy,*cumulative])
            np.savetxt(destination/'history.csv',rows,delimiter=',',comments='',header=','.join([
                'step','time','end_displacement','maximum_damage','top_reaction',
                'nodal_kinetic','elastic','fracture',*fields]))
            print(f'step={step} damage={float(jnp.max(pd)):.6f} kkt={float(solved.relative_kkt):.3e}',flush=True)
        # Preserve the old guard failure state for strict physical-state comparison.
        if step % 200 == 0 or step == count or step == 3274:
            save(step)
            checkpoint(step)
    summary = dict(status='numerically_completed_unvalidated_candidate', particles=len(x),nodes=n,
        dx=dx,length=length,density=density,gc=gc,speed=speed,dt=dt,steps=count,
        requested_displacement=displacement, nominal_cfl=dt*np.sqrt((law.lam+2*law.mu)/density)/dx,reached_requested_displacement=count>=required,cfl=cfl,
        maximum_kkt=max_kkt,maximum_strain=max_strain,audit=dict(zip(fields,cumulative.tolist())),
        resumed_from_step=start_step, restart_source=None if restart is None else str(restart),
        checkpoint=f'checkpoint_{count:06d}.npz', checkpoint_binding=binding,
        maximum_particle_damage=float(jnp.max(pd)),particles_above_point9=int(jnp.sum(pd>.9)),
        final_energy=dict(zip(['nodal_kinetic','elastic','fracture'],energy.tolist())),
        kinetic_energy_definition='lumped nodal; particle interpolated velocity is export only',
        transfer='persistent nodal unknowns; zero repeated P2G/G2P velocity reset',
        method='reference-grid lumped-mass Galerkin/MPM quadrature; symplectic Euler',
        validated_crack_propagation=False, energy_accuracy_validated=False,
        reference_geometry=preflight(dx), strain_qualification_policy=POLICY, immutable_initial_unit_seed_points=int(initial_unit_seed.sum()), initial_fracture_energy=initial_fracture_energy, boundary='normal-only top/bottom; tangential pins Ax Az Bz', seed='quadratic coefficient rows abs(y-.3)<=h, x<=.3; finite seed, not new damage')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name, default in [('dx',.0125),('length',.0375),('density',1.),('gc',.0002),
                          ('speed',.02),('displacement',.0018)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--steps',type=int)
    p.add_argument('--cfl',type=float,choices=(.04,.02,.01),default=.04)
    p.add_argument('--output',default='outputs/at2-tension-grid')
    p.add_argument('--restart',help='Exact checkpoint; --steps remains an absolute global stop step')
    p.add_argument('--preflight',action='store_true')
    a = p.parse_args()
    if a.preflight:
        dt=a.cfl*a.dx/np.sqrt(1111.111111111111/a.density)
        duration = (np.sqrt(.12*a.displacement/a.speed) if a.displacement < .03*a.speed
                    else a.displacement/a.speed+.03)
        print(json.dumps(dict(**preflight(a.dx),dt=dt,
            target_steps=int(np.ceil(duration/dt)))))
    else:
        run(a.dx,a.length,a.density,a.gc,a.speed,a.displacement,a.steps,a.output,a.cfl,a.restart)
