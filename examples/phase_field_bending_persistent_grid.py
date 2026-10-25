"""Isolated persistent-grid AT2 bending qualification; no PIC velocity reset.

Particles supply reference quadrature. Mechanics is lumped-mass reference-grid
Galerkin dynamics, not the original particle-velocity PIC time integrator.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from examples.phase_field_bending_geometry import reference_cells, preflight


def run(dx=.025, length=.05, density=1., gc=.005, speed=.12,
        displacement=.035, steps=None, output='outputs/at2-bending-grid', cfl=.04):
    if length < 2*dx or density <= 0 or gc <= 0 or speed <= 0 or displacement <= 0 or cfl not in (.04,.02,.01):
        raise ValueError('Positive parameters and ell/h >=2 required')
    import jax
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
    ramp = .012
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
    left = active & jnp.isclose(q[:, 1], 0, atol=1e-10) & (jnp.abs(q[:, 0]-.2) <= .06)
    right = active & jnp.isclose(q[:, 1], 0, atol=1e-10) & (jnp.abs(q[:, 0]-1.4) <= .06)
    punch = active & jnp.isclose(q[:, 1], .4, atol=1e-10) & (jnp.abs(q[:, 0]-.8) <= .06)
    anchor = left & (jnp.abs(q[:, 2]-.08) <= dx/2+1e-10)
    if not all(bool(jnp.any(m)) for m in (left, right, punch, anchor)):
        raise RuntimeError('Missing boundary rows')
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
    damping = 0.  # expose undamped dynamics; do not mask energy with damping

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
        solved = solve_damage(hn, volume, basis, damage, law, no_fixed_damage, zeros)
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
            time=step*dt, gc=gc, length=length, density=density)
    save(0)
    rows, cumulative = [], np.zeros(len(fields))
    max_kkt, max_strain = 0., 0.
    for step in range(1, count+1):
        time = min(step*dt, duration)
        # Continuous prescribed displacement with a linearly ramped velocity.
        travel = speed*(time*time/(2*ramp) if time < ramp else time-ramp/2)
        previous_d, previous_h = damage, history
        u, v, history, damage, solved, reaction, audit, sg, pg, wrong = advance(u, v, history, damage, -travel)
        if not bool(solved.converged) or not np.isfinite(np.asarray(audit)).all():
            raise RuntimeError(f'Solver/audit failure at {step}')
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
        if step == 1 or step % 25 == 0 or step == count:
            state = particle_state(u,v,history,damage)
            energy = np.asarray(energies(state,volume,basis,law,density)).copy()
            energy[0] = float(ke(v))  # authoritative kinetic energy is nodal
            pd, _ = interpolate(damage,basis)
            rows.append([step,step*dt,travel,float(jnp.max(pd)),float(jnp.sum(reaction[punch,1])),*energy,*cumulative])
            np.savetxt(destination/'history.csv',rows,delimiter=',',comments='',header=','.join([
                'step','time','punch_displacement','maximum_damage','punch_reaction',
                'nodal_kinetic','elastic','fracture',*fields]))
            print(f'step={step} damage={float(jnp.max(pd)):.6f} kkt={float(solved.relative_kkt):.3e}',flush=True)
        if step % 200 == 0 or step == count:
            save(step)
    summary = dict(status='numerically_completed_unvalidated_candidate', particles=len(x),nodes=n,
        dx=dx,length=length,density=density,gc=gc,speed=speed,dt=dt,steps=count,
        requested_displacement=displacement,reached_requested_displacement=count>=required,cfl=cfl,
        maximum_kkt=max_kkt,maximum_strain=max_strain,audit=dict(zip(fields,cumulative.tolist())),
        maximum_particle_damage=float(jnp.max(pd)),particles_above_point9=int(jnp.sum(pd>.9)),
        final_energy=dict(zip(['nodal_kinetic','elastic','fracture'],energy.tolist())),
        kinetic_energy_definition='lumped nodal; particle interpolated velocity is export only',
        transfer='persistent nodal unknowns; zero repeated P2G/G2P velocity reset',
        method='reference-grid lumped-mass Galerkin/MPM quadrature; symplectic Euler',
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
    p.add_argument('--preflight',action='store_true')
    a = p.parse_args()
    if a.preflight:
        dt=a.cfl*a.dx/np.sqrt(1111.111111111111/a.density)
        duration = (np.sqrt(.024*a.displacement/a.speed) if a.displacement < .006*a.speed
                    else a.displacement/a.speed+.006)
        print(json.dumps(dict(**preflight(a.dx),dt=dt,
            target_steps=int(np.ceil(duration/dt)))))
    else:
        run(a.dx,a.length,a.density,a.gc,a.speed,a.displacement,a.steps,a.output,a.cfl)
