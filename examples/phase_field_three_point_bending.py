"""Unpublished 3D AT2 three-point bending qualification candidate.

Frictionless rigid flat shoes act on material-attached reference-grid rows.
This is small-strain nodal contact, not finite-sliding particle/surface contact.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def geometry(dx):
    if not np.isfinite(dx) or dx not in (.04, .025, .02):
        raise ValueError('Qualified candidate grids are .04, .025 and .02')
    spacing = dx / 2
    axes = [np.arange(spacing/2, end, spacing) for end in (1.6, .4, .16)]
    x = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    # A removed bottom notch; fixed physical width exceeds 3h on every grid.
    return x[~((np.abs(x[:, 0]-.8) < .08) & (x[:, 1] < .16))]


def run(dx=.04, length=.08, steps=1200, speed=.12, output='outputs/at2-bending'):
    if length < 2*dx or steps < 1 or speed <= 0:
        raise ValueError('Require length >= 2dx, positive steps and speed')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_layout, grid_coordinates
    from mpm_jax.phase_field import (State, material, reference_basis, solve_damage,
        spectral_split, interpolate, scatter, energies)
    from mpm_jax.phase_field_diagnostics import audited_step
    x = geometry(dx)
    layout = build_layout(x, np.zeros(len(x), np.int32), dx=dx, padding_cells=2)
    basis = reference_basis(jnp.asarray(x), layout)
    if not bool(basis.valid):
        raise RuntimeError('Incomplete reference basis')
    n = layout.node_count
    q, _ = grid_coordinates(layout, jnp.arange(n))
    volume = jnp.full(len(x), (dx/2)**3)
    density, damping = 1., 5.
    law = material(1000., .2, .02, length)
    dt = .08*dx/np.sqrt(law.lam+2*law.mu)
    mass = scatter(volume[:, None]*basis.weight, basis, n)
    active = mass > 0
    left = active & jnp.isclose(q[:, 1], 0, atol=dx*1e-6) & (jnp.abs(q[:, 0]-.2) <= .06)
    right = active & jnp.isclose(q[:, 1], 0, atol=dx*1e-6) & (jnp.abs(q[:, 0]-1.4) <= .06)
    punch = active & jnp.isclose(q[:, 1], .4, atol=dx*1e-6) & (jnp.abs(q[:, 0]-.8) <= .06)
    if not all(bool(jnp.any(m)) for m in (left, right, punch)):
        raise RuntimeError('Missing contact shoe rows')
    # Pin x on the left shoe; guide z on its central plane to suppress rigid modes.
    anchor = left & (jnp.abs(q[:, 2]-.08) <= dx/2+1e-10)
    state = State(jnp.zeros_like(jnp.asarray(x)), jnp.zeros_like(jnp.asarray(x)),
                  jnp.zeros((len(x), 3, 3)), jnp.zeros(len(x)), jnp.zeros(n))
    free_damage = jnp.zeros(n, bool)
    grid_u = jnp.zeros((n, 3))

    @jax.jit
    def advance(old, grid_u, punch_position):
        # Predict with the same updated damage as audited_step. The extra solve
        # is deliberate: core solver remains untouched and energy retains old d.
        tensile, sp, sn = spectral_split(old.strain, law)
        predicted_damage = solve_damage(jnp.maximum(old.history, tensile), volume,
            basis, old.damage, law, free_damage, jnp.zeros(n))
        dp, _ = interpolate(predicted_damage.damage, basis)
        stress = ((1-law.residual)*(1-dp)**2+law.residual)[:, None, None]*sp+sn
        momentum = scatter(volume[:, None, None]*basis.weight[..., None]*old.velocity[:, None], basis, n)
        force = scatter(-volume[:, None, None]*jnp.einsum('pij,pkj->pki', stress, basis.gradient), basis, n)
        free = jnp.where(active[:, None], (momentum+dt*force)/jnp.where(active[:, None], mass[:, None], 1)/(1+dt*damping), 0)
        support_hit = (left | right) & (grid_u[:, 1]+dt*free[:, 1] < 0)
        punch_hit = punch & (grid_u[:, 1]+dt*free[:, 1] > punch_position)
        mask = jnp.zeros((n, 3), bool).at[:, 1].set(support_hit | punch_hit)
        mask = mask.at[:, 0].set(anchor).at[:, 2].set(anchor)
        value = jnp.zeros((n, 3)).at[:, 1].set(jnp.where(punch_hit,
            (punch_position-grid_u[:, 1])/dt, -grid_u[:, 1]/dt))
        new, solved, reaction, audit = audited_step(old, volume, density, basis,
            law, dt, mask, value, free_damage, jnp.zeros(n), damping=damping)
        new_grid_u = grid_u+dt*jnp.where(mask, value, free)
        support_gap = jnp.min(jnp.where(left | right, new_grid_u[:, 1], jnp.inf))
        punch_gap = jnp.min(jnp.where(punch, punch_position-new_grid_u[:, 1], jnp.inf))
        wrong_sign = jnp.maximum(jnp.max(jnp.where(left | right, -reaction[:, 1], 0)),
                                 jnp.max(jnp.where(punch, reaction[:, 1], 0)))
        return new, new_grid_u, solved, reaction, audit, support_gap, punch_gap, wrong_sign

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'summary.json').exists() or any(destination.glob('frame_*.npz')):
        raise RuntimeError('Use a fresh output directory to avoid mixed-run evidence')
    np.savez_compressed(destination/'frame_000000.npz', reference=x,
        position=x, damage=np.zeros(len(x)), time=0.)
    rows, totals = [], np.zeros(8)
    punch_position, max_kkt, max_strain = 0., 0., 0.
    for step in range(steps):
        punch_position -= dt*speed*min(1., (step+1)*dt/.012)
        previous = state
        state, grid_u, solved, reaction, audit, sg, pg, wrong = advance(state, grid_u, punch_position)
        if not bool(solved.converged) or not np.isfinite(np.asarray(state.strain)).all():
            raise RuntimeError(f'Invalid solve at {step+1}')
        if float(jnp.min(state.damage-previous.damage)) < -1e-12:
            raise RuntimeError('Damage irreversibility failed')
        if float(jnp.min(state.history-previous.history)) < -1e-12:
            raise RuntimeError('Tensile history decreased')
        if min(float(sg), float(pg)) < -1e-10 or float(wrong) > 1e-8:
            raise RuntimeError('Contact feasibility/reaction sign failed')
        totals += np.asarray(audit)
        max_kkt = max(max_kkt, float(solved.relative_kkt))
        max_strain = max(max_strain, float(jnp.max(jnp.abs(state.strain))))
        if (step+1) % 25 == 0 or step == 0 or step+1 == steps:
            d, _ = interpolate(state.damage, basis)
            e = np.asarray(energies(state, volume, basis, law, density))
            rows.append([step+1, (step+1)*dt, -punch_position, float(jnp.sum(reaction[punch, 1])),
                float(jnp.sum(reaction[left, 1])), float(jnp.sum(reaction[right, 1])),
                float(jnp.max(d)), *e, *totals, float(sg), float(pg)])
            print(f'step={step+1} max_damage={float(jnp.max(d)):.6g} kkt={max_kkt:.3g}', flush=True)
        if (step+1) % 100 == 0 or step+1 == steps:
            d, _ = interpolate(state.damage, basis)
            np.savez_compressed(destination/f'frame_{step+1:06d}.npz', reference=x,
                position=x+np.asarray(state.displacement), damage=np.asarray(d),
                velocity=np.asarray(state.velocity), strain=np.asarray(state.strain),
                history=np.asarray(state.history), time=(step+1)*dt)
    np.savetxt(destination/'history.csv', rows, delimiter=',', header=','.join([
        'step','time','punch_displacement','punch_reaction','left_reaction','right_reaction',
        'max_damage','kinetic','elastic','fracture',*audit._fields,'support_gap','punch_gap']), comments='')
    summary = dict(status='candidate_run_complete_requires_physical_review', particles=len(x), nodes=n,
        dx=dx, length=length, steps=steps, dt=dt, maximum_kkt=max_kkt,
        maximum_strain=max_strain, energy_audit=dict(zip(audit._fields, totals.tolist())),
        crack_propagation_validated=False, contact='material-attached nodal frictionless flat shoes; no finite sliding',
        damage_solve='every step; duplicated contact predictor; no damage lag',
        model='3D small-strain reference-grid spectral AT2 PIC; damping 5/s')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dx', type=float, default=.04)
    parser.add_argument('--length', type=float, default=.08)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--speed', type=float, default=.12)
    parser.add_argument('--output', default='outputs/at2-bending')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(dict(particles=len(geometry(args.dx)), dx=args.dx,
            length=args.length, length_over_dx=args.length/args.dx, executed_solver=False)))
    else:
        run(args.dx, args.length, args.steps, args.speed, args.output)
