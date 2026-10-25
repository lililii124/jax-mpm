"""Geometric edge-notched 3D plate in tension with reference-grid AT2 MPM.

Run: python -m examples.phase_field_notched_plate --steps 500 --output outputs/at2
No initial damage is prescribed; the notch is removed material.
"""
import argparse
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.fibre_mpm import build_layout, grid_coordinates
from mpm_jax.phase_field import (material, reference_basis, State, explicit_step,
    interpolate, energies)
from mpm_jax.phase_field_diagnostics import audited_step


def run(dx=.05, steps=500, output='outputs/at2', length=.1, gc=.02, speed=.15, damage_interval=1, snapshot_every=100):
    if not isinstance(damage_interval, int) or damage_interval < 1:
        raise ValueError('damage_interval must be a positive integer')
    if not isinstance(snapshot_every, int) or snapshot_every < 1:
        raise ValueError('snapshot_every must be a positive integer')
    if dx > .05:
        raise ValueError('dx must be <= .05 to avoid notch-face basis bridging')
    spacing = dx/2
    axes = [np.arange(spacing/2, extent, spacing) for extent in (.6, 1., .1)]
    x = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    # The slit is wider than a complete B-spline support to prevent bridging.
    notch = (x[:, 0] < .25) & (np.abs(x[:, 1]-.5) < .08)
    x = x[~notch]
    layout = build_layout(x, np.zeros(len(x), np.int32), dx=dx, padding_cells=2)
    basis = reference_basis(jnp.asarray(x), layout)
    if not bool(basis.valid):
        raise RuntimeError('Invalid reference support')
    n = layout.node_count
    coordinates, _ = grid_coordinates(layout, jnp.arange(n))
    bottom = coordinates[:, 1] <= dx
    top = coordinates[:, 1] >= 1-dx
    mask = jnp.broadcast_to((bottom | top)[:, None], (n, 3))
    law = material(1000., .2, gc, length)
    density = 1.
    dt = .08*dx/np.sqrt((law.lam+2*law.mu)/density)
    volume = jnp.full(len(x), spacing**3)
    state = State(jnp.zeros_like(jnp.asarray(x)), jnp.zeros_like(jnp.asarray(x)),
        jnp.zeros((len(x), 3, 3)), jnp.zeros(len(x)), jnp.zeros(n))
    def advance(state, speed, update_damage):
        prescribed = jnp.zeros((n, 3)).at[:, 1].set(jnp.where(top, speed, 0))
        return audited_step(state, volume, density, basis, law, dt, mask, prescribed,
            jnp.zeros(n, bool), jnp.zeros(n), damping=5., update_damage=update_damage)
    advance = jax.jit(advance)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    maximum_residual = 0.
    boundary_work = 0.
    audit_totals = np.zeros(8)
    for step in range(steps):
        previous = state.damage
        ramp_speed = speed*min(1., (step+1)*dt/.012)
        state, result, reaction, audit = advance(state, ramp_speed,
            (step % damage_interval == 0) or (step == steps-1))
        audit_totals += np.asarray(audit, dtype=float)
        if not bool(result.converged):
            raise RuntimeError(f'Damage did not converge at step {step}: {float(result.relative_kkt)}')
        if float(jnp.min(state.damage-previous)) < -1e-12:
            raise RuntimeError('Damage healed')
        boundary_work += float(jnp.sum(reaction[top, 1]))*ramp_speed*dt
        maximum_residual = max(maximum_residual, float(result.relative_kkt))
        if (step+1) % snapshot_every == 0 or step == steps-1:
            snapshot_damage, _ = interpolate(state.damage, basis)
            np.savez_compressed(destination/f'frame_{step+1:06d}.npz',
                position=x+np.asarray(state.displacement), damage=np.asarray(snapshot_damage),
                time=(step+1)*dt)
        if step % 25 == 0 or step == steps-1:
            particle_damage, _ = interpolate(state.damage, basis)
            energy = np.asarray(energies(state, volume, basis, law, density))
            rows.append([step+1, (step+1)*dt, float(jnp.max(particle_damage)),
                float(jnp.sum(reaction[top, 1])), *energy.tolist(), boundary_work])
            print(f'step={step+1} damage={float(jnp.max(particle_damage)):.6f} kkt={float(result.relative_kkt):.3e} iterations={int(result.iterations)}', flush=True)
    d, gradient = interpolate(state.damage, basis)
    np.savez_compressed(destination/'particles.npz', reference=x,
        position=x+np.asarray(state.displacement), velocity=np.asarray(state.velocity),
        strain=np.asarray(state.strain), history=np.asarray(state.history),
        damage=np.asarray(d), damage_gradient=np.asarray(gradient))
    np.savetxt(destination/'history.csv', rows, delimiter=',',
        header='step,time,max_damage,top_reaction,kinetic,elastic,fracture,boundary_work', comments='')
    summary = dict(particles=len(x), nodes=n, steps=steps, dx=dx, length=length,
        dt=dt, gc=gc, speed=speed, damage_interval=damage_interval, snapshot_every=snapshot_every, time=steps*dt, maximum_damage=float(jnp.max(d)),
        maximum_kkt=maximum_residual, maximum_strain=float(jnp.max(jnp.abs(state.strain))),
        energy=rows[-1][4:7], end_velocity_boundary_work=boundary_work,
        energy_audit=dict(zip(audit._fields, audit_totals.tolist())),
        boundary_work=float(audit_totals[3]),
        energy_minus_work=float(sum(rows[-1][4:7])-audit_totals[3]),
        damaged_notch_band_particles=int(jnp.sum((d > .9) & (jnp.asarray(x[:, 0]) > .25) & (jnp.abs(jnp.asarray(x[:, 1])-.5) < length))),
        offband_maximum_damage=float(jnp.max(jnp.where(jnp.abs(jnp.asarray(x[:, 1])-.5) > .25, d, 0))),
        damaged_ligament_particles=int(jnp.sum((d > .9) & (jnp.asarray(x[:, 0]) > .25))),
        method='small-strain total-Lagrangian PIC MPM; dynamic ramp; damping 5/s')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dx', type=float, default=.05)
    parser.add_argument('--length', type=float, default=.1)
    parser.add_argument('--gc', type=float, default=.02)
    parser.add_argument('--speed', type=float, default=.15)
    parser.add_argument('--damage-interval', type=int, default=1)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--snapshot-every', type=int, default=100)
    parser.add_argument('--output', default='outputs/at2')
    args = parser.parse_args()
    run(args.dx, args.steps, args.output, args.length, args.gc, args.speed, args.damage_interval, args.snapshot_every)
