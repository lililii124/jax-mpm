"""Differentiable 3D perforated bracket sizing with combined dynamic loads.

Ten bounded width variables parameterise the upper/lower ligaments with five
overlapping Gaussian controls along x. A fixed through-hole and load/support topology are retained.
The smooth boundary occupancy is a linear-Voigt solid/void mixture; the same
occupancy scales reference volume, inertia, and internal-force quadrature.
This is a bounded parametric sizing problem, not free-form topology
optimisation.  The minimum ligament width is an explicit variable bound and is
also checked from the sampled solid mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config
from mpm_jax.materials import elastic, kirchhoff_stress
from mpm_jax.total_lagrangian import NodalBoundary, initial_state, reference_grid, sample_nodal, step


RHO, YOUNG, POISSON = 1000.0, 2.0e5, 0.25
X0, X1 = 0.10, 0.90
Y0, Y1 = 0.22, 0.66
Z0, Z1 = 0.38, 0.62
HOLE_CENTRE, HOLE_RADIUS, BOSS_RADIUS = (0.45, 0.44), 0.065, 0.17
TORSION_NM = 15.0
WIDTH_MIN, WIDTH_MAX = 0.060, 0.120
VOID_FLOOR = 0.02
STATIONS = np.linspace(0.18, 0.82, 5)
CONTROL_SIGMA = 1.65 / 4.0
RHO_BLEND = 0.004
SIGMA_ALLOW = 20_000.0
VOL_FRACTION_MAX = 0.75
HORIZON = 0.060
DT = 2.0e-4
CHUNK = 8


def stable_pnorm(values, p):
    """Stable unnormalised p-norm; stop-gradient scaling preserves its AD."""
    values = jnp.abs(values)
    vmax = jax.lax.stop_gradient(jnp.max(values))
    nonzero = vmax > 1e-30
    scale = jnp.where(nonzero, vmax, 1.0)
    total = jnp.sum((values / scale) ** p)
    total = jnp.maximum(total, 1e-300)
    norm = scale * total ** (1.0 / p)
    return jnp.where(nonzero, norm, 0.0)


def cloud(grid_n):
    """Fixed reference quadrature around an exact through-hole."""
    dx = 1.0 / grid_n
    h = dx / 2
    axes = [np.arange(lo + h / 2, hi, h) for lo, hi in ((X0, X1), (Y0, Y1), (Z0, Z1))]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    r2 = (xyz[:, 0] - HOLE_CENTRE[0]) ** 2 + (xyz[:, 1] - HOLE_CENTRE[1]) ** 2
    xyz = xyz[r2 >= HOLE_RADIUS**2]
    return jnp.asarray(xyz), float(h**3), dx


def widths(q):
    """Physical width values at the five Gaussian control centres."""
    return WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * jax.nn.sigmoid(q.reshape(2, 5))


def width_field(x, q):
    """Bounded Gaussian-RBF width profile; lower bound is geometric, not filter-derived."""
    s = (jnp.asarray(x) - STATIONS[0]) / (STATIONS[-1] - STATIONS[0])
    centers = jnp.linspace(0.0, 1.0, 5, dtype=s.dtype)
    weights = jnp.exp(-0.5 * ((s[:, None] - centers[None, :]) / CONTROL_SIGMA) ** 2)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    latent = weights @ jnp.asarray(q).reshape(2, 5).T
    return WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * jax.nn.sigmoid(latent)


def rail_occupancy(xyz, q):
    """Smooth, bounded upper/lower ligament occupancy at each point."""
    w = width_field(xyz[:, 0], q)
    wu, wl = w[:, 0], w[:, 1]
    upper = jax.nn.sigmoid((xyz[:, 1] - (Y1 - wu)) / RHO_BLEND)
    lower = jax.nn.sigmoid(((Y0 + wl) - xyz[:, 1]) / RHO_BLEND)
    return jnp.maximum(upper, lower)


def occupancy(xyz, q):
    """Union of variable rails, fixed end blocks and the perforated boss."""
    rail = rail_occupancy(xyz, q)
    clamp_pad = xyz[:, 0] <= .18
    load_pad = xyz[:, 0] >= .82
    radius2 = (xyz[:, 0] - HOLE_CENTRE[0])**2 + (xyz[:, 1] - HOLE_CENTRE[1])**2
    boss = radius2 <= BOSS_RADIUS**2  # the r<HOLE_RADIUS particles are omitted
    fixed_solid = (clamp_pad | load_pad | boss).astype(xyz.dtype)
    solid = jnp.maximum(rail, fixed_solid)
    return VOID_FLOOR + (1 - VOID_FLOOR) * solid


def setup(grid_n, horizon=HORIZON, dt=DT):
    config = Config(n_grid=grid_n, dt=dt, domain_size=1.0, gravity=(0, 0, 0),
                    boundary_cells=0, dim=3, material="elastic", failure_mode="retain")
    xyz, dv, dx = cloud(grid_n)
    if np.any(np.asarray(xyz) <= dx) or np.any(np.asarray(xyz) >= 1.0 - dx):
        raise ValueError('all particles must stay one grid cell inside the [0,1]^3 stencil domain')
    nsteps = int(round(horizon / dt))
    if nsteps < 1 or abs(nsteps * dt - horizon) > 1e-12:
        raise ValueError("horizon/dt must give an exact positive step count")
    # All three dead loads have fixed resultant, independent of design mass.
    end = xyz[:, 0] > .84
    upper = end & (xyz[:, 1] > .61)
    lower = end & (xyz[:, 1] < .27)
    masks = jnp.stack((end, end, upper | lower))
    counts = np.asarray(jnp.sum(masks, axis=1))
    if np.any(counts == 0):
        raise ValueError(f"empty load patch: counts={counts.tolist()}")
    directions = jnp.zeros((3, xyz.shape[0], 3), dtype=xyz.dtype)
    # 100 N axial tension, 100 N transverse bending, and a grid-normalised
    # pure 15 N m couple on the fixed load-end block.
    f0 = jnp.asarray((100., 0., 0.))
    f1 = jnp.asarray((0., 100., 0.))
    top_y = jnp.sum(upper * xyz[:, 1]) / jnp.sum(upper)
    bottom_y = jnp.sum(lower * xyz[:, 1]) / jnp.sum(lower)
    torsion_force = TORSION_NM / (top_y - bottom_y)
    f2 = jnp.asarray((0., 0., torsion_force))
    directions = directions.at[0].set(masks[0, :, None] * f0 / jnp.sum(masks[0]))
    directions = directions.at[1].set(masks[1, :, None] * f1 / jnp.sum(masks[1]))
    directions = directions.at[2].set(
        masks[2, :, None] * jnp.where(upper[:, None], f2, -f2) / jnp.sum(upper)
    )
    # Fixed-node clamp on x=0.10 face, carried unchanged across all cases.
    nn = (grid_n + 1) ** 3
    node_x = (jnp.arange(nn) // ((grid_n + 1) ** 2)) * dx
    fixed = jnp.broadcast_to((node_x <= .10 + 1e-12)[:, None], (nn, 3))
    zero = jnp.zeros((nn, 3), dtype=xyz.dtype)
    bc = NodalBoundary(fixed, zero, zero, zero)
    mat = elastic(jnp.asarray(YOUNG), jnp.asarray(POISSON))
    sound = math.sqrt(YOUNG * (1 - POISSON) / (RHO * (1 + POISSON) * (1 - 2 * POISSON)))
    cfl = sound * dt / dx
    if cfl > .30:
        raise ValueError(f"CFL={cfl:.4f} exceeds .30")
    t = jnp.arange(nsteps + 1, dtype=xyz.dtype) * dt
    ramp = min(.001, horizon / 8)
    rise = jnp.where(t < ramp, .5 * (1 - jnp.cos(jnp.pi * jnp.clip(t / ramp, 0, 1))), 1.)
    fall_t = jnp.clip((t - (horizon-ramp)) / ramp, 0, 1)
    pulse = rise * jnp.where(t > horizon-ramp, .5 * (1 + jnp.cos(jnp.pi * fall_t)), 1.)
    raw_pairs = jnp.stack((pulse[:-1], pulse[1:]), axis=1)
    nblocks = math.ceil(nsteps / CHUNK)
    padded = nblocks * CHUNK
    raw_pairs = jnp.pad(raw_pairs, ((0, padded-nsteps), (0, 0)))
    pairs = raw_pairs.reshape(nblocks, CHUNK, 2)
    active_steps = (jnp.arange(padded) < nsteps).reshape(nblocks, CHUNK)
    tips = xyz[:, 0] > .78

    def simulate(q, k, collect_stress=False):
        phi = occupancy(xyz, q)
        volume = dv * phi
        ref = reference_grid(xyz, volume, RHO, config)
        state = initial_state(ref, mat, config, jnp.zeros_like(xyz))
        force = directions[k]
        # Force/mass cancels in the TL body-force term; resultant remains fixed.
        acceleration = force / ref.particle_mass[:, None]
        axis = jnp.asarray(((1.,0.,0.), (0.,1.,0.), (1.,0.,0.)))[k]

        def block(carry, args):
            pp, active = args
            def body(s, inputs):
                pair, active = inputs
                def advance(current):
                    return step(current, ref, mat, config, pair[0]*acceleration,
                                pair[1]*acceleration, bc, bc)[0]
                sn = jax.lax.cond(active, advance, lambda current: current, s)
                up = sample_nodal(sn.displacement, ref)
                resp = jnp.where(active, jnp.mean(jnp.sum(up[tips]**2, axis=-1)), 0.)
                tau = kirchhoff_stress(sn.F, mat, "elastic")
                sig = tau / jnp.maximum(jnp.linalg.det(sn.F), 1e-12)[:, None, None]
                dev = sig - jnp.trace(sig, axis1=1, axis2=2)[:, None, None] * jnp.eye(3)[None] / 3
                vm = jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev*dev, axis=(1,2)), 1e-24))
                # Unnormalised p-norm gives a differentiable upper bound. For
                # every sampled solid particle (phi>=0.5), phi*vm/0.5 >= vm;
                # the residual floor therefore cannot hide a solid hot spot.
                p = 24
                ks = stable_pnorm((phi * vm) / (0.5 * SIGMA_ALLOW), p) * SIGMA_ALLOW
                solid_max = jnp.max(jnp.where(phi >= .5, vm, 0.))
                gray_max = jnp.max(jnp.where((phi >= .1) & (phi < .5), vm, 0.))
                floor_max = jnp.max(jnp.where(phi < .1, vm, 0.))
                ks = jnp.where(active, ks, 0.)
                solid_max = jnp.where(active, solid_max, 0.)
                gray_max = jnp.where(active, gray_max, 0.)
                floor_max = jnp.where(active, floor_max, 0.)
                return sn, (resp, ks, solid_max, gray_max, floor_max, jnp.where(active, sn.valid, True))
            return jax.lax.scan(body, carry, (pp, active))

        chunk = jax.checkpoint(block, prevent_cse=False)
        final, values = jax.lax.scan(chunk, state, (pairs, active_steps))
        # Displacement is time-averaged; stress is a peak constraint over the
        # complete load history, rather than a time-averaged stress surrogate.
        stress_bound = stable_pnorm(values[1] / SIGMA_ALLOW, 24) * SIGMA_ALLOW
        return final, (jnp.sum(values[0])/nsteps, stress_bound,
                       jnp.max(values[2]), jnp.max(values[3]),
                       jnp.max(values[4]), jnp.all(values[5]))

    def metrics(q):
        phi = occupancy(xyz, q)
        vol_frac = jnp.sum(phi) * dv / ((X1-X0)*(Y1-Y0)*(Z1-Z0))
        out = jax.lax.map(lambda k: simulate(q, k)[1], jnp.arange(3), batch_size=1)
        response = jnp.mean(out[0]) / (.01**2)
        # Another unnormalised p-norm combines the load cases.  Together with
        # the per-time, per-particle p-norm this upper-bounds the peak across
        # every sampled particle, time step, and load case, with a smooth AD path.
        ks = stable_pnorm(out[1] / SIGMA_ALLOW, 24) * SIGMA_ALLOW
        # Feasibility merit: exact design fields and fixed loads; stress/volume
        # constraint violations are exposed separately in the saved record.
        merit = response + 100. * jax.nn.relu((vol_frac - VOL_FRACTION_MAX)/VOL_FRACTION_MAX)**2
        merit += 20. * jax.nn.relu((ks - SIGMA_ALLOW)/SIGMA_ALLOW)**2
        return merit, (response, vol_frac, ks, out)
    return config, xyz, dv, directions, masks, counts, metrics, simulate, bc, cfl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grid', type=int, default=24)
    ap.add_argument('--horizon', type=float, default=HORIZON)
    ap.add_argument('--dt', type=float, default=DT)
    ap.add_argument('--output', type=Path, default=Path('outputs/combined-bracket-design'))
    ap.add_argument('--iterations', type=int, default=0,
                    help='number of constrained Adam sizing updates after the mandatory AD/FD check')
    ap.add_argument('--learning-rate', type=float, default=0.12)
    ap.add_argument('--initial-width-m', type=float, default=0.090,
                    help='uniform initial ligament width used for a sizing/feasibility run (0.06 to 0.12 m)')
    args = ap.parse_args()
    jax.config.update('jax_enable_x64', True)
    args.output.mkdir(parents=True, exist_ok=True)
    model = setup(args.grid, args.horizon, args.dt)
    config, xyz, dv, loads, masks, counts, metrics, simulate, bc, cfl = model
    if not WIDTH_MIN <= args.initial_width_m <= WIDTH_MAX:
        raise ValueError(f'initial-width-m must lie in [{WIDTH_MIN},{WIDTH_MAX}]')
    initial_fraction = np.clip((args.initial_width_m-WIDTH_MIN)/(WIDTH_MAX-WIDTH_MIN), 1e-12, 1-1e-12)
    initial_logit = math.log(initial_fraction/(1-initial_fraction))
    q0 = jnp.full((10,), initial_logit, dtype=jnp.float64)
    phi0 = np.asarray(occupancy(xyz, q0))
    ref0 = reference_grid(xyz, dv*jnp.asarray(phi0), RHO, config)
    if (not bool(ref0.valid) or not np.isfinite(phi0).all() or
            phi0.min() < VOID_FLOOR-1e-12 or np.any(np.asarray(ref0.particle_mass) <= 0)):
        raise RuntimeError('reference-grid preflight failed domain, stencil, mass or finite checks')
    loads_np = np.asarray(loads)
    xyz_np = np.asarray(xyz)
    resultants = loads_np.sum(axis=1)
    moments = np.cross(xyz_np[None, :, :], loads_np).sum(axis=1)
    mass_particle = RHO*dv*phi0
    patch_mass = np.asarray([mass_particle[np.asarray(masks[k])].sum() for k in range(3)])
    if (np.any(counts <= 0) or np.any(patch_mass <= 0) or
            np.linalg.norm(resultants[2]) > 1e-10 or abs(moments[2, 0]-TORSION_NM) > 1e-10 or
            np.min(phi0[np.asarray(masks[0])]) < .99):
        raise RuntimeError('load/support preflight failed (patch, fixed pad, or exact target torque)')
    print(f"PREFLIGHT_DONE particles={len(xyz)} nodes={(args.grid+1)**3} patch_counts={counts.tolist()} patch_mass_kg={patch_mass.tolist()} torsion_Nm={moments[2,0]:.12g} grid_valid={bool(ref0.valid)}", flush=True)
    evaluate = jax.jit(jax.value_and_grad(lambda q: metrics(q)[0], has_aux=False))
    objective = jax.jit(lambda q: metrics(q)[0])
    print(f"GRADIENT_START grid={args.grid} particles={len(xyz)} steps={int(args.horizon/args.dt)} dt={args.dt:g} CFL={cfl:.5g}", flush=True)
    value, grad = evaluate(q0)
    value_finite = bool(np.isfinite(float(value)))
    grad_finite = bool(np.isfinite(np.asarray(grad)).all())
    print(f"AD_FINITE value={value_finite} gradient={grad_finite} gradient_l2={float(jnp.linalg.norm(grad))}", flush=True)
    if not value_finite or not grad_finite:
        raise RuntimeError('nonfinite initial merit or JAX gradient')
    direction = grad / (jnp.linalg.norm(grad) + 1e-30)
    eps = 1e-3
    plus, minus = objective(q0 + eps*direction), objective(q0 - eps*direction)
    fd = (plus-minus)/(2*eps)
    ad = jnp.vdot(grad, direction)
    rel = abs(fd-ad)/jnp.maximum(abs(fd), 1e-10)
    print(f"FD_CHECK direction={np.asarray(direction).tolist()} ad={float(ad):.12g} fd={float(fd):.12g} rel_error={float(rel):.6g}", flush=True)
    if not np.isfinite(float(rel)) or float(rel) > 2e-3:
        raise RuntimeError(f'directional FD failed, relative error={float(rel):g}')
    q_final = q0
    optimization = None
    if args.iterations:
        if args.iterations < 0 or args.iterations > 100 or args.learning_rate <= 0:
            raise ValueError('iterations must be in [0,100] and learning rate positive')
        eval_metrics = jax.jit(metrics)

        def unpack(q, result_value):
            merit_value, (response_value, volume_value, stress_value, case_tuple) = result_value
            cases = np.column_stack([np.asarray(column) for column in case_tuple])
            control_width_value = np.asarray(widths(q), dtype=np.float64)
            width_value = np.asarray(width_field(jnp.asarray(STATIONS), q), dtype=np.float64).T
            valid_value = bool(np.all(cases[:, 5]))
            feasible_value = bool(
                volume_value <= VOL_FRACTION_MAX + 1e-9
                and stress_value <= SIGMA_ALLOW + 1e-6
                and valid_value
                and np.all(control_width_value >= WIDTH_MIN - 1e-12)
                and np.all(control_width_value <= WIDTH_MAX + 1e-12)
            )
            violation_value = max(
                0.0,
                (float(volume_value) - VOL_FRACTION_MAX) / VOL_FRACTION_MAX,
                (float(stress_value) - SIGMA_ALLOW) / SIGMA_ALLOW,
                0.0 if valid_value else 1.0,
            )
            return {
                'merit': float(merit_value), 'response': float(response_value),
                'volume_fraction': float(volume_value), 'stress_bound_pa': float(stress_value),
                'maximum_solid_stress_pa': float(np.max(cases[:, 2])),
                'maximum_gray_stress_pa': float(np.max(cases[:, 3])),
                'maximum_floor_stress_pa': float(np.max(cases[:, 4])),
                'feasible': feasible_value, 'violation': violation_value,
                'valid': valid_value, 'widths_m': width_value.tolist(),
                'control_width_targets_m': control_width_value.tolist(),
                'controls': np.asarray(q, dtype=np.float64).tolist(),
                'case_metrics': cases.tolist(),
            }

        q = q0
        first = jnp.zeros_like(q)
        second = jnp.zeros_like(q)
        history = []
        current = unpack(q, eval_metrics(q))
        initial_response = current['response']
        accepted_updates = 0
        stop_reason = 'iteration_budget'
        for iteration in range(1, args.iterations + 1):
            merit_value, gradient = evaluate(q)
            if not np.isfinite(float(merit_value)) or not np.isfinite(np.asarray(gradient)).all():
                raise RuntimeError(f'nonfinite optimizer value/gradient at update {iteration}')
            first = .9 * first + .1 * gradient
            second = .999 * second + .001 * gradient**2
            mhat = first / (1 - .9**iteration)
            vhat = second / (1 - .999**iteration)
            direction_adam = mhat / (jnp.sqrt(vhat) + 1e-8)
            accepted = False
            accepted_state = None
            for backtrack in range(8):
                alpha = args.learning_rate * (0.5**backtrack)
                proposal = q - alpha * direction_adam
                candidate = unpack(proposal, eval_metrics(proposal))
                if current['feasible']:
                    accept = candidate['feasible'] and candidate['response'] < current['response'] - 1e-12
                else:
                    accept = (candidate['merit'] < current['merit']
                              and candidate['violation'] < current['violation'] - 1e-10)
                if accept:
                    accepted = True
                    accepted_state = (proposal, candidate)
                    break
            if accepted:
                q, current = accepted_state
                accepted_updates += 1
            row = {
                'iteration': iteration, 'accepted': accepted,
                **{k: current[k] for k in ('merit','response','volume_fraction','stress_bound_pa',
                                            'maximum_solid_stress_pa','maximum_gray_stress_pa',
                                            'maximum_floor_stress_pa','feasible','violation','valid')},
                'widths_m': json.dumps(current['widths_m']),
                'controls': json.dumps(current['controls']),
                'gradient_l2': float(jnp.linalg.norm(gradient)),
            }
            history.append(row)
            history_path = args.output / 'design-history.csv'
            temporary_csv = history_path.with_suffix('.tmp.csv')
            with temporary_csv.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            os.replace(temporary_csv, history_path)
            state_path = args.output / 'optimizer-state.npz'
            temporary_state = state_path.with_name(state_path.stem + '.tmp.npz')
            np.savez(temporary_state, controls=np.asarray(q), first=np.asarray(first),
                     second=np.asarray(second), iteration=np.asarray(iteration),
                     history_json=np.asarray(json.dumps(history)),
                     signature_json=np.asarray(json.dumps({
                         'grid':args.grid,'dt':args.dt,'horizon':args.horizon,
                         'stations_m':STATIONS.tolist(),'width_bounds_m':[WIDTH_MIN,WIDTH_MAX],
                         'volume_fraction_max':VOL_FRACTION_MAX,'stress_allow_pa':SIGMA_ALLOW,
                         'load_cases':['100N tension','100N bending',f'{TORSION_NM:g}Nm torsion'],
                     },sort_keys=True)))
            os.replace(temporary_state, state_path)
            print(f"OPT_UPDATE iteration={iteration} accepted={accepted} response={current['response']:.9g} vf={current['volume_fraction']:.6g} stress_bound={current['stress_bound_pa']:.6g} feasible={current['feasible']}",flush=True)
            if not accepted:
                stop_reason = 'projected_line_search_stalled'
                break
        q_final = q
        optimization = {
            'iterations_requested':args.iterations, 'updates_accepted':accepted_updates,
            'stop_reason':stop_reason, 'initial_response':initial_response,
            **current, 'history_csv':'design-history.csv',
            'optimizer_checkpoint':'optimizer-state.npz',
            'convergence_claim':False,
        }
        (args.output/'optimization-summary.json').write_text(
            json.dumps(optimization,indent=2)+'\n',encoding='utf-8')
        if not current['feasible']:
            raise RuntimeError('optimizer ended without a design satisfying volume, stress, member-size and state-validity constraints')
    initial_metrics = metrics(q0)
    _, (response, vf, ks, per_case_tuple) = initial_metrics
    per_case = np.column_stack([np.asarray(column) for column in per_case_tuple])
    phi = phi0
    width = np.asarray(width_field(jnp.asarray(STATIONS), q0)).T
    rail_phi = np.asarray(rail_occupancy(xyz, q0))
    solid_mask = rail_phi >= .5
    pitch = 0.5 / args.grid
    measured_widths = []
    for rail in range(2):
        for sx in STATIONS:
            near = np.abs(xyz_np[:, 0] - sx) <= pitch
            yy = xyz_np[:, 1]
            railmask = solid_mask & near & ((yy > HOLE_CENTRE[1]) if rail == 0 else (yy < HOLE_CENTRE[1]))
            if not railmask.any():
                raise RuntimeError(f'minimum-size audit found empty rail section at x={sx}')
            measured = (Y1 - yy[railmask].min()) if rail == 0 else (yy[railmask].max() - Y0)
            measured_widths.append(float(measured))
    if min(measured_widths) + pitch < WIDTH_MIN:
        raise RuntimeError(f'sampled minimum ligament width {min(measured_widths):.6g}m is below the 60mm bound')
    result = {
        'phase':'constrained-optimization' if optimization else 'directional-ad-fd', 'grid':args.grid, 'particles':int(len(xyz)),
        'steps':int(args.horizon/args.dt), 'dt_s':args.dt, 'horizon_s':args.horizon,
        'cfl':cfl, 'load_cases':['axial_tension_100N','transverse_bending_100N',f'pure_torsion_{TORSION_NM:g}Nm'],
        'geometry':{'outer_envelope_m':[X1-X0,Y1-Y0,Z1-Z0], 'through_hole_center_m':list(HOLE_CENTRE),
                    'through_hole_radius_m':HOLE_RADIUS, 'fixed_boss_outer_radius_m':BOSS_RADIUS,
                    'fixed_clamp_pad_x_m':[X0,.18], 'fixed_load_pad_x_m':[.82,X1]},
        'load_patch_particle_counts':counts.astype(int).tolist(),
        'load_patch_mass_kg':patch_mass.tolist(),
        'prescribed_resultant_forces_N':resultants.tolist(),
        'prescribed_resultant_moments_Nm_about_origin':moments.tolist(),
        'verified_torsion_moment_x_Nm':float(moments[2,0]),
        'design_variable_definition':'10 latent logits (5 upper, 5 lower). A normalized Gaussian RBF partition with centers at STATIONS and sigma=0.4125 maps the logits through sigmoid to local widths in [60,120]mm at every x; the minimum feature bound is this explicit width map, not the RBF length scale.',
        'control_stations_x_m':STATIONS.tolist(), 'rbf_sigma_normalized':CONTROL_SIGMA,
        'initial_uniform_width_target_m':args.initial_width_m,
        'initial_widths_at_control_centers_m':width.tolist(), 'minimum_width_bound_m':WIDTH_MIN,
        'minimum_geometric_width_m':float(width.min()),
        'measured_solid_widths_m_by_rail_station':measured_widths,
        'volume_fraction':float(vf), 'volume_fraction_upper_bound':VOL_FRACTION_MAX,
        'solid_fraction_sampled_minmax': [float(phi.min()),float(phi.max())],
        'gray_particle_count_phi_0p1_to_0p5':int(np.count_nonzero((phi>=.1)&(phi<.5))),
        'solid_particle_count_phi_ge_0p5':int(np.count_nonzero(phi>=.5)),
        'ersatz_particle_count_phi_eq_floor':int(np.count_nonzero(phi<=VOID_FLOOR+1e-8)),
        'void_floor_fraction':VOID_FLOOR, 'stress_definition':'Cauchy von Mises from the Neo-Hookean solid law; nested unnormalised p=24 norms over particles, time steps, and load cases of phi*sigma/0.5 form a smooth conservative upper bound on the maximum solid constituent stress for phi>=0.5. The declared 0.02 ersatz floor is included and cannot suppress the constrained stress. Direct discrete solid maximum is reported independently; gray 0.1<phi<0.5 and floor regions are separately quantified.',
        'stress_pnorm_upper_bound_pa':float(ks), 'stress_constraint_margin_pa':float(SIGMA_ALLOW-ks),
        'stress_allow_pa':SIGMA_ALLOW,
        'volume_constraint_margin':float(VOL_FRACTION_MAX-vf),
        'per_case_metrics':[{'response_m2':float(r[0]),'stress_pnorm_upper_bound_pa':float(r[1]),'peak_solid_von_mises_pa':float(r[2]),'peak_gray_von_mises_pa':float(r[3]),'peak_floor_von_mises_pa':float(r[4]),'all_steps_valid':bool(r[5])} for r in np.asarray(per_case)],
        'merit':float(value), 'gradient':np.asarray(grad).tolist(),
        'directional_ad':float(ad), 'directional_fd':float(fd), 'directional_relative_error':float(rel),
        'optimization':optimization,
        'optimized_widths_m':np.asarray(widths(q_final)).tolist(),
        'physical_scope':'bounded ligament sizing on fixed particle coordinates; exact through-hole is absent from material points and connected to upper/lower rails by a full-material annular boss; full-material clamp and loading pads; smooth p=1 Voigt mixture at rail boundaries; 20kPa cap equals E=200kPa times 10% nominal uniaxial strain and is an illustrative design limit, not a calibrated material failure property; no free topology claim',
    }
    (args.output/'gradient-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
