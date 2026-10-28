from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nlopt
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
import numpy as np

from examples.classic_cantilever_topopt import (
    E0, NU, FORCE_N, geometry, build_solver, gaussian_kernel, design_checks,
)
from mpm_jax.materials import linear_elastic, linear_stress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=48)
    parser.add_argument('--maxeval', type=int, default=48)
    parser.add_argument('--output', type=Path, default=Path('output/mma'))
    parser.add_argument('--initial-density', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = geometry(args.grid)
    cell = mesh['cell']
    _, ny, nz = mesh['shape']
    spine = (np.abs(cell[:, 1] - (ny - 1) / 2.0) <= 0.5) & (np.abs(cell[:, 2] - (nz - 1) / 2.0) <= 0.5)
    mesh['design_cell'][spine] = False
    mesh['fixed_cell'][spine] = True
    mesh['load'] *= 50000.0 / FORCE_N
    solver = build_solver(mesh, beta=12.0, vf=0.65, maxiter=2400, tol=1e-9)
    shape = mesh['shape']
    design = jnp.asarray(mesh['design_cell'].reshape(shape))
    kernel = jnp.asarray(gaussian_kernel(2.0))[None, None, ...]
    conn = jnp.asarray(mesh['indices'], dtype=jnp.int32)
    grad = jnp.asarray(mesh['gradients'])
    point_cell = jnp.asarray(mesh['cell_id_per_point'], dtype=jnp.int32)
    rhs = jnp.asarray(solver['rhs'])
    free = jnp.asarray(solver['free_flat'])
    material = linear_elastic(E0, NU)
    eye = jnp.eye(3)
    n = int(np.prod(shape))

    def density(q):
        q3 = q.reshape((1, 1) + shape)
        dimensions = ('NCDHW', 'OIDHW', 'NCDHW')
        num = lax.conv_general_dilated(q3, kernel, (1, 1, 1), 'SAME', dimension_numbers=dimensions)
        den = lax.conv_general_dilated(jnp.ones_like(q3), kernel, (1, 1, 1), 'SAME', dimension_numbers=dimensions)
        x = (num / den).reshape(shape)
        projected = (jnp.tanh(6.0) + jnp.tanh(12.0 * (x - .5))) / (2.0 * jnp.tanh(6.0))
        return jnp.clip(jnp.where(design, projected, 1.0).reshape(-1), 1e-8, 1.0)

    def responses(q):
        rho = density(q)
        u = jnp.where(free, solver['equilibrium'](rho), 0.0)
        residual = jnp.linalg.norm(jnp.where(free, solver['action'](rho, u) - rhs, 0.0)) / jnp.maximum(jnp.linalg.norm(rhs), 1.0)
        compliance = jnp.vdot(rhs, u).real
        gu = jnp.einsum('pki,pkj->pij', u.reshape((-1, 3))[conn], grad)
        strain = 0.5 * (gu + jnp.swapaxes(gu, 1, 2))
        stress = linear_stress(strain, material)
        dev = stress - jnp.trace(stress, axis1=1, axis2=2)[:, None, None] * eye / 3.0
        vm = jnp.sqrt(jnp.maximum(1e-30, 1.5 * jnp.sum(dev ** 2, axis=(1, 2))))
        mask = lax.stop_gradient(rho[point_cell] >= .10)
        selected = jnp.sqrt(jnp.maximum(rho[point_cell], 1e-8)) * vm / 1e8
        ks = logsumexp(64.0 * selected) / 64.0
        values = jnp.stack((jnp.mean(rho), compliance / 100.0 - 1.0, ks - 1.0, jnp.mean(4.0 * rho * (1.0 - rho)) / .15 - 1.0))
        return values, (rho, u, residual, jnp.max(jnp.where(mask, vm, 0.0)))

    calc = jax.jit(responses)
    grad_volume = jax.jit(jax.grad(lambda q: jnp.mean(density(q))))
    grad_binary = jax.jit(jax.grad(lambda q: jnp.mean(4.0 * density(q) * (1.0 - density(q))) / .15 - 1.0))
    grad_compliance = jax.jit(jax.grad(lambda q: responses(q)[0][1]))
    grad_stress = jax.jit(jax.grad(lambda q: responses(q)[0][2]))
    history = []
    snapshots = []
    cache = {}
    start = time.monotonic()
    best = None

    def evaluate(x):
        nonlocal best
        if 'x' in cache and np.array_equal(cache['x'], x):
            return cache
        q = jnp.asarray(x)
        values, aux = calc(q)
        derivative = jnp.stack((grad_volume(q), grad_compliance(q), grad_stress(q), grad_binary(q)))
        values.block_until_ready()
        vals = np.asarray(values)
        gradient = np.asarray(derivative)
        rho, u, residual, peak = (np.asarray(v) for v in aux)
        if not np.isfinite(vals).all() or not np.isfinite(gradient).all() or float(residual) > 1e-6:
            raise RuntimeError('Nonfinite response or equilibrium residual over limit')
        if not history:
            direction = np.random.default_rng(17).normal(size=n)
            direction /= np.linalg.norm(direction)
            step = 2e-4
            plus = np.asarray(calc(jnp.asarray(x + step * direction))[0])
            minus = np.asarray(calc(jnp.asarray(x - step * direction))[0])
            fd = (plus - minus) / (2 * step)
            ad = gradient @ direction
            errors = np.abs(fd - ad) / np.maximum(np.maximum(np.abs(fd), np.abs(ad)), 1e-10)
            check = dict(finite_difference=fd.tolist(), autodiff=ad.tolist(), relative_error=errors.tolist())
            (args.output / 'gradient-check.json').write_text(json.dumps(check, indent=2)+'\n')
            if not np.isfinite(errors).all() or np.any(errors > 0.005):
                raise RuntimeError(f'AD/finite-difference check failed: {check}')
        rec = dict(evaluation=len(history), volume_fraction=float(vals[0]),
                   deflection_m=float((vals[1]+1)*.002), stress_ks=float(vals[2]+1),
                   max_stress_pa=float(peak), relative_residual=float(residual),
                   feasible=bool(vals[1] <= 0 and vals[2] <= 0 and vals[3] <= 0), gray_indicator=float((vals[3]+1)*.15), wall_seconds=time.monotonic()-start)
        topology = design_checks(rho, shape, mesh['fixed_cell'], args.grid)
        rec['threshold_components'] = topology['six_connected_component_count']
        rec['threshold_clamp_to_pad_connected'] = topology['clamp_to_load_pad_connected']
        rec['stress_ks_definition'] = 'density-relaxed surrogate; actual and zero-void stresses require separate checks'
        rec['feasible'] = bool(rec['feasible'] and float(peak) <= 1e8 and rec['threshold_components'] == 1 and rec['threshold_clamp_to_pad_connected'])
        history.append(rec)
        snapshots.append(rho.reshape(shape).astype(np.float32))
        np.savez_compressed(args.output / 'design-evolution.npz', rho_history=np.stack(snapshots),
                            beta=np.full(len(snapshots), 12.0), iteration=np.arange(len(snapshots)))
        (args.output / 'evaluations.json').write_text(json.dumps(history, indent=2)+'\n')
        if rec['feasible'] and (best is None or rec['volume_fraction'] < best['volume_fraction']):
            best = rec.copy()
            np.savez_compressed(args.output / 'best-feasible.npz', q=x, rho=rho.reshape(shape), displacement=u.reshape((-1, 3)))
            (args.output / 'best-feasible.json').write_text(json.dumps(best, indent=2)+'\n')
        print(json.dumps({'event':'evaluation', **rec}, separators=(',', ':')), flush=True)
        cache.clear()
        cache.update(x=x.copy(), values=vals, derivative=gradient)
        return cache

    def callback(index):
        def f(x, grad_out):
            state = evaluate(x)
            if grad_out.size:
                grad_out[:] = state['derivative'][index] * (n if index == 0 else 1.0)
            return float(state['values'][index] * (n if index == 0 else 1.0))
        return f

    with np.load(args.initial_density) as seed:
        rho_seed = np.asarray(seed['rho']).reshape(-1)
    if rho_seed.shape != (n,):
        raise ValueError('Seed density shape differs from design grid')
    x0 = np.where(rho_seed >= .5, .9, .1)
    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.full(n, .0001))
    opt.set_upper_bounds(np.full(n, .9999))
    opt.set_min_objective(callback(0))
    opt.add_inequality_constraint(callback(1), 1e-6)
    opt.add_inequality_constraint(callback(2), 1e-6)
    opt.add_inequality_constraint(callback(3), 1e-6)
    opt.set_maxeval(args.maxeval)
    opt.set_ftol_rel(1e-7)
    try:
        final = opt.optimize(x0)
        status = int(opt.last_optimize_result())
        np.savez_compressed(args.output / 'final-controls.npz', q=final)
        (args.output / 'completion.json').write_text(json.dumps(dict(status=status, evaluations=len(history), best_feasible=best), indent=2)+'\n')
        print(json.dumps({'event':'completed','status':status,'evaluations':len(history),'best_feasible':best}), flush=True)
    except Exception as exc:
        (args.output / 'failure.json').write_text(json.dumps(dict(error=str(exc), evaluations=len(history), best_feasible=best), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
