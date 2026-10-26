"""Infer neo-Hookean E and nu from synthetic, multi-load 3D TL-MPM sensors.

Run with ``python -m examples.inverse_material --help`` from the repository root.
The default inverse discretization is intentionally moderate; ``--forward-grid``
and ``--forward-ppc`` configure an independent, higher-resolution holdout run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config
from mpm_jax.materials import neo_hookean
from mpm_jax.total_lagrangian import (
    NodalBoundary, deformation, initial_state, reference_grid, sample_particles, step,
)


RHO = 1050.0
TRUE_E = 1.25e6
TRUE_NU = 0.31
E_BOUNDS = (2.0e5, 3.0e6)
NU_BOUNDS = (0.12, 0.46)


def make_body(grid: int, ppc: int, domain: float):
    """Cubic material body, centered in the box, with structured reference points."""
    dx = domain / grid
    side = max(4, int(round(0.40 * grid * ppc)))
    spacing = 0.40 * domain / side
    start = 0.5 * (domain - 0.40 * domain) + 0.5 * spacing
    axis = start + spacing * np.arange(side)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    volume = np.full(len(xyz), spacing**3, dtype=np.float64)
    cfg = Config(n_grid=grid, dt=5e-5, domain_size=domain, dim=3, material="neo-hookean",
                 boundary_cells=0, gravity=(0.0, 0.0, 0.0))
    ref = reference_grid(jnp.asarray(xyz), jnp.asarray(volume), RHO, cfg)
    return cfg, ref, xyz, volume, dx


def make_loads(ref, cfg, steps: int, period_steps: int, amplitudes, directions):
    """Two independent harmonic boundary excitations with distinct directions."""
    nodes = np.asarray(ref.nodes)
    dx = cfg.dx
    occupied = np.asarray(ref.nodal_mass) > 0
    occupied_x = nodes[occupied, 0]
    if not occupied.any():
        raise ValueError("reference body has no occupied material nodes")
    # The cube is centered inside the background box. Derive end clamps from
    # occupied material nodes; fixing the box exterior would move empty air.
    xmin, xmax = float(occupied_x.min()), float(occupied_x.max())
    left = occupied & (nodes[:, 0] <= xmin + 1.01 * dx)
    right = occupied & (nodes[:, 0] >= xmax - 1.01 * dx)
    if not left.any() or not right.any():
        raise ValueError("both material end clamps must contain occupied nodes")
    if float(np.asarray(ref.nodal_mass)[left].sum()) <= 0 or float(np.asarray(ref.nodal_mass)[right].sum()) <= 0:
        raise ValueError("both material end clamps must carry positive nodal mass")
    fixed = jnp.asarray(left | right)
    signs = jnp.asarray(np.where(left, -1.0, np.where(right, 1.0, 0.0)))
    t = jnp.arange(steps + 1, dtype=jnp.float64) * cfg.dt
    omega = 2.0 * jnp.pi / (period_steps * cfg.dt)
    # Raised cosine starts and ends at rest, matching the zero-velocity state.
    phase = 0.5 * (1.0 - jnp.cos(omega * t))
    phase_v = 0.5 * omega * jnp.sin(omega * t)
    phase_a = 0.5 * omega**2 * jnp.cos(omega * t)
    def boundary(case):
        amp = amplitudes[case]
        direction = directions[case]
        scale = signs[:, None] * direction[None, :]
        u = amp * phase[:, None, None] * scale[None, :, :]
        v = amp * phase_v[:, None, None] * scale[None, :, :]
        a = amp * phase_a[:, None, None] * scale[None, :, :]
        return (fixed[None, :, None] * jnp.ones((steps + 1, len(nodes), 3), dtype=bool), u, v, a)
    return jax.vmap(boundary)(jnp.arange(len(amplitudes)))


def sensor_indices(xyz, count=12):
    # Choose distinct interior points nearest a fixed, reproducible pattern.
    targets = np.array([[.25,.25,.25],[.50,.25,.25],[.75,.25,.25],
                        [.25,.50,.50],[.50,.50,.50],[.75,.50,.50],
                        [.25,.75,.75],[.50,.75,.75],[.75,.75,.75],
                        [.25,.25,.75],[.75,.75,.25],[.50,.50,.75]])
    lo, hi = xyz.min(0), xyz.max(0)
    targets = lo + targets * (hi-lo)
    selected=[]
    for target in targets[:count]:
        d=np.linalg.norm(xyz-target,axis=1)
        order=np.argsort(d)
        selected.append(next(int(i) for i in order if int(i) not in selected))
    return np.asarray(selected,dtype=np.int32)


def trajectory(E, nu, cfg, ref, boundaries, sensor_ids, rematerialize=True):
    material = neo_hookean(E, nu)
    zero = jnp.zeros(3, dtype=jnp.float64)
    state = initial_state(ref, material, cfg, zero)
    fixed, disp, vel, acc = boundaries
    def advance_state(s, b0, b1):
        return step(s,ref,material,cfg,zero,zero,b0,b1)
    advance = jax.checkpoint(advance_state) if rematerialize else advance_state
    def scan_one(s, i):
        b0=NodalBoundary(fixed[i],disp[i],vel[i],acc[i])
        b1=NodalBoundary(fixed[i+1],disp[i+1],vel[i+1],acc[i+1])
        ns,_=advance(s,b0,b1)
        x,v=sample_particles(ns,ref).x,sample_particles(ns,ref).v
        return ns,(x[sensor_ids],v[sensor_ids],ns.valid)
    state, (x,v,valid)=jax.lax.scan(scan_one,state,jnp.arange(disp.shape[0]-1))
    return x,v,valid,state


def bounded(q):
    E=E_BOUNDS[0]+(E_BOUNDS[1]-E_BOUNDS[0])*jax.nn.sigmoid(q[0])
    nu=NU_BOUNDS[0]+(NU_BOUNDS[1]-NU_BOUNDS[0])*jax.nn.sigmoid(q[1])
    return E,nu


def unconstrained(E,nu):
    vals=[(E-E_BOUNDS[0])/(E_BOUNDS[1]-E_BOUNDS[0]),
          (nu-NU_BOUNDS[0])/(NU_BOUNDS[1]-NU_BOUNDS[0])]
    return jnp.asarray([math.log(x/(1-x)) for x in vals],dtype=jnp.float64)


def json_write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--grid',type=int,default=14,help='inverse grid cells per edge (moderate default)')
    ap.add_argument('--ppc',type=int,default=3,help='inverse particles per cell per axis')
    ap.add_argument('--forward-grid',type=int,default=24)
    ap.add_argument('--forward-ppc',type=int,default=4)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--iterations',type=int,default=40)
    ap.add_argument('--learning-rate',type=float,default=.08)
    ap.add_argument('--noise',type=float,default=0.0,help='Gaussian sensor noise, meters')
    ap.add_argument('--seed',type=int,default=12)
    ap.add_argument('--sensors',type=int,default=12)
    ap.add_argument('--fd-epsilon',type=float,default=1e-4)
    args=ap.parse_args(argv)
    if args.output.exists(): ap.error(f'output directory already exists: {args.output}')
    if args.steps<8 or args.iterations<1 or not 4<=args.sensors<=12 or args.noise<0: ap.error('invalid run controls')
    args.output.mkdir(parents=True)
    domain=1.0; period_steps=max(8,args.steps//2)
    directions=jnp.asarray([[1.,0.,0.],[0.,1.,0.]])
    amplitudes=jnp.asarray([.0020,.0035])
    cfg,ref,xyz,vol,_=make_body(args.grid,args.ppc,domain)
    sensors=sensor_indices(xyz,args.sensors)
    boundary=make_loads(ref,cfg,args.steps,period_steps,amplitudes,directions)
    # vmap exercises the same compiled scan with independent load protocols.
    synth_fn=jax.jit(jax.vmap(lambda fixed,disp,vel,acc:
        trajectory(TRUE_E,TRUE_NU,cfg,ref,(fixed,disp,vel,acc),sensors)[0]))
    observed=np.asarray(synth_fn(*boundary)).copy()
    if not np.isfinite(observed).all() or float(np.max(np.abs(observed))) < 1e-12:
        raise ValueError("synthetic sensor response is zero or nonfinite; check load/sensor overlap")
    rng=np.random.default_rng(args.seed)
    observed+=rng.normal(0.,args.noise,size=observed.shape)
    obs=jnp.asarray(observed)
    # Distinct excitations are simulated together; vmap is inside the objective.
    def objective(q):
        E,nu=bounded(q)
        def one(fixed,disp,vel,acc):
            return trajectory(E,nu,cfg,ref,(fixed,disp,vel,acc),sensors)[0]
        pred=jax.vmap(one)(*boundary)
        residual=pred-obs
        return jnp.mean(residual**2)
    value_grad=jax.jit(jax.value_and_grad(objective))
    q=unconstrained(7.5e5,.22)
    initial_loss,initial_grad=value_grad(q); jax.block_until_ready(initial_loss)
    if not np.isfinite(float(initial_loss)) or not np.isfinite(np.asarray(initial_grad)).all() or float(jnp.linalg.norm(initial_grad)) < 1e-15:
        raise ValueError("inverse objective has a zero or nonfinite parameter gradient")
    direction=jnp.asarray([.6,-.8],dtype=jnp.float64); direction/=jnp.linalg.norm(direction)
    eps=args.fd_epsilon
    fd=(objective(q+eps*direction)-objective(q-eps*direction))/(2*eps)
    autodiff=jnp.dot(initial_grad,direction)
    fd_rel=float(abs(fd-autodiff)/jnp.maximum(jnp.maximum(abs(fd),abs(autodiff)),1e-20))
    # Adam on bounded coordinates; save a resumable numerical checkpoint per iteration.
    m=jnp.zeros_like(q);v=jnp.zeros_like(q);loss_rows=[]
    best_q=q;best_loss=initial_loss
    start=time.monotonic()
    for k in range(1,args.iterations+1):
        loss,g=value_grad(q)
        m=.9*m+.1*g;v=.999*v+.001*g*g
        q=q-args.learning_rate*(m/(1-.9**k))/(jnp.sqrt(v/(1-.999**k))+1e-12)
        next_loss,_=value_grad(q); jax.block_until_ready(next_loss)
        if float(next_loss)<float(best_loss):best_q,best_loss=q,next_loss
        E,nu=map(float,bounded(q))
        loss_rows.append(dict(iteration=k,loss=float(next_loss),young_pa=E,poisson=nu,grad_norm=float(jnp.linalg.norm(g))))
        np.savez(args.output/'checkpoint.npz',iteration=k,q=np.asarray(q),m=np.asarray(m),v=np.asarray(v),best_q=np.asarray(best_q),best_loss=np.asarray(best_loss))
    Ehat,nuhat=map(float,bounded(best_q))
    with (args.output/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=loss_rows[0]);w.writeheader();w.writerows(loss_rows)
    # Independent holdout: new grid, finer particle quadrature, transverse load,
    # new sensor set, and more steps. It is simulated at truth and inferred values.
    fcfg,fref,fxyz,fvol,_=make_body(args.forward_grid,args.forward_ppc,domain)
    fsensors=sensor_indices(fxyz,args.sensors)
    fsteps=args.steps+20;fperiod=max(8,fsteps//3)
    famplitudes=jnp.asarray([.0028,.0017]); fdirections=jnp.asarray([[0.,0.,1.],[1.,1.,0.]])/jnp.sqrt(2.)
    fboundary=make_loads(fref,fcfg,fsteps,fperiod,famplitudes,fdirections)
    def forward(E,nu):
        return jax.jit(jax.vmap(lambda fixed,disp,vel,acc: trajectory(E,nu,fcfg,fref,(fixed,disp,vel,acc),fsensors)[0]))(*fboundary)
    hold_truth=np.asarray(forward(TRUE_E,TRUE_NU));hold_fit=np.asarray(forward(Ehat,nuhat))
    holdout_rmse=float(np.sqrt(np.mean((hold_fit-hold_truth)**2)))
    np.savez_compressed(args.output/'observations_synthetic.npz',observed=observed,truth_parameters=np.asarray([TRUE_E,TRUE_NU]),
        sensor_reference_positions=xyz[sensors],sensor_particle_ids=sensors,load_directions=np.asarray(directions),
        time_s=np.arange(args.steps)*cfg.dt)
    np.savez_compressed(args.output/'holdout_responses.npz',truth=hold_truth,inferred=hold_fit,
        sensor_reference_positions=fxyz[fsensors],sensor_particle_ids=fsensors,
        load_directions=np.asarray(fdirections),time_s=np.arange(fsteps)*fcfg.dt)
    source_root=Path(__file__).resolve().parents[1]
    inputs=['examples/inverse_material.py','mpm_jax/total_lagrangian.py','mpm_jax/core.py','mpm_jax/materials/elastic.py']
    report=dict(status='completed',method='3D reference-grid total-Lagrangian MPM, neo-Hookean Kirchhoff stress, quadratic B-spline transfer, velocity-Verlet; jax.lax.scan + jax.checkpoint + jax.vmap + jax.jit(value_and_grad)',
        observation_label='SYNTHETIC: generated by this same MPM solver at disclosed parameters; not experimental data',
        known_density_kg_m3=RHO,true_parameters=dict(young_pa=TRUE_E,poisson=TRUE_NU),
        initial_parameters=dict(young_pa=7.5e5,poisson=.22),estimated_parameters=dict(young_pa=Ehat,poisson=nuhat),
        relative_parameter_error=dict(young=abs(Ehat/TRUE_E-1),poisson=abs(nuhat/TRUE_NU-1)),
        losses=dict(initial=float(initial_loss),final=float(best_loss)),
        directional_gradient_check=dict(autodiff=float(autodiff),central_difference=float(fd),relative_error=fd_rel,direction=np.asarray(direction).tolist(),epsilon=eps),
        holdout=dict(load_directions=np.asarray(fdirections).tolist(),steps=fsteps,grid=args.forward_grid,ppc=args.forward_ppc,
            particles=len(fxyz),sensors=len(fsensors),response_rmse_m=holdout_rmse),
        inverse=dict(grid=args.grid,ppc=args.ppc,particles=len(xyz),steps=args.steps,dt_s=cfg.dt,sensors=len(sensors),
            excitation_amplitudes_m=np.asarray(amplitudes).tolist(),noise_std_m=args.noise,iterations=args.iterations,
            loaded_end_clamp_nodes=int(np.count_nonzero(np.asarray(boundary[0][0,:,0]))),
            estimated_peak_state_bytes='not measured; reverse-mode trajectory uses checkpointed scan and device-dependent memory'),
        elapsed_s=time.monotonic()-start,source_sha256={name:sha(source_root/name) for name in inputs})
    json_write(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','initial_parameters','estimated_parameters','losses','directional_gradient_check','holdout')},indent=2))


if __name__=='__main__': main()
