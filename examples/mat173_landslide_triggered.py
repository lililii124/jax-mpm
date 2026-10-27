"""Gravity-balanced 3D slope followed by a recorded MAT173 strength reduction.

The calculation is an algorithm demonstration, not native MAT173 final-check
evidence or a site-specific prediction. It records the numerical damping used
only to settle the pre-trigger state and removes that damping before triggering.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.contact import coupled_step
from mpm_jax.core import Config
from mpm_jax.boundaries import PlaneWall, fixed_box
from mpm_jax.materials import mat173, neo_hookean
from examples.mat173_landslide import (
    ROOT, SOURCE_FILES, atomic_json, atomic_npz, checkpoint_arrays,
    constitutive_preflight, deformation_metrics, make_foundation_particles,
    make_particles, make_state, save_frame, sha256,
)
from functools import partial


@partial(jax.jit, static_argnames=("configs", "steps"))
def advance_chunk(states, materials, configs, friction, gravity_scale,
                  ramp_time, damping, steps):
    def body(carry, _):
        state_pair, contact_impulse_sum, contact_loss_sum, normal_sum, tangent_sum, damping_loss_sum, gravity_impulse_sum = carry
        next_time = state_pair[0].time + configs[0].dt
        phase = jnp.clip(next_time / ramp_time, 0.0, 1.0)
        ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
        scale = jnp.where(gravity_scale < 0.0, ramp, gravity_scale)
        gravity = jnp.asarray((0.0, -9.81, 0.0), dtype=state_pair[0].x.dtype)
        forces = tuple(state.mass[:, None] * scale * gravity[None, :] for state in state_pair)
        updated, impulse, contact_loss, normal_impulse, tangent_impulse = coupled_step(
            state_pair, materials, configs, friction, forces, return_diagnostics=True)

        def damp_state(state):
            kinetic = 0.5 * jnp.sum(state.mass[:, None] * state.v * state.v)
            damped = state._replace(v=state.v * damping, C=state.C * damping)
            loss = kinetic * (1.0 - damping * damping)
            return damped, loss

        damped_pair, damping_loss = zip(*(damp_state(state) for state in updated))
        grav_impulse = sum(state.mass.sum() for state in state_pair) * scale * gravity * configs[0].dt
        return (
            damped_pair,
            contact_impulse_sum + impulse,
            contact_loss_sum + contact_loss,
            normal_sum + normal_impulse,
            tangent_sum + tangent_impulse,
            damping_loss_sum + sum(damping_loss),
            gravity_impulse_sum + grav_impulse,
        ), None

    zero_vec = jnp.zeros((3,), dtype=states[0].x.dtype)
    initial = (states, zero_vec, jnp.asarray(0.0), jnp.asarray(0.0),
               jnp.asarray(0.0), jnp.asarray(0.0), zero_vec)
    final, _ = jax.lax.scan(body, initial, None, length=steps)
    return final


def state_momentum(states):
    return np.sum([np.sum(np.asarray(s.mass)[:, None] * np.asarray(s.v), axis=0)
                   for s in states], axis=0)


def make_summary(states, references, angle, foundation_nearest, basal_mask):
    soil, foundation = states
    soil_x, base_x = np.asarray(soil.x), np.asarray(foundation.x)
    soil_u, base_u = soil_x - references[0], base_x - references[1]
    soil_v, base_v = np.asarray(soil.v), np.asarray(foundation.v)
    direction = np.asarray((math.cos(angle), -math.sin(angle), 0.0))
    relative_com_u = soil_u.mean(axis=0) - base_u.mean(axis=0)
    relative_com_v = soil_v.mean(axis=0) - base_v.mean(axis=0)
    basal_relative_u = soil_u[basal_mask] - base_u[foundation_nearest[basal_mask]]
    basal_relative_v = soil_v[basal_mask] - base_v[foundation_nearest[basal_mask]]
    plastic = np.asarray(soil.history.plastic_strain)
    logstrain, min_j = deformation_metrics(soil)
    base_logstrain, base_min_j = deformation_metrics(foundation)
    return {
        "time_s": float(soil.time),
        "soil_com_displacement_m": soil_u.mean(axis=0).tolist(),
        "foundation_com_displacement_m": base_u.mean(axis=0).tolist(),
        "relative_com_displacement_m": relative_com_u.tolist(),
        "relative_com_horizontal_x_m": float(relative_com_u[0]),
        "relative_com_vertical_y_m": float(relative_com_u[1]),
        "relative_com_downslope_m": float(relative_com_u @ direction),
        "relative_com_speed_m_s": relative_com_v.tolist(),
        "relative_com_horizontal_x_speed_m_s": float(relative_com_v[0]),
        "relative_com_vertical_y_speed_m_s": float(relative_com_v[1]),
        "relative_com_downslope_speed_m_s": float(relative_com_v @ direction),
        "soil_dx_m_mean": float(soil_u[:, 0].mean()),
        "soil_dy_m_mean": float(soil_u[:, 1].mean()),
        "soil_dx_m_p95": float(np.quantile(soil_u[:, 0], 0.95)),
        "soil_dy_m_p05": float(np.quantile(soil_u[:, 1], 0.05)),
        "soil_plastic_strain_mean": float(plastic.mean()),
        "soil_plastic_strain_p95": float(np.quantile(plastic, 0.95)),
        "soil_plastic_strain_max": float(plastic.max()),
        "soil_plastic_gt_1pct_fraction": float(np.mean(plastic > 0.01)),
        "soil_log_strain_abs_max": logstrain,
        "soil_minimum_J": min_j,
        "foundation_log_strain_abs_max": base_logstrain,
        "foundation_minimum_J": base_min_j,
        "basal_band_particle_count": int(basal_mask.sum()),
        "basal_relative_horizontal_x_m_mean": float(basal_relative_u[:, 0].mean()),
        "basal_relative_vertical_y_m_mean": float(basal_relative_u[:, 1].mean()),
        "basal_relative_downslope_m_mean": float((basal_relative_u @ direction).mean()),
        "basal_relative_horizontal_x_speed_m_s_mean": float(basal_relative_v[:, 0].mean()),
        "basal_relative_vertical_y_speed_m_s_mean": float(basal_relative_v[:, 1].mean()),
        "basal_relative_downslope_speed_m_s_mean": float((basal_relative_v @ direction).mean()),
        "valid": bool(soil.valid & foundation.valid),
    }


def yield_excess(stress, friction, cohesion):
    principal = np.linalg.eigvalsh(stress)[:, ::-1]
    phi = math.radians(friction)
    s, t = math.sin(phi), math.cos(phi)
    excess = (1+s)*principal[:, 0, None] - (1-s)*principal[:, 2, None] - 2*cohesion*t
    return float(excess.max()), float(np.quantile(excess, .95)), float(np.mean(excess > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--balance-max-steps", type=int, default=4000)
    parser.add_argument("--balance-window", type=int, default=100)
    parser.add_argument("--settle-windows", type=int, default=3)
    parser.add_argument("--post-steps", type=int, default=800)
    parser.add_argument("--frames", type=int, default=31)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=7.5e-5)
    parser.add_argument("--interface-friction", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.995)
    parser.add_argument("--balance-friction-deg", type=float, default=38.0)
    parser.add_argument("--balance-cohesion-pa", type=float, default=10000.0)
    parser.add_argument("--trigger-friction-deg", type=float, default=28.0)
    parser.add_argument("--trigger-cohesion-pa", type=float, default=1000.0)
    parser.add_argument("--p95-speed-limit", type=float, default=0.005)
    parser.add_argument("--com-drift-limit", type=float, default=1e-4)
    parser.add_argument("--reaction-balance-tolerance", type=float, default=0.08)
    args = parser.parse_args()
    if args.balance_window < 1 or args.balance_max_steps < args.balance_window:
        parser.error("invalid balance step/window settings")
    if args.frames < 2 or args.post_steps < 1:
        parser.error("post-steps>=1 and frames>=2 required")
    if not (0 < args.damping <= 1):
        parser.error("damping must be in (0,1]")

    soil_x0 = make_particles(args.spacing)
    foundation_x0 = make_foundation_particles()
    if len(soil_x0) < 30000:
        raise ValueError("expected at least 30,000 soil particles")
    density, shear, poisson, dilation = 1800.0, 5.86e6, 0.28, math.radians(2.0)
    foundation_density, foundation_young = 2500.0, 50e6
    soil_balance = mat173(shear_modulus=shear, poisson=poisson,
        friction_angle=math.radians(args.balance_friction_deg),
        cohesion=args.balance_cohesion_pa, dilation_angle=dilation,
        density=density, novoid=1)
    soil_trigger = mat173(shear_modulus=shear, poisson=poisson,
        friction_angle=math.radians(args.trigger_friction_deg),
        cohesion=args.trigger_cohesion_pa, dilation_angle=dilation,
        density=density, novoid=1)
    base_material = neo_hookean(young=foundation_young, poisson=0.28)
    preflight = {"balance": constitutive_preflight(soil_balance),
                 "trigger": constitutive_preflight(soil_trigger)}
    dx = 4.0 / args.grid
    soil_speed = math.sqrt((2*shear*(1-poisson)/((1+poisson)*(1-2*poisson)))/density)
    base_speed = math.sqrt((base_material.lam+2*base_material.mu)/foundation_density)
    cfl = args.dt*max(soil_speed, base_speed)/dx
    if cfl > .18:
        raise ValueError(f"CFL {cfl:.5g} exceeds 0.18")
    configs = (
        Config(n_grid=args.grid, dt=args.dt, domain_size=4.0, gravity=(0.,0.,0.),
            boundary_cells=0, material="mat173", dim=3, boundaries=(
                PlaneWall(axis=0, position=.2, side="lower", friction=0.),
                PlaneWall(axis=2, position=.3, side="lower", friction=0.),
                PlaneWall(axis=2, position=2.7, side="upper", friction=0.))),
        Config(n_grid=args.grid, dt=args.dt, domain_size=4.0, gravity=(0.,0.,0.),
            boundary_cells=0, material="neo-hookean", dim=3, boundaries=(
                fixed_box((0.,0.,0.), (4.,.3,4.)),
                PlaneWall(axis=0, position=.2, side="lower", friction=0.),
                PlaneWall(axis=2, position=.3, side="lower", friction=0.),
                PlaneWall(axis=2, position=2.7, side="upper", friction=0.))),
    )
    materials = (soil_balance, base_material)
    states = (make_state(soil_x0, args.spacing, density, soil_balance, jnp.float64),
              make_state(foundation_x0, .1, foundation_density, base_material, jnp.float64))
    references = (soil_x0, foundation_x0)
    angle = math.atan2(1.3, 2.0)
    # The soil's actual lower face is horizontal at y=0.5 m; the sloping
    # equation defines its free upper surface. Particle centers are 0.075 m
    # above the foundation's top particle centers at y=0.45 m.
    basal_mask = np.abs(soil_x0[:, 1]-.5) <= .075
    nearest = np.asarray([np.argmin((foundation_x0[:,0]-x)**2 + (foundation_x0[:,2]-z)**2)
        for x,z in soil_x0[basal_mask][:, (0,2)]], dtype=np.int64)
    full_nearest = np.full(len(soil_x0), -1, dtype=np.int64)
    full_nearest[basal_mask] = nearest

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be new or empty")
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    phi = math.radians(args.balance_friction_deg)
    normal_depth = 1.3 / math.sin(angle)
    gamma = density*9.81
    driving = gamma*normal_depth*math.sin(angle)*math.cos(angle)
    resistance = args.balance_cohesion_pa + gamma*normal_depth*math.cos(angle)**2*math.tan(phi)
    balance_fs = resistance/driving
    total_mass = float(sum(np.asarray(s.mass).sum() for s in states))
    soil_weight = density*9.81*args.spacing**3*len(soil_x0)
    total_weight = total_mass*9.81
    sources = {name: sha256(ROOT/name) for name in (*SOURCE_FILES, "examples/mat173_landslide_triggered.py")}
    meta = {
        "schema": "mat173-3d-slope-balanced-strength-reduction-v1",
        "scope": "numerical experiment only; MAT173 source finalcheck is pending; no site calibration",
        "source_sha256": sources,
        "slope_particles": len(soil_x0), "foundation_particles": len(foundation_x0),
        "geometry": {"slope_height_m":1.3,"run_m":2.0,"width_m":2.0,
            "foundation": {"x_m":[.2,3.3],"y_m":[.2,.5],"z_m":[.3,2.7],
                "fixed_grid_region_y_m":[0.,.3],"young_modulus_pa":foundation_young},
            "soil_spacing_m":args.spacing,"grid":args.grid},
        "material": {"G_Pa":shear,"nu":poisson,"psi_deg":2.0,"density_kg_m3":density,
            "balance_phi_deg":args.balance_friction_deg,"balance_cohesion_pa":args.balance_cohesion_pa,
            "trigger_phi_deg":args.trigger_friction_deg,"trigger_cohesion_pa":args.trigger_cohesion_pa,
            "interface_mu":args.interface_friction},
        "initial_condition": {"stress":"zero stress in both bodies; gravity is ramped over 0.02s, then numerically settled",
            "damping":"state v and affine C multiplied by damping factor each balance step only; removed before strength trigger",
            "damping_factor_per_step":args.damping,"translational_kinetic_energy_loss_recorded":True,
            "trigger":"instant, spatially uniform soil strength reduction at accepted balance checkpoint; no imposed displacement/velocity"},
        "equilibrium_screen": {"infinite_slope_factor_of_safety_balance_material":balance_fs,
            "driving_shear_pa":driving,"resistance_pa":resistance,
            "window_steps":args.balance_window,"settle_windows":args.settle_windows,
            "p95_speed_limit_m_s":args.p95_speed_limit,"COM_drift_limit_m":args.com_drift_limit,
            "total_vertical_reaction_weight_tolerance":args.reaction_balance_tolerance,
            "contact_vertical_support_reported_as_fraction_of_soil_weight":True,
            "interpretation":"screen only; finite bench, foundation and initial stress transient differ from infinite slope"},
        "cfl":cfl,"dt_s":args.dt,"post_steps_target":args.post_steps,
        "constitutive_preflight":preflight,
    }
    atomic_json(out/"model.json",meta)
    atomic_npz(out/"reference.npz",soil_x=soil_x0,foundation_x=foundation_x0)
    balance_rows=[]
    current=0
    prior_contact_ratio=None
    consecutive=0
    cumulative_contact=np.zeros(3)
    cumulative_contact_loss=0.0
    cumulative_normal=0.0
    cumulative_tangent=0.0
    cumulative_damping_loss=0.0
    balance_started=time.monotonic()
    while current < args.balance_max_steps:
        n=min(args.balance_window,args.balance_max_steps-current)
        before=state_momentum(states)
        prev_com=np.asarray(states[0].x).mean(axis=0)
        result=advance_chunk(states,materials,configs,args.interface_friction,-1.0,.02,args.damping,n)
        states, impulse, contact_loss, normal_impulse, tangent_impulse, damping_loss, gravity_impulse = result
        states=tuple(states)
        current+=n
        impulse=np.asarray(impulse);gravity_impulse=np.asarray(gravity_impulse)
        cumulative_contact+=impulse
        cumulative_contact_loss+=float(contact_loss)
        cumulative_normal+=float(normal_impulse)
        cumulative_tangent+=float(tangent_impulse)
        cumulative_damping_loss+=float(damping_loss)
        after=state_momentum(states)
        momentum_change=after-before
        boundary_impulse=momentum_change-gravity_impulse
        dt_window=n*args.dt
        soil_v=np.asarray(states[0].v);base_v=np.asarray(states[1].v)
        soil_p95=float(np.quantile(np.linalg.norm(soil_v,axis=1),.95))
        base_p95=float(np.quantile(np.linalg.norm(base_v,axis=1),.95))
        com_drift=float(np.linalg.norm(np.asarray(states[0].x).mean(axis=0)-prev_com))
        reaction_ratio=float(boundary_impulse[1]/max(-gravity_impulse[1],1e-30))
        contact_ratio=float(-impulse[1]/max(soil_weight*dt_window,1e-30))
        contact_stable=(prior_contact_ratio is not None and abs(contact_ratio-prior_contact_ratio)<.1)
        valid=bool(states[0].valid & states[1].valid)
        settled=(valid and soil_p95<args.p95_speed_limit and base_p95<args.p95_speed_limit
            and com_drift<args.com_drift_limit
            and abs(reaction_ratio-1.)<args.reaction_balance_tolerance
            and .35<contact_ratio<1.3 and contact_stable)
        consecutive=consecutive+1 if settled else 0
        row={"step":current,"time_s":float(states[0].time),"valid":valid,
            "soil_p95_speed_m_s":soil_p95,"foundation_p95_speed_m_s":base_p95,
            "soil_com_drift_m_window":com_drift,"contact_impulse_on_foundation_Ns":impulse.tolist(),
            "contact_impulse_rate_N":(impulse/dt_window).tolist(),
            "contact_normal_impulse_Ns":float(normal_impulse),"contact_tangent_impulse_Ns":float(tangent_impulse),
            "contact_vertical_support_fraction_of_soil_weight":contact_ratio,
            "contact_energy_loss_J_window":float(contact_loss),
            "cumulative_contact_energy_loss_J":cumulative_contact_loss,
            "damping_translational_energy_loss_J_window":float(damping_loss),
            "cumulative_damping_translational_energy_loss_J":cumulative_damping_loss,
            "gravity_impulse_Ns_window":gravity_impulse.tolist(),
            "boundary_reaction_impulse_Ns_inferred":boundary_impulse.tolist(),
            "vertical_boundary_reaction_to_weight":reaction_ratio,
            "momentum_change_Ns":momentum_change.tolist(),"settled_window":settled,
            "consecutive_settled_windows":consecutive}
        balance_rows.append(row)
        atomic_json(out/"balance_metrics.json",{"windows":balance_rows})
        if not valid:
            break
        if consecutive>=args.settle_windows:
            break
        prior_contact_ratio=contact_ratio

    balanced=(consecutive>=args.settle_windows and bool(states[0].valid & states[1].valid))
    if not balanced:
        summary={"status":"BALANCE_NOT_REACHED","balanced":False,"balance_steps":current,
            "balance_time_s":float(states[0].time),"valid":bool(states[0].valid & states[1].valid),
            "last_balance_window":balance_rows[-1] if balance_rows else None,
            "balance_wall_s":time.monotonic()-balance_started,"source_sha256":sources}
        atomic_npz(out/"checkpoint.npz",**checkpoint_arrays(states))
        atomic_json(out/"summary.json",summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
        raise SystemExit(2)

    atomic_npz(out/"balance_checkpoint.npz",**checkpoint_arrays(states))
    balanced_summary=make_summary(states,references,angle,full_nearest,basal_mask)
    balanced_summary["step"]=current
    balanced_summary["cumulative_contact_impulse_on_foundation_Ns"]=cumulative_contact.tolist()
    balanced_summary["cumulative_contact_energy_loss_J"]=cumulative_contact_loss
    balanced_summary["cumulative_damping_translational_energy_loss_J"]=cumulative_damping_loss
    atomic_json(out/"balance_summary.json",balanced_summary)
    old_stress=np.asarray(states[0].history.stress)
    start_excess=yield_excess(old_stress,args.trigger_friction_deg,args.trigger_cohesion_pa)
    states_after_trigger=(states[0],states[1])
    materials=(soil_trigger,base_material)
    post_start_step=current
    post_start_time=float(states[0].time)
    # The first undamped step is retained as an explicit validity/return-map check.
    first_result=advance_chunk(states_after_trigger,materials,configs,args.interface_friction,1.0,.02,1.0,1)
    states_after_trigger, first_impulse, first_contact_loss, first_normal, first_tangent, _, first_gravity = first_result
    states=tuple(states_after_trigger)
    current+=1
    cumulative_contact+=np.asarray(first_impulse)
    cumulative_contact_loss+=float(first_contact_loss)
    cumulative_normal+=float(first_normal)
    cumulative_tangent+=float(first_tangent)
    first_update={"valid":bool(states[0].valid & states[1].valid),
        "pretrigger_yield_excess_max_Pa":start_excess[0],
        "pretrigger_yield_excess_p95_Pa":start_excess[1],
        "pretrigger_fraction_outside_new_surface":start_excess[2],
        "first_step_plastic_strain_max":float(np.asarray(states[0].history.plastic_strain).max()),
        "first_step_contact_energy_loss_J":float(first_contact_loss),
        "first_step_contact_impulse_on_foundation_Ns":np.asarray(first_impulse).tolist(),
        "first_step_gravity_impulse_Ns":np.asarray(first_gravity).tolist(),
        "time_after_step_s":float(states[0].time),"step_after":current}
    atomic_json(out/"trigger_transition.json",{
        "trigger_time_before_first_update_s":post_start_time,"trigger_step_before_first_update":post_start_step,
        "old_material":{"friction_deg":args.balance_friction_deg,"cohesion_Pa":args.balance_cohesion_pa},
        "new_material":{"friction_deg":args.trigger_friction_deg,"cohesion_Pa":args.trigger_cohesion_pa},
        "update_check":first_update})
    if not first_update["valid"]:
        summary={"status":"INVALID_ON_TRIGGER","balanced":True,"balance_steps":post_start_step,
            "trigger_update":first_update,"source_sha256":sources}
        atomic_json(out/"summary.json",summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
        raise SystemExit(2)

    metrics_path=out/"metrics.csv"
    fields=("frame","step","time_s","soil_com_dx_m","soil_com_dy_m","foundation_com_dx_m",
        "foundation_com_dy_m","relative_com_dx_m","relative_com_dy_m","relative_com_downslope_m",
        "relative_com_downslope_speed_m_s","soil_dx_m_mean","soil_dy_m_mean","soil_dx_m_p95",
        "soil_dy_m_p05","soil_plastic_strain_mean","soil_plastic_strain_p95","soil_plastic_strain_max",
        "soil_plastic_gt_1pct_fraction","basal_relative_horizontal_x_m_mean","basal_relative_vertical_y_m_mean",
        "basal_relative_downslope_m_mean","basal_relative_horizontal_x_speed_m_s_mean",
        "basal_relative_vertical_y_speed_m_s_mean","basal_relative_downslope_speed_m_s_mean",
        "soil_log_strain_abs_max","soil_minimum_J","foundation_log_strain_abs_max","foundation_minimum_J",
        "cumulative_contact_energy_loss_J","cumulative_contact_normal_impulse_Ns",
        "cumulative_contact_tangent_impulse_Ns","cumulative_contact_impulse_x_Ns",
        "cumulative_contact_impulse_y_Ns","cumulative_contact_impulse_z_Ns","valid")
    writer=metrics_path.open("w",newline="",encoding="utf-8")
    csvw=csv.DictWriter(writer,fieldnames=fields);csvw.writeheader()
    post_frame_steps=np.unique(np.rint(np.linspace(0,args.post_steps,args.frames)).astype(int))
    # Frame zero is the post-trigger state after the one-step return-map check.
    frame=0
    def record_frame(frame_no):
        save_frame(out,frame_no,current,references,states)
        s=make_summary(states,references,angle,full_nearest,basal_mask)
        soil_u=s["soil_com_displacement_m"];base_u=s["foundation_com_displacement_m"]
        rel=s["relative_com_displacement_m"]
        row={"frame":frame_no,"step":current,"time_s":s["time_s"],
            "soil_com_dx_m":soil_u[0],"soil_com_dy_m":soil_u[1],
            "foundation_com_dx_m":base_u[0],"foundation_com_dy_m":base_u[1],
            "relative_com_dx_m":rel[0],"relative_com_dy_m":rel[1],
            "relative_com_downslope_m":s["relative_com_downslope_m"],
            "relative_com_downslope_speed_m_s":s["relative_com_downslope_speed_m_s"],
            "soil_dx_m_mean":s["soil_dx_m_mean"],"soil_dy_m_mean":s["soil_dy_m_mean"],
            "soil_dx_m_p95":s["soil_dx_m_p95"],"soil_dy_m_p05":s["soil_dy_m_p05"],
            "soil_plastic_strain_mean":s["soil_plastic_strain_mean"],
            "soil_plastic_strain_p95":s["soil_plastic_strain_p95"],
            "soil_plastic_strain_max":s["soil_plastic_strain_max"],
            "soil_plastic_gt_1pct_fraction":s["soil_plastic_gt_1pct_fraction"],
            "basal_relative_horizontal_x_m_mean":s["basal_relative_horizontal_x_m_mean"],
            "basal_relative_vertical_y_m_mean":s["basal_relative_vertical_y_m_mean"],
            "basal_relative_downslope_m_mean":s["basal_relative_downslope_m_mean"],
            "basal_relative_horizontal_x_speed_m_s_mean":s["basal_relative_horizontal_x_speed_m_s_mean"],
            "basal_relative_vertical_y_speed_m_s_mean":s["basal_relative_vertical_y_speed_m_s_mean"],
            "basal_relative_downslope_speed_m_s_mean":s["basal_relative_downslope_speed_m_s_mean"],
            "soil_log_strain_abs_max":s["soil_log_strain_abs_max"],"soil_minimum_J":s["soil_minimum_J"],
            "foundation_log_strain_abs_max":s["foundation_log_strain_abs_max"],"foundation_minimum_J":s["foundation_minimum_J"],
            "cumulative_contact_energy_loss_J":cumulative_contact_loss,
            "cumulative_contact_normal_impulse_Ns":cumulative_normal,
            "cumulative_contact_tangent_impulse_Ns":cumulative_tangent,
            "cumulative_contact_impulse_x_Ns":cumulative_contact[0],
            "cumulative_contact_impulse_y_Ns":cumulative_contact[1],
            "cumulative_contact_impulse_z_Ns":cumulative_contact[2],"valid":s["valid"]}
        csvw.writerow(row);writer.flush()
        atomic_npz(out/"checkpoint.npz",**checkpoint_arrays(states))
        return s
    s=record_frame(frame)
    post_current=0
    start=time.monotonic()
    small_strain=False
    for target in post_frame_steps[1:]:
        remaining=int(target-post_current)
        while remaining>0:
            n=min(10,remaining)
            result=advance_chunk(states,materials,configs,args.interface_friction,1.0,.02,1.0,n)
            states,impulse,loss,normal_impulse,tangent_impulse,_,_=result
            states=tuple(states);current+=n;post_current+=n;remaining-=n
            cumulative_contact+=np.asarray(impulse);cumulative_contact_loss+=float(loss)
            cumulative_normal+=float(normal_impulse);cumulative_tangent+=float(tangent_impulse)
            if not bool(states[0].valid & states[1].valid):
                break
        frame+=1
        s=record_frame(frame)
        if not s["valid"]:
            break
        if s["soil_log_strain_abs_max"]>=.05 or s["soil_plastic_strain_max"]>=.05:
            small_strain=True
            break
    writer.close()
    summary={"status":"INVALID" if not bool(states[0].valid & states[1].valid) else
        "SMALL_STRAIN_LIMIT" if small_strain else "COMPLETE" if post_current==args.post_steps else "INCOMPLETE",
        "balanced":True,"balance_steps":post_start_step,"balance_time_s":post_start_time,
        "trigger_update":first_update,"post_trigger_steps":post_current,"post_trigger_target_steps":args.post_steps,
        "actual_total_steps":current,"time_s":float(states[0].time),"valid":bool(states[0].valid & states[1].valid),
        "elapsed_after_trigger_s":time.monotonic()-start,"final_state":s,
        "cumulative_contact_impulse_on_foundation_Ns":cumulative_contact.tolist(),
        "cumulative_contact_energy_loss_J":cumulative_contact_loss,
        "cumulative_contact_normal_impulse_Ns":cumulative_normal,
        "cumulative_contact_tangent_impulse_Ns":cumulative_tangent,
        "balance_cumulative_damping_translational_energy_loss_J":cumulative_damping_loss,
        "source_sha256":sources,"constitutive_preflight":preflight}
    atomic_json(out/"summary.json",summary)
    print(json.dumps(summary,sort_keys=True),flush=True)
    if summary["status"] in ("INVALID","INCOMPLETE"):
        raise SystemExit(2)


if __name__=="__main__":
    main()
