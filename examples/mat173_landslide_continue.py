"""Continue the accepted triggered MAT173 slope state without re-equilibration."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config, State
from mpm_jax.boundaries import PlaneWall, fixed_box
from mpm_jax.materials.history import History
from mpm_jax.materials import mat173, neo_hookean
from examples.mat173_landslide import (
    ROOT, atomic_json, atomic_npz, checkpoint_arrays, deformation_metrics,
    save_frame, sha256,
)
from examples.mat173_landslide_triggered import advance_chunk


def load_states(path):
    arrays = np.load(path)
    states = []
    for prefix in ("soil_", "foundation_"):
        history = History(*(jnp.asarray(arrays[prefix + "history_" + name])
            for name in History._fields))
        states.append(State(
            jnp.asarray(arrays[prefix + "x"]), jnp.asarray(arrays[prefix + "v"]),
            jnp.asarray(arrays[prefix + "F"]), jnp.asarray(arrays[prefix + "C"]),
            jnp.asarray(arrays[prefix + "mass"]), jnp.asarray(arrays[prefix + "volume"]),
            history, jnp.asarray(arrays[prefix + "time"]),
            jnp.asarray(arrays[prefix + "valid"])))
    return tuple(states)


def summarize(states, balance, reference, balance_history, angle, basal_mask, nearest):
    soil, foundation = states
    soil_u = np.asarray(soil.x) - balance[0]
    base_u = np.asarray(foundation.x) - balance[1]
    soil_v, base_v = np.asarray(soil.v), np.asarray(foundation.v)
    direction = np.asarray((math.cos(angle), -math.sin(angle), 0.))
    relcom = soil_u.mean(0)-base_u.mean(0)
    relcom_v = soil_v.mean(0)-base_v.mean(0)
    basal_u = soil_u[basal_mask]-base_u[nearest[basal_mask]]
    basal_v = soil_v[basal_mask]-base_v[nearest[basal_mask]]
    plastic = np.asarray(soil.history.plastic_strain)-balance_history
    logstrain, minimum_j = deformation_metrics(soil)
    base_logstrain, base_j = deformation_metrics(foundation)
    return {
        "step": int(round(float(soil.time)/0.000075)), "time_s": float(soil.time),
        "soil_com_dx_since_balance_m": float(soil_u[:,0].mean()),
        "soil_com_dy_since_balance_m": float(soil_u[:,1].mean()),
        "foundation_com_dx_since_balance_m": float(base_u[:,0].mean()),
        "foundation_com_dy_since_balance_m": float(base_u[:,1].mean()),
        "relative_com_dx_since_balance_m": float(relcom[0]),
        "relative_com_dy_since_balance_m": float(relcom[1]),
        "relative_com_downslope_since_balance_m": float(relcom@direction),
        "relative_com_horizontal_projection_m": float(relcom[0]*direction[0]),
        "relative_com_vertical_projection_m": float(relcom[1]*direction[1]),
        "relative_com_dx_speed_m_s": float(relcom_v[0]),
        "relative_com_dy_speed_m_s": float(relcom_v[1]),
        "relative_com_downslope_speed_m_s": float(relcom_v@direction),
        "basal_particle_count": int(basal_mask.sum()),
        "basal_relative_dx_mean_m": float(basal_u[:,0].mean()),
        "basal_relative_dy_mean_m": float(basal_u[:,1].mean()),
        "basal_relative_downslope_mean_m": float((basal_u@direction).mean()),
        "basal_relative_dx_p90_m": float(np.quantile(basal_u[:,0],.9)),
        "basal_relative_dx_speed_mean_m_s": float(basal_v[:,0].mean()),
        "basal_relative_dy_speed_mean_m_s": float(basal_v[:,1].mean()),
        "basal_relative_downslope_speed_mean_m_s": float((basal_v@direction).mean()),
        "plastic_increment_mean": float(plastic.mean()),
        "plastic_increment_p95": float(np.quantile(plastic,.95)),
        "plastic_increment_max": float(plastic.max()),
        "plastic_increment_gt_0.5pct_fraction": float(np.mean(plastic>.005)),
        "plastic_increment_gt_1pct_fraction": float(np.mean(plastic>.01)),
        "plastic_by_depth": {
            "bottom_y_0.525": {"n":int(np.sum(reference[:,1]<.55)),
                "p95":float(np.quantile(plastic[reference[:,1]<.55],.95)),
                "max":float(plastic[reference[:,1]<.55].max())},
            "second_y_0.575": {"n":int(np.sum((reference[:,1]>=.55)&(reference[:,1]<.60))),
                "p95":float(np.quantile(plastic[(reference[:,1]>=.55)&(reference[:,1]<.60)],.95)),
                "max":float(plastic[(reference[:,1]>=.55)&(reference[:,1]<.60)].max())}},
        "soil_log_strain_abs_max": logstrain, "soil_minimum_J": minimum_j,
        "foundation_log_strain_abs_max": base_logstrain, "foundation_minimum_J": base_j,
        "valid": bool(soil.valid & foundation.valid),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=Path("resume"))
    parser.add_argument("--output",type=Path,default=Path("output"))
    parser.add_argument("--steps",type=int,default=1600)
    parser.add_argument("--frames",type=int,default=41)
    parser.add_argument("--grid",type=int,default=40)
    parser.add_argument("--spacing",type=float,default=.05)
    parser.add_argument("--dt",type=float,default=7.5e-5)
    parser.add_argument("--interface-friction",type=float,default=.1)
    args=parser.parse_args()
    parent=args.parent.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be new or empty")
    parent_summary=json.loads((parent/"summary.json").read_text())
    assert parent_summary["status"]=="COMPLETE" and parent_summary["valid"]
    assert parent_summary["balanced"] and parent_summary["post_trigger_steps"]==800
    parent_checkpoint=parent/"checkpoint.npz"
    parent_reference=np.load(parent/"reference.npz")
    reference=(parent_reference["soil_x"],parent_reference["foundation_x"])
    states=load_states(parent_checkpoint)
    balance_checkpoint=np.load(parent/"balance_checkpoint.npz")
    balance=(balance_checkpoint["soil_x"],balance_checkpoint["foundation_x"])
    balance_history=balance_checkpoint["soil_history_plastic_strain"]
    model=json.loads((parent/"model.json").read_text())
    params=model["material"]
    density=float(params["density_kg_m3"])
    base_density=float(model["geometry"]["foundation"].get("density_kg_m3",2500.))
    shear=float(params["G_Pa"]); poisson=float(params["nu"])
    soil=mat173(shear_modulus=shear,poisson=poisson,
        friction_angle=math.radians(params["trigger_phi_deg"]),
        cohesion=params["trigger_cohesion_pa"],
        dilation_angle=math.radians(params["psi_deg"]),density=density,novoid=1)
    foundation=neo_hookean(young=model["geometry"]["foundation"]["young_modulus_pa"],poisson=.28)
    configs=(
        Config(n_grid=args.grid,dt=args.dt,domain_size=4.,gravity=(0.,0.,0.),boundary_cells=0,
            material="mat173",dim=3,boundaries=(
                PlaneWall(axis=0,position=.2,side="lower",friction=0.),
                PlaneWall(axis=2,position=.3,side="lower",friction=0.),
                PlaneWall(axis=2,position=2.7,side="upper",friction=0.))),
        Config(n_grid=args.grid,dt=args.dt,domain_size=4.,gravity=(0.,0.,0.),boundary_cells=0,
            material="neo-hookean",dim=3,boundaries=(
                fixed_box((0.,0.,0.),(4.,.3,4.)),
                PlaneWall(axis=0,position=.2,side="lower",friction=0.),
                PlaneWall(axis=2,position=.3,side="lower",friction=0.),
                PlaneWall(axis=2,position=2.7,side="upper",friction=0.))),
    )
    materials=(soil,foundation)
    angle=math.atan2(1.3,2.)
    basal_mask=np.abs(reference[0][:,1]-.5)<=.075
    nearest=np.full(len(reference[0]),-1,dtype=np.int64)
    for i in np.flatnonzero(basal_mask):
        nearest[i]=np.argmin((reference[1][:,0]-reference[0][i,0])**2+
                             (reference[1][:,2]-reference[0][i,2])**2)
    out.mkdir(parents=True,exist_ok=True);(out/"frames").mkdir(exist_ok=True)
    sources={name:sha256(ROOT/name) for name in (
        "examples/mat173_landslide.py","examples/mat173_landslide_triggered.py",
        "examples/mat173_landslide_continue.py","mpm_jax/contact.py",
        "mpm_jax/core.py","mpm_jax/boundaries.py","mpm_jax/materials/mat173.py",
        "mpm_jax/materials/history.py","mpm_jax/materials/elastic.py")}
    parent_sha=sha256(parent_checkpoint)
    meta={"schema":"mat173-slope-strength-reduction-continuation-v1",
        "parent_task":"mat173-slope-ls-3d-triggered-v1",
        "parent_checkpoint_sha256":parent_sha,
        "parent_total_steps":parent_summary["actual_total_steps"],
        "parent_source_sha256":parent_summary["source_sha256"],
        "continuation_source_sha256":sources,"steps":args.steps,"dt_s":args.dt,
        "gravity_m_s2":[0.,-9.81,0.],"gravity_scale":1.,"damping":0.,
        "material":{"phi_deg":params["trigger_phi_deg"],"cohesion_Pa":params["trigger_cohesion_pa"],
            "dilation_deg":params["psi_deg"],"G_Pa":shear,"nu":poisson},
        "contact_friction":args.interface_friction,
        "restarts_from":"actual final NPZ state including F,C,velocity,stress and all MAT173 history; no solver rerun",
        "scope":"continuation of strength-reduction numerical response; not a validated landslide prediction"}
    atomic_json(out/"model.json",meta)
    soil_x0=reference[0]; foundation_x0=reference[1]
    frame_steps=np.unique(np.rint(np.linspace(0,args.steps,args.frames)).astype(int))
    summary_metrics=[]
    cumulative_contact=np.asarray(parent_summary["cumulative_contact_impulse_on_foundation_Ns"],dtype=float)
    cumulative_contact_loss=float(parent_summary["cumulative_contact_energy_loss_J"])
    cumulative_normal=float(parent_summary["cumulative_contact_normal_impulse_Ns"])
    cumulative_tangent=float(parent_summary["cumulative_contact_tangent_impulse_Ns"])
    continuation_contact=np.zeros(3);continuation_loss=0.;continuation_normal=0.;continuation_tangent=0.
    csvfile=(out/"metrics.csv").open("w",newline="",encoding="utf-8")
    fields=("frame","step","time_s","relative_com_dx_since_balance_m","relative_com_dy_since_balance_m",
        "relative_com_downslope_since_balance_m","relative_com_horizontal_projection_m",
        "relative_com_vertical_projection_m","basal_relative_dx_mean_m","basal_relative_dy_mean_m",
        "basal_relative_downslope_mean_m","basal_relative_dx_speed_mean_m_s",
        "basal_relative_dy_speed_mean_m_s","basal_relative_downslope_speed_mean_m_s",
        "plastic_increment_p95","plastic_increment_max","plastic_increment_gt_1pct_fraction",
        "soil_log_strain_abs_max","soil_minimum_J","cumulative_contact_energy_loss_J",
        "cumulative_contact_normal_impulse_Ns","cumulative_contact_tangent_impulse_Ns","valid")
    writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
    current=int(parent_summary["actual_total_steps"]);start_step=current
    state_time=float(states[0].time);start_wall=time.monotonic();small_strain=False
    frame=0
    def record(frame_index):
        save_frame(out,frame_index,current,reference,states)
        row=summarize(states,balance,soil_x0,balance_history,angle,basal_mask,nearest)
        row.update({"frame":frame_index,"cumulative_contact_energy_loss_J":cumulative_contact_loss,
            "cumulative_contact_normal_impulse_Ns":cumulative_normal,
            "cumulative_contact_tangent_impulse_Ns":cumulative_tangent})
        summary_metrics.append(row)
        writer.writerow({k:row.get(k) for k in fields});csvfile.flush()
        atomic_npz(out/"checkpoint.npz",**checkpoint_arrays(states))
        return row
    last=record(frame)
    post_done=0
    for target in frame_steps[1:]:
        remaining=int(target-post_done)
        while remaining>0:
            n=min(10,remaining)
            result=advance_chunk(states,materials,configs,args.interface_friction,1.,.02,1.,n)
            states,impulse,loss,norm,tangent,_,_=result;states=tuple(states)
            current+=n;post_done+=n;remaining-=n
            impulse=np.asarray(impulse);continuation_contact+=impulse;cumulative_contact+=impulse
            continuation_loss+=float(loss);cumulative_contact_loss+=float(loss)
            continuation_normal+=float(norm);cumulative_normal+=float(norm)
            continuation_tangent+=float(tangent);cumulative_tangent+=float(tangent)
            if not bool(states[0].valid & states[1].valid):break
        frame+=1;last=record(frame)
        if not last["valid"]:break
        if last["soil_log_strain_abs_max"]>=.05 or last["plastic_increment_max"]+float(balance_history.max())>=.05:
            small_strain=True;break
    csvfile.close()
    status="INVALID" if not bool(states[0].valid & states[1].valid) else \
        "SMALL_STRAIN_LIMIT" if small_strain else "COMPLETE" if post_done==args.steps else "INCOMPLETE"
    summary={"status":status,"valid":bool(states[0].valid & states[1].valid),
        "parent_task":"mat173-slope-ls-3d-triggered-v1","parent_checkpoint_sha256":parent_sha,
        "parent_total_steps":start_step,"continuation_steps":post_done,
        "continuation_target_steps":args.steps,"actual_total_steps":current,
        "start_time_s":state_time,"time_s":float(states[0].time),
        "elapsed_s":time.monotonic()-start_wall,"final_state":last,
        "continuation_contact_impulse_on_foundation_Ns":continuation_contact.tolist(),
        "continuation_contact_energy_loss_J":continuation_loss,
        "continuation_contact_normal_impulse_Ns":continuation_normal,
        "continuation_contact_tangent_impulse_Ns":continuation_tangent,
        "total_chain_contact_energy_loss_J":cumulative_contact_loss,
        "total_chain_contact_impulse_on_foundation_Ns":cumulative_contact.tolist(),
        "total_chain_contact_normal_impulse_Ns":cumulative_normal,
        "total_chain_contact_tangent_impulse_Ns":cumulative_tangent,
        "source_sha256":sources}
    atomic_json(out/"summary.json",summary)
    print(json.dumps(summary,sort_keys=True),flush=True)
    if status in ("INVALID","INCOMPLETE"):raise SystemExit(2)


if __name__=="__main__":main()
