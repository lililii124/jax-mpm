"""Homogeneous softening soil: gravity balance followed by full-height toe removal."""
import argparse
import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from examples.mat173_toe_helpers import (
    particles,
    foundation,
    soil_state,
    base_state,
    advance,
    save_checkpoint,
    atomic_json,
)
from mpm_jax.core import Config
from mpm_jax.boundaries import PlaneWall, fixed_box
from mpm_jax.materials import mat173_softening, neo_hookean
from mpm_jax.materials.mat173_tensile_cutoff import with_tensile_cutoff

PLAN = json.loads(Path(__file__).with_name("mat173_toe_plan.json").read_text())
OUT = Path("outputs/mat173-toe")
DT = 3e-5
SP = 0.0125
MU = 0.7


def momentum(s):
    return np.sum(np.asarray(s.mass)[:, None] * np.asarray(s.v), axis=0)


def kinetic(s):
    return float(0.5 * np.sum(np.asarray(s.mass)[:, None] * np.asarray(s.v) ** 2))


def frame(step, pair, refs, previous_plastic):
    soil = pair[0]
    arrays = {"step": step, "time_s": float(soil.time)}
    for name, s, ref in zip(("soil", "base"), pair, refs):
        arrays.update(
            {
                name + "_reference": ref,
                name + "_x": np.asarray(s.x),
                name + "_displacement": np.asarray(s.x) - ref,
                name + "_F": np.asarray(s.F),
                name + "_velocity": np.asarray(s.v),
                name + "_reference_volume": np.asarray(s.volume),
                name + "_failed": np.asarray(s.history.failed),
            }
        )
    arrays.update(
        soil_damage=np.asarray(soil.history.damage),
        soil_kappa=np.asarray(soil.history.hardening),
        soil_stress_Pa=np.asarray(soil.history.stress),
        soil_plastic_strain=np.asarray(soil.history.plastic_strain),
        soil_plastic_strain_increment=np.asarray(soil.history.plastic_strain) - previous_plastic,
        soil_tensile_plastic_strain=np.asarray(soil.history.tensile_plastic_strain),
        soil_partition_ambiguous_steps=np.asarray(soil.history.partition_ambiguous_steps),
    )
    np.savez_compressed(OUT / f"frame_{step:05d}.npz", **arrays)


def guard_check(g, valid, phase):
    g = np.asarray(g)
    cap = 0.05 if phase.startswith("balance") else 0.2
    return bool(
        bool(valid)
        and np.isfinite(g).all()
        and g[0] < cap
        and g[1] < 0.000625
        and g[2] < 0.0035
        and g[3] < 0
    )


def window(pair, mats, configs, steps, damping, phase):
    P0 = [momentum(s) for s in pair]
    com0 = np.asarray(pair[0].x).mean(0)
    result = advance(pair, mats, configs, MU, DT, 0.02, damping, steps)
    pair, imp, loss, normal, tangent, dl, gi, dps, dpb, gps, gpb, gmax, gvalid = result
    pair = tuple(pair)
    dP = [momentum(s) - p for s, p in zip(pair, P0)]
    damp = [np.asarray(dps), np.asarray(dpb)]
    grav = [np.asarray(gps), np.asarray(gpb)]
    total_weight = sum(float(np.asarray(s.mass).sum()) for s in pair) * 9.81
    soil_weight = float(np.asarray(pair[0].mass).sum()) * 9.81
    # Net physical force estimates exclude the separately tracked artificial damping impulse.
    net = np.linalg.norm(dP[0] + dP[1] - damp[0] - damp[1]) / (total_weight * steps * DT)
    soil_net = np.linalg.norm(dP[0] - damp[0]) / (soil_weight * steps * DT)
    speeds = [float(np.quantile(np.linalg.norm(np.asarray(s.v), axis=1), 0.95)) for s in pair]
    ke = sum(kinetic(s) for s in pair)
    drift = float(np.linalg.norm(np.asarray(pair[0].x).mean(0) - com0))
    g = np.asarray(gmax)
    row = {
        "phase": phase,
        "step": int(round(float(pair[0].time) / DT)),
        "time_s": float(pair[0].time),
        "p95_velocity_each_body_m_s": speeds,
        "soil_COM_drift_m": drift,
        "global_net_force_weight_ratio": float(net),
        "soil_net_force_weight_ratio": float(soil_net),
        "kinetic_J_per_m": ke,
        "kinetic_to_weight_height_ratio": ke / (total_weight * 1.3),
        "contact_support_fraction": float(-np.asarray(imp)[1] / (soil_weight * steps * DT)),
        "soil_contact_momentum_closure_residual_Ns": (
            dP[0] - grav[0] - damp[0] + np.asarray(imp)
        ).tolist(),
        "contact_impulse_on_foundation_Ns": np.asarray(imp).tolist(),
        "gravity_impulse_Ns": np.asarray(gi).tolist(),
        "damping_impulse_Ns": (damp[0] + damp[1]).tolist(),
        "contact_loss_J": float(loss),
        "damping_loss_J": float(dl),
        "max_total_Hencky": float(g[0]),
        "max_incremental_Hencky": float(g[1]),
        "max_soil_elastic_compliance": float(g[2]),
        "minimum_J": float(-g[3]),
        "valid": bool(gvalid),
        "guards_pass": guard_check(g, gvalid, phase),
        "kappa_max": float(np.asarray(pair[0].history.hardening).max()),
        "damage_max": float(np.asarray(pair[0].history.damage).max()),
    }
    row["balance_checks_pass"] = bool(
        row["guards_pass"]
        and float(pair[0].time) >= 0.02
        and max(speeds) < 0.005
        and drift < 1e-4
        and net < 0.02
        and soil_net < 0.02
        and row["kinetic_to_weight_height_ratio"] < 1e-6
        and 0.35 < row["contact_support_fraction"] < 1.3
    )
    row.update(
        tensile_diagnostic_max=float(np.asarray(pair[0].history.tensile_plastic_strain).max()),
        tensile_diagnostic_p95=float(
            np.quantile(np.asarray(pair[0].history.tensile_plastic_strain), 0.95)
        ),
        partition_ambiguous_points=int(
            np.count_nonzero(np.asarray(pair[0].history.partition_ambiguous_steps))
        ),
        partition_ambiguous_steps_sum=int(
            np.asarray(pair[0].history.partition_ambiguous_steps).sum()
        ),
    )
    return pair, row


def retain(s, keep):
    h = s.history._replace(**{k: getattr(s.history, k)[keep] for k in s.history._fields})
    return s._replace(
        x=s.x[keep],
        v=s.v[keep],
        F=s.F[keep],
        C=s.C[keep],
        mass=s.mass[keep],
        volume=s.volume[keep],
        history=h,
    )


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-balance-steps", type=int, default=9000)
    parser.add_argument("--postcut-steps", type=int, default=3000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed model and limits without advancing particles",
    )
    args = parser.parse_args(argv)
    if (
        args.max_balance_steps <= 0
        or args.max_balance_steps % 100
        or args.postcut_steps < 0
        or args.postcut_steps % 100
    ):
        parser.error("step limits must be multiples of 100; balance steps must be positive")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    PLAN,
                    balance=dict(PLAN["balance"], max_steps=args.max_balance_steps),
                    max_balance_steps=args.max_balance_steps,
                    postcut_steps=args.postcut_steps,
                ),
                indent=2,
            )
        )
        return
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    m = PLAN["material"]
    sm = with_tensile_cutoff(
        mat173_softening(
            shear_modulus=m["G_Pa"],
            poisson=m["nu"],
            friction_peak=np.deg2rad(m["phi_peak_deg"]),
            friction_residual=np.deg2rad(m["phi_residual_deg"]),
            cohesion_peak=m["cohesion_peak_Pa"],
            cohesion_residual=m["cohesion_residual_Pa"],
            dilation_peak=np.deg2rad(m["dilation_peak_deg"]),
            dilation_residual=np.deg2rad(m["dilation_residual_deg"]),
            density=m["rho_kg_m3"],
            characteristic_length=m["L_m"],
            fracture_slip=m["Dc_m"],
        ),
        0.0,
        separate_shear_softening=True,
    )
    bm = neo_hookean(young=50e6, poisson=0.28)
    xs = particles(SP)
    xb = foundation()
    refs = (xs, xb)
    pair = (
        soil_state(xs, SP, 1800.0, sm, jnp.float64),
        base_state(xb, 0.025, 2500.0, bm, jnp.float64),
    )
    mats = (sm, bm)
    configs = (
        Config(
            n_grid=160,
            dt=DT,
            domain_size=4.0,
            gravity=(0.0, 0.0),
            boundary_cells=0,
            material="mat173-softening-tensile-cutoff",
            dim=2,
            boundaries=(PlaneWall(axis=0, position=0.2, side="lower", friction=0.0),),
        ),
        Config(
            n_grid=160,
            dt=DT,
            domain_size=4.0,
            gravity=(0.0, 0.0),
            boundary_cells=0,
            material="neo-hookean",
            dim=2,
            boundaries=(
                fixed_box((0.0, 0.0), (4.0, 0.3)),
                PlaneWall(axis=0, position=0.2, side="lower", friction=0.0),
            ),
        ),
    )
    atomic_json(
        OUT / "model.json",
        dict(
            PLAN,
            balance=dict(PLAN["balance"], max_steps=args.max_balance_steps),
            postcut_steps=args.postcut_steps,
        ),
    )
    rows = []
    consecutive = 0
    accepted = pair
    excavation_performed = False
    save_checkpoint(OUT / "virgin_checkpoint.npz", pair)

    def record(row):
        rows.append(row)
        atomic_json(OUT / "metrics.json", {"windows": rows})
        print(
            json.dumps(
                row,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ),
            flush=True,
        )

    def stop(status, row):
        atomic_json(
            OUT / "summary.json",
            {
                "status": status,
                "excavation_performed": excavation_performed,
                "last_accepted_step": int(round(float(accepted[0].time) / DT)),
                "last_window": row,
                "scope": PLAN["scope"],
            },
        )

    while float(pair[0].time) < args.max_balance_steps * DT:
        candidate, row = window(pair, mats, configs, 100, 0.995, "balance_damped")
        record(row)
        if not row["guards_pass"]:
            save_checkpoint(OUT / "failed_attempt.npz", candidate)
            stop("BALANCE_GUARD_STOP_NO_EXCAVATION", row)
            return
        pair = candidate
        accepted = pair
        save_checkpoint(OUT / "checkpoint.npz", pair)
        consecutive = consecutive + 1 if row["balance_checks_pass"] else 0
        if consecutive >= 3:
            break
    if consecutive < 3:
        stop("BALANCE_NOT_REACHED_NO_EXCAVATION", rows[-1])
        return
    candidate, row = window(pair, mats, configs, 100, 1.0, "balance_undamped_verification")
    record(row)
    if not row["balance_checks_pass"]:
        save_checkpoint(OUT / "failed_verification.npz", candidate)
        stop("UNDAMPED_VERIFICATION_FAILED_NO_EXCAVATION", row)
        return
    pair = candidate
    accepted = pair
    save_checkpoint(OUT / "balanced_pre_cut_checkpoint.npz", pair)
    # Select from virgin reference geometry; remove the full toe, including its cap.
    remove = refs[0][:, 0] >= 2.2
    keep = ~remove
    soil = pair[0]
    removed_energy = {
        k: float(
            np.sum(np.asarray(soil.mass)[remove] * np.asarray(getattr(soil.history, k))[remove])
        )
        for k in ("work", "energy")
    }
    removed = {
        "cut": PLAN["cut"],
        "count": int(remove.sum()),
        "mass_kg_per_m": float(np.asarray(soil.mass)[remove].sum()),
        "reference_volume_m2": float(np.asarray(soil.volume)[remove].sum()),
        "kinetic_J_per_m": float(
            0.5 * np.sum(np.asarray(soil.mass)[remove, None] * np.asarray(soil.v)[remove] ** 2)
        ),
        "gravitational_potential_J_per_m": float(
            np.sum(np.asarray(soil.mass)[remove] * 9.81 * np.asarray(soil.x)[remove, 1])
        ),
        "history_specific_energy_mass_integrals": removed_energy,
        "removed_momentum_Ns_per_m": np.sum(
            np.asarray(soil.mass)[remove, None] * np.asarray(soil.v)[remove], axis=0
        ).tolist(),
        "history_preservation": "every retained state/history array exactly sliced; time and validity unchanged; no history reset",
        "energy_scope": "removed terms are separate diagnostics; history.energy may not represent recoverable elastic energy; no total energy closure claim",
    }
    np.savez_compressed(
        OUT / "removed_particle_records.npz",
        reference=refs[0][remove],
        **{k: np.asarray(getattr(soil, k))[remove] for k in ("x", "v", "F", "C", "mass", "volume")},
        **{
            "history_" + k: np.asarray(getattr(soil.history, k))[remove]
            for k in soil.history._fields
        },
    )
    atomic_json(OUT / "excavation-ledger.json", removed)
    retained = retain(soil, keep)
    for key in soil.history._fields:
        if not np.array_equal(
            np.asarray(getattr(retained.history, key)), np.asarray(getattr(soil.history, key))[keep]
        ):
            raise RuntimeError("retained history changed")
    pair = (retained, pair[1])
    refs = (refs[0][keep], refs[1])
    accepted = pair
    excavation_performed = True
    save_checkpoint(OUT / "post_cut_initial_checkpoint.npz", pair)
    save_checkpoint(OUT / "checkpoint.npz", pair)
    cut_step = int(round(float(pair[0].time) / DT))
    pee = np.asarray(pair[0].history.plastic_strain).copy()
    frame(cut_step, pair, refs, pee)
    for n in range(args.postcut_steps // 100):
        candidate, row = window(pair, mats, configs, 100, 1.0, "postcut")
        record(row)
        if not row["guards_pass"]:
            save_checkpoint(OUT / "failed_attempt.npz", candidate)
            stop("POSTCUT_GUARD_STOP", row)
            return
        pair = candidate
        accepted = pair
        save_checkpoint(OUT / "checkpoint.npz", pair)
        frame(row["step"], pair, refs, pee)
        pee = np.asarray(pair[0].history.plastic_strain).copy()
    stop("REQUESTED_POSTCUT_STAGE_COMPLETE", rows[-1])


if __name__ == "__main__":
    main()
