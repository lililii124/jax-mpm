"""Audit saved fibre-run records and extract explicitly dynamic tensile measures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


INPUTS = ("plan.json", "summary.json", "status.json", "observables.csv", "diagnostics.csv")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty record: {path}")
    return {key: np.array([np.nan if row[key] == "" else
            1. if row[key] == "True" else 0. if row[key] == "False" else float(row[key])
            for row in rows]) for key in rows[0]}


def strain_at(t, plan):
    phase = np.clip(np.asarray(t)/plan["load_time_s"], 0., 1.)
    return .5*plan["max_engineering_strain"]*(1-np.cos(np.pi*phase))


def audit(folder):
    folder = Path(folder)
    plan, summary, status = [json.loads((folder/name).read_text(encoding="utf-8"))
                             for name in INPUTS[:3]]
    obs, diag = [table(folder/name) for name in INPUTS[3:]]
    t, steps = obs["time"], obs["accepted_steps"]
    if len(t) < 2 or len(t) != len(diag["time"]):
        raise ValueError("Two or more matching saved observation/diagnostic states are required")
    finite_optional = {"first_material_failure_time_s", "kinetic_to_elastic_ratio"}
    checks = {"finite_records": all(np.isfinite(v).all() for data in (obs, diag)
                for k, v in data.items() if k not in finite_optional)}
    checks["monotone_time_and_steps"] = bool(t[0] == steps[0] == 0 and
        np.all(np.diff(t) > 0) and np.all(np.diff(steps) > 0) and np.all(steps == np.rint(steps)))
    checks["matching_times"] = bool(np.allclose(t, diag["time"], rtol=1e-10, atol=1e-15) and
        np.allclose(t, steps*plan["dt"], rtol=1e-9, atol=1e-15))
    checks["planned_saved_states"] = bool(len(t) == len(plan["frame_steps"]) and
        np.array_equal(steps, plan["frame_steps"]))
    probe = bool(plan["probe_steps"])
    checks["complete_requested_scope"] = bool(steps[-1] == plan["stop_steps"] and
        status["steps"] == summary["steps"] == int(steps[-1]) and
        (summary.get("probe_complete") is True and status.get("probe_complete") is True and
         summary.get("completed") is False if probe else
         summary.get("completed") is True and status.get("completed") is True))
    checks["source_identity"] = bool(plan["source"] == summary["source"] and
        plan["source"]["fingerprint"] == hashlib.sha256(json.dumps(
            plan["source"]["files"], sort_keys=True).encode()).hexdigest())
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    checks["status_signature"] = status.get("signature") == signature
    checks["definition_unchanged"] = all(summary["definition"].get(k) == v for k, v in plan.items())
    checks["waveform"] = bool(np.allclose(obs["prescribed_engineering_strain"], strain_at(t, plan),
                                          rtol=1e-9, atol=1e-12))
    checks["valid_states"] = bool(np.all(diag["state_valid"] == 1) and
        np.all(diag["coupling_valid"] == 1) and np.all(diag["failure_reason_code"] == 0))
    checks["positive_jacobian"] = bool(np.all(diag["min_J"] > 0) and
        np.all(diag["max_J"] >= diag["min_J"]))
    mass = diag["mass"][0]
    checks["mass_conservation"] = bool(mass > 0 and np.allclose(diag["mass"], mass,
        rtol=1e-10, atol=0) and np.max(np.abs(diag["mass_relative_drift"])) <= 1e-10)
    checks["bounded_damage"] = bool(np.all((obs["max_damage"] >= 0) & (obs["max_damage"] <= 1))
        and np.all(np.diff(obs["max_damage"]) >= -1e-12) and
        np.all(np.diff(obs["failed_particles"]) >= 0) and
        np.all((diag["failed_mass"] >= 0) & (diag["failed_mass"] <= mass*(1+1e-12))))
    checks["kinetic_decomposition"] = bool(np.all(obs["kinetic_energy_J"] >= 0) and
        np.all(obs["elastic_energy_J"] >= 0) and np.allclose(obs["kinetic_energy_J"],
        obs["kinetic_translational_energy_J"]+obs["kinetic_apic_affine_energy_J"], rtol=1e-10, atol=1e-18))
    energetic = obs["elastic_energy_J"] > 1e-30
    checks["energy_ratio"] = bool(np.allclose(obs["kinetic_to_elastic_ratio"][energetic],
        obs["kinetic_energy_J"][energetic]/obs["elastic_energy_J"][energetic], rtol=1e-9, atol=1e-12))
    checks["reaction_definition"] = bool(np.allclose(obs["mean_grip_tensile_force_N"],
        .5*(obs["right_applied_force_x_N"]-obs["left_applied_force_x_N"]), rtol=1e-10, atol=1e-12))
    checks["impulse_reactions"] = all(np.allclose(
        np.diff(obs[f"{side}_accumulated_impulse_{axis}_Ns"]),
        obs[f"{side}_applied_force_{axis}_N"][1:]*np.diff(t), rtol=1e-8, atol=1e-16)
        for side in ("left", "right") for axis in "xyz")
    checks["momentum_balance"] = bool(np.max(diag["momentum_balance_residual_Ns"])
                                      <= 1e-8*mass*plan["wave_speed_m_per_s"])
    checks["grip_errors"] = bool(np.max(obs["grip_position_error_m"]) <= .02*plan["grid"]["dx_m"]
        and np.max(obs["grip_velocity_error_m_per_s"]) <= 1e-9*plan["wave_speed_m_per_s"])
    peak = obs["peak_step_dynamic_grip_force_N"]
    peak_t = float(summary["peak_dynamic_grip_force_time_s"])
    recorded_peak_t = obs["peak_step_dynamic_grip_force_time_s"]
    time_tolerance = plan["dt"]*1e-5
    checks["peak_consistency"] = bool(np.all(np.diff(peak) >= -1e-12) and
        np.isclose(peak[-1], summary["peak_dynamic_grip_force_N"], rtol=1e-10, atol=1e-12) and
        peak[-1]+1e-10 >= np.max(obs["mean_grip_tensile_force_N"]) and 0 <= peak_t <= t[-1])
    changed_peak = np.diff(peak) > 0
    checks["peak_event_identity"] = bool(abs(peak_t-recorded_peak_t[-1]) <= time_tolerance and
        recorded_peak_t[0] == 0 and
        np.all(np.abs(recorded_peak_t/plan["dt"]-np.rint(recorded_peak_t/plan["dt"])) <= 1e-5) and
        np.all(recorded_peak_t[1:][changed_peak] > t[:-1][changed_peak]+.5*plan["dt"]) and
        np.all(recorded_peak_t[1:][changed_peak] <= t[1:][changed_peak]+time_tolerance) and
        np.all(np.abs(np.diff(recorded_peak_t)[~changed_peak]) <= time_tolerance))
    first = summary.get("first_material_failure_time_s")
    failed = np.flatnonzero(obs["failed_particles"] > 0)
    checks["first_failure_bracket"] = bool((first is None and not len(failed)) or
        (first is not None and len(failed) and failed[0] > 0 and
         t[failed[0]-1] < first <= t[failed[0]]*(1+1e-10)))
    recorded_first = obs["first_material_failure_time_s"]
    checks["first_failure_event_identity"] = bool(
        np.isnan(recorded_first).all() if first is None else len(failed) > 0 and
        np.isnan(recorded_first[:failed[0]]).all() and
        np.all(np.isfinite(recorded_first[failed[0]:])) and
        np.all(np.abs(recorded_first[failed[0]:]-first) <= time_tolerance) and
        abs(first/plan["dt"]-round(first/plan["dt"])) <= 1e-5)
    geometry = plan["geometry"]
    thickness = 2*geometry["crimp_amplitude_m"]+geometry["diameter_m"]
    area = geometry["dimensions_m"][1]*thickness
    peak_frame = int(np.argmax(obs["mean_grip_tensile_force_N"]))
    first_cut = np.flatnonzero(obs["active_fibre_x_support_cuts"] > 0)
    cut_bracket = [float(t[first_cut[0]-1]), float(t[first_cut[0]])] if len(first_cut) and first_cut[0] else None
    result = {
        "schema": "triaxial-record-audit-v1", "passed": all(checks.values()),
        "production_eligible": False, "checks": checks, "scope": plan["scope"],
        "input_sha256": {name: digest(folder/name) for name in INPUTS},
        "auditor_sha256": digest(Path(__file__)), "source": plan["source"],
        "theta_deg": plan["theta_deg"], "law": plan["law"],
        "area_definition": "Nominal rectangular envelope: specimen width times (two crimp amplitudes plus fibre diameter), including voids; not a sum of fibre cross-sections.",
        "nominal_reference_area_m2": area,
        "dynamic_protocol_max_force_N": float(peak[-1]),
        "dynamic_protocol_max_nominal_stress_Pa": float(peak[-1]/area),
        "dynamic_protocol_max_time_s": peak_t,
        "prescribed_strain_at_dynamic_max": float(strain_at(peak_t, plan)),
        "maximum_force_at_final_step": bool(abs(peak_t-t[-1]) <= plan["dt"]*.5),
        "sampled_interval_force_max_time_s": float(t[peak_frame]),
        "sampled_interval_force_max_kinetic_to_elastic_ratio": None if not np.isfinite(
            obs["kinetic_to_elastic_ratio"][peak_frame]) else float(obs["kinetic_to_elastic_ratio"][peak_frame]),
        "first_material_failure_time_s": first, "first_sampled_axial_cut_bracket_s": cut_bracket,
        "interpretation": "Saved-record consistency audit, not an independent reconstruction of particle states, contact coverage or constitutive integration. Protocol maxima are dynamic and may be censored by the chosen duration. Section graphs and axial cuts are approximate connectivity diagnostics. Size, resolution and loading-rate checks are required for a material failure envelope.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.folder)
    output = args.output or args.folder/"record-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "scope": report["scope"],
                      "failed_checks": [k for k, v in report["checks"].items() if not v]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
