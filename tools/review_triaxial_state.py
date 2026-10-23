"""Independently reconstruct a saved fibre state with bounded-memory NumPy checks."""

import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


RECORDS = ("plan.json", "summary.json", "status.json", "observables.csv", "diagnostics.csv")
PARTICLE_SHAPES = {
    "x": (3,), "v": (3,), "F": (3, 3), "C": (3, 3), "tau": (3, 3),
    "mass": (), "volume": (), "fibre_id": (), "history_kappa": (),
    "history_damage": (), "history_failed": (), "history_valid": (),
}
STATE_SHAPES = {"time": (), "valid": (), "boundary_impulse": (3,),
                "external_impulse": (3,), "boundary_work": ()}
HISTORY_FIELDS = ("x", "damage", "failed")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty table: {path}")
    return [{key: np.nan if value == "" else 1. if value == "True" else
             0. if value == "False" else float(value) for key, value in row.items()} for row in rows]


@contextmanager
def mapped_npz(path, names):
    """Stream only named NPY members to temporary files and map them read-only."""
    arrays = {}
    with tempfile.TemporaryDirectory(prefix="triaxial-state-") as directory:
        try:
            with zipfile.ZipFile(path) as archive:
                for name in names:
                    member = name + ".npy"
                    if archive.namelist().count(member) != 1:
                        raise ValueError(f"Expected exactly one {member} in {path}")
                    target = Path(directory) / member
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024**2)
                    arrays[name] = np.load(target, mmap_mode="r", allow_pickle=False)
            yield arrays
        finally:
            for value in arrays.values():
                value._mmap.close()


def biot_response(F, damage, material):
    """SVD polar reconstruction, independent of the solver's eigensystem implementation."""
    left, stretch, right = np.linalg.svd(F, full_matrices=False)
    strain = stretch - 1.
    young, poisson = material["young"], material["poisson"]
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    trace = strain.sum(axis=-1)
    effective_principal = lam * trace[:, None] + 2 * mu * strain
    P = ((left * effective_principal[:, None, :]) @ right) * (1 - damage[:, None, None])
    tau = P @ np.swapaxes(F, -1, -2)
    psi0 = .5 * lam * trace**2 + mu * np.sum(strain**2, axis=-1)
    equivalent = np.maximum(effective_principal.max(axis=-1), 0.) / young
    return P, tau, (1 - damage) * psi0, equivalent


def softening_damage(kappa, material):
    delta = material["characteristic_length"] * np.maximum(kappa, 0.)
    onset = material["characteristic_length"] * material["tensile_strength"] / material["young"]
    final = 2 * material["fracture_energy"] / material["tensile_strength"]
    return np.clip(final * (delta - onset) / (np.maximum(delta, onset) * (final - onset)), 0., 1.)


def review(folder, *, chunk_size=16384):
    folder = Path(folder)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    plan, summary, status = [json.loads((folder / name).read_text(encoding="utf-8")) for name in RECORDS[:3]]
    observations, diagnostics = [read_table(folder / name) for name in RECORDS[3:]]
    record_audit = json.loads((folder / "record-audit.json").read_text(encoding="utf-8"))
    names = (*RECORDS, "record-audit.json", "final_state_0.npz", "particle_fields.npz", "trajectory.npz")
    input_hashes = {name: digest(folder / name) for name in names}
    present = {name: (folder / f"trajectory_{name}.npy").is_file() for name in HISTORY_FIELDS}
    for name, exists in present.items():
        if exists:
            filename = f"trajectory_{name}.npy"
            input_hashes[filename] = digest(folder / filename)
    source = plan["source"]
    source_fingerprint = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    checks = {
        "source_identity": source == summary["source"] == record_audit["source"] and source["fingerprint"] == source_fingerprint,
        "record_audit_passed": record_audit.get("passed") is True and bool(record_audit.get("checks")) and
            all(value is True for value in record_audit["checks"].values()),
        "record_audit_input_identity": record_audit.get("input_sha256") == {name: input_hashes[name] for name in RECORDS},
        "status_signature": status.get("signature") == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "definition_unchanged": all(summary["definition"].get(key) == value for key, value in plan.items()),
        "completed_scope": (summary.get("completed") is True and status.get("completed") is True) or
            (bool(plan.get("probe_steps")) and summary.get("probe_complete") is True and status.get("probe_complete") is True),
    }
    material = plan["material"]
    young, poisson, rho, strength, energy, length = [material[name] for name in
        ("young", "poisson", "density", "tensile_strength", "fracture_energy", "characteristic_length")]
    valid_material = all(np.ndim(v) == 0 and np.isfinite(v) for v in (young, poisson, rho, strength, energy, length))
    valid_material = valid_material and min(young, rho, strength, energy, length) > 0 and -1 < poisson < .5
    valid_material = valid_material and length < 2 * young * energy / strength**2
    if not valid_material:
        raise ValueError("Invalid or unresolved crack-band material")
    dx = plan["grid"]["dx_m"]
    checks["band_length_matches_grid"] = bool(np.isclose(length, dx, rtol=1e-12, atol=0))
    count = plan["particles"]
    checks["particle_count"] = summary["particles"] == [count]
    checks["matching_saved_records"] = len(observations) == len(diagnostics) == len(plan["frame_steps"])
    if not checks["matching_saved_records"]:
        raise ValueError("Saved tables and frame plan differ")
    final_obs, final_diag = observations[-1], diagnostics[-1]
    time_tol = plan["dt"] * 1e-5
    time_values = np.asarray([row["time"] for row in observations])
    checks["saved_time_identity"] = bool(np.allclose(time_values, [row["time"] for row in diagnostics], rtol=0, atol=time_tol) and
        np.allclose(time_values, np.asarray(plan["frame_steps"]) * plan["dt"], rtol=0, atol=time_tol))
    checks["initial_zero_motion"] = observations[0]["kinetic_energy_J"] == 0 and all(
        observations[0][f"{side}_accumulated_impulse_{axis}_Ns"] == 0 for side in ("left", "right") for axis in "xyz")
    for key in ("finite_particle_state", "positive_particle_mass_volume", "mass_density_relation", "positive_J",
                "particle_history_valid", "kappa_bounds_current_driving_strain", "softening_damage_matches_kappa",
                "failed_matches_damage", "stored_Kirchhoff_stress", "stored_Piola_stress", "particle_identity",
                "material_reconstruction_complete"):
        checks[key] = True
    totals = {name: 0. for name in ("mass", "reference_volume", "current_volume", "failed_mass", "elastic_energy_J",
              "kinetic_translational_energy_J", "kinetic_apic_affine_energy_J")}
    totals.update(failed_particles=0, max_damage=0., min_J=float("inf"), max_J=-float("inf"),
                  maximum_tau_error_Pa=0., maximum_P_error_Pa=0., maximum_damage_error=0.,
                  maximum_current_driving_strain=0., maximum_kappa=0.)
    momentum = np.zeros(3)
    fibre_counts = np.zeros(plan["geometry"]["fibre_count"], dtype=np.int64)
    history_checks = {}
    with ExitStack() as stack:
        state = stack.enter_context(mapped_npz(folder / "final_state_0.npz", (*PARTICLE_SHAPES, *STATE_SHAPES)))
        fields = stack.enter_context(mapped_npz(folder / "particle_fields.npz", ("yarn_id", "family_id")))
        index = stack.enter_context(mapped_npz(folder / "trajectory.npz", ("time", "body", "failed")))
        for name, shape in {**{key: (count, *tail) for key, tail in PARTICLE_SHAPES.items()}, **STATE_SHAPES}.items():
            if state[name].shape != shape:
                raise ValueError(f"Unexpected {name} shape: {state[name].shape}")
        if any(fields[name].shape != (count,) for name in fields) or index["body"].shape != (count,):
            raise ValueError("Static particle field shape mismatch")
        if index["time"].shape != time_values.shape or index["failed"].shape != (len(time_values), count):
            raise ValueError("Trajectory index shape mismatch")
        checks["index_times_and_body"] = bool(np.allclose(index["time"], time_values, rtol=0, atol=time_tol) and np.all(index["body"] == 0))
        checks["state_valid"] = bool(np.all(np.isfinite([float(state[name]) for name in ("time", "valid", "boundary_work")])) and state["valid"] == 1)
        checks["state_time"] = bool(abs(float(state["time"]) - summary["simulation_time"]) <= time_tol and
                                    abs(float(state["time"]) - time_values[-1]) <= time_tol)
        checks["finite_impulses"] = bool(np.isfinite(state["boundary_impulse"]).all() and np.isfinite(state["external_impulse"]).all())
        expected_family = np.asarray([fibre["family_id"] for fibre in plan["geometry"]["fibres"]])
        for start in range(0, count, chunk_size):
            part = slice(start, min(start + chunk_size, count))
            values = {name: np.asarray(array[part]) for name, array in state.items() if name in PARTICLE_SHAPES}
            finite = all(np.isfinite(value).all() for value in values.values())
            checks["finite_particle_state"] &= bool(finite)
            if not finite:
                checks["material_reconstruction_complete"] = False
                continue
            F, D, kappa = values["F"], values["history_damage"], values["history_kappa"]
            mass, volume, failed, ids = values["mass"], values["volume"], values["history_failed"], values["fibre_id"]
            checks["positive_particle_mass_volume"] &= bool(np.all(mass > 0) and np.all(volume > 0))
            checks["mass_density_relation"] &= bool(np.allclose(mass, rho * volume, rtol=1e-10, atol=0))
            valid_ids = ids.dtype.kind in "iu" and bool(np.all((ids >= 0) & (ids < len(fibre_counts))))
            checks["particle_identity"] &= bool(valid_ids and np.array_equal(ids, fields["yarn_id"][part]) and
                np.array_equal(fields["family_id"][part], expected_family[ids] if valid_ids else np.zeros(len(ids))))
            if valid_ids:
                fibre_counts += np.bincount(ids, minlength=len(fibre_counts))
            J = np.linalg.det(F)
            checks["positive_J"] &= bool(np.all(J > 0))
            checks["material_reconstruction_complete"] &= bool(np.all(J > 0))
            checks["particle_history_valid"] &= bool(np.all(values["history_valid"] == 1) and
                np.all(kappa >= 0) and np.all((D >= 0) & (D <= 1)))
            expected_D = softening_damage(kappa, material)
            checks["softening_damage_matches_kappa"] &= bool(np.allclose(D, expected_D, rtol=1e-9, atol=1e-10))
            checks["failed_matches_damage"] &= bool(np.all((failed == 0) | (failed == 1)) and np.array_equal(failed.astype(bool), D >= 1))
            totals["maximum_damage_error"] = max(totals["maximum_damage_error"], float(np.max(np.abs(D - expected_D))))
            if np.all(J > 0):
                P, tau, elastic, equivalent = biot_response(F, D, material)
                stored_P = np.swapaxes(np.linalg.solve(F, np.swapaxes(values["tau"], -1, -2)), -1, -2)
                checks["kappa_bounds_current_driving_strain"] &= bool(np.all(kappa + 2e-10 * strength / young >= equivalent))
                checks["stored_Kirchhoff_stress"] &= bool(np.allclose(values["tau"], tau, rtol=2e-8, atol=2e-11 * young))
                checks["stored_Piola_stress"] &= bool(np.allclose(stored_P, P, rtol=2e-8, atol=2e-11 * young))
                totals["maximum_tau_error_Pa"] = max(totals["maximum_tau_error_Pa"], float(np.max(np.abs(values["tau"] - tau))))
                totals["maximum_P_error_Pa"] = max(totals["maximum_P_error_Pa"], float(np.max(np.abs(stored_P - P))))
                totals["maximum_current_driving_strain"] = max(totals["maximum_current_driving_strain"], float(equivalent.max()))
                totals["elastic_energy_J"] += float(np.sum(volume * elastic))
            totals["mass"] += float(mass.sum())
            totals["reference_volume"] += float(volume.sum())
            totals["current_volume"] += float(np.sum(volume * J))
            totals["failed_mass"] += float(mass[failed.astype(bool)].sum())
            totals["failed_particles"] += int(np.count_nonzero(failed))
            totals["max_damage"] = max(totals["max_damage"], float(D.max()))
            totals["maximum_kappa"] = max(totals["maximum_kappa"], float(kappa.max()))
            totals["min_J"] = min(totals["min_J"], float(J.min()))
            totals["max_J"] = max(totals["max_J"], float(J.max()))
            totals["kinetic_translational_energy_J"] += float(.5 * np.sum(mass * np.sum(values["v"]**2, axis=1)))
            totals["kinetic_apic_affine_energy_J"] += float(dx**2 / 8 * np.sum(mass * np.sum(values["C"]**2, axis=(1, 2))))
            momentum += np.sum(mass[:, None] * values["v"], axis=0)
        totals["kinetic_energy_J"] = totals["kinetic_translational_energy_J"] + totals["kinetic_apic_affine_energy_J"]
        checks["fibre_particle_counts"] = np.array_equal(fibre_counts, [fibre["particle_count"] for fibre in plan["geometry"]["fibres"]])
        checks["reference_geometry_totals"] = bool(np.isclose(totals["mass"], plan["geometry"]["mass_kg"], rtol=1e-10, atol=0) and
            np.isclose(totals["reference_volume"], plan["geometry"]["reference_volume_m3"], rtol=1e-10, atol=0))
        for name in ("mass", "reference_volume", "current_volume", "min_J", "max_J", "failed_mass", "max_damage"):
            checks["final_diagnostic_" + name] = bool(np.isclose(totals[name], final_diag[name], rtol=2e-9, atol=0))
        for name in ("elastic_energy_J", "kinetic_energy_J", "kinetic_translational_energy_J", "kinetic_apic_affine_energy_J", "max_damage", "failed_particles"):
            checks["final_observation_" + name] = bool(np.isclose(totals[name], final_obs[name], rtol=2e-9, atol=1e-18))
        checks["mass_conservation"] = bool(np.isclose(totals["mass"], diagnostics[0]["mass"], rtol=1e-10, atol=0) and abs(final_diag["mass_relative_drift"]) <= 1e-10)
        boundary = np.asarray(state["boundary_impulse"])
        residual = float(np.linalg.norm(momentum - boundary - state["external_impulse"]))
        tolerance = 1e-8 * totals["mass"] * plan["wave_speed_m_per_s"]
        checks["independent_momentum_balance"] = bool(np.isfinite(residual) and residual <= tolerance and
            abs(residual - final_diag["momentum_balance_residual_Ns"]) <= tolerance)
        recorded_impulse = np.array([sum(final_obs[f"{side}_accumulated_impulse_{axis}_Ns"] for side in ("left", "right")) for axis in "xyz"])
        checks["boundary_impulse_matches_records"] = bool(np.allclose(boundary, recorded_impulse, rtol=1e-8, atol=tolerance))
        checks["grid_kinetic_correction_matches_state"] = bool(np.isclose(float(state["boundary_work"]), final_obs["grid_kinetic_correction_J"], rtol=1e-8, atol=1e-18))
        totals.update(momentum_Ns=momentum.tolist(), momentum_residual_Ns=residual, momentum_tolerance_Ns=tolerance)
        history = {}
        if all(present.values()):
            for name in HISTORY_FIELDS:
                history[name] = np.load(folder / f"trajectory_{name}.npy", mmap_mode="r", allow_pickle=False)
                stack.callback(history[name]._mmap.close)
            expected_shape = (len(time_values), count)
            if any(history[name].shape != expected_shape + ((3,) if name == "x" else ()) for name in HISTORY_FIELDS):
                raise ValueError("Complete history files have inconsistent shapes")
            history_checks = {name: True for name in ("finite_positions_damage", "bounded_damage", "irreversible_damage",
                "failed_matches_damage", "initial_undamaged", "failure_archive_matches", "final_state_matches", "saved_damage_statistics")}
            initial_hash = hashlib.sha256()
            for frame in range(len(time_values)):
                maximum, failed_count, failed_mass = 0., 0, 0.
                for start in range(0, count, chunk_size):
                    part = slice(start, min(start + chunk_size, count))
                    x, D, failed = [np.asarray(history[name][frame, part]) for name in HISTORY_FIELDS]
                    history_checks["finite_positions_damage"] &= bool(np.isfinite(x).all() and np.isfinite(D).all())
                    history_checks["bounded_damage"] &= bool(np.all((D >= 0) & (D <= 1)))
                    history_checks["failed_matches_damage"] &= bool(np.all((failed == 0) | (failed == 1)) and np.array_equal(failed.astype(bool), D >= 1))
                    history_checks["failure_archive_matches"] &= np.array_equal(failed, index["failed"][frame, part])
                    if frame:
                        history_checks["irreversible_damage"] &= bool(np.all(D >= history["damage"][frame-1, part] - 1e-12))
                    else:
                        initial_hash.update(np.ascontiguousarray(x).tobytes())
                        history_checks["initial_undamaged"] &= bool(np.all(D == 0) and np.all(failed == 0))
                    if frame == len(time_values)-1:
                        history_checks["final_state_matches"] &= bool(np.array_equal(x, state["x"][part]) and
                            np.array_equal(D, state["history_damage"][part]) and np.array_equal(failed, state["history_failed"][part]))
                    maximum = max(maximum, float(np.max(D)))
                    failed_count += int(np.count_nonzero(failed))
                    failed_mass += float(np.sum(state["mass"][part][failed.astype(bool)]))
                history_checks["saved_damage_statistics"] &= bool(np.isclose(maximum, observations[frame]["max_damage"], rtol=0, atol=1e-12) and
                    failed_count == observations[frame]["failed_particles"] and
                    np.isclose(failed_mass, diagnostics[frame]["failed_mass"], rtol=1e-10, atol=0))
            descriptor = plan["sampling_hash"]
            history_checks["initial_sampling_identity"] = bool(initial_hash.hexdigest() == descriptor["sha256"] and
                descriptor["shape"] == [count, 3] and descriptor["dtype"] == str(history["x"].dtype))
            checks.update({"history_" + key: bool(value) for key, value in history_checks.items()})
        else:
            checks["available_failure_archive_final_matches"] = bool(np.array_equal(index["failed"][-1], state["history_failed"]))
    full_history = bool(all(present.values()) and history_checks and all(history_checks.values()))
    totals = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in totals.items()}
    return {
        "schema": "triaxial-independent-state-review-v1", "passed": all(bool(value) for value in checks.values()),
        "production_eligible": False, "full_history_checked": full_history,
        "checks": {key: bool(value) for key, value in checks.items()}, "metrics": totals,
        "input_sha256": input_hashes, "producer_source": source, "review_source_sha256": digest(Path(__file__)),
        "record_audit_source_sha256": record_audit.get("auditor_sha256"),
        "history_scope": {"present_mmap_fields": [name for name, exists in present.items() if exists],
            "omitted_mmap_fields": [name for name, exists in present.items() if not exists],
            "saved_states_checked": len(time_values) if all(present.values()) else 0,
            "description": "All saved x, damage and failed frames are checked when complete mmap files exist; per-step or saved F/C/stress histories are not available and are not reconstructed."},
        "scope": "Independent final-state SVD reconstruction of Biot stress, crack-band damage from kappa, stored energy, mass, APIC kinetic energy and momentum; kappa is checked against the current driving strain, not reconstructed over the missing deformation history.",
        "limitations": "No fracture-energy balance, contact-force/coverage reintegration, macroscopic rupture, quasi-static strength or angle-envelope validation. Dynamic curvature work and contact snap dissipation are not reconstructed. Hertz/JKR differences at illustrative W=0.1 J/m2 may be very small.",
        "chunk_size": chunk_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = review(args.folder)
    output = args.output or args.folder / "state-review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "full_history_checked": report["full_history_checked"], "report": output.as_posix(),
                      "failed_checks": [key for key, value in report["checks"].items() if not value]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
