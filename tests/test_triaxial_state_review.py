import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import review_triaxial_state as reviewer


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_table(path, rows):
    import csv
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bind_records(folder):
    plan = json.loads((folder / "plan.json").read_text())
    save(folder / "record-audit.json", {"passed": True, "checks": {"synthetic_protocol_fixture": True},
        "source": plan["source"], "auditor_sha256": "b" * 64,
        "input_sha256": {name: reviewer.digest(folder / name) for name in reviewer.RECORDS}})


def fixture(folder, *, history=True):
    """Analytical uniaxial states, not an MPM simulation or a numerical validation run."""
    folder.mkdir(parents=True, exist_ok=True)
    young, nu, rho, dx = 10e9, .25, 1500., 25e-6
    material = dict(young=young, poisson=nu, density=rho, tensile_strength=100e6,
                    fracture_energy=100., characteristic_length=dx)
    eps = np.array([0., .005, .02, .09])
    damage = np.array([0., 0., .08 * (.02 - .01) / (.02 * (.08 - .01)), 1.])
    F = np.zeros((4, 3, 3))
    F[:, 0, 0] = 1 + eps
    F[:, 1, 1] = F[:, 2, 2] = 1 - nu * eps
    tau = np.zeros_like(F)
    tau[:, 0, 0] = (1 - damage) * young * eps * (1 + eps)
    volume = np.array([1., 2., 1.5, 1.2]) * 1e-12
    mass = rho * volume
    x0 = np.array([[.001, .002, .003], [.002, .002, .003], [.003, .002, .003], [.004, .002, .003]])
    v = np.array([[.1, 0., 0.], [-.1, .2, 0.], [0., 0., .3], [.2, .1, 0.]])
    C = np.broadcast_to(np.diag([.3, -.2, .1]), F.shape).copy()
    dt = 1e-8
    x = x0 + 2 * dt * v
    momentum = np.sum(mass[:, None] * v, axis=0)
    state = dict(x=x, v=v, F=F, C=C, tau=tau, mass=mass, volume=volume, fibre_id=np.array([0, 0, 1, 1], np.int32),
        history_kappa=eps, history_damage=damage, history_failed=damage == 1, history_valid=np.ones(4, bool),
        time=np.array(2 * dt), valid=np.array(True), boundary_impulse=momentum, external_impulse=np.zeros(3),
        boundary_work=np.array(.001))
    np.savez_compressed(folder / "final_state_0.npz", **state)
    np.savez_compressed(folder / "particle_fields.npz", yarn_id=state["fibre_id"], family_id=np.array([0, 0, 1, 1]))
    source_files = {"fixture.py": "a" * 64}
    source = {"files": source_files, "fingerprint": hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()}
    plan = {"schema": "triaxial-fibre-extension-v1", "material": material, "grid": {"dx_m": dx}, "particles": 4,
        "dt": dt, "frame_steps": [0, 1, 2], "probe_steps": 0, "wave_speed_m_per_s": 2828.42712474619,
        "source": source, "geometry": {"fibre_count": 2, "fibres": [{"family_id": 0, "particle_count": 2},
            {"family_id": 1, "particle_count": 2}], "mass_kg": float(mass.sum()), "reference_volume_m3": float(volume.sum())},
        "sampling_hash": {"shape": list(x0.shape), "dtype": str(x0.dtype), "sha256": hashlib.sha256(x0.tobytes()).hexdigest()}}
    summary = {"completed": True, "particles": [4], "definition": plan, "source": source, "simulation_time": 2 * dt}
    save(folder / "plan.json", plan)
    save(folder / "summary.json", summary)
    save(folder / "status.json", {"completed": True, "signature": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()})
    damage_trace = np.stack([np.zeros(4), damage / 2, damage])
    failed_trace = damage_trace == 1
    x_trace = np.stack([x0, (x0 + x) / 2, x])
    np.savez_compressed(folder / "trajectory.npz", time=np.array([0., dt, 2 * dt]), body=np.zeros(4, np.int32), failed=failed_trace)
    if history:
        for name, array in (("x", x_trace), ("damage", damage_trace), ("failed", failed_trace)):
            np.save(folder / f"trajectory_{name}.npy", array)
    elastic = float(np.sum(volume * (1 - damage) * .5 * young * eps**2))
    translation = float(.5 * np.sum(mass * np.sum(v**2, axis=1)))
    affine = float(dx**2 / 8 * np.sum(mass * np.sum(C**2, axis=(1, 2))))
    J = (1 + eps) * (1 - nu * eps)**2
    obs, diag = [], []
    for frame in range(3):
        row = {"time": frame * dt, "max_damage": float(damage_trace[frame].max()),
            "failed_particles": int(failed_trace[frame].sum()), "elastic_energy_J": elastic if frame == 2 else 0.,
            "kinetic_energy_J": translation + affine if frame == 2 else 0.,
            "kinetic_translational_energy_J": translation if frame == 2 else 0.,
            "kinetic_apic_affine_energy_J": affine if frame == 2 else 0.,
            "grid_kinetic_correction_J": .001 if frame == 2 else 0.}
        for side in ("left", "right"):
            for axis, coordinate in enumerate("xyz"):
                row[f"{side}_accumulated_impulse_{coordinate}_Ns"] = float(momentum[axis]) if frame == 2 and side == "right" else 0.
        obs.append(row)
        diag.append({"time": frame * dt, "mass": float(mass.sum()), "reference_volume": float(volume.sum()),
            "current_volume": float(np.sum(volume * J)) if frame == 2 else float(volume.sum()),
            "min_J": float(J.min()) if frame == 2 else 1., "max_J": float(J.max()) if frame == 2 else 1.,
            "failed_mass": float(mass[failed_trace[frame]].sum()), "max_damage": float(damage_trace[frame].max()),
            "mass_relative_drift": 0., "momentum_balance_residual_Ns": 0.})
    write_table(folder / "observables.csv", obs)
    write_table(folder / "diagnostics.csv", diag)
    bind_records(folder)
    return state, material


def mutate_state(folder, change):
    path = folder / "final_state_0.npz"
    with np.load(path) as data:
        arrays = {key: data[key].copy() for key in data.files}
    change(arrays)
    np.savez_compressed(path, **arrays)


def test_uniaxial_analytic_state_full_history_and_chunk_partition(tmp_path):
    _, material = fixture(tmp_path)
    small = reviewer.review(tmp_path, chunk_size=1)
    larger = reviewer.review(tmp_path, chunk_size=3)
    assert small["passed"] is True and small["full_history_checked"] is True
    assert small["production_eligible"] is False
    assert all(small["checks"].values())
    for key in ("elastic_energy_J", "kinetic_energy_J", "mass", "failed_mass"):
        np.testing.assert_allclose(small["metrics"][key], larger["metrics"][key], rtol=1e-13)
    assert small["metrics"]["failed_particles"] == 1
    assert small["review_source_sha256"] == reviewer.digest(Path(reviewer.__file__))
    assert set(small["input_sha256"]) >= {"record-audit.json", "final_state_0.npz", "trajectory_x.npy"}
    assert "F/C/stress histories are not available" in small["history_scope"]["description"]


def test_svd_reconstruction_obeys_rigid_rotation_and_mirror(tmp_path):
    state, material = fixture(tmp_path)
    P, tau, energy, equivalent = reviewer.biot_response(state["F"], state["history_damage"], material)
    np.testing.assert_allclose(tau, state["tau"], atol=1e-5)
    angle = .71
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = reviewer.biot_response(Q @ state["F"], state["history_damage"], material)
    np.testing.assert_allclose(rotated[0], Q @ P, rtol=1e-12, atol=1e-5)
    np.testing.assert_allclose(rotated[1], Q @ tau @ Q.T, rtol=1e-12, atol=1e-5)
    np.testing.assert_allclose(rotated[2], energy, rtol=1e-12, atol=1e-20)
    mirror = np.diag([-1., 1., 1.])
    mirrored = reviewer.biot_response(mirror @ state["F"] @ mirror, state["history_damage"], material)
    np.testing.assert_allclose(mirrored[1], mirror @ tau @ mirror)
    np.testing.assert_allclose(mirrored[3], equivalent)


@pytest.mark.parametrize("problem,check", [
    ("tau", "stored_Kirchhoff_stress"), ("kappa", "kappa_bounds_current_driving_strain"),
    ("damage", "softening_damage_matches_kappa"), ("failed", "failed_matches_damage"),
    ("mass", "mass_density_relation"), ("affine", "final_observation_kinetic_energy_J"),
    ("momentum", "independent_momentum_balance"), ("negative-J", "positive_J"),
    ("invalid-history", "particle_history_valid"), ("nan-F", "finite_particle_state"),
])
def test_corrupted_final_state_is_rejected(tmp_path, problem, check):
    fixture(tmp_path)
    def change(state):
        if problem == "tau": state["tau"][1, 0, 0] += 1e6
        elif problem == "kappa": state["history_kappa"][2] = .019
        elif problem == "damage": state["history_damage"][2] = .2
        elif problem == "failed": state["history_failed"][2] = True
        elif problem == "mass": state["mass"][2] *= 1.1
        elif problem == "affine": state["C"] *= 1e4
        elif problem == "momentum": state["external_impulse"][0] = 1e-5
        elif problem == "negative-J": state["F"][1, 0, 0] = -1.
        elif problem == "invalid-history": state["history_valid"][2] = False
        elif problem == "nan-F": state["F"][:] = np.nan
    mutate_state(tmp_path, change)
    result = reviewer.review(tmp_path, chunk_size=2)
    assert result["passed"] is False and result["checks"][check] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("problem,check", [
    ("healing", "history_irreversible_damage"), ("failed", "history_failed_matches_damage"),
    ("initial-damage", "history_initial_undamaged"), ("final-x", "history_final_state_matches"),
    ("initial-x", "history_initial_sampling_identity"),
])
def test_saved_history_is_checked_per_particle(tmp_path, problem, check):
    fixture(tmp_path)
    field = "x" if problem.endswith("-x") else "failed" if problem == "failed" else "damage"
    path = tmp_path / f"trajectory_{field}.npy"
    values = np.load(path)
    if problem == "healing": values[1, 2] = .9
    elif problem == "failed": values[1, 2] = True
    elif problem == "initial-damage": values[0, 1] = .001
    elif problem == "final-x": values[-1, 1, 0] += 1e-6
    elif problem == "initial-x": values[0, 1, 0] += 1e-6
    np.save(path, values)
    result = reviewer.review(tmp_path, chunk_size=1)
    assert result["passed"] is False and result["full_history_checked"] is False
    assert result["checks"][check] is False


def test_missing_history_is_explicit_without_claiming_reconstruction(tmp_path):
    fixture(tmp_path, history=False)
    result = reviewer.review(tmp_path)
    assert result["passed"] is True and result["full_history_checked"] is False
    assert result["history_scope"]["omitted_mmap_fields"] == ["x", "damage", "failed"]
    assert result["history_scope"]["saved_states_checked"] == 0


def test_stale_record_audit_cannot_bind_changed_inputs(tmp_path):
    fixture(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["extra"] = "changed after record audit"
    save(tmp_path / "summary.json", summary)
    result = reviewer.review(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["record_audit_input_identity"] is False


def test_reviewer_never_reads_payload_with_path_read_bytes(tmp_path, monkeypatch):
    fixture(tmp_path)
    def forbid(*args, **kwargs):
        raise AssertionError("Large-file hashes must stream")
    monkeypatch.setattr(Path, "read_bytes", forbid)
    assert reviewer.review(tmp_path)["passed"] is True
