import csv
import hashlib
import json

import pytest

from tools.audit_triaxial_records import audit


@pytest.fixture
def records(tmp_path):
    files = {"material.py": "a"*64}
    source = {"files": files, "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    plan = {"scope": "smoke", "theta_deg": 0, "law": "hertz", "dt": .25,
            "load_time_s": 1., "max_engineering_strain": .02, "stop_steps": 4,
            "probe_steps": 0, "frame_steps": [0, 2, 4], "wave_speed_m_per_s": 100.,
            "grid": {"dx_m": .001}, "source": source,
            "geometry": {"crimp_amplitude_m": .0001, "diameter_m": .0002,
                         "dimensions_m": [.02, .01]}}
    summary = {"steps": 4, "completed": True, "probe_complete": False,
               "definition": plan, "source": source, "peak_dynamic_grip_force_N": 2.5,
               "peak_dynamic_grip_force_time_s": .75, "first_material_failure_time_s": .75}
    status = {"completed": True, "steps": 4, "signature": hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest()}
    for name, value in (("plan", plan), ("summary", summary), ("status", status)):
        (tmp_path/f"{name}.json").write_text(json.dumps(value))
    rows, diagnostics = [], []
    for i in range(3):
        t = .5*i
        row = {"time": t, "accepted_steps": 2*i,
               "prescribed_engineering_strain": [0, .01, .02][i],
               "mean_grip_tensile_force_N": i, "peak_step_dynamic_grip_force_N": [0, 1.5, 2.5][i],
               "first_material_failure_time_s": .75 if i == 2 else "",
               "failed_particles": int(i == 2), "max_damage": i*.5,
               "kinetic_energy_J": .1*i, "kinetic_translational_energy_J": .08*i,
               "kinetic_apic_affine_energy_J": .02*i, "elastic_energy_J": .2*i,
               "kinetic_to_elastic_ratio": .5 if i else "",
               "grip_position_error_m": 0., "grip_velocity_error_m_per_s": 0.,
               "active_fibre_x_support_cuts": int(i == 2)}
        for side, sign in (("left", -1), ("right", 1)):
            for axis in "xyz":
                row[f"{side}_applied_force_{axis}_N"] = sign*i if axis == "x" else 0.
                row[f"{side}_accumulated_impulse_{axis}_Ns"] = sign*[0, .5, 1.5][i] if axis == "x" else 0.
        rows.append(row)
        diagnostics.append({"time": t, "mass": .001, "mass_relative_drift": 0.,
                            "state_valid": True, "coupling_valid": True, "failure_reason_code": 0,
                            "min_J": 1., "max_J": 1., "failed_mass": .00001 if i == 2 else 0.,
                            "momentum_balance_residual_Ns": 0.})
    for name, values in (("observables", rows), ("diagnostics", diagnostics)):
        with (tmp_path/f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=values[0])
            writer.writeheader()
            writer.writerows(values)
    return tmp_path


def change_csv(folder, file, field, value):
    path = folder/file
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_dynamic_observations_never_become_an_accepted_material_envelope(records):
    result = audit(records)
    assert result["passed"]
    assert result["production_eligible"] is False
    assert result["nominal_reference_area_m2"] == pytest.approx(4e-6)
    assert result["dynamic_protocol_max_nominal_stress_Pa"] == pytest.approx(625000.)
    assert result["first_sampled_axial_cut_bracket_s"] == [.5, 1.]
    assert result["sampled_interval_force_max_kinetic_to_elastic_ratio"] == .5
    assert not result["maximum_force_at_final_step"]


@pytest.mark.parametrize("file,field,value,check", [
    ("observables.csv", "right_accumulated_impulse_x_Ns", 2., "impulse_reactions"),
    ("observables.csv", "mean_grip_tensile_force_N", 3., "reaction_definition"),
    ("observables.csv", "kinetic_to_elastic_ratio", "", "energy_ratio"),
    ("observables.csv", "kinetic_apic_affine_energy_J", 0., "kinetic_decomposition"),
    ("diagnostics.csv", "mass", .0009, "mass_conservation"),
    ("diagnostics.csv", "min_J", -.1, "positive_jacobian"),
    ("diagnostics.csv", "momentum_balance_residual_Ns", .01, "momentum_balance"),
    ("diagnostics.csv", "coupling_valid", False, "valid_states"),
])
def test_corrupted_physical_records_are_rejected(records, file, field, value, check):
    change_csv(records, file, field, value)
    result = audit(records)
    assert not result["passed"] and not result["checks"][check]


def test_stale_plan_identity_and_impossible_failure_time_rejected(records):
    path = records/"summary.json"
    summary = json.loads(path.read_text())
    summary["source"]["fingerprint"] = "b"*64
    summary["first_material_failure_time_s"] = .25
    path.write_text(json.dumps(summary))
    result = audit(records)
    assert not result["checks"]["source_identity"]
    assert not result["checks"]["first_failure_bracket"]


def test_missing_final_states_cannot_be_called_completed(records):
    change_csv(records, "observables.csv", "accepted_steps", 3)
    result = audit(records)
    assert not result["checks"]["complete_requested_scope"]
    assert not result["checks"]["planned_saved_states"]
