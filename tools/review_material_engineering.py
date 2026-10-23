"""Check completed material-engineering runs without importing the solver."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CASES = {
    "jc-tension": "johnson-cook",
    "jc-necking": "johnson-cook",
    "concrete-bending": "hjc",
    "concrete-bending-large": "hjc",
    "concrete-bending-symmetric": "hjc",
    "ceramic-indentation": "jh2",
    "ceramic-indentation-wide": "jh2",
}

CERAMIC_PARAMETERS = {
    "shear": 20e9, "A": .79, "B": .3, "C": .007, "M": .6, "N": .6,
    "reference_rate": 1., "tensile_strength": 100e6, "maximum_fractured_strength": 1.,
    "hel": 2.6e9, "pressure_hel": 1.6e9, "bulking_fraction": .5,
    "D1": .01, "D2": .5, "K1": 130e9, "K2": 0., "K3": 0.,
    "failure_strain": 0., "density": 2500.,
}

HJC_PARAMETERS = {
    "shear": 14.86e9, "A": .79, "B": 1.6, "C": .007, "N": .61,
    "compressive_strength": 48e6, "tensile_strength": 4e6, "reference_rate": 1.,
    "minimum_fracture_strain": .01, "maximum_strength": 7.,
    "crush_pressure": 16e6, "crush_strain": .001, "lock_pressure": .8e9,
    "lock_strain": .1, "D1": .04, "D2": 1., "K1": 85e9,
    "K2": -171e9, "K3": 208e9, "density": 2440., "initial_damage": 0.,
    "legacy_v3": False,
}


def rows(path):
    def number(value):
        if value in ("", "None"):
            return None
        if value in ("True", "False"):
            return value == "True"
        return float(value)

    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: number(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def extrema(table, key):
    values = np.array([row[key] for row in table])
    low, high = int(values.argmin()), int(values.argmax())
    return {
        "min": float(values[low]), "min_time_s": table[low]["time"],
        "max": float(values[high]), "max_time_s": table[high]["time"],
        "final": float(values[-1]),
    }


def hjc_loading_patch(definition, config):
    dx = config["domain_size"] / config["n_grid"]
    coordinates = np.arange(config["n_grid"] + 1) * dx
    boundary = definition["loading_boundary"]
    selected = np.flatnonzero((coordinates >= boundary["lower"][0]) & (coordinates <= boundary["upper"][0]))
    revision = definition.get("loading_revision", "legacy-exact-bounds-v1")
    report = {"revision": revision, "selected_x_indices": selected.tolist(),
              "selected_x_coordinates_m": coordinates[selected].tolist(),
              "actual_grid_center_x_m": float(coordinates[selected].mean()) if len(selected) else None}
    checks = {"recognized_loading_revision": revision in ("legacy-exact-bounds-v1", "centered-grid-patch-v2")}
    if revision == "centered-grid-patch-v2":
        nominal = np.array([.1875, .2125])
        effective = np.array([np.nextafter(nominal[0], -np.inf), np.nextafter(nominal[1], np.inf)])
        checks.update(
            specified_nominal_loading_interval=bool(np.array_equal(definition.get("loading_nominal_x_bounds_m"), nominal)),
            one_ulp_expansion_only=bool(np.array_equal(definition.get("loading_effective_x_bounds_m"), effective) and np.array_equal([boundary["lower"][0], boundary["upper"][0]], effective)),
            actual_loading_nodes_match_record=bool(np.array_equal(selected, definition.get("loading_selected_grid_x_indices")) and np.array_equal(coordinates[selected], definition.get("loading_selected_grid_x_coordinates_m"))),
            centered_actual_loading_nodes=bool(len(selected) > 0 and np.array_equal(selected, config["n_grid"] - selected[::-1]) and np.isclose(coordinates[selected].mean(), .2, rtol=0, atol=1e-14)),
            recorded_loading_center_matches=bool(len(selected) > 0 and np.isclose(definition.get("loading_grid_center_x_m", np.nan), coordinates[selected].mean(), rtol=0, atol=1e-14)),
        )
    return checks, report


def review(folder, name):
    family = {
        "jc-necking": "jc-tension", "concrete-bending-large": "concrete-bending",
        "concrete-bending-symmetric": "concrete-bending",
        "ceramic-indentation-wide": "ceramic-indentation",
    }.get(name, name)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError(f"Incomplete run: {folder}")
    diagnostics = rows(folder / "diagnostics.csv")
    observations = rows(folder / "observables.csv")
    with np.load(folder / "final_state_0.npz") as archive:
        state = {key: archive[key] for key in archive.files}
    time = np.array([row["time"] for row in observations])
    mass = np.array([row["mass"] for row in diagnostics])
    checks = {
        "final_arrays_finite": all(bool(np.isfinite(value).all()) for value in state.values()),
        "final_state_valid": bool(state["valid"] and state["history_valid"].all()),
        "positive_particle_mass_and_volume": bool(
            (state["mass"] > 0).all() and (state["volume"] > 0).all()
        ),
        "positive_saved_J": all(row["min_J"] > 0 for row in diagnostics),
        "positive_final_J": bool((np.linalg.det(state["F"]) > 0).all()),
        "finite_saved_observables": all(
            value is None or np.isfinite(value)
            for row in observations + diagnostics for value in row.values()
        ),
        "matching_table_times": np.array_equal(time, [row["time"] for row in diagnostics]),
        "strictly_increasing_times": bool((np.diff(time) > 0).all()),
        "full_duration": bool(np.isclose(time[-1], summary["simulation_time"], rtol=1e-12)),
        "constant_saved_mass": bool(np.allclose(mass, mass[0], rtol=1e-13, atol=0)),
        "saved_mass_matches_final_state": bool(np.allclose(mass, state["mass"].sum(), rtol=1e-13, atol=0)),
    }
    columns = {
        "jc-tension": (
            "mean_grip_tensile_force_N", "central_material_section_force_N",
            "engineering_strain", "load_bearing_axial_gap_m", "max_damage",
        ),
        "concrete-bending": (
            "loading_reaction_y_N", "midspan_bottom_deflection_m", "max_damage",
            "damaged_reference_volume_fraction",
        ),
        "ceramic-indentation": (
            "indenter_reaction_y_N", "nominal_indentation_m", "max_damage",
            "damaged_reference_volume_fraction",
        ),
    }
    result = {
        "case": summary["case"],
        "scope": "Saved-frame diagnostics and final particle state; no mesh-convergence or experimental-calibration claim.",
        "sources": {
            filename: hashlib.sha256((folder / filename).read_text(encoding="utf-8").encode()).hexdigest()
            for filename in ("summary.json", "diagnostics.csv", "observables.csv")
        },
        "checks": checks,
        "saved_frames": len(time),
        "min_J": min(row["min_J"] for row in diagnostics),
        "maximum_relative_mass_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "failed_mass_fraction": float(state["mass"][state["history_failed"]].sum() / mass[0]),
        "observables": {key: extrema(observations, key) for key in columns[family]},
    }
    if family == "jc-tension":
        positions = np.load(folder / "trajectory_x.npy", mmap_mode="r")
        failed = np.load(folder / "trajectory_failed.npy", mmap_mode="r")
        dx = summary["configs"][0]["domain_size"] / summary["configs"][0]["n_grid"]
        active_x = np.sort(state["x"][~state["history_failed"], 0])
        split = np.flatnonzero(np.diff(active_x) > 3 * dx) + 1
        groups = np.split(active_x, split) if len(active_x) else []
        levels, sections = np.unique(positions[0, :, 0], return_inverse=True)
        active_count = np.bincount(sections, weights=~state["history_failed"], minlength=len(levels))
        spacing = float(np.diff(levels).mean())
        middle = np.abs(levels - np.mean(summary["definition"]["bar_ends_x_m"])) < 0.51 * spacing
        F = state["F"]
        normal = np.linalg.det(F)[:, None] * np.linalg.inv(F).transpose(0, 2, 1)[:, :, 0]
        traction = np.einsum("pi,pi->p", state["history_stress"][:, 0, :], normal)
        force = np.bincount(
            sections,
            weights=traction * state["volume"] * (~state["history_failed"]) / spacing,
            minlength=len(levels),
        )
        central_force = float(force[middle].mean())
        checks.update(
            final_failure_field_matches=bool(np.array_equal(failed[-1], state["history_failed"])),
            failure_is_irreversible=all(bool((failed[i] >= failed[i - 1]).all()) for i in range(1, len(failed))),
            central_reference_section_has_no_active_particles=bool((active_count[middle] == 0).all()),
            separated_grid_support=bool(len(groups) >= 2),
            zero_final_central_section_force=bool(abs(central_force) < 1e-8),
            section_force_matches_observer=bool(np.isclose(
                central_force, observations[-1]["central_material_section_force_N"], atol=1e-8
            )),
        )
        first = next((i for i, frame in enumerate(failed) if frame.any()), None)
        result["separation"] = {
            "first_saved_failure_time_s": None if first is None else float(time[first]),
            "support_separation_threshold_m": 3 * dx,
            "active_groups": [
                {"particles": len(group), "x_min_m": float(group[0]), "x_max_m": float(group[-1])}
                for group in groups
            ],
            "failed_reference_layer_centres_m": levels[active_count == 0].tolist(),
            "final_central_section_force_N": central_force,
            "interpretation": "Mass remains in noncontacting debris. Grip reactions include vibration of the separated pieces; localization is not fracture-energy regularized.",
        }
        positions._mmap.close()
        failed._mmap.close()
    else:
        result["interpretation"] = (
            "Constitutive damage with retained grid participation. D=1 is not a resolved crack surface or a detached fragment."
        )
    if name in ("jc-necking", "concrete-bending-large", "concrete-bending-symmetric", "ceramic-indentation-wide"):
        review_revision(result, folder, name, summary, observations, diagnostics, state)
    result["passed"] = all(bool(value) for value in checks.values())
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    return result


def review_revision(result, folder, name, summary, observations, diagnostics, state):
    checks = result["checks"]
    definition, config = summary["definition"], summary["configs"][0]
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    time = np.asarray([row["time"] for row in observations])
    dx = config["domain_size"] / config["n_grid"]
    total_volume = float(state["volume"].sum())
    checks.update(
        complete_status=bool(status.get("completed")),
        complete_saved_frame_count=len(time) == status["frames"] + 1 and status["frame"] == status["frames"],
        step_count_matches_duration=bool(np.isclose(summary["steps"] * config["dt"], time[-1], rtol=1e-12, atol=1e-14)),
        complete_step_count=status["steps_completed"] == summary["steps"],
        duration_matches_definition=bool(np.isclose(time[-1], definition["duration_s"], rtol=1e-12, atol=1e-14)),
        at_least_one_hundred_thousand_particles=len(state["x"]) >= 100000,
        initial_time_is_zero=bool(np.isclose(time[0], 0.0, rtol=0, atol=1e-14)),
        uniform_saved_times=bool(np.allclose(np.diff(time), time[-1] / status["frames"], rtol=1e-12, atol=1e-14)),
        status_time_matches=bool(np.isclose(status["time"], time[-1], rtol=1e-12, atol=1e-14)),
    )
    mapped = []
    try:
        for field in ("x", "damage"):
            mapped.append(np.load(folder / f"trajectory_{field}.npy", mmap_mode="r"))
        positions, damage = mapped
        checks["saved_shapes_match"] = positions.shape == (len(time), len(state["x"]), 3) and damage.shape == (len(time), len(state["x"]))
        checks["final_fields_match"] = bool(np.array_equal(positions[-1], state["x"]) and np.array_equal(damage[-1], state["history_damage"]))
        checks["finite_saved_positions_and_damage"] = all(bool(np.isfinite(x).all() and np.isfinite(d).all()) for x, d in zip(positions, damage))
        checks["bounded_irreversible_damage"] = bool(np.min(damage) >= 0 and np.max(damage) <= 1) and all(bool((damage[i] >= damage[i - 1]).all()) for i in range(1, len(time)))
        checks["initially_undamaged"] = bool(np.count_nonzero(damage[0]) == 0)
        if name == "jc-necking":
            mapped.append(np.load(folder / "trajectory_failed.npy", mmap_mode="r"))
            check_necking(result, definition, config, time, observations, state, positions, mapped[-1])
        elif name == "ceramic-indentation-wide":
            check_ceramic_wide(result, definition, config, time, observations, state, positions, damage, summary["materials"][0])
        else:
            reference = positions[0]
            patch_checks, patch_report = hjc_loading_patch(definition, config)
            checks.update(patch_checks)
            result["loading_grid_patch"] = patch_report
            loading = definition["loading_boundary"]
            center_x = 0.5 * (loading["lower"][0] + loading["upper"][0])
            bottom = np.isclose(reference[:, 1], reference[:, 1].min(), atol=1e-12, rtol=0)
            middle = bottom & (np.abs(reference[:, 0] - center_x) <= max(dx, 0.0125))
            deflection, damaged_fraction, maximum = [], [], []
            for x, d in zip(positions, damage):
                displacement = x[:, 1] - reference[:, 1]
                deflection.append(float(-displacement[middle].mean()))
                maximum.append(float(-displacement.min()))
                damaged_fraction.append(float(state["volume"][d >= 0.5].sum() / total_volume))
            checks.update(
                explicit_strain_floor_profile=summary["materials"][0].get("legacy_v3") is False,
                retained_compressive_contact=config["failure_mode"] == "retain" and not bool(state["history_failed"].any()),
                deflection_matches_positions=bool(np.allclose(deflection, [row["midspan_bottom_deflection_m"] for row in observations], rtol=1e-11, atol=1e-12)),
                maximum_displacement_matches_positions=bool(np.allclose(maximum, [row["maximum_downward_displacement_m"] for row in observations], rtol=1e-11, atol=1e-12)),
                damaged_volume_matches_particles=bool(np.allclose(damaged_fraction, [row["damaged_reference_volume_fraction"] for row in observations], rtol=1e-11, atol=1e-13)),
                visible_actual_midspan_deflection=max(deflection) >= 0.008,
                substantial_damaged_volume=max(damaged_fraction) >= 0.05,
                prescribed_twelve_mm_stroke=bool(np.isclose(observations[-1]["prescribed_displacement_y_m"], -0.012, rtol=0, atol=1e-12)),
                stationary_final_loading_patch=abs(observations[-1]["prescribed_speed_y_m_per_s"]) < 1e-12,
            )
            result["deformation_and_damage"] = {
                "maximum_actual_midspan_deflection_m": max(deflection),
                "maximum_downward_displacement_m": max(maximum),
                "maximum_reference_volume_fraction_with_damage_at_least_half": max(damaged_fraction),
                "saved_midspan_deflection_m": deflection,
                "saved_damaged_volume_fraction": damaged_fraction,
                "display_requirement": "At least 8 mm actual midspan deflection and 5% reference volume at D >= 0.5; geometric scale remains one.",
            }
            if name == "concrete-bending-symmetric":
                check_hjc_symmetric(result, definition, config, time, observations, state, positions, damage, summary["materials"][0])
                result["runtime_symmetry"]["sources"] = {
                    filename: binary_digest(folder / filename) for filename in (
                        "summary.json", "observables.csv", "diagnostics.csv",
                        "trajectory_x.npy", "trajectory_damage.npy", "final_state_0.npz",
                    )
                }
                result["runtime_symmetry"]["review_source_sha256"] = binary_digest(Path(__file__))
    finally:
        for value in mapped:
            value._mmap.close()


def binary_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def hjc_symmetric_motion(boundary, time):
    """Independent cosine-ramp integral for the specified loading and hold."""
    ramp, duration = boundary["ramp_time"], boundary["loading_duration"]
    elapsed = np.clip(np.asarray(time) - boundary.get("start", 0.), 0., duration)
    a = np.minimum(elapsed, ramp)
    b = np.clip(elapsed - duration + ramp, 0., ramp)
    integral = .5 * (a - ramp / np.pi * np.sin(np.pi * a / ramp))
    integral += np.clip(elapsed - ramp, 0., duration - 2 * ramp)
    integral += .5 * (b + ramp / np.pi * np.sin(np.pi * b / ramp))
    rate = np.where(elapsed < ramp, .5 * (1 - np.cos(np.pi * a / ramp)),
                    np.where(elapsed <= duration - ramp, 1., .5 * (1 + np.cos(np.pi * b / ramp))))
    rate = np.where((elapsed > 0) & (elapsed < duration), rate, 0.)
    target = np.asarray(boundary["velocity"])
    return integral[..., None] * target, rate[..., None] * target


def hjc_symmetric_boundaries(definition, config):
    """Reconstruct the fixed BVP and evaluate its actual grid-node masks."""
    grid, domain = config["n_grid"], config["domain_size"]
    dx, center = domain / grid, domain / 2
    names = ["left_support", "right_support", "x_symmetry_plane", "z_symmetry_plane", "loading"]
    boundaries = config.get("boundaries", [])
    checks = {"symmetric_five_boundaries": len(boundaries) == 5,
              "symmetric_support_profile": definition.get("supports_profile") == "mirror-rollers-with-centre-planes",
              "symmetric_boundary_names": definition.get("boundary_names") == names}
    report = {"supports_profile": definition.get("supports_profile"), "node_planes": {}, "operator_reflection": []}
    if len(boundaries) != 5:
        return checks, report

    def fixed(lower, upper, components):
        return {"lower": list(lower), "upper": list(upper), "velocity": [0., 0., 0.],
                "components": list(components), "start": 0.}

    nominal = {name: {"lower": [x - .009375, 0., 0.], "upper": [x + .009375, .15 + .25 * dx, domain]}
               for name, x in (("left", .075), ("right", .325))}
    expected = []
    for name in ("left", "right"):
        lower, upper = nominal[name]["lower"].copy(), nominal[name]["upper"].copy()
        lower[0], upper[0] = float(np.nextafter(lower[0], -np.inf)), float(np.nextafter(upper[0], np.inf))
        expected.append(fixed(lower, upper, (False, True, False)))
    expected.extend((
        fixed((center - .25 * dx, 0., 0.), (center + .25 * dx, domain, domain), (True, False, False)),
        fixed((0., 0., center - .25 * dx), (domain, domain, center + .25 * dx), (False, False, True)),
    ))
    loader = fixed((float(np.nextafter(.1875, -np.inf)), .21 - .5 * dx, 0.),
                   (float(np.nextafter(.2125, np.inf)), .21 + 2 * dx, domain), (False, True, False))
    loader.update(velocity=[0., -.012 / (.0012 - .0003), 0.], ramp_time=.0003, loading_duration=.0012)
    expected.append(loader)
    actual = [{key: value for key, value in boundary.items() if key != "end"} for boundary in boundaries]
    checks.update(
        symmetric_exact_boundary_geometry_components_and_motion=actual == expected,
        symmetric_boundaries_active_for_full_run=all(boundary.get("end") is None and boundary.get("start") == 0. for boundary in boundaries),
        symmetric_definition_matches_actual_boundaries=definition.get("boundary_definitions") == actual,
        symmetric_support_nominal_bounds=definition.get("support_nominal_bounds_m") == nominal,
        symmetric_loading_record_matches_actual_boundary=definition.get("loading_boundary") == actual[-1],
        symmetric_single_plane_metadata=definition.get("symmetry_plane_coordinates_m") == {"x": center, "z": center}
            and definition.get("symmetry_plane_selection_half_width_m") == .25 * dx,
    )
    coordinates = np.arange(grid + 1) * dx
    selections = [
        [np.flatnonzero((coordinates >= boundary["lower"][axis]) & (coordinates <= boundary["upper"][axis])).tolist()
         for axis in range(3)] for boundary in boundaries
    ]
    report["selected_node_indices_by_axis"] = dict(zip(names, selections))
    report["node_planes"] = {"x": selections[2][0], "z": selections[3][2]}
    checks.update(
        symmetric_one_middle_node_plane=grid % 2 == 0 and selections[2][0] == [grid // 2] and selections[3][2] == [grid // 2],
        symmetric_plane_node_record=definition.get("symmetry_plane_selected_indices") == report["node_planes"],
        symmetric_support_node_record=definition.get("support_selected_grid_x_indices") == {"left": selections[0][0], "right": selections[1][0]},
        symmetric_mirrored_support_nodes=bool(selections[0][0] and np.array_equal(selections[0][0], grid - np.asarray(selections[1][0][::-1]))),
        symmetric_five_centered_loading_nodes=grid == 64 and selections[4][0] == [30, 31, 32, 33, 34],
    )
    # Component-wise affine grid operator v -> where(mask, prescribed, v).
    for time in (0., .00015, .0006, .00105, .0012, .0015):
        constrained = np.zeros((grid + 1, grid + 1, grid + 1, 3), dtype=bool)
        prescribed = np.zeros(constrained.shape)
        for boundary in boundaries:
            shift, speed = (hjc_symmetric_motion(boundary, time) if "loading_duration" in boundary
                            else (np.zeros(3), np.asarray(boundary["velocity"])))
            axes = [(coordinates >= boundary["lower"][axis] + shift[axis])
                    & (coordinates <= boundary["upper"][axis] + shift[axis]) for axis in range(3)]
            mask = axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
            if not boundary.get("start", 0.) <= time or boundary.get("end") is not None and time > boundary["end"]:
                mask[:] = False
            components = np.asarray(boundary.get("components", [True] * 3), dtype=bool)
            active = mask[..., None] & components
            constrained |= active
            prescribed = np.where(active, speed, prescribed)
        for axis, name in ((0, "x"), (2, "z")):
            sign = np.ones(3)
            sign[axis] = -1.
            masks_match = bool(np.array_equal(constrained, np.flip(constrained, axis=axis)))
            error = float(np.max(np.abs(prescribed - np.flip(prescribed, axis=axis) * sign)))
            report["operator_reflection"].append({"time_s": time, "axis": name, "mask_equal": masks_match, "prescribed_velocity_error_m_per_s": error})
    checks["symmetric_complete_grid_operator_reflection"] = all(item["mask_equal"] and item["prescribed_velocity_error_m_per_s"] == 0. for item in report["operator_reflection"])
    return checks, report


def check_hjc_symmetric(result, definition, config, time, observations, state, positions, damage, material):
    checks, boundary_report = hjc_symmetric_boundaries(definition, config)
    lower, upper = np.array([.05, .15, .17]), np.array([.35, .21, .23])
    dx = config["domain_size"] / config["n_grid"]
    reference = np.asarray(positions[0])
    counts = np.ceil((upper - lower) / dx * 3).astype(int)
    axes = [lower[axis] + (np.arange(counts[axis]) + .5) * (upper[axis] - lower[axis]) / counts[axis] for axis in range(3)]
    expected_reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    checks.update(
        symmetric_fixed_domain_grid_and_time=config["n_grid"] == 64 and config["domain_size"] == .4
            and config["dim"] == 3 and config["boundary_cells"] == 0 and config["gravity"] == [0., 0., 0.]
            and config["material"] == "hjc" and config["failure_mode"] == "retain"
            and np.isclose(config["dt"], .0015 / 10500, rtol=1e-14, atol=0),
        symmetric_fixed_material=material == HJC_PARAMETERS and material.get("legacy_v3") is False
            and definition.get("material_parameters_SI") == HJC_PARAMETERS,
        symmetric_fixed_geometry_and_sampling=definition.get("lower") == lower.tolist() and definition.get("upper") == upper.tolist()
            and definition.get("particles_per_cell_per_axis") == 3 and len(reference) == 121104,
        symmetric_original_loading_history=definition.get("loading_revision") == "centered-grid-patch-v2"
            and definition.get("material_profile") == "strain-floor" and np.isclose(definition.get("duration_s", np.nan), .0015, rtol=1e-14, atol=0)
            and definition.get("loading_duration_s") == .0012 and definition.get("hold_duration_s") == .0003
            and definition.get("startup_ramp_s") == .0003 and definition.get("stopping_ramp_s") == .0003
            and definition.get("final_prescribed_displacement_m") == .012,
        symmetric_initial_reference_lattice=reference.shape == expected_reference.shape
            and np.allclose(reference, expected_reference, rtol=0, atol=1e-14),
        symmetric_scene_identity=definition.get("purpose") == "concrete-bending-symmetric"
            and definition.get("comparison_baseline_case") == "concrete-bending-large",
        symmetric_complete_sampling=len(time) == 61 and np.isclose(time[0], 0., rtol=0, atol=1e-14)
            and np.isclose(time[-1], .0015, rtol=1e-12, atol=1e-14),
    )
    report = {"scope": "Independent saved-frame reflection audit of the explicitly symmetry-constrained BVP; no field symmetrization, mesh-convergence or unconstrained-instability claim.",
              "thresholds": {"maximum_saved_damage_mirror_rms": .01, "maximum_saved_left_right_damaged_fraction_difference": .01},
              "boundary_audit": boundary_report, "history": []}
    result["runtime_symmetry"] = report
    if not checks["symmetric_initial_reference_lattice"]:
        checks["symmetric_runtime_data_auditable"] = False
        result["checks"].update({key: bool(value) for key, value in checks.items()})
        report.update(checks={key: bool(value) for key, value in checks.items()}, passed=False)
        return
    indices = np.arange(len(reference)).reshape(tuple(counts))
    pairs = {name: np.flip(indices, axis=axis).ravel() for axis, name in ((0, "x"), (2, "z"))}
    volume, mass = state["volume"], state["mass"]
    expected_volume = np.prod(upper - lower) / len(reference)
    checks["symmetric_original_mass_and_volume"] = bool(np.allclose(volume, expected_volume, rtol=1e-14, atol=0)
        and np.allclose(mass, expected_volume * HJC_PARAMETERS["density"], rtol=1e-14, atol=0))
    tolerance = 32 * np.finfo(reference.dtype).eps
    halves = {"left": reference[:, 0] < .2 - tolerance, "right": reference[:, 0] > .2 + tolerance}
    initial_error = {}
    for axis, name in ((0, "x"), (2, "z")):
        reflected = reference[pairs[name]].copy()
        reflected[:, axis] = .4 - reflected[:, axis]
        initial_error[name] = float(np.max(np.abs(reflected - reference)))
    checks["symmetric_initial_particles_mirror_equivalent"] = max(initial_error.values()) <= 1e-14
    report["initial_reflection_max_error_m"] = initial_error
    report["half_reference_particle_counts"] = {key: int(mask.sum()) for key, mask in halves.items()}

    def metrics(x, d):
        values = {}
        displacement = x - reference
        for name, mask in halves.items():
            half_volume = volume[mask].sum()
            values[f"{name}_mean_damage_reference_volume"] = float(np.sum(volume[mask] * d[mask]) / half_volume)
            values[f"{name}_damaged_reference_volume_fraction"] = float(np.sum(volume[mask] * (d[mask] >= .5)) / half_volume)
            values[f"{name}_fully_damaged_reference_volume_fraction"] = float(np.sum(volume[mask] * (d[mask] >= 1.)) / half_volume)
        for axis, name in ((0, "x"), (2, "z")):
            dd = d - d[pairs[name]]
            values[f"damage_mirror_{name}_mae"] = float(np.sum(volume * np.abs(dd)) / volume.sum())
            values[f"damage_mirror_{name}_rms"] = float(np.sqrt(np.sum(volume * dd**2) / volume.sum()))
            values[f"damage_mirror_{name}_max"] = float(np.max(np.abs(dd)))
            reflected = displacement[pairs[name]].copy()
            reflected[:, axis] *= -1
            values[f"displacement_mirror_{name}_rms_m"] = float(np.sqrt(np.sum(volume[:, None] * (displacement - reflected)**2) / volume.sum()))
        com = np.sum(mass[:, None] * displacement, axis=0) / mass.sum()
        values.update({f"center_of_mass_displacement_{axis}_m": float(value) for axis, value in zip("xyz", com)})
        return values

    support_inside = True
    for t, x, d in zip(time, positions, damage):
        report["history"].append({"time_s": float(t), **metrics(x, d)})
        base = np.floor(x / dx - .5)
        support_inside &= bool(np.all((base >= 0) & (base + 2 <= config["n_grid"])))
    final = metrics(state["x"], state["history_damage"])
    report["final_state_metrics"] = final
    report["maximum_saved_damage_mirror_rms"] = {name: max(row[f"damage_mirror_{name}_rms"] for row in report["history"]) for name in ("x", "z")}
    half_difference = [abs(row["left_damaged_reference_volume_fraction"] - row["right_damaged_reference_volume_fraction"]) for row in report["history"]]
    report["maximum_saved_left_right_damaged_fraction_difference"] = max(half_difference)
    prescribed, velocity = hjc_symmetric_motion(definition["loading_boundary"], time)
    checks.update(
        symmetric_all_saved_statistics_match_observer=all(all(key in row and row[key] is not None and np.isclose(value, row[key], rtol=1e-10, atol=1e-12)
            for key, value in computed.items() if key != "time_s") for computed, row in zip(report["history"], observations)),
        symmetric_final_state_statistics_match_observer=all(key in observations[-1] and observations[-1][key] is not None
            and np.isclose(value, observations[-1][key], rtol=1e-10, atol=1e-12) for key, value in final.items()),
        symmetric_runtime_damage_x_rms_within_one_percent=report["maximum_saved_damage_mirror_rms"]["x"] <= .01,
        symmetric_runtime_damage_z_rms_within_one_percent=report["maximum_saved_damage_mirror_rms"]["z"] <= .01,
        symmetric_runtime_half_damage_fraction_within_one_percent=max(half_difference) <= .01,
        symmetric_loading_observer_matches_analytic_motion=bool(np.allclose(prescribed[:, 1], [row["prescribed_displacement_y_m"] for row in observations], rtol=1e-12, atol=1e-14)
            and np.allclose(velocity[:, 1], [row["prescribed_speed_y_m_per_s"] for row in observations], rtol=1e-12, atol=1e-14)),
        symmetric_boundary_sample_time_matches_next_step=bool(np.allclose([row["boundary_sample_time_s"] for row in observations], time + config["dt"], rtol=1e-12, atol=1e-14)),
        symmetric_particles_keep_complete_grid_support=support_inside,
        symmetric_final_momentum_matches_observer=bool(np.isclose(np.sum(mass * state["v"][:, 1]), observations[-1]["momentum_y_kg_m_per_s"], rtol=1e-11, atol=1e-12)),
    )
    report["checks"] = {key: bool(value) for key, value in checks.items()}
    report["passed"] = all(report["checks"].values())
    result["checks"].update(report["checks"])


def check_ceramic_wide(result, definition, config, time, observations, state, positions, damage, material):
    checks = result["checks"]
    lower = np.array((0.035, 0.075, 0.035))
    upper = np.array((0.215, 0.100, 0.215))
    lengths = upper - lower
    dx = config["domain_size"] / config["n_grid"]
    radius, stroke, duration = 0.0375, 0.004, 0.0003
    initial_center = np.array((0.125, 0.1375, 0.125))
    displacement = np.array((0.0, -stroke, 0.0))
    counts = np.ceil(lengths / dx * definition["particles_per_cell_per_axis"]).astype(int)
    spacings = lengths / counts
    reference, volume = positions[0], state["volume"]
    top = np.isclose(reference[:, 1], reference[:, 1].max(), rtol=0, atol=1e-12)
    radial2 = (reference[:, 0] - 0.125)**2 + (reference[:, 2] - 0.125)**2
    central = top & (radial2 <= 0.004**2)
    local = radial2 <= 0.030**2
    surface = top & local
    if not central.any() or not local.any() or not surface.any():
        raise ValueError("The fixed ceramic observation regions contain no particles")

    def same(actual, expected, *, atol=1e-13):
        return bool(np.allclose(actual, expected, rtol=1e-11, atol=atol))

    sphere = config["boundaries"][1]
    backing = config["boundaries"][0]
    expected_sphere = {
        "center": initial_center, "radius": radius,
        "displacement": displacement, "duration": duration,
    }
    lattice_ok = len(reference) == int(np.prod(counts))
    for axis in range(3):
        levels = np.unique(reference[:, axis])
        expected = lower[axis] + (np.arange(counts[axis]) + 0.5) * spacings[axis]
        lattice_ok &= levels.shape == expected.shape and same(levels, expected)
    lattice_ok &= len(np.unique(reference, axis=0)) == len(reference)
    checks.update(
        specified_wide_plate=same(definition["lower"], lower) and same(definition["upper"], upper),
        specified_domain_and_no_added_loading=bool(
            same(config["domain_size"], .25) and config["dim"] == 3
            and same(config["gravity"], (0, 0, 0)) and config["boundary_cells"] == 0
            and len(config["boundaries"]) == 2
        ),
        unchanged_jh2_parameters=all(
            key in material and key in definition["material_parameters_SI"]
            and same(material[key], value) and same(definition["material_parameters_SI"][key], value)
            for key, value in CERAMIC_PARAMETERS.items()
        ),
        specified_smaller_sphere=all(same(sphere[key], value) and same(definition["rigid_sphere"][key], value) for key, value in expected_sphere.items()),
        full_loading_unloading_cycle=same(time[-1], duration),
        ideal_bonded_backing=bool(
            same(backing["lower"], (0, 0, 0)) and same(backing["upper"], (.25, .075, .25))
            and same(backing["velocity"], (0, 0, 0)) and backing["components"] is None
            and backing["start"] == 0 and backing["end"] is None
        ),
        jh2_retained_participation=config["material"] == "jh2" and config["failure_mode"] == "retain",
        unscaled_geometry=definition["geometry_scale"] == 1.0,
        correct_regular_reference_lattice=bool(lattice_ok),
        reference_volume_matches_plate=same(volume.sum(), np.prod(lengths), atol=1e-16),
        uniform_reference_cell_volumes=same(volume, np.prod(spacings), atol=1e-20),
        mass_matches_material_density=same(state["mass"], volume * definition["material_parameters_SI"]["density"], atol=1e-18),
        correct_particle_spacing_metadata=same(definition["particle_spacings_m"], spacings) and same(definition["particle_spacing_m"], spacings.min()),
        correct_particle_counts=np.array_equal(definition["particle_counts_per_axis"], counts),
        fixed_physical_observation_regions=same(definition["central_surface_radius_m"], .004) and same(definition["local_region_radius_m"], .030),
        correct_region_population=bool(
            definition["central_surface_particles"] == central.sum()
            and definition["local_region_particles"] == local.sum()
            and definition["local_surface_particles"] == surface.sum()
        ),
        resolved_plate_thickness=lengths[1] / dx >= 8,
        resolved_nominal_contact_radius=np.sqrt(radius * stroke) / dx >= 4,
        geometry_resolution_metadata=bool(
            same(definition["sphere_diameter_to_plate_width_ratio"], 2 * radius / lengths[0])
            and same(definition["plate_thickness_in_grid_cells"], lengths[1] / dx)
            and same(definition["nominal_contact_radius_m"], np.sqrt(radius * stroke))
            and same(definition["nominal_contact_radius_in_grid_cells"], np.sqrt(radius * stroke) / dx)
            and same(definition["nominal_contact_radius_to_plate_half_width_ratio"], np.sqrt(radius * stroke) / (lengths[0] / 2))
            and same(definition["maximum_nominal_indentation_m"], stroke)
        ),
    )
    curves = {name: [] for name in (
        "nominal_indentation_m", "indenter_speed_y_m_per_s", "central_surface_displacement_y_m",
        "maximum_downward_displacement_m", "central_surface_max_damage", "local_surface_displacement_y_m",
        "local_region_maximum_downward_displacement_m", "local_region_max_damage",
        "local_region_mean_damage_reference_volume", "local_region_damaged_reference_volume_m3",
        "indenter_particle_overlap_max_m", "indenter_surface_particle_overlap_max_m",
        "particle_centers_inside_indenter", "damaged_reference_volume_fraction",
    )}
    for instant, x, d in zip(time, positions, damage):
        fraction = np.clip(instant / duration, 0.0, 1.0)
        center = initial_center + displacement * (1 - np.cos(2 * np.pi * fraction)) / 2
        speed = displacement[1] * np.pi / duration * np.sin(2 * np.pi * fraction)
        dy = x[:, 1] - reference[:, 1]
        overlap = np.maximum(radius - np.linalg.norm(x - center, axis=1), 0.0)
        values = (
            upper[1] - (center[1] - radius), speed, dy[central].mean(), -dy.min(),
            d[central].max(), dy[surface].mean(), -dy[local].min(), d[local].max(),
            np.dot(volume[local], d[local]) / volume[local].sum(), volume[local & (d >= .5)].sum(),
            overlap.max(), overlap[top].max(), np.count_nonzero(overlap > 0),
            volume[d >= .5].sum() / volume.sum(),
        )
        for name, value in zip(curves, values):
            curves[name].append(float(value))
    for name, values in curves.items():
        checks[f"reconstructed_{name}"] = same(values, [row[name] for row in observations])
    peak_overlap = max(curves["indenter_particle_overlap_max_m"])
    central_depth = -min(curves["central_surface_displacement_y_m"])
    checks.update(
        actual_central_indentation_at_least_one_mm=central_depth >= .001,
        visible_local_damage=max(curves["local_region_max_damage"]) >= .5,
        nonzero_local_damaged_volume=max(curves["local_region_damaged_reference_volume_m3"]) > 0,
        positive_compressive_indenter_reaction=max(row["indenter_reaction_y_N"] for row in observations) > 0,
        particle_sphere_overlap_within_grid_support=peak_overlap <= 1.5 * dx,
        reaction_sample_times=same([row["boundary_sample_time_s"] for row in observations], time + config["dt"]),
    )
    result["ceramic_indentation"] = {
        "sphere_diameter_m": 2 * radius,
        "plate_dimensions_m": lengths.tolist(),
        "maximum_actual_central_indentation_m": float(central_depth),
        "maximum_particle_center_overlap_m": float(peak_overlap),
        "overlap_guard_m": 1.5 * dx,
        "saved_curves": curves,
        "interpretation": (
            "Dynamic prescribed spherical indentation with bonded backing and retained JH-2 material. "
            "Damage and displacement are measured in fixed reference regions. Particle-center overlap "
            "is a grid-contact diagnostic, not contact area; 1.5 grid cells is a support-scale alarm, "
            "not a surface-accuracy tolerance or a Hertz validation."
        ),
    }


def check_necking(result, definition, config, time, observations, state, positions, failures):
    checks = result["checks"]
    reference = positions[0]
    levels, layers = np.unique(reference[:, 0], return_inverse=True)
    spacing = definition["particle_spacing_m"]
    center = np.mean(definition["bar_ends_x_m"])
    central = np.abs(levels - center) <= 0.51 * spacing
    counts = np.bincount(layers)

    def radii(x):
        cy = np.bincount(layers, weights=x[:, 1]) / counts
        cz = np.bincount(layers, weights=x[:, 2]) / counts
        squared = (x[:, 1] - cy[layers])**2 + (x[:, 2] - cz[layers])**2
        return np.sqrt(np.bincount(layers, weights=squared) / counts)

    initial_radius = radii(reference)
    radius_ratio = [float((radii(x)[central] / initial_radius[central]).mean()) for x in positions]
    ratio_observed = [row["central_geometric_rms_radius_ratio"] for row in observations]
    maximum_velocity_error = max(row["grip_inner_velocity_error_m_per_s"] for row in observations)
    maximum_position_error = max(row["grip_inner_axial_position_error_m"] for row in observations)
    maximum_transverse_motion = max(row["grip_inner_transverse_displacement_m"] for row in observations)
    speed = definition["grip_speed_m_per_s"]
    central_particles = central[layers]
    if failures.shape != (len(time), len(state["x"])):
        raise ValueError("The necking failure history has a different shape")
    active_counts = np.asarray([np.count_nonzero(frame[central_particles] == 0) for frame in failures])
    intact = active_counts == counts[central].sum()
    minimum_intact_radius = float(np.min(np.asarray(radius_ratio)[intact])) if intact.any() else None
    checks.update(
        binary_failure_history=all(bool(((frame == 0) | (frame == 1)).all()) for frame in failures),
        no_initial_failures=bool(np.count_nonzero(failures[0]) == 0),
        central_active_count_matches_observer=bool(np.array_equal(active_counts, [row["central_active_particles"] for row in observations])),
        specified_pr_revision=definition.get("source_commit") == "c3c1556ec86f9ebc6f0697149dff052702d20a75",
        full_reference_duration=bool(np.isclose(time[-1], 150e-6, rtol=0, atol=1e-14)),
        uniform_axial_lattice=bool(np.allclose(np.diff(levels), spacing, rtol=1e-12, atol=1e-15)),
        uniform_reference_particle_volumes=bool(np.allclose(state["volume"], spacing**3, rtol=1e-13, atol=0)),
        coordinate_radius_matches_observer=bool(np.allclose(radius_ratio, ratio_observed, rtol=1e-10, atol=1e-12)),
        resolved_initial_diameter=definition["grid_cells_across_nominal_diameter"] >= 12,
        measurable_neck_before_central_failure=minimum_intact_radius is not None and minimum_intact_radius <= 0.9,
        inner_grip_sample_exists=min(row["grip_inner_particles"] for row in observations) > 0,
        intact_grips=max(row["failed_grip_particles"] for row in observations) == 0,
        constrained_grip_velocity=maximum_velocity_error <= max(1e-8, 1e-8 * speed),
        constrained_grip_axial_motion=maximum_position_error <= 1.1 * speed * config["dt"],
        constrained_grip_transverse_motion=maximum_transverse_motion <= 1e-8 * config["domain_size"],
    )
    result["necking"] = {
        "saved_central_geometric_rms_radius_ratio": radius_ratio,
        "minimum_central_radius_ratio_before_any_central_failure": minimum_intact_radius,
        "maximum_inner_grip_velocity_error_m_per_s": maximum_velocity_error,
        "maximum_inner_grip_axial_position_error_m": maximum_position_error,
        "maximum_inner_grip_transverse_displacement_m": maximum_transverse_motion,
        "interpretation": "Coordinate radii are reference-layer RMS measures. Post-failure debris geometry is not inferred from its frozen deformation gradient. The PR is an implementation reference, not experimental validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/material-engineering"))
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    args = parser.parse_args()
    names = CASES if args.case == "all" else (args.case,)
    records = {}
    for name in names:
        folder = args.root / name / CASES[name]
        result = review(folder, name)
        if "runtime_symmetry" in result:
            audit_path = folder / "runtime-symmetry.json"
            audit_path.write_text(json.dumps(result["runtime_symmetry"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
            result["runtime_symmetry_report"] = {"file": audit_path.name, "sha256": binary_digest(audit_path)}
        (folder / "interpretation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        records[name] = result
        print(f"{name}: {'passed' if result['passed'] else 'FAILED'}", flush=True)
    if not all(record["passed"] for record in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
