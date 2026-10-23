"""Reconstruct woven-composite geometry and elastic response using NumPy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.woven_geometry import ARCHITECTURES, sample_weave


LOADS = ("warp-tension", "weft-tension", "in-plane-shear", "transverse-compression", "transverse-bending")
DOMAIN = 0.016
ORIGIN = np.full(3, 0.008)
LOWER = np.array((0.003, 0.007, 0.003))
UPPER = np.array((0.013, 0.009, 0.013))
LOAD_TIME = 0.45e-3 / np.sqrt(1000.0)
END_TIME = 0.60e-3 / np.sqrt(1000.0)
YARN_STRAIN_LIMIT = 0.1
SOURCE_NAMES = ("summary.json", "diagnostics.csv", "observables.csv")


def file_hash(path, *, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_descriptor(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{key: None if value in ("", "None") else float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def equivalent_record(first, second):
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(equivalent_record(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(equivalent_record(a, b) for a, b in zip(first, second))
    if first is None or second is None or isinstance(first, bool) or isinstance(second, bool):
        return first is second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return bool(np.isfinite(first) and np.isfinite(second) and np.isclose(first, second, rtol=1e-12, atol=1e-15))
    return first == second


def motion(time, boundary):
    if "displacement" not in boundary:
        return np.zeros(3), np.zeros(3)
    elapsed = time - boundary["start"]
    fraction = np.clip(elapsed / boundary["loading_duration"], 0.0, 1.0)
    target = np.asarray(boundary["displacement"])
    distance = target * (1 - np.cos(np.pi * fraction)) / 2
    velocity = target * np.pi / (2 * boundary["loading_duration"]) * np.sin(np.pi * fraction)
    if elapsed <= 0 or elapsed >= boundary["loading_duration"]:
        velocity = np.zeros(3)
    return distance, velocity


def expected_boundaries(loading, dx, loading_revision=1):
    if type(loading_revision) is not int or loading_revision not in (1, 2) or (loading_revision == 2 and loading != "in-plane-shear"):
        raise ValueError("unknown woven loading revision")
    def moving(lower, upper, displacement, components=None):
        return {"lower": list(lower), "upper": list(upper), "displacement": list(displacement), "loading_duration": LOAD_TIME, "components": components, "start": 0.0}

    def fixed(lower, upper, components):
        return {"lower": list(lower), "upper": list(upper), "velocity": [0.0] * 3, "components": list(components), "start": 0.0, "end": None}

    if loading in LOADS[:3]:
        face_axis = 0 if loading == "warp-tension" else 2
        direction_axis = 2 if loading == "weft-tension" else 0
        stroke = (0.00015 if loading_revision == 2 else 0.0002) if loading == "in-plane-shear" else 0.00008
        high_low, low_high = np.zeros(3), np.full(3, DOMAIN)
        high_low[face_axis], low_high[face_axis] = UPPER[face_axis] - 0.001, LOWER[face_axis] + 0.001
        target = np.eye(3)[direction_axis] * stroke
        definitions = [moving(np.zeros(3), low_high, -target), moving(high_low, np.full(3, DOMAIN), target)]
        names = ["negative_grip", "positive_grip"]
    elif loading == "transverse-compression":
        definitions = [
            moving((0, 0, 0), (DOMAIN, LOWER[1] + 0.000375, DOMAIN), (0, 0.000025, 0), [False, True, False]),
            moving((0, UPPER[1] - 0.000375, 0), (DOMAIN,) * 3, (0, -0.000025, 0), [False, True, False]),
        ]
        names = ["lower_platen", "upper_platen"]
    elif loading == "transverse-bending":
        definitions = [
            fixed((0.0038, 0, 0), (0.0042, LOWER[1] + 0.25 * dx, DOMAIN), (True, True, False)),
            fixed((0.0118, 0, 0), (0.0122, LOWER[1] + 0.25 * dx, DOMAIN), (False, True, False)),
            fixed((0.0038, 0, ORIGIN[2] - 0.55 * dx), (0.0042, LOWER[1] + 0.25 * dx, ORIGIN[2] + 0.55 * dx), (False, False, True)),
            moving((0.0075, UPPER[1] - 0.5 * dx, 0), (0.0085, DOMAIN, DOMAIN), (0, -0.00025, 0), [False, True, False]),
        ]
        names = ["left_support", "right_support", "lateral_anchor", "loading_patch"]
    else:
        raise ValueError("unknown woven loading")
    types = ["SmoothDisplacementBox" if "displacement" in boundary else "VelocityBox" for boundary in definitions]
    return definitions, names, types


def stiffness_from_constants(parameters):
    E, nu, G = (np.asarray(parameters[key], dtype=float) for key in ("yarn_young", "yarn_poisson", "yarn_shear"))
    if E.shape != (3,) or G.shape != (3,) or nu.shape != (3,) or np.any(E <= 0) or np.any(G <= 0):
        raise ValueError("invalid yarn engineering constants")
    compliance = np.diag(np.concatenate((1 / E, 1 / G[[2, 1, 0]])))
    for i, j, k in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
        compliance[i, j] = compliance[j, i] = -nu[k] / E[i]
    if not np.isfinite(compliance).all() or np.linalg.eigvalsh(compliance).min() <= 0:
        raise ValueError("yarn compliance is not positive definite")
    return np.linalg.inv(compliance)


def elastic_response(F, orientation, phase, material):
    """Return W, Green-strain norm and Kirchhoff stress in reference-volume units."""
    F, orientation, phase = np.asarray(F), np.asarray(orientation), np.asarray(phase)
    if F.shape != orientation.shape or F.ndim != 3 or F.shape[1:] != (3, 3) or phase.shape != (len(F),):
        raise ValueError("inconsistent constitutive array shapes")
    if not np.all((phase == 0) | (phase == 1)):
        raise ValueError("phase must be matrix=0 or yarn=1")
    green = (F.swapaxes(-1, -2) @ F - np.eye(3)) / 2
    local = orientation.swapaxes(-1, -2) @ green @ orientation
    engineering = np.stack((local[:, 0, 0], local[:, 1, 1], local[:, 2, 2], 2 * local[:, 1, 2], 2 * local[:, 0, 2], 2 * local[:, 0, 1]), axis=-1)
    second_voigt = engineering @ np.asarray(material["yarn_stiffness"]).T
    yarn_energy = np.einsum("ni,ni->n", engineering, second_voigt) / 2
    second = np.stack((second_voigt[:, 0], second_voigt[:, 5], second_voigt[:, 4], second_voigt[:, 5], second_voigt[:, 1], second_voigt[:, 3], second_voigt[:, 4], second_voigt[:, 3], second_voigt[:, 2]), axis=-1).reshape(F.shape)
    yarn_stress = F @ orientation @ second @ orientation.swapaxes(-1, -2) @ F.swapaxes(-1, -2)
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError("nonpositive deformation determinant")
    logJ = np.log(J)
    mu, lam = material["matrix_mu"], material["matrix_lam"]
    matrix_energy = mu * (np.sum(F * F, axis=(-1, -2)) - 3) / 2 - mu * logJ + lam * logJ**2 / 2
    matrix_stress = mu * (F @ F.swapaxes(-1, -2) - np.eye(3)) + lam * logJ[:, None, None] * np.eye(3)
    return np.where(phase == 1, yarn_energy, matrix_energy), np.sqrt(np.sum(green**2, axis=(-1, -2))), np.where((phase == 1)[:, None, None], yarn_stress, matrix_stress)


def positive_stored_energy(total):
    return bool(np.isfinite(total) and total > 0)


def section_layout(reference, spacing):
    masks = [np.abs(reference[:, axis] - ORIGIN[axis]) <= 0.51 * spacing for axis in range(3)]
    widths = np.array([len(np.unique(reference[selected, axis])) * spacing for axis, selected in enumerate(masks)])
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("each central reference section must contain particle layers")
    return masks, widths


def section_resultants(F, tau, reference, volume, masks, widths):
    """One chunk's reference-plane forces and reference moment about z."""
    piola = tau @ np.linalg.inv(F).swapaxes(-1, -2)
    force = np.zeros((3, 3))
    moment_z = 0.0
    for axis, selected in enumerate(masks):
        weighted = piola[selected, :, axis] * (volume[selected] / widths[axis])[:, None]
        force[axis] = weighted.sum(axis=0)
        if axis == 0:
            moment_z = -float(np.sum((reference[selected, 1] - ORIGIN[1]) * weighted[:, 0]))
    return force, moment_z


def support_mask(reference, dx, boundary):
    if "displacement" in boundary:
        return np.all((reference >= np.asarray(boundary["lower"]) + 1.5 * dx) & (reference <= np.asarray(boundary["upper"]) - 1.5 * dx), axis=1)
    lower_node = np.floor(reference / dx - 0.5) * dx
    return np.all((lower_node >= np.asarray(boundary["lower"]) - 1e-14) & (lower_node + 2 * dx <= np.asarray(boundary["upper"]) + 1e-14), axis=1)


class Comparisons:
    def __init__(self):
        self.errors = {}
        self.matches = {}

    def add(self, name, actual, expected, *, rtol=1e-9, atol=1e-12):
        if actual is None or expected is None:
            matched, error = actual is expected, 0.0
        else:
            actual, expected = np.asarray(actual), np.asarray(expected)
            finite = np.isfinite(actual).all() and np.isfinite(expected).all()
            error = float(np.max(np.abs(actual - expected))) if finite else None
            matched = bool(finite and actual.shape == expected.shape and np.allclose(actual, expected, rtol=rtol, atol=atol))
        previous = self.errors.get(name, 0.0)
        self.errors[name] = max(previous, error) if previous is not None and error is not None else None
        self.matches[name] = self.matches.get(name, True) and matched


def review(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    definition, config, material = summary["definition"], summary["configs"][0], summary["materials"][0]
    architecture, loading = definition["architecture"], definition["loading"]
    if definition["series"] != "woven-elastic-v1" or architecture not in ARCHITECTURES or loading not in LOADS:
        raise ValueError("unknown woven series, architecture or loading")
    rows, diagnostics = read_rows(folder / "observables.csv"), read_rows(folder / "diagnostics.csv")
    if len(rows) < 2 or len(rows) != len(diagnostics):
        raise ValueError("expected matching complete observable and diagnostic tables")
    time = np.asarray([row["time"] for row in rows])
    count = sum(summary["particles"])
    frames = status["frames"]
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("status.frames must be a positive integer")
    dx = config["domain_size"] / config["n_grid"]
    boundaries, names, types = expected_boundaries(loading, dx, definition.get("loading_revision", 1))
    comparisons = Comparisons()
    checks = {
        "completed_numerical_run": summary.get("completed") is True and status.get("completed") is True,
        "single_three_dimensional_woven_body": len(summary["configs"]) == len(summary["materials"]) == len(summary["particles"]) == 1 and config["dim"] == 3 and config["material"] == "woven-composite",
        "complete_saved_frame_count": len(time) == frames + 1 and status["frame"] == frames,
        "complete_step_count": summary["steps"] == status["steps_completed"],
        "initial_time_zero": bool(np.isclose(time[0], 0, atol=1e-14, rtol=0)),
        "strict_uniform_saved_times": bool(np.all(np.diff(time) > 0) and np.allclose(np.diff(time), END_TIME / frames, rtol=1e-11, atol=1e-14)),
        "full_prescribed_duration": all(bool(np.isclose(value, END_TIME, rtol=1e-12, atol=1e-14)) for value in (time[-1], summary["simulation_time"], status["time"], summary["steps"] * config["dt"], definition["duration_s"])),
        "matching_diagnostic_times": bool(np.array_equal(time, [row["time"] for row in diagnostics])),
        "finite_saved_tables": all(value is None or np.isfinite(value) for row in rows + diagnostics for value in row.values()),
        "at_least_one_hundred_thousand_particles": count >= 100000,
        "status_particle_count_matches": status["particles"] == count,
        "fixed_domain_and_no_body_force": bool(np.isclose(config["domain_size"], DOMAIN, atol=1e-15, rtol=0)) and config["boundary_cells"] == 0 and np.array_equal(config["gravity"], np.zeros(3)),
        "prescribed_boundary_types": definition["boundary_types"] == types,
        "prescribed_boundary_names": definition["boundary_names"] == names,
        "prescribed_boundary_values": equivalent_record(definition["boundaries"], boundaries) and equivalent_record(config["boundaries"], boundaries),
        "loading_and_hold_times": bool(np.isclose(definition["loading_duration_s"], LOAD_TIME, rtol=1e-12, atol=1e-14)) and bool(np.isclose(definition["hold_duration_s"], END_TIME - LOAD_TIME, rtol=1e-12, atol=1e-14)),
        "actual_geometry_scale": definition["geometry_scale"] == 1.0,
        "zero_damage_and_failure_observations": all(row["max_damage"] == 0 and row["failed_particles"] == 0 for row in rows),
        "zero_damage_and_plastic_strain_diagnostics": all(row["max_damage"] == 0 and row["max_plastic_strain"] == 0 for row in diagnostics),
        "small_saved_yarn_strain": max(row["yarn_maximum_green_strain_norm"] for row in rows) <= YARN_STRAIN_LIMIT,
        "positive_saved_J": min(row["min_J"] for row in diagnostics) > 0,
    }
    geometry = definition["geometry"]
    geometry_expected = {
        "size_xz_m": [0.010, 0.010], "pitch_m": 0.0025,
        "yarn_width_m": 0.000625, "yarn_height_m": 0.0005,
        "crossing_gap_m": 0.0005, "matrix_cover_m": 0.00025,
        "thickness_m": 0.002, "origin_m": ORIGIN.tolist(),
    }
    checks["specified_geometry"] = all(equivalent_record(geometry[key], expected) for key, expected in geometry_expected.items()) and geometry["architecture"] == architecture
    sample = sample_weave(
        architecture, size=geometry["size_xz_m"], pitch=geometry["pitch_m"],
        yarn_width=geometry["yarn_width_m"], yarn_height=geometry["yarn_height_m"],
        gap=geometry["crossing_gap_m"], matrix_cover=geometry["matrix_cover_m"],
        spacing=geometry["particle_spacing_m"], origin=geometry["origin_m"],
    )
    if len(sample.positions) != count:
        raise ValueError("reconstructed geometry has a different particle count")
    checks["grid_resolves_yarn_height"] = geometry["yarn_height_m"] / dx >= 3 - 1e-12
    checks["reported_grid_resolution_matches"] = bool(np.isclose(definition["grid_cells_across_yarn_height"], geometry["yarn_height_m"] / dx, rtol=1e-12))
    checks["particle_spacing_matches_grid"] = bool(np.isclose(definition["particle_spacing_m"], dx / definition["particles_per_cell_per_axis"], rtol=1e-12)) and definition["particle_spacing_m"] == geometry["particle_spacing_m"]
    checks["geometry_occupancy_has_no_overlap"] = geometry["sampled_yarn_intersections"] == geometry["unresolved_normal_projections"] == sample.metadata["sampled_yarn_intersections"] == sample.metadata["unresolved_normal_projections"] == 0
    checks["yarn_volume_fraction_matches_geometry"] = bool(np.isclose(geometry["yarn_volume_fraction"], sample.metadata["yarn_volume_fraction"], rtol=1e-13, atol=1e-15))
    parameters = definition["material_parameters"]
    constants = {"matrix_young": 2e9, "matrix_poisson": 0.3, "matrix_density": 1100., "yarn_young": [40e9, 4e9, 4e9], "yarn_poisson": [.25, .25, .30], "yarn_shear": [1.5e9, 2e9, 2e9], "yarn_density": 1400.}
    checks["specified_illustrative_material"] = equivalent_record(parameters, constants)
    stiffness = stiffness_from_constants(parameters)
    mu = parameters["matrix_young"] / (2 * (1 + parameters["matrix_poisson"]))
    lam = parameters["matrix_young"] * parameters["matrix_poisson"] / ((1 + parameters["matrix_poisson"]) * (1 - 2 * parameters["matrix_poisson"]))
    checks["material_stiffness_matches_engineering_constants"] = bool(np.allclose(stiffness, material["yarn_stiffness"], rtol=1e-12, atol=1e-4) and np.isclose(mu, material["matrix_mu"], rtol=1e-12) and np.isclose(lam, material["matrix_lam"], rtol=1e-12))
    checks["material_densities_match"] = material["matrix_density"] == parameters["matrix_density"] and material["yarn_density"] == parameters["yarn_density"]
    q_descriptor = material["orientation"]
    if not isinstance(q_descriptor, dict):
        raise ValueError("formal woven material must fingerprint its particle orientation field")
    orientation = np.asarray(sample.orientation, dtype=np.dtype(q_descriptor["dtype"]))
    checks["orientation_matches_rebuilt_geometry_raw_hash"] = array_descriptor(orientation) == q_descriptor
    checks["constitutive_phase_matches_rebuilt_geometry_raw_hash"] = array_descriptor(sample.phase) == material["phase"]
    scale = np.diag([1., 1., 1., np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    speed = np.sqrt(max(np.linalg.eigvalsh(scale @ stiffness @ scale).max() / parameters["yarn_density"], max(3 * (lam + 2 * mu / 3), 2 * mu) / parameters["matrix_density"]))
    checks["initial_acoustic_CFL_within_limit"] = bool(0 < config["dt"] * speed / dx <= 0.15 + 1e-13)
    checks["reported_acoustic_bound_matches"] = bool(np.isclose(definition["initial_acoustic_speed_bound_m_per_s"], speed, rtol=1e-12))
    if summary["particle_fields"]["file"] != "particle_fields.npz":
        raise ValueError("unexpected static-field filename")
    with np.load(folder / "particle_fields.npz", allow_pickle=False) as archive:
        checks["exact_static_field_set"] = set(archive.files) == {"phase", "family_id", "yarn_id"}
        field_descriptors = {}
        for name in ("phase", "family_id", "yarn_id"):
            field = archive[name]
            field_descriptors[name] = array_descriptor(field)
            checks[f"static_{name}_matches_geometry_and_manifest"] = bool(np.array_equal(field, getattr(sample, name))) and field_descriptors[name] == summary["particle_fields"]["fields"][name]

    state_names = ("x", "v", "F", "mass", "volume", "valid", "time", "history_valid", "history_damage", "history_failed", "history_plastic_strain")
    with np.load(folder / "final_state_0.npz", allow_pickle=False) as archive:
        final_finite = all(bool(np.isfinite(archive[name]).all()) for name in archive.files)
        state = {name: archive[name] for name in state_names}
    if state["x"].shape != (count, 3) or state["F"].shape != (count, 3, 3) or state["v"].shape != (count, 3):
        raise ValueError("invalid final state shapes")
    checks["finite_valid_final_state"] = final_finite and bool(state["valid"]) and bool(state["history_valid"].all())
    checks["final_state_time_matches"] = bool(np.isclose(state["time"], END_TIME, rtol=1e-12, atol=1e-14))
    checks["zero_final_damage_failure_plasticity"] = not any(np.count_nonzero(state[name]) for name in ("history_damage", "history_failed", "history_plastic_strain"))
    density = np.where(sample.phase == 1, parameters["yarn_density"], parameters["matrix_density"])
    checks["reference_volume_and_mass_match_phases"] = bool(np.all(state["volume"] > 0) and np.allclose(state["volume"], sample.volume, rtol=1e-13, atol=0) and np.allclose(state["mass"], sample.volume * density, rtol=1e-13, atol=0))
    saved_mass = np.array([row["mass"] for row in diagnostics])
    checks["constant_saved_mass_matches_state"] = bool(np.allclose(saved_mass, state["mass"].sum(), rtol=1e-13, atol=0))
    checks["positive_final_J"] = bool(np.all(np.linalg.det(state["F"]) > 0))
    energy, strain_norm = np.empty(count), np.empty(count)
    sections, widths = section_layout(sample.positions, geometry["particle_spacing_m"])
    section_forces, reference_moment_z = np.zeros((3, 3)), 0.0
    for start in range(0, count, 16384):
        end = min(start + 16384, count)
        energy[start:end], strain_norm[start:end], tau = elastic_response(state["F"][start:end], orientation[start:end], sample.phase[start:end], material)
        force, moment = section_resultants(
            state["F"][start:end], tau, sample.positions[start:end], state["volume"][start:end],
            [selected[start:end] for selected in sections], widths,
        )
        section_forces += force
        reference_moment_z += moment
    del tau
    total_energy = float(np.sum(energy * state["volume"]))
    checks["positive_final_stored_elastic_energy"] = positive_stored_energy(total_energy)
    checks["small_final_yarn_strain_independent"] = float(strain_norm[sample.phase == 1].max()) <= YARN_STRAIN_LIMIT
    comparisons.add("final_maximum_green_strain_norm", rows[-1]["maximum_green_strain_norm"], strain_norm.max())
    for phase, name in ((0, "matrix"), (1, "yarn")):
        selected = sample.phase == phase
        comparisons.add(f"final_{name}_maximum_green_strain_norm", rows[-1][f"{name}_maximum_green_strain_norm"], strain_norm[selected].max())
        comparisons.add(f"final_{name}_strain_energy_J", rows[-1][f"{name}_strain_energy_J"], np.sum(energy[selected] * state["volume"][selected]), atol=1e-10)
    comparisons.add("final_total_strain_energy_J", diagnostics[-1]["strain_energy"], total_energy, atol=1e-10)
    force_checks = []
    for axis, normal in enumerate("xyz"):
        for component, direction in enumerate("xyz"):
            key = f"central_{normal}_section_force_{direction}_N"
            comparisons.add("final_" + key, rows[-1][key], section_forces[axis, component], atol=1e-8)
            force_checks.append(comparisons.matches["final_" + key])
    moment_key = "central_x_section_moment_z_Nm"
    comparisons.add("final_" + moment_key, rows[-1][moment_key], reference_moment_z, atol=1e-11)
    checks["final_piola_section_forces_match"] = all(force_checks)
    checks["final_reference_section_moment_matches"] = comparisons.matches["final_" + moment_key]
    comparisons.add("final_kinetic_energy_J", diagnostics[-1]["kinetic_energy"], np.sum(state["mass"][:, None] * state["v"]**2) / 2, atol=1e-10)
    comparisons.add("final_minimum_J", diagnostics[-1]["min_J"], np.linalg.det(state["F"]).min())
    comparisons.add("final_maximum_J", diagnostics[-1]["max_J"], np.linalg.det(state["F"]).max())
    checks["initial_energy_and_speed_zero"] = diagnostics[0]["kinetic_energy"] == 0 and abs(diagnostics[0]["strain_energy"]) < 1e-12 and diagnostics[0]["max_speed"] == 0

    mapped = []
    boundary_report = {name: {"samples": 0, "maximum_position_error_m": 0., "maximum_reported_velocity_error_m_per_s": 0., "full_support_at_all_saved_frames": True} for name in names}
    closure, central_deflection, final_grip_motion = [], [], {}
    finite_fields, zero_damage = True, True
    checks.update(constrained_positions_within_time_step_error=True, constrained_reported_velocities_match=True, constrained_final_velocities_independent=True, required_moving_samples_exist=True, required_moving_samples_keep_complete_support=True)
    try:
        for name in ("x", "speed", "damage"):
            mapped.append(np.load(folder / f"trajectory_{name}.npy", mmap_mode="r", allow_pickle=False))
        positions, speeds, damage = mapped
        if positions.shape != (len(time), count, 3) or speeds.shape != (len(time), count) or damage.shape != (len(time), count):
            raise ValueError("saved field shapes do not match the run")
        checks["initial_positions_match_geometry"] = bool(np.array_equal(positions[0], sample.positions))
        checks["final_saved_fields_match_state"] = bool(np.array_equal(positions[-1], state["x"]) and np.array_equal(damage[-1], state["history_damage"]) and np.allclose(speeds[-1], np.linalg.norm(state["v"], axis=1), rtol=1e-12, atol=1e-12))
        checks["initial_saved_speed_zero"] = bool(np.count_nonzero(speeds[0]) == 0)
        reference = sample.positions
        masks = [support_mask(reference, dx, boundary) for boundary in boundaries]
        center = (np.abs(reference[:, 0] - ORIGIN[0]) <= .0005) & (np.abs(reference[:, 2] - ORIGIN[2]) <= .0005)
        bottom, top = reference[:, 1] == reference[:, 1].min(), reference[:, 1] == reference[:, 1].max()
        initial_height = float(reference[top, 1].mean() - reference[bottom, 1].mean())
        for index, (instant, row, diagnostic) in enumerate(zip(time, rows, diagnostics)):
            current = positions[index]
            change = current - reference
            magnitude = np.linalg.norm(change, axis=1)
            finite_fields &= bool(np.isfinite(current).all() and np.isfinite(speeds[index]).all() and np.isfinite(damage[index]).all())
            zero_damage &= np.count_nonzero(damage[index]) == 0
            closure.append(initial_height - float(current[top, 1].mean() - current[bottom, 1].mean()))
            central_deflection.append(float(-change[center, 1].mean()))
            comparisons.add("maximum_displacement_m", row["maximum_displacement_m"], magnitude.max())
            comparisons.add("rms_displacement_m", row["rms_displacement_m"], np.sqrt(np.mean(magnitude**2)))
            comparisons.add("center_mean_displacement_y_m", row["center_mean_displacement_y_m"], -central_deflection[-1])
            comparisons.add("reference_yarn_volume_fraction", row["reference_yarn_volume_fraction"], sample.metadata["yarn_volume_fraction"])
            comparisons.add("minimum_J_table", row["minimum_J"], diagnostic["min_J"])
            comparisons.add("maximum_J_table", row["maximum_J"], diagnostic["max_J"])
            comparisons.add("boundary_sample_time_s", row["boundary_sample_time_s"], instant + config["dt"], atol=1e-14)
            comparisons.add("sum_phase_strain_energies", row["matrix_strain_energy_J"] + row["yarn_strain_energy_J"], diagnostic["strain_energy"], atol=1e-10)
            for boundary, name, selected in zip(boundaries, names, masks):
                target, velocity = motion(instant, boundary)
                components = np.ones(3, dtype=bool) if boundary["components"] is None else np.asarray(boundary["components"], dtype=bool)
                selected_count = int(selected.sum())
                report = boundary_report[name]
                report["samples"] = selected_count
                moving_required = "displacement" in boundary and loading != "transverse-bending"
                checks["required_moving_samples_exist"] &= not moving_required or selected_count > 0
                peak_velocity = np.abs(np.asarray(boundary.get("displacement", (0.,) * 3))) * np.pi / (2 * LOAD_TIME)
                position_limit = np.maximum(1e-10, 1.1 * peak_velocity * config["dt"])
                velocity_limit = max(1e-10, 1e-8 * float(peak_velocity.max()))
                for axis, label in enumerate("xyz"):
                    comparisons.add(f"{name}_prescribed_displacement_{label}_m", row[f"{name}_prescribed_displacement_{label}_m"], target[axis])
                    comparisons.add(f"{name}_prescribed_velocity_{label}_m_per_s", row[f"{name}_prescribed_velocity_{label}_m_per_s"], velocity[axis], atol=1e-9)
                comparisons.add(f"{name}_interior_particles", row[f"{name}_interior_particles"], selected_count, rtol=0, atol=0)
                if selected_count:
                    difference = np.abs(change[selected] - target)
                    position_error = float(difference[:, components].max())
                    report["maximum_position_error_m"] = max(report["maximum_position_error_m"], position_error)
                    checks["constrained_positions_within_time_step_error"] &= bool(np.all(difference[:, components] <= position_limit[components]))
                    reported_velocity_error = row[f"{name}_interior_velocity_error_m_per_s"]
                    checks["constrained_reported_velocities_match"] &= reported_velocity_error is not None and 0 <= reported_velocity_error <= velocity_limit
                    if reported_velocity_error is not None:
                        report["maximum_reported_velocity_error_m_per_s"] = max(report["maximum_reported_velocity_error_m_per_s"], reported_velocity_error)
                    if components.all():
                        checks["constrained_reported_velocities_match"] &= bool(np.max(np.abs(speeds[index][selected] - np.linalg.norm(velocity))) <= velocity_limit)
                    if index == len(time) - 1:
                        final_error = float(np.abs((state["v"][selected] - velocity)[:, components]).max())
                        checks["constrained_final_velocities_independent"] &= final_error <= velocity_limit
                        comparisons.add(f"final_{name}_interior_velocity_error_m_per_s", reported_velocity_error, final_error, atol=1e-10)
                        final_grip_motion[name] = change[selected].mean(axis=0).tolist()
                    low = np.floor(current[selected] / dx - .5) * dx
                    complete = np.all((low >= np.asarray(boundary["lower"]) + target - 1e-14) & (low + 2 * dx <= np.asarray(boundary["upper"]) + target + 1e-14), axis=1)
                    complete_count = int(complete.sum())
                    report["full_support_at_all_saved_frames"] &= complete_count == selected_count
                    checks["required_moving_samples_keep_complete_support"] &= not moving_required or complete_count == selected_count
                else:
                    position_error, complete_count = None, 0
                    checks["constrained_reported_velocities_match"] &= row[f"{name}_interior_velocity_error_m_per_s"] is None
                comparisons.add(f"{name}_interior_position_error_m", row[f"{name}_interior_position_error_m"], position_error)
                comparisons.add(f"{name}_interior_current_full_support_particles", row[f"{name}_interior_current_full_support_particles"], complete_count, rtol=0, atol=0)
        checks["finite_saved_particle_fields"] = finite_fields
        checks["zero_saved_damage"] = zero_damage
    finally:
        for array in mapped:
            array._mmap.close()
    for report in boundary_report.values():
        report["exact_particle_tracking_available"] = report["samples"] > 0
        if not report["samples"]:
            report["maximum_position_error_m"] = None
            report["maximum_reported_velocity_error_m_per_s"] = None
            report["full_support_at_all_saved_frames"] = None
    if loading in LOADS[:3]:
        for boundary, name in zip(boundaries, names):
            target = np.asarray(boundary["displacement"])
            limit = np.maximum(1e-10, 1.1 * np.abs(target) * np.pi / (2 * LOAD_TIME) * config["dt"])
            checks[f"{name}_actual_final_target_reached"] = name in final_grip_motion and bool(np.all(np.abs(np.asarray(final_grip_motion[name]) - target) <= limit))
    elif loading == "transverse-compression":
        checks["actual_reference_surface_closure"] = bool(1e-6 < closure[-1] < 1e-4)
    else:
        checks["actual_center_deflection_at_least_point_one_mm"] = bool(max(central_deflection) >= 1e-4)
    checks["independent_observables_match"] = all(comparisons.matches.values())
    return {
        "case": summary["case"], "passed": all(bool(value) for value in checks.values()),
        "checks": {name: bool(value) for name, value in checks.items()},
        "sources": {name: file_hash(folder / name, text=True) for name in SOURCE_NAMES},
        "static_fields": {"filename": "particle_fields.npz", "sha256": file_hash(folder / "particle_fields.npz"), "fields": field_descriptors},
        "geometry_orientation": q_descriptor,
        "other_sources": {"status.json": file_hash(folder / "status.json", text=True), "final_state_0.npz": file_hash(folder / "final_state_0.npz")},
        "saved_frames": len(time), "particles": count,
        "grid_cells_across_yarn_height": geometry["yarn_height_m"] / dx,
        "yarn_volume_fraction": sample.metadata["yarn_volume_fraction"],
        "min_J": min(row["min_J"] for row in diagnostics),
        "maximum_relative_mass_drift": float(np.max(np.abs(saved_mass / saved_mass[0] - 1))),
        "final_independent_yarn_maximum_green_strain_norm": float(strain_norm[sample.phase == 1].max()),
        "maximum_saved_yarn_green_strain_norm": max(row["yarn_maximum_green_strain_norm"] for row in rows),
        "maximum_allowed_yarn_green_strain_norm": YARN_STRAIN_LIMIT,
        "final_independent_strain_energy_J": total_energy,
        "final_reference_section_resultants": {
            "force_N": {normal: section_forces[axis].tolist() for axis, normal in enumerate("xyz")},
            "x_section_moment_z_Nm": reference_moment_z,
            "measure": "P=tau*F^-T integrated over reference slabs; moment uses reference Y, not a current-configuration angular-momentum balance",
        },
        "boundary_tracking": boundary_report,
        "response": {"saved_reference_surface_closure_m": closure, "saved_center_deflection_m": central_deflection, "final_grip_mean_displacement_m": final_grip_motion},
        "maximum_observable_reconstruction_error": comparisons.errors,
        "scope": "NumPy reconstruction of initial geometry, per-particle labels and reference axes, saved coordinate motion and final elastic state. Saved strain-history limits use observer records; only final strain and energy are independently reconstructed because intermediate deformation gradients are not archived.",
        "interpretation": "Fully bonded elastic yarn and matrix under dynamic prescribed loading; no Z binders, debonding, damage, calibration or convergence claim. Yarn volume fractions differ between architectures. Thin support/loading patches with no full-stencil particle samples have no exact particle-clamping claim. Fixture reactions are next-step diagnostics, not a quasi-static material curve.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    (args.folder / "interpretation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failed = [key for key, value in result["checks"].items() if not value]
    print(json.dumps({"case": result["case"], "passed": result["passed"], "failed_checks": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
