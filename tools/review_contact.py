"""Summarize completed spherical-contact examples without loading trajectories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def boolean_column(rows, key):
    return np.asarray([row[key].lower() == "true" for row in rows])


def completed(folder):
    path = folder / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No completed summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError(f"Incomplete run: {folder}")
    return summary


def sphere_review(folder):
    summary = completed(folder)
    rows = read_rows(folder / "observables.csv")
    diagnostics = read_rows(folder / "diagnostics.csv")
    time = column(rows, "time")
    load = column(rows, "indenter_reaction_y_N")
    transverse = np.hypot(
        column(rows, "indenter_reaction_x_N"), column(rows, "indenter_reaction_z_N")
    )
    hertz = column(rows, "Hertz_reference_load_N")
    radius = column(rows, "Hertz_reference_contact_radius_m")
    indentation = column(rows, "nominal_indentation_m")
    config = summary["configs"][0]
    dx = config["domain_size"] / config["n_grid"]
    definition = summary["definition"]
    sphere = definition["rigid_sphere"]
    peak_time = sphere["duration"] / 2
    peak_speed = np.pi * np.linalg.norm(sphere["displacement"]) / sphere["duration"]
    reference_volume = np.prod(np.asarray(definition["upper"]) - definition["lower"])
    density = summary["initial"][0]["mass"] / reference_volume
    material = summary["materials"][0]
    wave_speed = np.sqrt((material["lam"] + 2 * material["mu"]) / density)
    loading = time <= peak_time
    unloading = time > peak_time
    resolved = radius >= 4 * dx
    comparison = loading & resolved & (hertz >= 1.0)
    tolerance = 1e-9 * max(1.0, float(np.abs(load).max()))
    separated = unloading & (load <= tolerance)
    i = int(np.argmax(load))
    j = int(np.argmax(indentation))
    relative = (load[comparison] - hertz[comparison]) / hertz[comparison]
    energy = column(diagnostics, "kinetic_energy")
    strain_energy = column(diagnostics, "strain_energy")
    return {
        "case": summary["case"],
        "particles": sum(summary["particles"]),
        "steps": summary["steps"],
        "simulation_time_s": summary["simulation_time"],
        "minimum_J": summary["min_J"],
        "peak_prescribed_speed_m_per_s": float(peak_speed),
        "peak_prescribed_speed_to_initial_wave_speed": float(peak_speed / wave_speed),
        "peak_prescribed_movement_per_step_in_grid_cells": float(peak_speed * config["dt"] / dx),
        "force_sampling": "next-step grid impulse / dt at each saved output frame",
        "minimum_reaction_y_N": float(load.min()),
        "maximum_transverse_reaction_N": float(transverse.max()),
        "tensile_reaction_frames": int(np.sum(load < -tolerance)),
        "maximum_reaction_y_N": float(load[i]),
        "maximum_reaction_frame_time_s": float(time[i]),
        "maximum_nominal_indentation_m": float(indentation[j]),
        "reaction_at_maximum_indentation_N": float(load[j]),
        "Hertz_at_maximum_indentation_N": float(hertz[j]),
        "relative_difference_at_maximum_indentation": float((load[j] - hertz[j]) / hertz[j]),
        "maximum_reference_contact_radius_grid_cells": float((radius / dx).max()),
        "loading_frames_with_radius_at_least_4_grid_cells": int(comparison.sum()),
        "loading_median_relative_difference_in_that_interval": float(np.median(relative))
        if relative.size
        else None,
        "loading_maximum_absolute_relative_difference_in_that_interval": float(
            np.abs(relative).max()
        )
        if relative.size
        else None,
        "first_sampled_separation_time_s": float(time[separated][0]) if np.any(separated) else None,
        "positive_nominal_indentation_after_separation": bool(
            np.any(separated & (indentation > dx / 10))
        ),
        "maximum_kinetic_energy_J": float(energy.max()),
        "maximum_strain_energy_J": float(strain_energy.max()),
        "kinetic_to_strain_energy_at_maximum_indentation": float(energy[j] / strain_energy[j])
        if strain_energy[j] > 0
        else None,
        "assessment": "finite-block dynamic contact demonstration; Hertz is a reference curve, not a verified error bound",
    }


def law_review(folder):
    parameters = json.loads((folder / "jkr_law_parameters.json").read_text(encoding="utf-8"))
    rows = read_rows(folder / "jkr_law_cycle.csv")
    delta = column(rows, "nominal_indentation_m")
    radius = column(rows, "contact_radius_m")
    force = column(rows, "force_N")
    attached = column(rows, "attached") > 0.5
    R, E, work = parameters["radius_m"], parameters["modulus_Pa"], parameters["work_J_per_m2"]
    critical_radius = (np.pi * work * R**2 / (8 * E)) ** (1 / 3)
    critical_indentation = -3 * critical_radius**2 / R
    reconstructed_indentation = radius**2 / R - np.sqrt(2 * np.pi * work * radius / E)
    reconstructed_force = 4 * E * radius**3 / (3 * R) - np.sqrt(
        8 * np.pi * work * E * radius**3
    )
    transitions = np.diff(attached.astype(int))
    hertz_radius = np.sqrt(R * np.maximum(delta, 0.0))
    hertz_force = 4 * E * hertz_radius**3 / (3 * R)
    np.savetxt(
        folder / "law_comparison.csv",
        np.column_stack((delta, radius, force, attached.astype(int), hertz_radius, hertz_force)),
        delimiter=",",
        header="nominal_indentation_m,JKR_contact_radius_m,JKR_force_N,attached,Hertz_contact_radius_m,Hertz_force_N",
        comments="",
    )
    detach_indices = np.flatnonzero(transitions == -1)
    return {
        "samples": len(rows),
        "parameters": parameters,
        "maximum_attached_indentation_equation_residual_m": float(
            np.abs(reconstructed_indentation[attached] - delta[attached]).max()
        ),
        "maximum_force_equation_residual_N": float(np.abs(reconstructed_force - force).max()),
        "attachment_events": int(np.sum(transitions == 1)),
        "detachment_events": int(np.sum(transitions == -1)),
        "displacement_control_critical_radius_m": float(critical_radius),
        "displacement_control_critical_indentation_m": float(critical_indentation),
        "displacement_control_force_at_instability_N": float(-5 / 6 * np.pi * work * R),
        "load_control_minimum_force_N": float(-1.5 * np.pi * work * R),
        "minimum_sampled_force_N": float(force.min()),
        "last_attached_indentation_before_release_m": float(delta[detach_indices[0]])
        if len(detach_indices)
        else None,
        "first_separated_indentation_after_release_m": float(delta[detach_indices[0] + 1])
        if len(detach_indices)
        else None,
        "assessment": "algebraic consistency of exported reduced-law history; independent root checks are in the unit tests",
    }


def footprint_review(folder):
    summary = completed(folder)
    rows = read_rows(folder / "observables.csv")
    force = column(rows, "prescribed_law_force_N")
    assigned = -column(rows, "applied_force_y_N")
    attached = boolean_column(rows, "contact_attached")
    resolved = boolean_column(rows, "footprint_resolved")
    loaded = column(rows, "loaded_surface_particles")
    radius = column(rows, "prescribed_contact_radius_m")
    config = summary["configs"][0]
    grid_radius = radius * config["n_grid"] / config["domain_size"]
    displacement = column(rows, "central_surface_displacement_y_m")
    return {
        "case": summary["case"],
        "particles": sum(summary["particles"]),
        "steps": summary["steps"],
        "simulation_time_s": summary["simulation_time"],
        "minimum_J": summary["min_J"],
        "minimum_prescribed_force_N": float(force.min()),
        "maximum_prescribed_force_N": float(force.max()),
        "maximum_force_assignment_error_N": float(np.abs(assigned - force).max()),
        "relative_force_assignment_error_at_peak": float(
            np.abs(assigned - force).max() / np.abs(force).max()
        ),
        "attached_frames": int(attached.sum()),
        "tensile_load_frames": int(np.sum(force < 0)),
        "unresolved_attached_frames": int(np.sum(attached & ~resolved)),
        "minimum_active_contact_radius_grid_cells": float(grid_radius[attached].min())
        if attached.any()
        else None,
        "maximum_active_contact_radius_grid_cells": float(grid_radius.max()),
        "active_frames_with_radius_below_2_grid_cells": int(np.sum(attached & (grid_radius < 2))),
        "minimum_loaded_particles_when_attached": int(loaded[attached].min())
        if attached.any()
        else 0,
        "minimum_central_surface_displacement_y_m": float(displacement.min()),
        "maximum_central_surface_displacement_y_m": float(displacement.max()),
        "maximum_force_while_detached_N": float(np.abs(assigned[~attached]).max())
        if (~attached).any()
        else None,
        "assessment": "prescribed pressure footprint with a computed MPM response; no geometric adhesive closure",
        "law_cycle": law_review(folder),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("outputs/contact-models"))
    args = parser.parse_args()
    # Require both complete runs before creating the combined acceptance file.
    completed(args.directory / "sphere-contact")
    completed(args.directory / "jkr-footprint")
    result = {
        "sphere_contact": sphere_review(args.directory / "sphere-contact"),
        "jkr_footprint": footprint_review(args.directory / "jkr-footprint"),
    }
    path = args.directory / "contact-review.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
