import copy

import numpy as np
import pytest

from tools.review_material_engineering import CERAMIC_PARAMETERS, check_ceramic_wide


@pytest.fixture
def indentation_record():
    lower, upper = np.array((.035, .075, .035)), np.array((.215, .1, .215))
    lengths = upper - lower
    dx = .25 / 96
    counts = np.ceil(lengths / dx).astype(int)
    spacing = lengths / counts
    axes = [lower[i] + (np.arange(counts[i]) + .5) * spacing[i] for i in range(3)]
    reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    top = reference[:, 1] == axes[1][-1]
    radial2 = (reference[:, 0] - .125)**2 + (reference[:, 2] - .125)**2
    central = top & (radial2 <= .004**2)
    local = radial2 <= .030**2
    surface = top & local
    volume = np.full(len(reference), np.prod(spacing))
    state = {"volume": volume, "mass": volume * 2500}
    positions = np.tile(reference, (3, 1, 1))
    positions[1:, central, 1] -= .002
    damage = np.zeros((3, len(reference)))
    damage[1:, central] = .8
    sphere = {"center": [.125, .1375, .125], "radius": .0375, "displacement": [0, -.004, 0], "duration": .0003}
    backing = {"lower": [0, 0, 0], "upper": [.25, .075, .25], "velocity": [0, 0, 0], "components": None, "start": 0, "end": None}
    config = {"domain_size": .25, "n_grid": 96, "dt": 5e-8, "material": "jh2", "failure_mode": "retain", "boundaries": [backing, sphere]}
    config.update(dim=3, gravity=[0., 0., 0.], boundary_cells=0)
    definition = {
        "lower": lower, "upper": upper, "particles_per_cell_per_axis": 1,
        "rigid_sphere": copy.deepcopy(sphere), "geometry_scale": 1.,
        "material_parameters_SI": dict(CERAMIC_PARAMETERS),
        "particle_spacings_m": spacing, "particle_spacing_m": spacing.min(),
        "particle_counts_per_axis": counts, "central_surface_radius_m": .004,
        "local_region_radius_m": .030, "central_surface_particles": central.sum(),
        "local_region_particles": local.sum(), "local_surface_particles": surface.sum(),
        "sphere_diameter_to_plate_width_ratio": .075 / .18,
        "plate_thickness_in_grid_cells": .025 / dx,
        "nominal_contact_radius_m": np.sqrt(.0375 * .004),
        "nominal_contact_radius_in_grid_cells": np.sqrt(.0375 * .004) / dx,
        "nominal_contact_radius_to_plate_half_width_ratio": np.sqrt(.0375 * .004) / .09,
        "maximum_nominal_indentation_m": .004,
    }
    time = np.array([0., .00015, .0003])
    observations = []
    for index, instant in enumerate(time):
        loaded = index > 0
        center = np.array([.125, .1335 if index == 1 else .1375, .125])
        overlap = np.maximum(.0375 - np.linalg.norm(positions[index] - center, axis=1), 0)
        observations.append({
            "nominal_indentation_m": .004 if index == 1 else 0.,
            "indenter_speed_y_m_per_s": 0.,
            "central_surface_displacement_y_m": -.002 if loaded else 0.,
            "maximum_downward_displacement_m": .002 if loaded else 0.,
            "central_surface_max_damage": .8 if loaded else 0.,
            "local_surface_displacement_y_m": -.002 * central.sum() / surface.sum() if loaded else 0.,
            "local_region_maximum_downward_displacement_m": .002 if loaded else 0.,
            "local_region_max_damage": .8 if loaded else 0.,
            "local_region_mean_damage_reference_volume": .8 * central.sum() / local.sum() if loaded else 0.,
            "local_region_damaged_reference_volume_m3": volume[central].sum() if loaded else 0.,
            "indenter_particle_overlap_max_m": overlap.max(),
            "indenter_surface_particle_overlap_max_m": overlap[top].max(),
            "particle_centers_inside_indenter": np.count_nonzero(overlap),
            "damaged_reference_volume_fraction": central.sum() / len(reference) if loaded else 0.,
            "indenter_reaction_y_N": 1. if index == 1 else 0.,
            "boundary_sample_time_s": instant + config["dt"],
        })
    return definition, config, time, observations, state, positions, damage, dict(CERAMIC_PARAMETERS)


def inspect_record(record):
    result = {"checks": {}}
    check_ceramic_wide(result, *record)
    return result


def test_fixed_region_displacement_and_damage_have_analytic_values(indentation_record):
    result = inspect_record(indentation_record)
    assert all(result["checks"].values()), {key: value for key, value in result["checks"].items() if not value}
    assert result["ceramic_indentation"]["maximum_actual_central_indentation_m"] == pytest.approx(.002)


def test_changed_geometry_and_one_tampered_curve_cannot_pass(indentation_record):
    definition, config, _, observations, *_ = indentation_record
    config["boundaries"][1]["radius"] = .075
    definition["local_region_radius_m"] = .060
    observations[1]["local_region_mean_damage_reference_volume"] *= 2
    result = inspect_record(indentation_record)
    assert not result["checks"]["specified_smaller_sphere"]
    assert not result["checks"]["fixed_physical_observation_regions"]
    assert not result["checks"]["reconstructed_local_region_mean_damage_reference_volume"]


def test_static_positions_and_severe_overlap_are_detected(indentation_record):
    positions = indentation_record[5]
    positions[:] = positions[0].copy()
    result = inspect_record(indentation_record)
    assert not result["checks"]["actual_central_indentation_at_least_one_mm"]
    assert not result["checks"]["reconstructed_central_surface_displacement_y_m"]
    positions[1, 0] = (.125, .1335, .125)
    result = inspect_record(indentation_record)
    assert not result["checks"]["particle_sphere_overlap_within_grid_support"]


def test_density_and_duplicated_lattice_points_cannot_pass(indentation_record):
    state, positions = indentation_record[4:6]
    state["mass"][0] *= 2
    positions[0, 1] = positions[0, 0]
    result = inspect_record(indentation_record)
    assert not result["checks"]["mass_matches_material_density"]
    assert not result["checks"]["correct_regular_reference_lattice"]


def test_extra_loading_and_changed_jh2_material_cannot_pass(indentation_record):
    definition, config, *_, material = indentation_record
    config["gravity"][1] = -9.81
    material["D1"] *= 2
    result = inspect_record(indentation_record)
    assert not result["checks"]["specified_domain_and_no_added_loading"]
    assert not result["checks"]["unchanged_jh2_parameters"]
    material["D1"] = CERAMIC_PARAMETERS["D1"]
    definition["material_parameters_SI"]["A"] = .1
    result = inspect_record(indentation_record)
    assert not result["checks"]["unchanged_jh2_parameters"]


def test_annular_damage_does_not_require_failure_of_the_compressed_centre(indentation_record):
    observations, reference, damage = indentation_record[3], indentation_record[5][0], indentation_record[6]
    top = reference[:, 1] == reference[:, 1].max()
    radial2 = (reference[:, 0] - .125)**2 + (reference[:, 2] - .125)**2
    central = top & (radial2 <= .004**2)
    annulus = np.flatnonzero(top & (radial2 >= .005**2) & (radial2 <= .009**2))
    assert len(annulus) >= central.sum()
    damage[1:, central] = 0.
    damage[1:, annulus[:central.sum()]] = .8
    for row in observations[1:]:
        row["central_surface_max_damage"] = 0.
    result = inspect_record(indentation_record)
    assert all(result["checks"].values()), {key: value for key, value in result["checks"].items() if not value}
