"""Perforated structures, cushioning and an accelerated liquid tank in SI units."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, PlaneWall, block, fixed_box, fluid, neo_hookean
from .benchmarks import hydrostatic, select
from .scenarios import Case, tank_walls


def wave_speed(material, density):
    return float(np.sqrt((material.bulk + 4 * material.mu / 3) / density))


def perforated_panel(grid, ppc):
    material = neo_hookean(1e5, 0.3)
    density = 1100.0
    boundaries = (
        fixed_box((0.0,) * 3, (0.21, 1.0, 1.0)),
        fixed_box((0.79, 0.0, 0.0), (1.0,) * 3),
        fixed_box((0.0,) * 3, (1.0, 1.0, 0.21)),
        fixed_box((0.0, 0.0, 0.79), (1.0,) * 3),
    )
    config = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / wave_speed(material, density),
        material="neo-hookean",
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=boundaries,
    )
    state = block(
        config,
        (0.16, 0.46, 0.16),
        (0.84, 0.54, 0.84),
        particles_per_cell=ppc,
        density=density,
        dtype=jnp.float64,
    )
    points = np.asarray(state.x)
    keep = np.ones(len(points), dtype=bool)
    centers = [(x, z) for x in (0.32, 0.5, 0.68) for z in (0.32, 0.5, 0.68)]
    for x, z in centers:
        keep &= (points[:, 0] - x) ** 2 + (points[:, 2] - z) ** 2 >= 0.05**2
    state = select(state, keep)
    reference = np.asarray(state.x)
    middle = (np.abs(reference[:, 0] - 0.5) < 0.12) & (np.abs(reference[:, 2] - 0.5) < 0.12)
    top = jnp.isclose(state.x[:, 1], state.x[:, 1].max(), atol=1e-10)
    area = float(state.volume.sum()) / 0.08
    tributary_area = area / jnp.sum(top)

    def force(states, time):
        phase = jnp.clip(time / 0.08, 0.0, 1.0)
        pressure = 3000.0 * jnp.sin(jnp.pi * phase) ** 2
        return ((pressure * tributary_area * top)[:, None] * jnp.array([0.0, -1.0, 0.0]),)

    def observe(states, time):
        displacement = np.asarray(states[0].x) - reference
        return {
            "central_displacement_y": float(displacement[middle, 1].mean()),
            "maximum_displacement": float(np.linalg.norm(displacement, axis=1).max()),
            "nominal_pressure": float(3000.0 * np.sin(np.pi * np.clip(time / 0.08, 0.0, 1.0)) ** 2),
        }

    return Case(
        (state,),
        (material,),
        (config,),
        0.35,
        {
            "purpose": "clamped perforated panel under a smooth pressure pulse",
            "particles_per_cell_per_axis": ppc,
            "holes": centers,
            "hole_radius": 0.05,
            "thickness": 0.08,
            "loaded_reference_area": area,
            "traction": "fixed negative-y dead load on the reference upper face",
            "pressure_peak_Pa": 3000.0,
            "pulse_duration_s": 0.08,
        },
        forces=force,
        observer=observe,
    )


def cellular_cushion(grid, ppc):
    rubber, platen = neo_hookean(8e4, 0.28), neo_hookean(8e5, 0.25)
    c = max(wave_speed(rubber, 1000.0), wave_speed(platen, 1200.0))
    config = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="neo-hookean",
        boundary_cells=0,
        boundaries=(fixed_box((0.0,) * 3, (1.0, 0.15, 1.0)),),
    )
    state = block(
        config,
        (0.2, 0.125, 0.2),
        (0.8, 0.4, 0.8),
        particles_per_cell=ppc,
        density=1000.0,
        dtype=jnp.float64,
    )
    points = np.asarray(state.x)
    keep = (points[:, 1] < 0.16) | (points[:, 1] > 0.365)
    centers = [(x, z) for x in (0.275, 0.425, 0.575, 0.725) for z in (0.275, 0.425, 0.575, 0.725)]
    for x, z in centers:
        keep |= (points[:, 0] - x) ** 2 + (points[:, 2] - z) ** 2 < 0.04**2
    state = select(state, keep)
    plate_config = replace(config, boundaries=(PlaneWall(1, 0.125),))
    plate = block(
        plate_config,
        (0.22, 0.58, 0.22),
        (0.78, 0.645, 0.78),
        particles_per_cell=ppc,
        density=1200.0,
        velocity=(0.0, -0.2, 0.0),
        dtype=jnp.float64,
    )
    initial = np.asarray(state.x)
    upper = initial[:, 1] > 0.365

    def observe(states, time):
        body = states[1]
        return {
            "platen_center_y": float(jnp.sum(body.mass * body.x[:, 1]) / body.mass.sum()),
            "platen_velocity_y": float(jnp.sum(body.mass * body.v[:, 1]) / body.mass.sum()),
            "cushion_upper_displacement_y": float(
                (np.asarray(states[0].x) - initial)[upper, 1].mean()
            ),
        }

    return Case(
        (state, plate),
        (rubber, platen),
        (config, plate_config),
        0.34,
        {
            "purpose": "flat-mass drop onto a cellular elastomer cushion",
            "particles_per_cell_per_axis": ppc,
            "column_centers": centers,
            "column_radius": 0.04,
            "skins": [[0.125, 0.16], [0.365, 0.4]],
            "contact_friction": 0.2,
            "initial_gap": 0.18,
            "scope": "elastic cushioning; no material fracture or strut self-contact",
        },
        observer=observe,
        friction=0.2,
    )


def accelerated_tank(scene, grid, ppc):
    liquid, solid = fluid(2e5), neo_hookean(1e5, 0.3)
    c = max(np.sqrt(200.0), wave_speed(solid, 1100.0))
    config = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="fluid",
        boundary_cells=0,
        boundaries=tank_walls(),
    )
    structure_config = replace(
        config,
        material="neo-hookean",
        boundaries=tank_walls() + (fixed_box((0.0,) * 3, (1.0, 0.16, 1.0)),),
    )
    water = block(
        config,
        (0.125, 0.125, 0.25),
        (0.875, 0.425, 0.75),
        particles_per_cell=ppc,
        density=1000.0,
        dtype=jnp.float64,
    )
    baffle_centers = (0.325, 0.5, 0.675)

    def baffle_mask(x):
        sheets = np.zeros(len(x), dtype=bool)
        for center in baffle_centers:
            sheets |= np.abs(x[:, 0] - center) < 0.025
        sheets &= (x[:, 1] <= 0.52) & (x[:, 2] >= 0.25) & (x[:, 2] <= 0.75)
        windows = (x[:, 1] > 0.24) & (x[:, 1] < 0.34) & (x[:, 2] > 0.4) & (x[:, 2] < 0.6)
        return sheets & ~windows

    states, materials, configs = (water,), (liquid,), (config,)
    reference = None
    if scene == "baffled-tank":
        water = select(water, ~baffle_mask(np.asarray(water.x)))
        baffles = block(
            structure_config,
            (0.3, 0.125, 0.25),
            (0.7, 0.52, 0.75),
            particles_per_cell=ppc,
            density=1100.0,
            dtype=jnp.float64,
        )
        baffles = select(baffles, baffle_mask(np.asarray(baffles.x)))
        reference = np.asarray(baffles.x)
        states, materials, configs = (water, baffles), (liquid, solid), (config, structure_config)
    states = (hydrostatic(states[0], 0.425, liquid.bulk),) + states[1:]

    def acceleration(time):
        ramp = 0.5 * (1 - jnp.cos(jnp.pi * jnp.clip(time / 0.1, 0.0, 1.0)))
        return 6.0 * ramp * jnp.sin(2 * jnp.pi * 1.5 * time)

    def force(states, time):
        direction = jnp.array([-acceleration(time), 0.0, 0.0])
        return tuple(state.mass[:, None] * direction for state in states)

    def observe(states, time):
        water = np.asarray(states[0].x)
        values = {
            "tank_acceleration_x": float(acceleration(time)),
            "free_surface_y": float(water[:, 1].max()),
            "water_center_x": float(
                jnp.sum(states[0].mass * states[0].x[:, 0]) / states[0].mass.sum()
            ),
        }
        if reference is not None:
            displacement = np.asarray(states[1].x) - reference
            values["maximum_baffle_displacement"] = float(
                np.linalg.norm(displacement, axis=1).max()
            )
        return values

    return Case(
        states,
        materials,
        configs,
        0.7,
        {
            "purpose": "liquid tank under horizontal harmonic base acceleration",
            "particles_per_cell_per_axis": ppc,
            "frame": "tank-fixed translating frame",
            "applied_inertial_force": "-mass * tank_acceleration on each body",
            "acceleration_peak_m_s2": 6.0,
            "frequency_Hz": 1.5,
            "ramp_s": 0.1,
            "water_height": 0.425,
            "initial_hydrostatic_equilibrium": True,
            "baffle_centers": baffle_centers if reference is not None else [],
            "baffle_thickness": 0.05,
            "baffle_window_y": [0.24, 0.34],
            "baffle_window_z": [0.4, 0.6],
        },
        forces=force,
        observer=observe,
    )


def setup(scene, grid, ppc):
    if scene == "perforated-panel":
        return perforated_panel(grid, ppc)
    if scene == "cellular-cushion":
        return cellular_cushion(grid, ppc)
    return accelerated_tank(scene, grid, ppc)
