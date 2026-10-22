"""Public-benchmark-inspired contact and fluid–structure examples."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, PlaneWall, block, fixed_box, fluid, neo_hookean
from .scenarios import Case, solid_case, tank_walls


REFERENCES = {
    "elastic-gate": [
        "https://www.sphinxsys.org/html/examples/example2_elasticGate.html",
        "https://lsdyna.ansys.com/dam5/",
    ],
    "floating-block": ["https://github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases"],
    "rubber-ball": ["https://www.dynaexamples.com/introduction/intro-by-j.-day/misc/ball"],
    "ball-beam": ["https://www.dynasupport.com/manuals/additional/ls-dyna-examples-manual"],
    "large-bending": ["https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html"],
    "large-twist": ["https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html"],
}


def select(state, keep):
    return jax.tree.map(lambda a: a[keep] if a.ndim else a, state)


def sphere(config, center, radius, *, ppc, density, material):
    center = np.asarray(center)
    state = block(
        config,
        center - radius,
        center + radius,
        particles_per_cell=ppc,
        density=density,
        material=material,
        dtype=jnp.float64,
    )
    keep = np.sum((np.asarray(state.x) - center) ** 2, axis=1) <= radius**2
    state = select(state, keep)
    volume = jnp.full(len(state.x), 4 * np.pi * radius**3 / (3 * len(state.x)))
    return state._replace(volume=volume, mass=density * volume)


def hydrostatic(state, surface, bulk, density=1000.0):
    depth = jnp.maximum(surface - state.x[:, 1], 0.0)
    J = jnp.sqrt(1 - 2 * density * 9.81 * depth / bulk)
    volume = state.volume / J
    return state._replace(
        F=J[:, None, None] ** (1 / 3) * jnp.eye(3),
        volume=volume,
        mass=density * volume,
        history=state.history._replace(density=density / J),
    )


def gate(grid, ppc):
    solid, liquid = neo_hookean(5e5, 0.3), fluid(2e5)
    c = np.sqrt((solid.bulk + 4 * solid.mu / 3) / 1100.0)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="fluid",
        boundary_cells=0,
        boundaries=tank_walls(),
    )
    scfg = replace(
        cfg,
        material="neo-hookean",
        boundaries=tank_walls() + (fixed_box((0.0, 0.65, 0.0), (1.0,) * 3),),
    )
    water = block(
        cfg, (0.125, 0.125, 0.25), (0.5, 0.6, 0.75), particles_per_cell=ppc, dtype=jnp.float64
    )
    water = hydrostatic(water, 0.6, liquid.bulk)
    panel = block(
        scfg,
        (0.5, 0.125, 0.25),
        (0.6, 0.7, 0.75),
        particles_per_cell=ppc,
        density=1100.0,
        dtype=jnp.float64,
    )
    reference = np.asarray(panel.x)
    tip = reference[:, 1] < 0.18

    def observe(states, time):
        displacement = np.asarray(states[1].x) - reference
        return {
            "gate_tip_displacement_x": float(displacement[tip, 0].mean()),
            "gate_tip_displacement_y": float(displacement[tip, 1].mean()),
            "fluid_front_x": float(states[0].x[:, 0].max()),
        }

    return Case(
        (water, panel),
        (liquid, solid),
        (cfg, scfg),
        0.35,
        {
            "purpose": "top-clamped elastic water gate",
            "sources": REFERENCES["elastic-gate"],
            "adaptation": "3D extruded geometry; Neo-Hookean solid; compressible inviscid MPM contact",
            "particles_per_cell_per_axis": ppc,
            "initial_hydrostatic_equilibrium": True,
            "gate_lower": (0.5, 0.125, 0.25),
            "gate_upper": (0.6, 0.7, 0.75),
        },
        observer=observe,
    )


def floating(grid, ppc):
    solid, liquid = neo_hookean(2e5, 0.3), fluid(2e5)
    c = np.sqrt((solid.bulk + 4 * solid.mu / 3) / 500.0)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="fluid",
        boundary_cells=0,
        boundaries=tank_walls(),
    )
    scfg = replace(cfg, material="neo-hookean")
    water = block(
        cfg, (0.125, 0.125, 0.25), (0.875, 0.45, 0.75), particles_per_cell=ppc, dtype=jnp.float64
    )
    lo, hi = np.array([0.4, 0.4, 0.4]), np.array([0.6, 0.6, 0.6])
    keep = ~np.all((np.asarray(water.x) >= lo) & (np.asarray(water.x) <= hi), axis=1)
    water = hydrostatic(select(water, keep), 0.45, liquid.bulk)
    body = block(scfg, lo, hi, particles_per_cell=ppc, density=500.0, dtype=jnp.float64)

    def observe(states, time):
        s = states[1]
        return {
            "floating_center_y": float(jnp.sum(s.mass * s.x[:, 1]) / s.mass.sum()),
            "floating_mean_velocity_y": float(jnp.sum(s.mass * s.v[:, 1]) / s.mass.sum()),
            "water_maximum_height": float(states[0].x[:, 1].max()),
        }

    return Case(
        (water, body),
        (liquid, solid),
        (cfg, scfg),
        0.5,
        {
            "purpose": "floating block heave release",
            "sources": REFERENCES["floating-block"],
            "adaptation": "deformable block, finite tank, initial heave imbalance; no wavemaker",
            "particles_per_cell_per_axis": ppc,
            "initial_hydrostatic_equilibrium": True,
            "density_ratio": 0.5,
            "initial_submerged_fraction": 0.25,
        },
        observer=observe,
    )


def ball(scene, grid, ppc):
    material = neo_hookean(5e4, 0.2)
    c = np.sqrt((material.bulk + 4 * material.mu / 3) / 800.0)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="neo-hookean",
        boundary_cells=0,
        boundaries=(PlaneWall(1, 0.125),),
    )
    state = sphere(
        cfg,
        (0.5, 0.6, 0.5),
        0.1 if scene == "ball-beam" else 0.15,
        ppc=ppc,
        density=800.0,
        material=material,
    )
    states, materials, configs = (state,), (material,), (cfg,)
    if scene == "ball-beam":
        scfg = replace(
            cfg,
            boundaries=(
                fixed_box((0.0,) * 3, (0.25, 1.0, 1.0)),
                fixed_box((0.75, 0.0, 0.0), (1.0,) * 3),
            ),
        )
        beam = block(
            scfg,
            (0.2, 0.25, 0.3),
            (0.8, 0.35, 0.7),
            particles_per_cell=ppc,
            density=1000.0,
            dtype=jnp.float64,
        )
        states += (beam,)
        materials += (neo_hookean(2e4, 0.2),)
        configs += (scfg,)

    def observe(states, time):
        values = {
            "ball_center_y": float(
                jnp.sum(states[0].mass * states[0].x[:, 1]) / states[0].mass.sum()
            ),
            "ball_lowest_y": float(states[0].x[:, 1].min()),
        }
        if len(states) == 2:
            values["beam_lowest_y"] = float(states[1].x[:, 1].min())
        return values

    return Case(
        states,
        materials,
        configs,
        0.45,
        {
            "purpose": scene,
            "sources": REFERENCES[scene],
            "particles_per_cell_per_axis": ppc,
            "adaptation": "filled elastic sphere and solid beam; no pressurized shell or airbag",
        },
        observer=observe,
    )


def setup(scene, grid, ppc):
    if scene == "elastic-gate":
        return gate(grid, ppc)
    if scene == "floating-block":
        return floating(grid, ppc)
    if scene in ("rubber-ball", "ball-beam"):
        return ball(scene, grid, ppc)
    case = solid_case(
        "cantilever" if scene == "large-bending" else "torsion", "neo-hookean", grid, ppc
    )
    factor = 20.0 if scene == "large-bending" else 15.0
    case.states = (case.states[0]._replace(v=case.states[0].v * factor),)
    case.duration = 0.35
    case.description.update(
        purpose=scene,
        sources=REFERENCES[scene],
        initial_velocity_multiplier=factor,
        adaptation="dynamic release with the package's compressible Neo-Hookean law",
    )
    return case
