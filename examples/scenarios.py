"""Three-dimensional mechanics problems in SI units."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mpm_jax import (
    Config,
    PlaneWall,
    VelocityBox,
    block,
    fixed_box,
    fluid,
    linear_elastic,
    neo_hookean,
    orthotropic,
    transverse_isotropic,
    blatz_ko,
    bilinear_plastic,
    thermal_plastic,
    steinberg_guinan,
    johnson_cook,
    jh2,
    hjc,
    rht,
)


ELASTIC = ("linear", "orthotropic", "transverse-isotropic", "neo-hookean", "blatz-ko")
PLASTIC = (
    "bilinear",
    "isotropic-plastic",
    "thermal-plastic",
    "steinberg-guinan",
    "johnson-cook",
    "jh2",
    "hjc",
    "rht",
)
SCENES = (
    "wave",
    "cantilever",
    "torsion",
    "shear",
    "compression",
    "surface-load",
    "head-on",
    "oblique",
    "sliding",
    "rebound",
    "dam-break",
    "sloshing",
    "fsi",
    "water-entry",
    "elastic-gate",
    "floating-block",
    "rubber-ball",
    "ball-beam",
    "large-bending",
    "large-twist",
    "perforated-panel",
    "cellular-cushion",
    "baffled-tank",
    "unbaffled-tank",
)


@dataclass
class Case:
    states: tuple
    materials: tuple
    configs: tuple
    duration: float
    description: dict
    forces: object = None
    observer: object = None
    friction: float = 0.0


def material_model(name):
    a = np.pi / 6
    rotation = jnp.array(
        [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    )
    factories = {
        "linear": lambda: linear_elastic(2e4, 0.2),
        "orthotropic": lambda: orthotropic(
            (8e4, 2e4, 3e4), (0.2, 0.15, 0.25), (1e4, 1.5e4, 8e3), rotation
        ),
        "transverse-isotropic": lambda: transverse_isotropic(8e4, 2e4, 0.25, 0.3, 1.2e4, rotation),
        "neo-hookean": lambda: neo_hookean(2e4, 0.2),
        "blatz-ko": lambda: blatz_ko(8e3),
        "bilinear": lambda: bilinear_plastic(isotropic_fraction=0.5),
        "isotropic-plastic": bilinear_plastic,
        "thermal-plastic": thermal_plastic,
        "steinberg-guinan": steinberg_guinan,
        "johnson-cook": johnson_cook,
        "jh2": jh2,
        "hjc": hjc,
        "rht": rht,
    }
    material = factories[name]()
    density = float(getattr(material, "density", 1000.0))
    kind = "bilinear" if name == "isotropic-plastic" else name
    if hasattr(material, "stiffness"):
        modulus = float(np.linalg.eigvalsh(material.stiffness).max())
    elif name == "thermal-plastic":
        modulus = float(max(material.young) / (1 - 2 * max(material.poisson)))
    elif name == "blatz-ko":
        modulus = material.shear * (2 + 2 * material.poisson / (1 - 2 * material.poisson))
    else:
        # The brittle EOS coefficients can exceed the initial elastic bulk modulus.
        bulk = float(getattr(material, "bulk", getattr(material, "K1", 200e9)))
        shear = float(getattr(material, "mu", getattr(material, "shear", 100e9)))
        modulus = bulk + 4 * shear / 3
    return material, density, kind, float(np.sqrt(modulus / density))


def tank_walls(friction=0.0):
    return tuple(
        PlaneWall(axis, value, side, friction=friction)
        for axis in range(3)
        for value, side in ((0.125, "lower"), (0.875, "upper"))
    )


def solid_case(scene, model, grid, ppc):
    material, density, kind, c = material_model(model)
    boundaries = ()
    forces = None
    duration = 0.15
    lower, upper = (0.15, 0.4, 0.35), (0.8, 0.6, 0.65)
    if scene in ("cantilever", "torsion"):
        boundaries = (fixed_box((0.0, 0.0, 0.0), (0.2, 1.0, 1.0)),)
    elif scene == "shear":
        lower, upper = (0.3, 0.25, 0.3), (0.7, 0.75, 0.7)
        boundaries = (
            fixed_box((0.0, 0.0, 0.0), (1.0, 0.28, 1.0)),
            VelocityBox((0.0, 0.72, 0.0), (1.0, 1.0, 1.0), (0.05, 0.0, 0.0)),
        )
    elif scene == "compression":
        lower, upper = (0.35,) * 3, (0.65,) * 3
        duration = 0.3 / c
        boundaries = (
            fixed_box((0.0, 0.0, 0.0), (1.0, 0.35, 1.0)),
            VelocityBox((0.0, 0.65, 0.0), (1.0, 1.0, 1.0), (0.0, -0.005 * c, 0.0)),
        )
    elif scene == "surface-load":
        lower, upper = (0.2, 0.45, 0.2), (0.8, 0.55, 0.8)
        boundaries = (
            fixed_box((0.0,) * 3, (0.25, 1.0, 1.0)),
            fixed_box((0.75, 0.0, 0.0), (1.0,) * 3),
            fixed_box((0.0,) * 3, (1.0, 1.0, 0.25)),
            fixed_box((0.0, 0.0, 0.75), (1.0,) * 3),
        )
    elif scene == "wave":
        material, kind = linear_elastic(2e4, 0.0), "linear"
        density, c = 1000.0, np.sqrt(20.0)
        lower, upper = (0.125, 0.375, 0.375), (0.875, 0.625, 0.625)
        boundaries = (
            fixed_box((0.0,) * 3, (1.0,) * 3, components=(False, True, True)),
            fixed_box((0.0,) * 3, (0.125, 1.0, 1.0)),
            fixed_box((0.875, 0.0, 0.0), (1.0,) * 3),
        )
        duration = 0.8 * (upper[0] - lower[0]) / c
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material=kind,
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=boundaries,
    )
    state = block(
        cfg,
        lower,
        upper,
        particles_per_cell=ppc,
        density=density,
        material=material,
        dtype=jnp.float64,
    )
    reference = np.asarray(state.x)
    tip = reference[:, 0] > 0.72
    observer = None
    if scene == "surface-load":
        face = jnp.isclose(state.x[:, 1], state.x[:, 1].max(), atol=1e-10)
        nodal_area = 0.6**2 / jnp.sum(face)

        def pressure_load(states, time):
            pressure = 20.0 * 0.5 * (1 - jnp.cos(jnp.pi * jnp.clip(time / 0.05, 0.0, 1.0)))
            return ((pressure * nodal_area * face)[:, None] * jnp.array([0.0, -1.0, 0.0]),)

        forces = pressure_load
    if scene == "wave":
        L, amplitude = upper[0] - lower[0], 0.02
        mode = jnp.sin(jnp.pi * (state.x[:, 0] - lower[0]) / L)
        state = state._replace(v=state.v.at[:, 0].set(amplitude * mode))
        omega = np.pi * c / L
        mode = np.asarray(mode)

        def observe(states, time):
            velocity = np.asarray(states[0].v[:, 0])
            exact = amplitude * mode * np.cos(omega * time)
            displacement = np.asarray(states[0].x[:, 0]) - reference[:, 0]
            exact_u = amplitude / omega * mode * np.sin(omega * time)
            return {
                "velocity_l2_over_initial_amplitude": float(
                    np.sqrt(np.mean((velocity - exact) ** 2)) / amplitude
                ),
                "displacement_l2_over_initial_amplitude": float(
                    np.sqrt(np.mean((displacement - exact_u) ** 2)) / (amplitude / omega)
                ),
                "mode_velocity": float(velocity @ mode / (mode @ mode)),
            }

        observer = observe
    elif scene == "cantilever":
        shape = jnp.clip((state.x[:, 0] - 0.2) / 0.6, 0.0, 1.0) ** 2
        state = state._replace(v=state.v.at[:, 1].set(0.03 * shape))
    elif scene == "torsion":
        omega = 0.4 * jnp.clip((state.x[:, 0] - 0.2) / 0.6, 0.0, 1.0)
        v = state.v.at[:, 1].set(-omega * (state.x[:, 2] - 0.5))
        state = state._replace(v=v.at[:, 2].set(omega * (state.x[:, 1] - 0.5)))
    if observer is None:
        tracked = (
            tip
            if scene in ("cantilever", "torsion")
            else reference[:, 1] > (lower[1] + upper[1]) / 2
        )

        def observe(states, time):
            displacement = np.asarray(states[0].x) - reference
            return {
                f"tracked_displacement_{a}": float(displacement[tracked, d].mean())
                for d, a in enumerate("xyz")
            }

        observer = observe
    description = {
        "geometry": {"lower": lower, "upper": upper},
        "model": model if scene != "wave" else "linear",
        "particles_per_cell_per_axis": ppc,
        "initial_acoustic_speed": c,
        "boundary_method": "grid velocity constraints",
        "units": "m, s, kg, Pa",
        "purpose": "prescribed-platen material compression" if scene == "compression" else scene,
    }
    if scene == "surface-load":
        description["surface_pressure"] = {
            "peak_Pa": 20.0,
            "cosine_ramp_seconds": 0.05,
            "area_measure": "reference top-face area",
        }
    return Case(
        (state,), (material,), (cfg,), duration, description, observer=observer, forces=forces
    )


def contact_case(scene, grid, ppc):
    material = neo_hookean(5e4, 0.2)
    c = np.sqrt((material.bulk + 4 * material.mu / 3) / 1000.0)
    walls = (PlaneWall(1, 0.125, friction=0.3 if scene == "sliding" else 0.0),)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / c,
        material="neo-hookean",
        gravity=(0.0, -9.81, 0.0) if scene in ("sliding", "rebound") else (0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=walls if scene in ("sliding", "rebound") else (),
    )
    if scene in ("head-on", "oblique"):
        tangent = 0.2 if scene == "oblique" else 0.0
        a = block(
            cfg,
            (0.22, 0.35, 0.35),
            (0.45, 0.65, 0.65),
            velocity=(0.5, tangent, 0.0),
            particles_per_cell=ppc,
            dtype=jnp.float64,
        )
        b = block(
            cfg,
            (0.55, 0.35, 0.35),
            (0.78, 0.65, 0.65),
            velocity=(-0.5, -tangent, 0.0),
            particles_per_cell=ppc,
            dtype=jnp.float64,
        )
        return Case(
            (a, b),
            (material,) * 2,
            (cfg,) * 2,
            0.24,
            {
                "purpose": "elastic block collision",
                "boundary_method": "traction-free exterior",
                "particles_per_cell_per_axis": ppc,
                "contact_friction": 0.3 if tangent else 0.0,
            },
            friction=0.3 if tangent else 0.0,
        )
    lower = (0.25, 0.125, 0.35) if scene == "sliding" else (0.35, 0.35, 0.35)
    upper = (0.5, 0.375, 0.65) if scene == "sliding" else (0.65, 0.65, 0.65)
    state = block(
        cfg,
        lower,
        upper,
        particles_per_cell=ppc,
        velocity=(0.7, 0.0, 0.0) if scene == "sliding" else (0.0, 0.0, 0.0),
        dtype=jnp.float64,
    )

    def observe(states, time):
        s = states[0]
        return {
            "mean_speed_x": float(jnp.sum(s.mass * s.v[:, 0]) / s.mass.sum()),
            "lowest_particle_y": float(s.x[:, 1].min()),
        }

    return Case(
        (state,),
        (material,),
        (cfg,),
        0.25 if scene == "sliding" else 0.36,
        {
            "purpose": scene,
            "wall_y": 0.125,
            "friction": 0.3 if scene == "sliding" else 0.0,
            "particles_per_cell_per_axis": ppc,
        },
        observer=observe,
    )


def fluid_case(scene, grid, ppc):
    material = fluid(2e5)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=0.15 / grid / np.sqrt(200.0),
        material="fluid",
        boundary_cells=0,
        boundaries=tank_walls(),
    )
    lower, upper, velocity = (0.125, 0.125, 0.25), (0.4, 0.65, 0.75), (0.0, 0.0, 0.0)
    if scene == "sloshing":
        upper, velocity = (0.875, 0.375, 0.75), (0.5, 0.0, 0.0)
    elif scene == "water-entry":
        upper = (0.875, 0.45, 0.75)
    state = block(
        cfg,
        lower,
        upper,
        particles_per_cell=ppc,
        density=1000.0,
        velocity=velocity,
        dtype=jnp.float64,
    )
    states, materials, configs = (state,), (material,), (cfg,)
    if scene in ("fsi", "water-entry"):
        anchors = (fixed_box((0.0,) * 3, (1.0, 0.18, 1.0)),) if scene == "fsi" else ()
        scfg = Config(
            dim=3,
            n_grid=grid,
            dt=cfg.dt,
            material="neo-hookean",
            boundary_cells=0,
            boundaries=tank_walls() + anchors,
        )
        solid = neo_hookean(2e5, 0.3)
        lo, hi = (
            ((0.55, 0.125, 0.25), (0.65, 0.55, 0.75))
            if scene == "fsi"
            else ((0.4, 0.55, 0.4), (0.6, 0.75, 0.6))
        )
        structure = block(scfg, lo, hi, particles_per_cell=ppc, density=800.0, dtype=jnp.float64)
        states, materials, configs = states + (structure,), materials + (solid,), configs + (scfg,)
    initial_solid = np.asarray(states[-1].x)
    tracked = initial_solid[:, 1] > 0.45

    def observe(states, time):
        s = states[0]
        pressure = 2e5 * (1 - np.linalg.det(s.F))
        values = {
            "water_front_x": float(s.x[:, 0].max()),
            "free_surface_y": float(s.x[:, 1].max()),
            "maximum_pressure": float(pressure.max()),
        }
        if len(states) > 1:
            values["solid_top_displacement_x"] = float(
                (np.asarray(states[1].x)[tracked, 0] - initial_solid[tracked, 0]).mean()
            )
        return values

    return Case(
        states,
        materials,
        configs,
        0.35,
        {
            "purpose": scene,
            "tank_lower": (0.125,) * 3,
            "tank_upper": (0.875,) * 3,
            "liquid_bulk_modulus": 2e5,
            "particles_per_cell_per_axis": ppc,
            "interface": "separate-grid nonadhesive normal contact"
            if len(states) > 1
            else "free surface",
            "initial_hydrostatic_equilibrium": False,
        },
        observer=observe,
    )


def setup(scene, *, model="neo-hookean", grid=48, ppc=2):
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene}")
    if grid < 24:
        raise ValueError("these geometries need grid >= 24")
    if scene in ("perforated-panel", "cellular-cushion", "baffled-tank", "unbaffled-tank"):
        from .engineering import setup as engineering_setup

        return engineering_setup(scene, grid, ppc)
    if scene in (
        "elastic-gate",
        "floating-block",
        "rubber-ball",
        "ball-beam",
        "large-bending",
        "large-twist",
    ):
        from .benchmarks import setup as benchmark_setup

        return benchmark_setup(scene, grid, ppc)
    if scene in ("wave", "cantilever", "torsion", "shear", "compression", "surface-load"):
        if model in PLASTIC and scene != "compression":
            raise ValueError("use compression for the incremental material models")
        return solid_case(scene, model, grid, ppc)
    if scene in ("head-on", "oblique", "sliding", "rebound"):
        return contact_case(scene, grid, ppc)
    return fluid_case(scene, grid, ppc)
