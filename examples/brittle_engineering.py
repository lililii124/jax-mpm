"""Dynamic concrete bending and ceramic spherical indentation in SI units."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, block, fixed_box, hjc
from mpm_jax.contact_laws import RigidSphereBoundary, boundary_reaction
from .common import run
from .scenarios import Case, material_model


SCENES = ("concrete-bending", "concrete-bending-large", "ceramic-indentation")


@dataclass(frozen=True)
class LoadingBox(MovingVelocityBox):
    """Translate with cosine velocity ramps, then hold the final position."""

    loading_duration: float = 0.0012

    def validate(self, dim, domain_size):
        super().validate(dim, domain_size)
        if not np.isfinite(self.loading_duration) or self.loading_duration <= 0:
            raise ValueError("loading_duration must be finite and positive")
        if not 0 < self.ramp_time <= self.loading_duration / 2:
            raise ValueError("loading ramps must fit within the loading interval")
        if self.end < self.start + self.loading_duration:
            raise ValueError("loading boundary must remain active through the loading interval")

    def motion(self, time):
        elapsed = jnp.clip(jnp.asarray(time) - self.start, 0.0, self.loading_duration)
        ramp = self.ramp_time
        rising = jnp.minimum(elapsed, ramp)
        falling = jnp.clip(elapsed - (self.loading_duration - ramp), 0.0, ramp)
        distance = 0.5 * (rising - ramp / jnp.pi * jnp.sin(jnp.pi * rising / ramp))
        distance += jnp.clip(elapsed - ramp, 0.0, self.loading_duration - 2 * ramp)
        distance += 0.5 * (falling + ramp / jnp.pi * jnp.sin(jnp.pi * falling / ramp))
        speed = jnp.where(
            elapsed < ramp,
            0.5 * (1 - jnp.cos(jnp.pi * rising / ramp)),
            jnp.where(
                elapsed <= self.loading_duration - ramp,
                1.0,
                0.5 * (1 + jnp.cos(jnp.pi * falling / ramp)),
            ),
        )
        speed = jnp.where((time > self.start) & (time < self.start + self.loading_duration), speed, 0.0)
        target = jnp.asarray(self.velocity, dtype=elapsed.dtype)
        return distance * target, speed * target


def _config(model, grid, domain, duration, frames, boundaries, *, profile=None):
    material, density, kind, wave_speed = material_model(model)
    if profile is not None:
        if model != "hjc":
            raise ValueError("an HJC profile requires the HJC material")
        material = hjc(profile=profile)
    stable_dt = 0.15 * domain / (grid * wave_speed)
    substeps = int(np.ceil(duration / (frames * stable_dt)))
    config = Config(
        dim=3,
        n_grid=grid,
        domain_size=domain,
        dt=duration / (frames * substeps),
        material=kind,
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=boundaries,
    )
    return material, density, config, wave_speed


def _definition(scene, model, material, config, ppc, lower, upper, duration, wave_speed):
    return {
        "purpose": scene,
        "model": model,
        "units": "m, s, kg, N, Pa",
        "parameter_status": (
            "repository constitutive defaults; demonstration geometry and loading, "
            "not a calibrated experiment"
        ),
        "material_parameters_SI": {
            key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in material._asdict().items()
        },
        "particles_per_cell_per_axis": ppc,
        "lower": lower,
        "upper": upper,
        "grid_spacing_m": config.dx,
        "duration_s": duration,
        "time_step_s": config.dt,
        "estimated_acoustic_speed_m_per_s": float(wave_speed),
        "acoustic_CFL": float(wave_speed * config.dt / config.dx),
        "gravity": "omitted for the short dynamic loading interval",
        "other_faces": "traction-free",
        "reaction_sampling": (
            "force on each rigid boundary from next-step grid momentum correction; "
            "force on specimen has opposite sign"
        ),
        "failure_mode": config.failure_mode,
        "damage_region_threshold": 0.5,
        "damage_region_geometry": "bounds of current particle centers above the damage threshold",
        "damage_interpretation": (
            "local constitutive damage and plastic strain; "
            "D=1 is not a resolved crack or a separated fragment"
        ),
        "mesh_dependence": (
            "local softening has no fracture-energy regularization; "
            "damage widths and post-peak response require refinement studies"
        ),
    }


def _damage_observables(state):
    damage = np.asarray(state.history.damage)
    volume = np.asarray(state.volume)
    positions = np.asarray(state.x)
    failed = np.asarray(state.history.failed)
    selected = damage >= 0.5
    result = {
        "max_damage": float(damage.max()),
        "mean_damage_reference_volume": float(np.dot(volume, damage) / volume.sum()),
        "damaged_reference_volume_m3": float(volume[selected].sum()),
        "damaged_reference_volume_fraction": float(volume[selected].sum() / volume.sum()),
        "fully_damaged_reference_volume_fraction": float(
            volume[damage >= 1.0].sum() / volume.sum()
        ),
        "material_failed_reference_volume_fraction": float(volume[failed].sum() / volume.sum()),
        "max_equivalent_plastic_strain": float(np.asarray(state.history.plastic_strain).max()),
    }
    for axis, name in enumerate("xyz"):
        result[f"damaged_region_min_{name}_m"] = (
            float(positions[selected, axis].min()) if selected.any() else None
        )
        result[f"damaged_region_max_{name}_m"] = (
            float(positions[selected, axis].max()) if selected.any() else None
        )
    return result


def _reactions(state, material, config, names):
    result = {}
    for index, name in enumerate(names):
        force = np.asarray(boundary_reaction(state, material, config, boundary_index=index))
        for axis, value in zip("xyz", force):
            result[f"{name}_reaction_{axis}_N"] = float(value)
    return result


def _concrete_beam(*, grid, ppc, model, frames, displacement, loading_duration,
                   hold_duration=0.0, ramp_time=None, profile=None):
    if model not in ("hjc", "rht"):
        raise ValueError("concrete-bending requires model='hjc' or 'rht'")
    domain, duration = 0.4, loading_duration + hold_duration
    large = profile is not None
    dx = domain / grid
    lower, upper = (0.05, 0.15, 0.17), (0.35, 0.21, 0.23)
    support_half_width = 0.009375
    left = fixed_box(
        (0.075 - support_half_width, 0.0, 0.0),
        (0.075 + support_half_width, lower[1] + 0.25 * dx, domain),
        components=(True, True, False),
    )
    right = fixed_box(
        (0.325 - support_half_width, 0.0, 0.0),
        (0.325 + support_half_width, lower[1] + 0.25 * dx, domain),
        components=(False, True, False),
    )
    lateral_anchor = fixed_box(
        (0.075 - support_half_width, 0.0, 0.2 - 0.55 * dx),
        (0.075 + support_half_width, lower[1] + 0.25 * dx, 0.2 + 0.55 * dx),
        components=(False, False, True),
    )
    ramp_time = 0.25 * loading_duration if ramp_time is None else ramp_time
    speed = displacement / (loading_duration - (ramp_time if large else 0.5 * ramp_time))
    loading_type = LoadingBox if large else MovingVelocityBox
    loading = loading_type(
        (0.1875, upper[1] - 0.5 * dx, 0.0),
        (0.2125, upper[1] + 2.0 * dx, domain),
        (0.0, -speed, 0.0),
        components=(False, True, False),
        ramp_time=ramp_time,
        **({"loading_duration": loading_duration} if large else {}),
    )
    boundaries = (left, right, lateral_anchor, loading)
    material, density, cfg, wave_speed = _config(
        model, grid, domain, duration, frames, boundaries, profile=profile
    )
    state = block(
        cfg, lower, upper, particles_per_cell=ppc, density=density,
        material=material, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    bottom = np.isclose(reference[:, 1], reference[:, 1].min(), atol=1e-12, rtol=0)
    midspan = bottom & (np.abs(reference[:, 0] - 0.2) <= max(dx, 0.0125))
    definition = _definition(
        "concrete-bending-large" if large else "concrete-bending",
        model, material, cfg, ppc, lower, upper, duration, wave_speed
    )
    central_lower = (np.abs(reference[:, 0] - 0.2) <= 0.025) & (reference[:, 1] <= lower[1] + 0.01)
    central_upper = (np.abs(reference[:, 0] - 0.2) <= 0.025) & (reference[:, 1] >= upper[1] - 0.01)
    definition.update(
        {
            "loading_regime": "dynamic three-point bending; inertia and constitutive rate effects retained",
            "support_span_m": 0.25,
            "support_patch_width_m": 2 * support_half_width,
            "supports": (
                "left patch fixes x/y; right patch fixes y and permits axial sliding; "
                "a small left mid-width patch fixes z"
            ),
            "support_centers_x_m": [0.075, 0.325],
            "loading_patch_width_m": 0.025,
            "loading": (
                "central moving velocity patch, y component only; "
                "half-cosine speed startup followed by constant speed"
            ),
            "loading_boundary": {
                key: value for key, value in asdict(loading).items() if key != "end"
            },
            "startup_ramp_s": ramp_time,
            "final_prescribed_displacement_m": displacement,
            "plateau_loading_speed_m_per_s": speed,
            "axial_wave_transits": float(duration * wave_speed / (upper[0] - lower[0])),
            "support_idealization": (
                "bilateral finite grid patches represent mechanical fixtures; "
                "finite patch width affects local stress"
            ),
        }
    )
    if large:
        definition.update({
            "material_profile": profile,
            "loading": "central translating y-velocity patch; cosine startup and stopping ramps, then fixed-displacement hold",
            "loading_duration_s": loading_duration,
            "hold_duration_s": hold_duration,
            "stopping_ramp_s": ramp_time,
            "loading_motion": "cosine velocity ramp up, constant velocity, cosine velocity ramp down, hold",
            "loading_distance_integral_s": loading_duration - ramp_time,
            "geometry_scale": 1.0,
        })

    def observe(states, time):
        current = states[0]
        result = _damage_observables(current)
        result.update(_reactions(
            current, material, cfg,
            ("left_support", "right_support", "lateral_anchor", "loading")
        ))
        prescribed, velocity = loading.motion(time)
        displacement_y = np.asarray(current.x)[:, 1] - reference[:, 1]
        result.update(
            {
                "boundary_sample_time_s": float(time + cfg.dt),
                "prescribed_displacement_y_m": float(prescribed[1]),
                "prescribed_speed_y_m_per_s": float(velocity[1]),
                "midspan_bottom_deflection_m": float(-displacement_y[midspan].mean()),
                "maximum_downward_displacement_m": float(-displacement_y.min()),
                "momentum_y_kg_m_per_s": float(
                    np.sum(np.asarray(current.mass) * np.asarray(current.v)[:, 1])
                ),
            }
        )
        if large:
            damage = np.asarray(current.history.damage)
            reference_volume = np.asarray(current.volume)
            for region, mask in (("central_lower", central_lower), ("central_upper", central_upper)):
                result[f"{region}_max_damage"] = float(damage[mask].max()) if mask.any() else None
                result[f"{region}_mean_damage"] = (
                    float(np.dot(reference_volume[mask], damage[mask]) / reference_volume[mask].sum())
                    if mask.any() else None
                )
        return result

    return Case((state,), (material,), (cfg,), duration, definition, observer=observe)


def concrete_bending(*, grid=64, ppc=3, model="hjc", frames=40):
    """The original 2 mm dynamic beam loading with the legacy material profile."""
    return _concrete_beam(
        grid=grid, ppc=ppc, model=model, frames=frames,
        displacement=0.002, loading_duration=3e-4,
    )


def concrete_bending_large(*, grid=80, ppc=3, model="hjc", frames=60,
                           displacement=0.012, loading_duration=0.0012,
                           hold_duration=0.0003):
    """A 12 mm loading and hold of the same beam using the strain-floor profile."""
    if model != "hjc":
        raise ValueError("concrete-bending-large requires model='hjc'")
    if not np.isfinite(displacement) or not 0 < displacement <= 0.03:
        raise ValueError("displacement must be positive and no greater than half the beam depth")
    if not np.isfinite(loading_duration) or loading_duration <= 0:
        raise ValueError("loading_duration must be positive")
    if not np.isfinite(hold_duration) or hold_duration < 0:
        raise ValueError("hold_duration must be nonnegative")
    return _concrete_beam(
        grid=grid, ppc=ppc, model=model, frames=frames, displacement=displacement,
        loading_duration=loading_duration, hold_duration=hold_duration,
        ramp_time=loading_duration / 4, profile="strain-floor",
    )


def ceramic_indentation(*, grid=64, ppc=3, model="jh2", frames=40):
    """A backed 100 x 25 x 100 mm tile under a 75 mm radius rigid sphere."""
    if model != "jh2":
        raise ValueError("ceramic-indentation requires model='jh2'")
    domain, duration, indentation = 0.25, 3e-4, 0.004
    lower, upper = (0.075, 0.075, 0.075), (0.175, 0.1, 0.175)
    fixed_base = fixed_box((0.0, 0.0, 0.0), (domain, lower[1], domain))
    sphere = RigidSphereBoundary(
        (0.125, 0.175, 0.125), 0.075, (0.0, -indentation, 0.0), duration
    )
    material, density, cfg, wave_speed = _config(
        model, grid, domain, duration, frames, (fixed_base, sphere)
    )
    state = block(
        cfg, lower, upper, particles_per_cell=ppc, density=density,
        material=material, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    top = np.isclose(reference[:, 1], reference[:, 1].max(), atol=1e-12, rtol=0)
    radial2 = (reference[:, 0] - 0.125) ** 2 + (reference[:, 2] - 0.125) ** 2
    central_surface = top & (radial2 <= (1.5 * cfg.dx) ** 2)
    definition = _definition(
        "ceramic-indentation", model, material, cfg, ppc, lower, upper, duration, wave_speed
    )
    definition.update(
        {
            "loading_regime": "dynamic loading/unloading indentation; not a quasi-static hardness test",
            "contact": "frictionless unilateral grid-node rigid sphere",
            "base": "all velocity components fixed at y <= 0.075 m; ideal rigid bonded backing",
            "rigid_sphere": asdict(sphere),
            "rigid_sphere_motion": "center + displacement * (1-cos(2*pi*clip(t/duration,0,1)))/2",
            "maximum_nominal_indentation_m": indentation,
            "peak_indenter_speed_m_per_s": float(np.pi * indentation / duration),
            "nominal_contact_radius_m": float(np.sqrt(sphere.radius * indentation)),
            "nominal_contact_radius_in_grid_cells": float(
                np.sqrt(sphere.radius * indentation) / cfg.dx
            ),
            "contact_radius_interpretation": (
                "sqrt(R*delta) is only a geometric resolution estimate; "
                "no elastic Hertz validation is claimed for damaged ceramic"
            ),
            "thickness_wave_transits": float(duration * wave_speed / (upper[1] - lower[1])),
        }
    )

    def observe(states, time):
        current = states[0]
        result = _damage_observables(current)
        result.update(_reactions(current, material, cfg, ("backing", "indenter")))
        center, velocity = sphere.motion(time)
        displacement_y = np.asarray(current.x)[:, 1] - reference[:, 1]
        result.update(
            {
                "boundary_sample_time_s": float(time + cfg.dt),
                "nominal_indentation_m": float(upper[1] - (center[1] - sphere.radius)),
                "indenter_speed_y_m_per_s": float(velocity[1]),
                "central_surface_displacement_y_m": float(
                    displacement_y[central_surface].mean()
                ),
                "maximum_downward_displacement_m": float(-displacement_y.min()),
                "momentum_y_kg_m_per_s": float(
                    np.sum(np.asarray(current.mass) * np.asarray(current.v)[:, 1])
                ),
            }
        )
        return result

    return Case((state,), (material,), (cfg,), duration, definition, observer=observe)


def setup(scene, *, grid=64, ppc=3, model=None, frames=40):
    if not isinstance(grid, int) or grid < 12:
        raise ValueError("brittle engineering examples require grid >= 12")
    if not isinstance(ppc, int) or ppc < 1 or not isinstance(frames, int) or frames < 1:
        raise ValueError("ppc and frames must be positive integers")
    if scene == "concrete-bending":
        return concrete_bending(grid=grid, ppc=ppc, model=model or "hjc", frames=frames)
    if scene == "concrete-bending-large":
        return concrete_bending_large(grid=grid, ppc=ppc, model=model or "hjc", frames=frames)
    if scene == "ceramic-indentation":
        return ceramic_indentation(grid=grid, ppc=ppc, model=model or "jh2", frames=frames)
    raise ValueError(f"unknown brittle engineering case: {scene}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SCENES, default="concrete-bending")
    parser.add_argument("--model", choices=("hjc", "jh2", "rht"))
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("outputs/brittle-engineering"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    try:
        case = setup(args.case, grid=args.grid, ppc=args.ppc, model=args.model, frames=args.frames)
    except ValueError as error:
        parser.error(str(error))
    model = case.configs[0].material
    substeps = round(case.duration / (args.frames * case.configs[0].dt))
    print(
        f"{args.case}/{model}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=args.output / args.case / model,
        label=f"{args.case} / {model}",
        observer=case.observer,
        metadata=case.description,
        render_output=not args.no_render,
        gif=args.gif,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
