"""Finite-range normal adhesion on advected surface quadrature points."""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DugdaleLaw:
    """Reversible nominal traction law with penalty contact, in SI units."""

    work_of_adhesion: float
    interaction_range: float
    penalty_stiffness: float

    def __post_init__(self):
        values = (self.work_of_adhesion, self.interaction_range, self.penalty_stiffness)
        if not all(isfinite(value) for value in values):
            raise ValueError("Dugdale parameters must be finite")
        if (
            self.work_of_adhesion < 0
            or self.interaction_range <= 0
            or self.penalty_stiffness <= 0
        ):
            raise ValueError(
                "work must be nonnegative; range and penalty stiffness must be positive"
            )

    @property
    def cohesive_stress(self):
        return self.work_of_adhesion / self.interaction_range

    def potential(self, gap):
        """Energy per reference area; zero outside the interaction range."""
        gap = jnp.asarray(gap)
        return jnp.where(
            gap < 0.0,
            0.5 * self.penalty_stiffness * gap**2 - self.work_of_adhesion,
            jnp.where(
                gap < self.interaction_range,
                self.cohesive_stress * gap - self.work_of_adhesion,
                0.0,
            ),
        )

    def traction(self, gap):
        """Outward traction on the deformable body; negative means attraction."""
        gap = jnp.asarray(gap)
        return jnp.where(
            gap < 0.0,
            -self.penalty_stiffness * gap,
            jnp.where(gap < self.interaction_range, -self.cohesive_stress, 0.0),
        )


@dataclass(frozen=True)
class SpherePath:
    """A rigid sphere following cosine-interpolated position waypoints."""

    radius: float
    times: tuple[float, ...]
    centers: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if not isfinite(self.radius) or self.radius <= 0:
            raise ValueError("sphere radius must be finite and positive")
        if len(self.times) < 2 or len(self.times) != len(self.centers):
            raise ValueError(
                "matching times and centers require at least two waypoints"
            )
        if not all(isfinite(time) for time in self.times) or any(
            b <= a for a, b in zip(self.times, self.times[1:])
        ):
            raise ValueError("waypoint times must be finite and strictly increasing")
        if any(
            len(center) != 3 or not all(isfinite(x) for x in center)
            for center in self.centers
        ):
            raise ValueError("sphere centers must contain three finite coordinates")

    def motion(self, time):
        times, centers = jnp.asarray(self.times), jnp.asarray(self.centers)
        intervals = times[1:] - times[:-1]
        phase = jnp.clip((time - times[:-1]) / intervals, 0.0, 1.0)
        scale = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
        rate = 0.5 * jnp.pi * jnp.sin(jnp.pi * phase) / intervals
        rate = jnp.where((time > times[:-1]) & (time < times[1:]), rate, 0.0)
        changes = centers[1:] - centers[:-1]
        return centers[0] + jnp.sum(scale[:, None] * changes, axis=0), jnp.sum(
            rate[:, None] * changes, axis=0
        )


class SurfaceInteraction(NamedTuple):
    forces: jax.Array
    gap: jax.Array
    reaction: jax.Array
    reaction_moment: jax.Array
    potential: jax.Array
    penalty_potential: jax.Array
    adhesive_potential: jax.Array
    contact_reference_area: jax.Array
    adhesive_reference_area: jax.Array
    compressed_points: jax.Array
    adhesive_points: jax.Array


@partial(jax.jit, static_argnames=("law",))
def sphere_interaction(positions, reference_areas, center, radius, law):
    """Integrate a Dugdale potential over a material surface sampling plane.

    Areas are fixed reference quadrature weights. Positions are the actual
    advected sampling points, without a deformation-dependent surface offset.
    The sphere supplies the gap normal. No tangential traction is applied.
    """
    radial = positions - jnp.asarray(center)
    distance = jnp.linalg.norm(radial, axis=-1)
    normal = radial / jnp.maximum(distance, jnp.finfo(positions.dtype).tiny)[:, None]
    gap = distance - radius
    areas = jnp.broadcast_to(jnp.asarray(reference_areas), gap.shape)
    forces = (areas * law.traction(gap))[:, None] * normal
    penalty = areas * 0.5 * law.penalty_stiffness * jnp.minimum(gap, 0.0) ** 2
    potential = areas * law.potential(gap)
    compressed = gap < 0.0
    adhesive = (gap >= 0.0) & (gap < law.interaction_range) & (law.work_of_adhesion > 0)
    return SurfaceInteraction(
        forces,
        gap,
        -jnp.sum(forces, axis=0),
        -jnp.sum(jnp.cross(radial, forces), axis=0),
        jnp.sum(potential),
        jnp.sum(penalty),
        jnp.sum(potential - penalty),
        jnp.sum(jnp.where(compressed, areas, 0.0)),
        jnp.sum(jnp.where(adhesive, areas, 0.0)),
        jnp.sum(compressed),
        jnp.sum(adhesive),
    )


def penalty_step_number(masses, reference_areas, dt, law):
    """k A0 dt**2 / m for each surface point; a local explicit stability indicator."""
    return (
        law.penalty_stiffness
        * jnp.asarray(reference_areas)
        * dt**2
        / jnp.asarray(masses)
    )
