"""Spherical contact laws, a rigid obstacle, and particle surface loads.

Compression is positive. JKR uses work of adhesion in J/m² and the reduced
Young modulus E*, not the Maugis modulus 4 E*/3. See docs/CONTACT.md.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class SphericalContact(NamedTuple):
    radius: jax.Array
    force: jax.Array
    compression: jax.Array
    adhesion: jax.Array
    active: jax.Array
    detachment: jax.Array


def reduced_modulus(young_a, poisson_a, young_b=jnp.inf, poisson_b=0.0):
    """The second body is rigid when young_b is infinite."""
    return 1 / ((1 - poisson_a**2) / young_a + (1 - poisson_b**2) / young_b)


def hertz_contact(indentation, radius, modulus):
    indentation = jnp.asarray(indentation)
    positive = indentation > 0
    a = jnp.where(positive, jnp.sqrt(radius * jnp.where(positive, indentation, 1.0)), 0.0)
    force = 4 * modulus * a**3 / (3 * radius)
    zero = jnp.zeros_like(force)
    return SphericalContact(a, force, force, zero, indentation > 0, zero)


def jkr_indentation(contact_radius, radius, modulus, work):
    return contact_radius**2 / radius - jnp.sqrt(2 * jnp.pi * work * contact_radius / modulus)


def jkr_force(contact_radius, radius, modulus, work):
    return 4 * modulus * contact_radius**3 / (3 * radius) - jnp.sqrt(
        8 * jnp.pi * work * modulus * contact_radius**3
    )


def jkr_detachment(radius, modulus, work):
    """Critical (radius, indentation) for a displacement-controlled contact."""
    a = jnp.cbrt(jnp.pi * work * radius**2 / (8 * modulus))
    return a, -3 * a**2 / radius


@jax.custom_jvp
def _stable_radius(indentation, radius, modulus, work):
    critical, minimum = jkr_detachment(radius, modulus, work)
    target = jnp.maximum(indentation, minimum)
    lo = critical
    hi = 4 * critical + 2 * jnp.sqrt(radius * jnp.maximum(target, 0.0))

    def bisect(_, bounds):
        lo, hi = bounds
        middle = (lo + hi) / 2
        below = jkr_indentation(middle, radius, modulus, work) < target
        return jnp.where(below, middle, lo), jnp.where(below, hi, middle)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (lo, hi))
    return (lo + hi) / 2


@_stable_radius.defjvp
def _stable_radius_jvp(primals, tangents):
    indentation, radius, modulus, work = primals
    ddelta, dradius, dmodulus, dwork = tangents
    a = _stable_radius(indentation, radius, modulus, work)
    adhesive_displacement = jnp.sqrt(2 * jnp.pi * work * a / modulus)
    slope = 2 * a / radius - adhesive_displacement / (2 * a)
    numerator = (
        ddelta
        + a**2 / radius**2 * dradius
        - adhesive_displacement / (2 * modulus) * dmodulus
        + adhesive_displacement / (2 * work) * dwork
    )
    return a, numerator / slope


def jkr_contact(indentation, radius, modulus, work, *, attached=True):
    """Largest stable root, with separation and reattachment at zero gap.

    Carry result.active into attached on the next step. At the instability,
    the contact opens. Derivatives apply inside a smooth attached branch.
    """
    indentation, work = jnp.asarray(indentation), jnp.asarray(work)
    positive_work = jnp.where(work > 0, work, 1.0)
    _, minimum = jkr_detachment(radius, modulus, positive_work)
    active = (indentation > minimum) & (jnp.asarray(attached) | (indentation >= 0))
    # The inactive branch is evaluated at a regular point to avoid NaNs in AD.
    target = jnp.where(active, indentation, 0.0)
    a = _stable_radius(target, radius, modulus, positive_work)
    a = jnp.where(active, a, 0.0)
    compression = 4 * modulus * a**3 / (3 * radius)
    adhesion = jnp.where(
        active,
        jnp.sqrt(8 * jnp.pi * positive_work * modulus * jnp.where(active, a, 1.0) ** 3),
        0.0,
    )
    result = SphericalContact(a, compression - adhesion, compression, adhesion, active, minimum)
    hertz = hertz_contact(indentation, radius, modulus)
    return jax.tree.map(lambda j, h: jnp.where(work > 0, j, h), result, hertz)


def jkr_history(indentations, radius, modulus, work, *, initially_attached=False):
    """Displacement-controlled approach/retraction including contact history."""

    def update(attached, indentation):
        result = jkr_contact(indentation, radius, modulus, work, attached=attached)
        return result.active, result

    return jax.lax.scan(update, jnp.asarray(initially_attached), indentations)[1]


class SurfaceLoad(NamedTuple):
    forces: jax.Array
    reaction: jax.Array
    resolved: jax.Array
    loaded_particles: jax.Array


def spherical_surface_load(positions, area, surface, center, normal, contact, cell_width):
    """Apply a reference-plane Hertz/JKR pressure footprint to surface particles.

    The compressive and tensile resultants are normalized independently.
    The JKR edge singularity is regularized over a particle-width band.
    A footprint smaller than the surface sampling uses the nearest particle
    and returns resolved=False. This function prescribes load, not contact.
    """
    normal = jnp.asarray(normal, dtype=positions.dtype)
    normal = normal / jnp.linalg.norm(normal)
    relative = positions - jnp.asarray(center, dtype=positions.dtype)
    tangent = relative - (relative @ normal)[:, None] * normal
    radial_squared = jnp.sum(tangent**2, axis=-1)
    safe_radius = jnp.maximum(contact.radius, cell_width * 1e-6)
    inside = jnp.asarray(surface) & (radial_squared < contact.radius**2)
    nearest = jnp.argmin(jnp.where(surface, radial_squared, jnp.inf))
    fallback = jnp.arange(positions.shape[0]) == nearest
    fallback = fallback & jnp.any(surface)
    selected = jnp.where(jnp.any(inside), inside, fallback) & contact.active
    edge = jnp.minimum(0.5, cell_width / safe_radius)
    radial_factor = jnp.maximum(1 - radial_squared / safe_radius**2, 0.0)
    compression_weights = jnp.where(selected, area * jnp.sqrt(radial_factor), 0.0)
    compression_weights = jnp.where(jnp.any(inside), compression_weights, selected * area)
    adhesion_weights = jnp.where(selected, area / jnp.sqrt(jnp.maximum(radial_factor, edge)), 0.0)
    compression_weights /= jnp.maximum(compression_weights.sum(), 1e-30)
    adhesion_weights /= jnp.maximum(adhesion_weights.sum(), 1e-30)
    magnitude = contact.compression * compression_weights - contact.adhesion * adhesion_weights
    forces = -magnitude[:, None] * normal
    return SurfaceLoad(
        forces,
        -forces.sum(axis=0),
        (~contact.active) | (jnp.sum(inside) >= 12),
        jnp.sum(selected),
    )


@dataclass(frozen=True)
class RigidSphereBoundary:
    """Frictionless unilateral rigid sphere with a prescribed smooth cycle."""

    center: tuple[float, ...]
    radius: float
    displacement: tuple[float, ...]
    duration: float

    def validate(self, dim, domain_size):
        if len(self.center) != dim or len(self.displacement) != dim:
            raise ValueError("sphere center and displacement must match config.dim")
        if not np.all(np.isfinite((*self.center, *self.displacement, self.radius, self.duration))):
            raise ValueError("sphere parameters must be finite")
        if self.radius <= 0 or self.duration <= 0:
            raise ValueError("sphere radius and duration must be positive")

    def motion(self, time):
        phase = jnp.clip(time / self.duration, 0.0, 1.0)
        scale = 0.5 * (1 - jnp.cos(2 * jnp.pi * phase))
        rate = jnp.pi * jnp.sin(2 * jnp.pi * phase) / self.duration
        rate = jnp.where((time > 0) & (time < self.duration), rate, 0.0)
        displacement = jnp.asarray(self.displacement)
        return jnp.asarray(self.center) + scale * displacement, rate * displacement

    def apply(self, velocity, coordinates, time):
        center, prescribed_velocity = self.motion(time)
        radial = coordinates - center
        distance = jnp.linalg.norm(radial, axis=-1)
        normal = radial / jnp.maximum(distance, 1e-30)[:, None]
        closing = jnp.sum((velocity - prescribed_velocity) * normal, axis=-1)
        correction = jnp.maximum(-closing, 0.0)[:, None] * normal
        inside = (distance <= self.radius) & (distance > 0)
        return velocity + jnp.where(inside[:, None], correction, 0.0)


@partial(jax.jit, static_argnames=("config", "boundary_index"))
def boundary_reaction(state, material, config, boundary_index=-1, force=None):
    """Force on one prescribed boundary during the next MPM step.

    Replays the grid stage and reads the impulse of the selected boundary;
    earlier constraints are included in the same order as grid_boundaries.
    No particle update is performed. Optional particle forces are included in
    the same P2G stage as step(); the result is the force on the boundary.
    """
    from .core import grid_update, particle_to_grid

    index = boundary_index % len(config.boundaries)
    mass, momentum = particle_to_grid(state, material, config, force=force)
    before = grid_update(
        mass,
        momentum,
        replace(config, boundaries=config.boundaries[:index]),
        state.time + config.dt,
    )
    node = jnp.arange((config.n_grid + 1) ** config.dim)
    strides = (config.n_grid + 1) ** jnp.arange(config.dim - 1, -1, -1)
    coordinates = ((node[:, None] // strides) % (config.n_grid + 1)) * config.dx
    after = config.boundaries[index].apply(before, coordinates, state.time + config.dt)
    return -jnp.sum(mass[:, None] * (after - before), axis=0) / config.dt
