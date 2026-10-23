"""Grid boundary conditions in physical coordinates."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PlaneWall:
    axis: int
    position: float
    side: str = "lower"
    friction: float = 0.0
    velocity: tuple[float, ...] | None = None

    def validate(self, dim, domain_size):
        if self.axis not in range(dim) or self.side not in ("lower", "upper"):
            raise ValueError("wall requires a valid axis and side='lower' or 'upper'")
        if not np.isfinite(self.position) or not 0 <= self.position <= domain_size:
            raise ValueError("wall position must lie inside the grid domain")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("wall friction must be finite and nonnegative")
        if self.velocity is not None and (
            len(self.velocity) != dim or not np.all(np.isfinite(self.velocity))
        ):
            raise ValueError("wall velocity must contain dim finite values")

    def apply(self, velocity, coordinates, time):
        wall_velocity = (
            jnp.zeros(velocity.shape[-1], dtype=velocity.dtype)
            if self.velocity is None
            else jnp.asarray(self.velocity, dtype=velocity.dtype)
        )
        position = self.position + wall_velocity[self.axis] * time
        normal = jnp.eye(velocity.shape[-1], dtype=velocity.dtype)[self.axis] * (
            1 if self.side == "lower" else -1
        )
        mask = (
            coordinates[:, self.axis] <= position
            if self.side == "lower"
            else coordinates[:, self.axis] >= position
        )
        relative = velocity - wall_velocity
        normal_speed = relative @ normal
        correction = jnp.maximum(-normal_speed, 0.0)
        tangent = relative - normal_speed[:, None] * normal
        speed = jnp.linalg.norm(tangent, axis=-1)
        reduction = jnp.minimum(speed, self.friction * correction) / jnp.maximum(speed, 1e-30)
        corrected = velocity + correction[:, None] * normal - reduction[:, None] * tangent
        return jnp.where(mask[:, None], corrected, velocity)


@dataclass(frozen=True)
class VelocityBox:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    velocity: tuple[float, ...]
    components: tuple[bool, ...] | None = None
    start: float = 0.0
    end: float = float("inf")

    def validate(self, dim, domain_size):
        if any(len(v) != dim for v in (self.lower, self.upper, self.velocity)):
            raise ValueError("velocity boundary vectors must match config.dim")
        if not np.all(np.isfinite((self.lower, self.upper, self.velocity))):
            raise ValueError("velocity boundary coordinates and velocity must be finite")
        if np.any(np.array(self.upper) < self.lower):
            raise ValueError("velocity box upper bounds must exceed lower bounds")
        if self.components is not None and len(self.components) != dim:
            raise ValueError("component mask must match config.dim")
        if (
            not np.isfinite(self.start)
            or self.start < 0
            or np.isnan(self.end)
            or self.end < self.start
        ):
            raise ValueError("invalid boundary activation interval")

    def apply(self, velocity, coordinates, time):
        mask = jnp.all(
            (coordinates >= jnp.array(self.lower)) & (coordinates <= jnp.array(self.upper)), axis=-1
        )
        mask = mask & (time >= self.start) & (time <= self.end)
        components = (
            jnp.ones(velocity.shape[-1], dtype=bool)
            if self.components is None
            else jnp.array(self.components)
        )
        return jnp.where(
            mask[:, None] & components, jnp.asarray(self.velocity, dtype=velocity.dtype), velocity
        )


@dataclass(frozen=True)
class MovingVelocityBox(VelocityBox):
    """Translating velocity box with an optional half-cosine startup ramp."""

    ramp_time: float = 0.0

    def validate(self, dim, domain_size):
        super().validate(dim, domain_size)
        if not np.isfinite(self.ramp_time) or self.ramp_time < 0:
            raise ValueError("boundary ramp_time must be finite and nonnegative")

    def motion(self, time):
        """Return displacement and velocity; motion stops at the activation end."""
        dtype = jnp.result_type(time, float)
        time = jnp.asarray(time, dtype=dtype)
        target = jnp.asarray(self.velocity, dtype=dtype)
        elapsed = jnp.clip(time - self.start, 0.0, self.end - self.start)
        active = (time >= self.start) & (time <= self.end)
        if self.ramp_time == 0:
            displacement = target * elapsed
            speed = target
        else:
            ramp_elapsed = jnp.minimum(elapsed, self.ramp_time)
            phase = jnp.pi * ramp_elapsed / self.ramp_time
            distance = 0.5 * (ramp_elapsed - self.ramp_time / jnp.pi * jnp.sin(phase))
            distance = distance + jnp.maximum(elapsed - self.ramp_time, 0.0)
            displacement = target * distance
            speed = target * (0.5 * (1.0 - jnp.cos(phase)))
        return displacement, jnp.where(active, speed, jnp.zeros_like(target))

    def apply(self, velocity, coordinates, time):
        displacement, prescribed = self.motion(time)
        lower = jnp.asarray(self.lower, dtype=coordinates.dtype) + displacement
        upper = jnp.asarray(self.upper, dtype=coordinates.dtype) + displacement
        mask = jnp.all((coordinates >= lower) & (coordinates <= upper), axis=-1)
        mask = mask & (time >= self.start) & (time <= self.end)
        components = (
            jnp.ones(velocity.shape[-1], dtype=bool)
            if self.components is None
            else jnp.asarray(self.components, dtype=bool)
        )
        return jnp.where(
            mask[:, None] & components, prescribed.astype(velocity.dtype), velocity
        )


def fixed_box(lower, upper, *, components=None):
    return VelocityBox(tuple(lower), tuple(upper), (0.0,) * len(lower), components)
