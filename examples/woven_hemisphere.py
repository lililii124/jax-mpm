"""Closed lower-hemisphere boundary for a frictionless indentation example."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RigidHemisphereBoundary:
    center: tuple[float, float, float]
    radius: float
    displacement: tuple[float, float, float]
    loading_duration: float
    start: float = 0.0

    def validate(self, dim, domain_size):
        if dim != 3 or len(self.center) != 3 or len(self.displacement) != 3:
            raise ValueError("hemisphere requires three-dimensional coordinates")
        if not np.isfinite((*self.center, *self.displacement, self.radius, self.loading_duration, self.start)).all():
            raise ValueError("hemisphere parameters must be finite")
        if self.radius <= 0 or self.loading_duration <= 0 or self.start < 0:
            raise ValueError("radius and loading_duration must be positive; start must be nonnegative")
        ends = np.array((self.center, np.asarray(self.center) + self.displacement))
        lower, upper = ends.min(axis=0) - (self.radius,) * 3, ends.max(axis=0) + (self.radius, 0, self.radius)
        if np.any(lower < 0) or np.any(upper > domain_size):
            raise ValueError("the complete hemisphere path must lie inside the grid domain")

    def motion(self, time):
        elapsed = jnp.asarray(time, dtype=jnp.result_type(time, float)) - self.start
        phase = jnp.clip(elapsed / self.loading_duration, 0.0, 1.0)
        scale = 0.5 * (1 - jnp.cos(jnp.pi * phase))
        rate = 0.5 * jnp.pi / self.loading_duration * jnp.sin(jnp.pi * phase)
        rate = jnp.where((elapsed > 0) & (elapsed < self.loading_duration), rate, 0.0)
        stroke = jnp.asarray(self.displacement, dtype=elapsed.dtype)
        return jnp.asarray(self.center, dtype=elapsed.dtype) + scale * stroke, rate * stroke

    def signed_distance(self, coordinates, time):
        """Euclidean signed distance to the curved cap and its closing disk."""
        center, _ = self.motion(time)
        radial = coordinates - center.astype(coordinates.dtype)
        distance = jnp.linalg.norm(radial, axis=-1)
        rho = jnp.linalg.norm(radial[:, (0, 2)], axis=-1)
        below = jnp.maximum(distance - self.radius, radial[:, 1])
        above = jnp.sqrt(jnp.maximum(rho - self.radius, 0.0)**2 + radial[:, 1]**2)
        return jnp.where(radial[:, 1] <= 0, below, above)

    def apply(self, velocity, coordinates, time):
        center, tool_velocity = self.motion(time)
        center, tool_velocity = center.astype(coordinates.dtype), tool_velocity.astype(velocity.dtype)
        radial = coordinates - center
        distance = jnp.linalg.norm(radial, axis=-1)
        curved_normal = radial / jnp.maximum(distance, 1e-30)[:, None]
        flat_normal = jnp.array((0.0, 1.0, 0.0), dtype=velocity.dtype)
        curved_gap, flat_gap = distance - self.radius, radial[:, 1]
        relative = velocity - tool_velocity
        curved_speed = jnp.sum(relative * curved_normal, axis=-1)
        flat_speed = relative[:, 1]
        # Inside the solid, its nearest exit is either the cap or the flat disk.
        # At equal distances, use the face requiring the smaller impulse; a
        # velocity leaving either face at the rim must remain unconstrained.
        use_curve = (curved_gap > flat_gap) | ((curved_gap == flat_gap) & (curved_speed >= flat_speed))
        normal = jnp.where(use_curve[:, None], curved_normal, flat_normal)
        closing = jnp.sum(relative * normal, axis=-1)
        inside = (curved_gap <= 0) & (flat_gap <= 0) & (time >= self.start)
        correction = jnp.maximum(-closing, 0)[:, None] * normal
        return velocity + jnp.where(inside[:, None], correction, 0).astype(velocity.dtype)
