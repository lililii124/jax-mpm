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


def _cycle(time, start, period):
    elapsed = jnp.asarray(time, dtype=jnp.result_type(time, float)) - start
    phase = jnp.clip(elapsed / period, 0.0, 1.0)
    value = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * phase))
    rate = jnp.pi / period * jnp.sin(2.0 * jnp.pi * phase)
    return value, jnp.where((elapsed > 0) & (elapsed < period), rate, 0.0)


def _validate_cycle(start, period):
    if not np.isfinite(start) or start < 0:
        raise ValueError("cycle start must be finite and nonnegative")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("cycle period must be finite and positive")


@dataclass(frozen=True)
class CyclicVelocityBox:
    """Translate to a prescribed displacement and back, then hold at rest."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    displacement: tuple[float, ...]
    period: float
    components: tuple[bool, ...] | None = None
    start: float = 0.0

    def validate(self, dim, domain_size):
        VelocityBox(
            self.lower, self.upper, self.displacement, self.components
        ).validate(dim, domain_size)
        _validate_cycle(self.start, self.period)

    def motion(self, time):
        value, rate = _cycle(time, self.start, self.period)
        target = jnp.asarray(self.displacement, dtype=value.dtype)
        return value * target, rate * target

    def apply(self, velocity, coordinates, time):
        displacement, speed = self.motion(time)
        reference = coordinates - displacement.astype(coordinates.dtype)
        inside = jnp.all(
            (reference >= jnp.asarray(self.lower))
            & (reference <= jnp.asarray(self.upper)), axis=-1
        ) & (time >= self.start)
        components = (
            jnp.ones(velocity.shape[-1], dtype=bool)
            if self.components is None else jnp.asarray(self.components, dtype=bool)
        )
        return jnp.where(inside[:, None] & components, speed.astype(velocity.dtype), velocity)


@dataclass(frozen=True)
class RotatingVelocityBox:
    """Rotate a reference box to an angle and back, then hold at rest."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    center: tuple[float, float, float]
    axis: int
    angle: float
    period: float
    start: float = 0.0

    def validate(self, dim, domain_size):
        if dim != 3:
            raise ValueError("rotating velocity boxes require three dimensions")
        VelocityBox(self.lower, self.upper, (0.0,) * 3).validate(dim, domain_size)
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("rotation center must contain three finite coordinates")
        if not isinstance(self.axis, (int, np.integer)) or self.axis not in range(3):
            raise ValueError("rotation axis must be 0, 1 or 2")
        if not np.isfinite(self.angle):
            raise ValueError("rotation angle must be finite")
        _validate_cycle(self.start, self.period)

    def rotation(self, time):
        value, rate = _cycle(time, self.start, self.period)
        return self.angle * value, self.angle * rate

    def apply(self, velocity, coordinates, time):
        angle, speed = self.rotation(time)
        axis = jnp.eye(3, dtype=coordinates.dtype)[self.axis]
        center = jnp.asarray(self.center, dtype=coordinates.dtype)
        radial = coordinates - center
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        reference = (
            center + cosine * radial - sine * jnp.cross(axis, radial)
            + (1.0 - cosine) * (radial @ axis)[:, None] * axis
        )
        inside = jnp.all(
            (reference >= jnp.asarray(self.lower))
            & (reference <= jnp.asarray(self.upper)), axis=-1
        ) & (time >= self.start)
        prescribed = speed * jnp.cross(axis, radial)
        return jnp.where(inside[:, None], prescribed.astype(velocity.dtype), velocity)


def fixed_box(lower, upper, *, components=None):
    return VelocityBox(tuple(lower), tuple(upper), (0.0,) * len(lower), components)


@dataclass(frozen=True)
class MovingWedgeTool:
    """Rigid 2D wedge extruded through z, enforced by grid velocity contact.

    ``tip`` is the initial cutting-edge position in x-y. The cutting direction
    is +x; rake and clearance faces extend to the left of the tip. Contact is a
    unilateral nodal velocity projection with Coulomb tangential slip. The tool
    is prescribed and has infinite mass; this boundary does not deform or wear.
    """

    tip: tuple[float, float]
    speed: float
    rake_angle_deg: float = 10.0
    clearance_angle_deg: float = 7.0
    length: float = 1.0
    friction: float = 0.0
    tip_radius: float = 0.0
    tip_arc_segments: int = 8
    translation_path: tuple[tuple[float, float, float], ...] | None = None

    def validate(self, dim, domain_size):
        if dim not in (2, 3):
            raise ValueError("MovingWedgeTool requires a 2D grid or a 3D grid extruded through z")
        if len(self.tip) != 2 or not np.all(np.isfinite(self.tip)):
            raise ValueError("tool tip must be a finite (x, y) pair")
        if not np.isfinite([self.speed, self.rake_angle_deg,
                            self.clearance_angle_deg, self.length, self.friction,
                            self.tip_radius]).all():
            raise ValueError("tool parameters must be finite")
        if self.speed <= 0 or self.length <= 0 or self.friction < 0 or self.tip_radius < 0:
            raise ValueError("tool speed/length must be positive and friction/tip radius nonnegative")
        if self.translation_path is not None:
            path = np.asarray(self.translation_path, dtype=float)
            if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3 or not np.isfinite(path).all():
                raise ValueError("translation_path needs finite (absolute_time,x,y) knots")
            if path[0, 0] != 0 or np.any(np.diff(path[:, 0]) <= 0) or not np.array_equal(path[0, 1:], np.asarray(self.tip)):
                raise ValueError("trajectory must start at time zero and virtual tip, with increasing times")
        if self.tip_radius >= self.length:
            raise ValueError("tool tip radius must be smaller than the face length")
        if type(self.tip_arc_segments) is not int or self.tip_arc_segments < 2:
            raise ValueError("rounded tip requires at least two arc segments")
        if self.rake_angle_deg + self.clearance_angle_deg >= 90:
            raise ValueError("rake plus positive clearance must be below 90 degrees")
        half_angle = .5 * np.deg2rad(90 - self.rake_angle_deg - self.clearance_angle_deg)
        if self.tip_radius > 0 and self.tip_radius / np.tan(half_angle) >= self.length:
            raise ValueError("rounded-tip tangent distance must be smaller than face length")
        if not 0 < self.rake_angle_deg < 80 or not 0 < self.clearance_angle_deg < 45:
            raise ValueError("rake must be in (0,80) deg and clearance in (0,45) deg")

    def tip_position(self, time, dtype):
        """Absolute virtual sharp-face intersection; the polygon fillet is unchanged."""
        if self.translation_path is None:
            return jnp.asarray(self.tip, dtype=dtype) + jnp.array([self.speed * time, 0.0], dtype=dtype)
        knots = jnp.asarray(self.translation_path, dtype=dtype)
        index = jnp.clip(jnp.searchsorted(knots[:, 0], time, side="right") - 1, 0, knots.shape[0] - 2)
        slope = (knots[index + 1, 1:] - knots[index, 1:]) / (knots[index + 1, 0] - knots[index, 0])
        position = knots[index, 1:] + slope * (time - knots[index, 0])
        return jnp.where(time >= knots[-1, 0], knots[-1, 1:], position)

    def velocity(self, time, dtype):
        """Right-continuous 2D translation velocity; zero after the final knot."""
        if self.translation_path is None:
            return jnp.asarray((self.speed, 0.0), dtype=dtype)
        knots = jnp.asarray(self.translation_path, dtype=dtype)
        index = jnp.clip(jnp.searchsorted(knots[:, 0], time, side="right") - 1, 0, knots.shape[0] - 2)
        value = (knots[index + 1, 1:] - knots[index, 1:]) / (knots[index + 1, 0] - knots[index, 0])
        return jnp.where(time >= knots[-1, 0], jnp.zeros(2, dtype=dtype), value)

    def interval_velocity(self, start_time, end_time, dtype):
        """A segment's exact velocity. Caller must satisfy interval_is_linear."""
        return self.velocity(start_time + .5 * (end_time - start_time), dtype)

    def interval_is_linear(self, start_time, end_time):
        """Reject velocity-kink crossings, allowing only explicitly bounded FP roundoff."""
        if self.translation_path is None:
            return jnp.asarray(True)
        knots = jnp.asarray(self.translation_path)
        tolerance = 64 * jnp.finfo(knots.dtype).eps * jnp.maximum(jnp.maximum(jnp.abs(start_time), jnp.abs(end_time)), 1e-12)
        crossings = (knots[1:, 0] > start_time + tolerance) & (knots[1:, 0] < end_time - tolerance)
        return (end_time >= start_time) & ~jnp.any(crossings)

    def _polygon(self, time, dtype):
        tip = self.tip_position(time, dtype)
        rake = jnp.deg2rad(self.rake_angle_deg)
        clearance = jnp.deg2rad(self.clearance_angle_deg)
        far = self.length
        # A convex finite wedge: tip, rake-face endpoint, clearance-face
        # endpoint.  Closing the two face endpoints forms the tool's back.
        rake_direction = jnp.array([-jnp.sin(rake), jnp.cos(rake)], dtype=dtype)
        clearance_direction = jnp.array(
            [-jnp.cos(clearance), jnp.sin(clearance)], dtype=dtype
        )
        rake_endpoint = tip + far * rake_direction
        clearance_endpoint = tip + far * clearance_direction
        if self.tip_radius == 0.0:
            return jnp.stack((tip, rake_endpoint, clearance_endpoint))

        # Positive clearance rises behind the virtual tip, above its swept cut plane.
        # Replace the virtual sharp intersection with the circular fillet
        # tangent to both faces. The point sequence remains convex and CCW.
        half_angle = 0.5 * jnp.arccos(
            jnp.clip(jnp.dot(rake_direction, clearance_direction), -1.0, 1.0)
        )
        tangent_distance = self.tip_radius / jnp.tan(half_angle)
        bisector = rake_direction + clearance_direction
        bisector = bisector / jnp.linalg.norm(bisector)
        center = tip + (self.tip_radius / jnp.sin(half_angle)) * bisector
        upper_tangent = tip + tangent_distance * rake_direction
        lower_tangent = tip + tangent_distance * clearance_direction
        upper_offset = upper_tangent - center
        lower_offset = lower_tangent - center
        phi_upper = jnp.arctan2(upper_offset[1], upper_offset[0])
        phi_lower = jnp.arctan2(lower_offset[1], lower_offset[0])
        delta = jnp.mod(phi_upper - phi_lower, 2.0 * jnp.pi)
        fractions = jnp.arange(self.tip_arc_segments, dtype=dtype) / self.tip_arc_segments
        angles = phi_lower + fractions * delta
        arc = center[None, :] + self.tip_radius * jnp.stack(
            (jnp.cos(angles), jnp.sin(angles)), axis=-1
        )
        return jnp.concatenate((upper_tangent[None, :], rake_endpoint[None, :],
                                clearance_endpoint[None, :], arc), axis=0)

    def _project(self, velocity, coordinates, time, mass=None):
        vertices = self._polygon(time, coordinates.dtype)
        starts = vertices
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        lengths = jnp.linalg.norm(edges, axis=-1)
        rel = coordinates[:, None, :2] - starts[None, :, :]
        signed = (edges[None, :, 0] * rel[:, :, 1]
                  - edges[None, :, 1] * rel[:, :, 0]) / lengths[None, :]
        inside = jnp.all(signed >= 0.0, axis=1)
        face = jnp.argmin(signed, axis=1)
        edge = edges[face]
        normal2 = jnp.stack((edge[:, 1], -edge[:, 0]), axis=-1)
        normal2 = normal2 / jnp.linalg.norm(normal2, axis=-1, keepdims=True)
        if velocity.shape[-1] == 3:
            normal = jnp.concatenate((normal2, jnp.zeros((len(velocity), 1), dtype=velocity.dtype)), axis=-1)
        else:
            normal = normal2
        tool_velocity = jnp.zeros((velocity.shape[-1],), dtype=velocity.dtype).at[:2].set(self.velocity(time, velocity.dtype))
        relative_velocity = velocity - tool_velocity
        normal_speed = jnp.sum(relative_velocity * normal, axis=-1)
        correction = jnp.maximum(-normal_speed, 0.0)
        tangent = relative_velocity - normal_speed[:, None] * normal
        tangent_speed = jnp.linalg.norm(tangent, axis=-1)
        friction_scale = jnp.minimum(tangent_speed, self.friction * correction) / jnp.maximum(tangent_speed, 1e-30)
        corrected = velocity + correction[:, None] * normal - friction_scale[:, None] * tangent
        active = inside if mass is None else inside & (mass > 0)
        corrected = jnp.where(active[:, None], corrected, velocity)
        impulse = jnp.zeros((velocity.shape[-1],), dtype=velocity.dtype)
        if mass is not None:
            impulse = jnp.sum(mass[:, None] * (corrected - velocity), axis=0)
        return corrected, impulse, inside

    def apply(self, velocity, coordinates, time):
        return self._project(velocity, coordinates, time)[0]

    def apply_grid(self, mass, velocity, coordinates, time):
        return self._project(velocity, coordinates, time, mass)[0]

    def impulse_on_body(self, mass, velocity, coordinates, time):
        """Impulse applied to the workpiece at the tool boundary during projection."""
        return self._project(velocity, coordinates, time, mass)[1]
