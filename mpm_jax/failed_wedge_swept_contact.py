"""Swept collision of already-failed ballistic particles with a translating wedge.

This small model handles tool contact only. Failed points remain outside the
active MPM transfer and are advanced freely except when their start-of-step
state sweeps through the rigid polygon. It does not model debris-to-workpiece
or debris-to-debris contact, and it freezes F and constitutive history.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from .convex_penetration import polygon_penetration


class SweptContactMetrics(NamedTuple):
    particle_impulse: jax.Array  # impulse applied to the ballistic workpiece points, N s / m
    hit_mask: jax.Array
    tool_work: jax.Array         # impulse dot prescribed tool velocity, J / m
    delta_kinetic: jax.Array     # inertial particle kinetic energy change, J / m
    collision_loss: jax.Array    # tool work - delta kinetic energy, J / m
    collisions: jax.Array
    invalid_start: jax.Array
    invalid_energy: jax.Array
    max_penetration: jax.Array
    max_all_particle_penetration: jax.Array
    all_particle_guard_failed: jax.Array
    persistent_touch_points: jax.Array
    persistent_touch_max_correction: jax.Array
    persistent_touch_max_rejected_correction: jax.Array
    post_reconstruction_max_penetration: jax.Array
    post_reconstruction_max_all_particle_penetration: jax.Array


def _polygon_gap(x, polygon):
    """Signed distance to a convex CCW polygon; negative values are inside."""
    edges = jnp.roll(polygon, -1, axis=0) - polygon
    rel = x[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * rel[..., 1] - edges[None, :, 1] * rel[..., 0]
    inside = jnp.all(cross >= 0.0, axis=1)
    edge_sq = jnp.sum(edges * edges, axis=1)
    parameter = jnp.clip(jnp.sum(rel * edges[None, :, :], axis=-1)
                         / edge_sq[None, :], 0.0, 1.0)
    closest = polygon[None, :, :] + parameter[..., None] * edges[None, :, :]
    distance = jnp.sqrt(jnp.min(jnp.sum((x[:, None, :] - closest) ** 2, axis=-1), axis=1))
    return jnp.where(inside, -distance, distance)


@jax.jit
def _swept_step(x_start, v_start, mass, failed_at_start, polygon_start,
                tool_velocity, dt, friction, dx):
    edges = jnp.roll(polygon_start, -1, axis=0) - polygon_start
    lengths = jnp.linalg.norm(edges, axis=1)
    normals = jnp.stack((edges[:, 1], -edges[:, 0]), axis=-1) / lengths[:, None]
    rel = x_start[:, None, :] - polygon_start[None, :, :]
    signed_out = jnp.sum(rel * normals[None, :, :], axis=-1)
    geometry_scale = jnp.maximum(jnp.max(lengths), 1e-12)
    geometry_tolerance = jnp.maximum(
        64.0 * jnp.finfo(x_start.dtype).eps * geometry_scale, 1e-9 * dx)
    # Only a point strictly inside all half-spaces is an invalid seed. A point
    # on a face after a prior impact is admissible; roundoff-scale signed gaps
    # are set to zero so inward motion gets a zero-time hit on the next step.
    strict_inside = jnp.all(signed_out < -geometry_tolerance, axis=1)
    sweep_signed_out = jnp.where(jnp.abs(signed_out) <= geometry_tolerance,
                                 0.0, signed_out)

    relative_velocity = v_start - tool_velocity[None, :]
    normal_velocity = jnp.einsum('nd,ed->ne', relative_velocity, normals)
    # Each face defines one time half-space. Their intersection gives the
    # entry/exit interval of the particle's line in tool-relative coordinates.
    closing = normal_velocity < 0.0
    opening = normal_velocity > 0.0
    entry_face_time = jnp.where(closing, -sweep_signed_out / jnp.where(closing, normal_velocity, -1.0), -jnp.inf)
    exit_face_time = jnp.where(opening, -sweep_signed_out / jnp.where(opening, normal_velocity, 1.0), jnp.inf)
    entry_face = jnp.argmax(entry_face_time, axis=1)
    entry_time = jnp.max(entry_face_time, axis=1)
    exit_time = jnp.min(exit_face_time, axis=1)
    parallel_outside = jnp.any((normal_velocity == 0.0) & (signed_out > 0.0), axis=1)
    speed_scale = jnp.maximum(jnp.linalg.norm(relative_velocity, axis=1), jnp.linalg.norm(tool_velocity))
    time_tol = (32.0 * jnp.finfo(x_start.dtype).eps * jnp.maximum(dt, 1e-30)
                + geometry_tolerance / jnp.maximum(speed_scale, 1e-30))
    hits = (failed_at_start & ~strict_inside & ~parallel_outside
            & jnp.any(closing, axis=1)
            & (entry_time >= -time_tol) & (entry_time <= dt + time_tol)
            & (entry_time <= exit_time + time_tol))
    invalid_start = jnp.any(failed_at_start & strict_inside)

    row = jnp.arange(x_start.shape[0])
    n = normals[entry_face]
    tangent = edges[entry_face] / lengths[entry_face, None]
    hit_time = jnp.clip(entry_time, 0.0, dt)
    vn = jnp.sum(relative_velocity * n, axis=1)
    jn = -mass * jnp.minimum(vn, 0.0)
    tangent_velocity = relative_velocity - vn[:, None] * n
    tangent_speed = jnp.linalg.norm(tangent_velocity, axis=1)
    jt_mag = jnp.minimum(mass * tangent_speed, friction * jn)
    tangent_direction = tangent_velocity / jnp.maximum(tangent_speed[:, None], 1e-30)
    particle_impulse_each = jn[:, None] * n - jt_mag[:, None] * tangent_direction
    particle_impulse_each = jnp.where(hits[:, None], particle_impulse_each, 0.0)
    v_after_hit = v_start + particle_impulse_each / jnp.maximum(mass[:, None], 1e-30)
    # Reconstruct the post-impact point in the moving face frame. This pins
    # the normal coordinate to the analytically translated face and evolves
    # only the tangential coordinate, avoiding cumulative world-space drift.
    face_start = polygon_start[entry_face]
    face_at_hit = face_start + tool_velocity[None, :] * hit_time[:, None]
    x_at_hit_rel = x_start + v_start * hit_time[:, None] - face_at_hit
    impact_s = jnp.sum(x_at_hit_rel * tangent, axis=1)
    relative_tangent_speed = jnp.sum((v_after_hit-tool_velocity[None,:])*tangent,axis=1)
    face_end = face_start + tool_velocity[None, :] * dt
    x_after_hit = face_end + (impact_s + relative_tangent_speed*(dt-hit_time))[:,None] * tangent
    x_after = jnp.where(hits[:, None], x_after_hit, x_start + v_start * dt)
    v_after = jnp.where(hits[:, None], v_after_hit, v_start)

    impulse = jnp.sum(particle_impulse_each, axis=0)
    tool_work = jnp.dot(impulse, tool_velocity)
    # Stable impulse-work identity avoids subtracting nearly equal squared
    # velocities in grazing collisions: 螖K = J路v_free + |J|虏/(2m).
    delta_ke_each = (jnp.sum(particle_impulse_each * v_start, axis=1)
                     + jnp.sum(particle_impulse_each**2, axis=1)
                     / (2.0 * jnp.maximum(mass, 1e-30)))
    delta_ke = jnp.sum(jnp.where(hits, delta_ke_each, 0.0))
    collision_loss = tool_work - delta_ke
    end_polygon = polygon_start + tool_velocity[None, :] * dt
    end_gap = -polygon_penetration(x_after, end_polygon)
    max_penetration = jnp.max(jnp.where(failed_at_start, jnp.maximum(-end_gap, 0.0), 0.0))
    max_all_particle_penetration = jnp.max(jnp.maximum(-end_gap, 0.0))
    all_particle_guard_failed = max_all_particle_penetration > 0.5 * dx
    energy_scale = jnp.maximum(jnp.abs(tool_work) + jnp.abs(delta_ke), 1e-30)
    invalid_energy = collision_loss < -1e-10 * energy_scale
    metrics = SweptContactMetrics(
        impulse, hits, tool_work, delta_ke, collision_loss,
        jnp.sum(hits.astype(jnp.int32)), invalid_start, invalid_energy,
        max_penetration, max_all_particle_penetration, all_particle_guard_failed,
        jnp.asarray(0,jnp.int32), jnp.asarray(0.0,x_start.dtype),
        jnp.asarray(0.0,x_start.dtype), jnp.asarray(0.0,x_start.dtype),
        jnp.asarray(0.0,x_start.dtype),
    )
    return x_after, v_after, metrics


@jax.jit
def preserve_persistent_failed_face_touch(x_start, x_free, v_free, mass,
                                          failed_start, hit_mask, polygon_start,
                                          polygon_end, tool_velocity, dt,
                                          residual_tolerance):
    """Reconstruct same-face endpoints in local coordinates without clipping gap.

    This avoids accumulating world-coordinate ULP drift while preserving both
    opening and closing normal motion as g1 = g0 + vn*dt. The swept CCD mask,
    finite-face checks, and the existing displacement bound remain unchanged.
    """
    edges = jnp.roll(polygon_start, -1, axis=0) - polygon_start
    lengths = jnp.linalg.norm(edges, axis=1)
    tangents = edges / lengths[:, None]
    normals = jnp.stack((edges[:, 1], -edges[:, 0]), axis=-1) / lengths[:, None]
    rel = x_start[:, None, :] - polygon_start[None, :, :]
    signed = jnp.sum(rel * normals[None, :, :], axis=-1)
    projection = jnp.sum(rel * tangents[None, :, :], axis=-1)
    relative_velocity = v_free - tool_velocity[None, :]
    normal_speed = jnp.einsum('nd,ed->ne', relative_velocity, normals)
    scale = jnp.maximum(jnp.max(lengths), 1e-12)
    geometry_roundoff = 64 * jnp.finfo(x_start.dtype).eps * scale
    normal_end = signed + normal_speed * dt
    end_s = projection + jnp.einsum('nd,ed->ne', relative_velocity, tangents) * dt
    other_halfspaces_inside = jnp.all(signed <= residual_tolerance, axis=1)
    same_face = ((projection > geometry_roundoff) & (projection < lengths[None, :] - geometry_roundoff)
                 & (end_s > geometry_roundoff) & (end_s < lengths[None, :] - geometry_roundoff))
    eligible = (failed_start[:, None] & ~hit_mask[:, None]
                & (jnp.abs(signed) <= residual_tolerance)
                & other_halfspaces_inside[:, None]
                & same_face)
    score = jnp.where(eligible, jnp.abs(signed), jnp.inf)
    face = jnp.argmin(score, axis=1)
    has_touch = jnp.any(eligible, axis=1)
    rows = jnp.arange(x_start.shape[0])
    tangent = tangents[face]
    face_end = polygon_end[face]
    candidate = (face_end + end_s[rows, face, None] * tangent
                 + normal_end[rows, face, None] * normals[face])
    correction = jnp.linalg.norm(candidate - x_free, axis=1)
    apply = has_touch & (correction <= residual_tolerance)
    x_out = jnp.where(apply[:, None], candidate, x_free)
    metrics = jnp.stack((jnp.sum(apply.astype(x_start.dtype)),
                         jnp.max(jnp.where(apply, correction, 0.0)),
                         jnp.max(jnp.where(has_touch & ~apply, correction, 0.0)),
                         jnp.max(jnp.where(apply, mass, 0.0))))
    return x_out, metrics


def collide_failed_particles_with_translating_wedge(state_start, state_updated,
                                                      config, tool):
    """Apply a continuous sweep only to particles failed before this step.

    The normal MPM update has already run for intact points. Since start-failed
    points are inactive, replace their free-flight x/v with the swept result;
    preserve the updated deformation and all history fields exactly.
    """
    failed_start = state_start.history.failed
    polygon_start = tool._polygon(state_start.time, state_start.x.dtype)
    tool_velocity = tool.interval_velocity(state_start.time, state_updated.time, state_start.x.dtype)
    # G2P has already provided the unimpeded end-of-substep velocity. Use it
    # for the swept trajectory while retaining the exact start position.
    x_ballistic, v_ballistic, metrics = _swept_step(
        state_start.x, state_updated.v, state_start.mass, failed_start,
        polygon_start, tool_velocity, config.dt,
        tool.friction, config.dx,
    )
    replace_mask = metrics.hit_mask
    x = jnp.where(replace_mask[:, None], x_ballistic, state_updated.x)
    v = jnp.where(replace_mask[:, None], v_ballistic, state_updated.v)
    end_polygon = tool._polygon(state_updated.time, state_updated.x.dtype)
    all_end_gap = -polygon_penetration(x, end_polygon)
    max_all_particle_penetration = jnp.max(jnp.maximum(-all_end_gap, 0.0))
    all_particle_guard_failed = ((max_all_particle_penetration > 0.5 * config.dx)
                                 | ~tool.interval_is_linear(state_start.time, state_updated.time))
    metrics = metrics._replace(max_all_particle_penetration=max_all_particle_penetration,
                               all_particle_guard_failed=all_particle_guard_failed)
    # Keep original CCD metrics intact, then reconstruct only x for old-
    # failed, no-hit points remaining on the same finite face. The signed
    # normal gap advances by vn*dt, including its sign; no velocity or history
    # is changed. Use the tool's actual endpoint polygon to avoid a second,
    # independently rounded translation of the face.
    x_touch, touch_metrics = preserve_persistent_failed_face_touch(
        state_start.x, x, v, state_start.mass, failed_start, metrics.hit_mask,
        polygon_start, end_polygon, tool_velocity, config.dt, 1e-9 * config.dx)
    post_end_gap = -polygon_penetration(x_touch, end_polygon)
    max_post_failed_penetration = jnp.max(jnp.where(
        failed_start, jnp.maximum(-post_end_gap, 0.0), 0.0))
    max_post_all_particle_penetration = jnp.max(jnp.maximum(-post_end_gap, 0.0))
    metrics = metrics._replace(
        persistent_touch_points=touch_metrics[0].astype(jnp.int32),
        persistent_touch_max_correction=touch_metrics[1],
        persistent_touch_max_rejected_correction=touch_metrics[2],
        post_reconstruction_max_penetration=max_post_failed_penetration,
        post_reconstruction_max_all_particle_penetration=max_post_all_particle_penetration,
        all_particle_guard_failed=(metrics.all_particle_guard_failed
                                   | (max_post_all_particle_penetration > 0.5 * config.dx)))
    x = x_touch
    bad = (metrics.invalid_start | metrics.invalid_energy
           | (metrics.max_penetration > 1e-9 * config.dx)
           | metrics.all_particle_guard_failed
           | (metrics.post_reconstruction_max_penetration > 1e-9 * config.dx))
    return state_updated._replace(x=x, v=v, valid=state_updated.valid & ~bad), metrics
