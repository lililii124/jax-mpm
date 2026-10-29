"""Isolated particle-to-convex-wedge projection for 2D plane-strain probes.

This is an after-G2P contact adapter.  It corrects particle position and
velocity and returns the corresponding particle impulse.  It deliberately
does not alter F or constitutive history, and it does not convert friction
work into heat.  Those omissions are explicit model limitations; compare
projected and nodal-contact runs before using this adapter in a production
case.
"""

import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def project_particles_to_wedge(x, v, mass, vertices_xy, tool_velocity_xy,
                                friction_coefficient, separation=0.0, particle_mask=None):
    """Project particles inside a CCW triangle out of its moving 2D boundary.

    Parameters
    ----------
    x, v : (N, 2) arrays
        Particle positions and velocities in SI units.
    mass : (N,) array
        MPM masses per unit out-of-plane thickness (kg/m).
    vertices_xy : (3, 2) array
        Current triangle vertices in counter-clockwise order (m).
    tool_velocity_xy : (2,) array
        Rigid tool translation velocity (m/s).
    friction_coefficient : scalar
        Coulomb coefficient applied to the particle-level normal impulse.
    separation : scalar
        Optional tiny outward clearance (m), normally 0 for the probe.

    Returns
    -------
    x_after, v_after : (N,2) arrays
    diagnostics : (7,) array
        ``[projected_count, max_depth_m, max_position_correction_m,
        impulse_x_Ns_per_m, impulse_y_Ns_per_m,
        delta_kinetic_energy_J_per_m, tool_work_J_per_m]``.  Impulse is the
        impulse applied to particles; tool reaction is its negative.

    The closest point on all three finite segments is used, so points near
    the wedge tip do not use an infinite half-plane projection.  The normal
    velocity is removed only for approaching particles; tangential velocity
    receives a Coulomb-limited reduction.  This per-particle projection
    bypasses the grid's nodal contact impulse and must not be combined with
    that impulse without an explicit double-contact study.
    """
    edges = jnp.roll(vertices_xy, -1, axis=0) - vertices_xy
    rel = x[:, None, :] - vertices_xy[None, :, :]
    edge_sq = jnp.sum(edges * edges, axis=1)
    parameter = jnp.clip(jnp.sum(rel * edges[None, :, :], axis=-1)
                         / edge_sq[None, :], 0.0, 1.0)
    closest = vertices_xy[None, :, :] + parameter[..., None] * edges[None, :, :]
    delta_to_boundary = closest - x[:, None, :]
    distance = jnp.linalg.norm(delta_to_boundary, axis=-1)

    cross = (edges[None, :, 0] * rel[..., 1]
             - edges[None, :, 1] * rel[..., 0])
    signed_plane_distance = cross / jnp.sqrt(edge_sq)[None, :]
    inside = jnp.all(signed_plane_distance >= -1e-14, axis=1)
    face = jnp.argmin(distance, axis=1)
    rows = jnp.arange(x.shape[0])
    nearest = closest[rows, face]
    depth = distance[rows, face]

    edge = edges[face]
    edge_length = jnp.sqrt(edge_sq[face])
    face_outward = jnp.stack((edge[:, 1], -edge[:, 0]), axis=-1) / edge_length[:, None]
    radial_outward = delta_to_boundary[rows, face] / jnp.maximum(depth[:, None], 1e-30)
    # Radial directions resolve the polygon-tip vertex without extending a
    # face plane behind its finite segment.  At zero distance use the face.
    outward = jnp.where((depth > 1e-30)[:, None], radial_outward, face_outward)
    mask = jnp.ones((x.shape[0],), dtype=bool) if particle_mask is None else particle_mask
    active = inside & (depth > 0.0) & mask
    x_after = jnp.where(active[:, None], nearest + separation * outward, x)

    relative_velocity = v - tool_velocity_xy[None, :]
    normal_speed = jnp.sum(relative_velocity * outward, axis=-1)
    approaching = active & (normal_speed < 0.0)
    remove_normal = jnp.where(approaching, -normal_speed, 0.0)
    tangent = relative_velocity - normal_speed[:, None] * outward
    tangent_speed = jnp.linalg.norm(tangent, axis=-1)
    friction_speed = jnp.minimum(tangent_speed,
                                 friction_coefficient * remove_normal)
    tangent_unit = tangent / jnp.maximum(tangent_speed[:, None], 1e-30)
    delta_v = remove_normal[:, None] * outward - friction_speed[:, None] * tangent_unit
    v_after = v + delta_v

    particle_impulse = mass[:, None] * delta_v
    impulse = jnp.sum(particle_impulse, axis=0)
    delta_ke = 0.5 * jnp.sum(mass * (jnp.sum(v_after * v_after, axis=-1)
                                    - jnp.sum(v * v, axis=-1)))
    max_depth = jnp.max(jnp.where(active, depth, 0.0))
    tool_work = jnp.dot(impulse, tool_velocity_xy)
    diagnostics = jnp.stack((jnp.sum(active).astype(x.dtype), max_depth,
                             max_depth + jnp.where(jnp.any(active), separation, 0.0),
                             impulse[0], impulse[1], delta_ke, tool_work))
    return x_after, v_after, diagnostics


def project_state_after_g2p(state, vertices_xy, tool_velocity_xy, friction_coefficient,
                            separation=0.0):
    """Apply the projection to a 2D state while leaving F/history untouched."""
    if state.x.shape[-1] != 2:
        raise ValueError("particle wedge adapter is 2D only")
    x, v, diagnostics = project_particles_to_wedge(
        state.x, state.v, state.mass, vertices_xy, tool_velocity_xy,
        friction_coefficient, separation,
    )
    return state._replace(x=x, v=v), diagnostics


@partial(jax.jit, static_argnames=("bucket_capacity", "max_search_cells", "grid_cells"))
def contact_failed_particles_with_intact(state, friction_coefficient, cell_size,
                                         bucket_capacity=32, max_search_cells=2,
                                         grid_cells=256,
                                         separation=0.0):
    """Bounded spatial-hash contact between failed and intact material points.

    Returns updated state and diagnostics ``[pairs, max_overlap,
    momentum_residual, delta_KE, overflow, required_search_cells]``. Any
    occupancy or search-radius overflow is explicit; callers must stop on it.
    The post-G2P adapter preserves pair COM and equal/opposite impulse, but
    does not update F/history or provide rotational/Coulomb multi-point crack
    contact. It is exploratory and must be gated by the overflow flag.
    """
    x0, v0 = state.x, state.v
    n = x0.shape[0]
    cell_counts = jnp.full((2,), grid_cells, jnp.int32)
    nx, ny = grid_cells, grid_cells
    key_xy = jnp.floor(state.x / cell_size).astype(jnp.int32)
    coordinate_overflow = jnp.any((key_xy < 0) | (key_xy >= grid_cells))
    key_xy = jnp.clip(key_xy, 0, cell_counts[None, :] - 1)
    keys = key_xy[:, 0] * ny + key_xy[:, 1]
    num_cells = nx * ny
    order = jnp.argsort(keys)
    sorted_keys = keys[order]
    ranks_sorted = jnp.arange(n, dtype=jnp.int32) - jnp.searchsorted(
        sorted_keys, sorted_keys, side="left")
    ranks = jnp.zeros((n,), jnp.int32).at[order].set(ranks_sorted)
    buckets = jnp.full((num_cells, bucket_capacity), n, dtype=jnp.int32)
    buckets = buckets.at[keys, ranks].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    occupancy = jnp.zeros((num_cells,), jnp.int32).at[keys].add(1)
    bucket_overflow = jnp.any(occupancy > bucket_capacity)

    current_area = state.volume * jnp.maximum(jnp.linalg.det(state.F), 0.0)
    radius = jnp.sqrt(jnp.maximum(current_area, 0.0) / jnp.pi)
    # Allow the maximum possible mass-weighted separation correction plus
    # sum of radii; ceil makes the cell stencil cover the full interaction.
    required_cells = jnp.ceil((2*jnp.max(radius) + separation) / cell_size).astype(jnp.int32)
    search_overflow = required_cells > max_search_cells
    search_radius = jnp.minimum(required_cells, max_search_cells)
    offsets = jnp.stack(jnp.meshgrid(jnp.arange(-max_search_cells, max_search_cells+1),
                                     jnp.arange(-max_search_cells, max_search_cells+1),
                                     indexing="ij"), axis=-1).reshape((-1,2))
    use_offset = jnp.max(jnp.abs(offsets), axis=1) <= search_radius
    failed = state.history.failed
    failed_indices = jnp.nonzero(failed, size=n, fill_value=0)[0]
    failed_count = jnp.sum(failed.astype(jnp.int32))

    def process(_):
        def body(i, carry):
            x, v, impulse_total, pair_count, max_overlap, stale = carry
            point = failed_indices[i]
            cell = key_xy[point]
            neighbor_xy = cell[None, :] + offsets
            inside = jnp.all((neighbor_xy >= 0) & (neighbor_xy < cell_counts[None, :]), axis=1)
            neighbor_keys = neighbor_xy[:, 0] * ny + neighbor_xy[:, 1]
            candidate_ids = buckets[jnp.clip(neighbor_keys, 0, num_cells-1)]
            candidate_mask = inside[:, None] & use_offset[:, None]
            candidate_ids = jnp.where(candidate_mask, candidate_ids, n).reshape((-1,))
            safe_ids = jnp.minimum(candidate_ids, n-1)
            delta = x[safe_ids] - x[point]
            distance = jnp.linalg.norm(delta, axis=-1)
            overlap = radius[safe_ids] + radius[point] - distance
            eligible = (candidate_ids < n) & (~failed[safe_ids]) & (safe_ids != point) & (overlap > 0)
            score = jnp.where(eligible, distance, jnp.inf)
            slot = jnp.argmin(score)
            j = safe_ids[slot]
            has_pair = jnp.isfinite(score[slot]) & ~bucket_overflow & ~search_overflow & ~coordinate_overflow & ~stale
            direction = delta[slot]
            fallback = jnp.where(jnp.linalg.norm(v[point]-v[j]) > 1e-30,
                                 v[point]-v[j], jnp.array([1.0,0.0],x.dtype))
            direction = jnp.where(distance[slot] > 1e-30,direction,fallback)
            normal = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
            mi, mj = state.mass[point], state.mass[j]
            reduced = mi*mj / jnp.maximum(mi+mj,1e-30)
            relative = v[point]-v[j]
            closing = jnp.dot(relative,normal)
            jn = reduced*jnp.maximum(closing,0.0)
            tangent = relative-closing*normal
            tangent_speed = jnp.linalg.norm(tangent)
            jt = jnp.minimum(reduced*tangent_speed,friction_coefficient*jn)
            impulse = jn*normal + jt*tangent/jnp.maximum(tangent_speed,1e-30)
            impulse = jnp.where(has_pair,impulse,0.0)
            correction = jnp.where(has_pair,jnp.maximum(overlap[slot],0.0)+separation,0.0)
            xi=x[point]-correction*mj/jnp.maximum(mi+mj,1e-30)*normal
            xj=x[j]+correction*mi/jnp.maximum(mi+mj,1e-30)*normal
            x=x.at[point].set(jnp.where(has_pair,xi,x[point])).at[j].set(jnp.where(has_pair,xj,x[j]))
            v=v.at[point].set(jnp.where(has_pair,v[point]-impulse/jnp.maximum(mi,1e-30),v[point]))
            v=v.at[j].set(jnp.where(has_pair,v[j]+impulse/jnp.maximum(mj,1e-30),v[j]))
            new_key_i=jnp.floor(x[point]/cell_size).astype(jnp.int32)
            new_key_j=jnp.floor(x[j]/cell_size).astype(jnp.int32)
            crossed=has_pair & (i+1<failed_count) & (jnp.any(new_key_i!=key_xy[point]) | jnp.any(new_key_j!=key_xy[j]))
            return (x,v,impulse_total+impulse,pair_count+has_pair,
                    jnp.maximum(max_overlap,jnp.where(has_pair,overlap[slot],0.0)),stale|crossed)
        initial=(x0,v0,jnp.zeros((2,),x0.dtype),jnp.array(0,jnp.int32),jnp.array(0.0,x0.dtype),jnp.array(False))
        return jax.lax.fori_loop(0,failed_count,body,initial)

    def no_contact(_):
        return (x0,v0,jnp.zeros((2,),x0.dtype),jnp.array(0,jnp.int32),jnp.array(0.0,x0.dtype),jnp.array(False))

    x,v,impulse,count,max_overlap,stale=jax.lax.cond(jnp.any(failed),process,no_contact,operand=None)
    total_before=jnp.sum(state.mass[:,None]*v0,axis=0)
    total_after=jnp.sum(state.mass[:,None]*v,axis=0)
    kinetic_before=.5*jnp.sum(state.mass*jnp.sum(v0*v0,axis=-1))
    kinetic_after=.5*jnp.sum(state.mass*jnp.sum(v*v,axis=-1))
    diagnostics=jnp.stack((count.astype(x0.dtype),max_overlap,
        jnp.linalg.norm(total_after-total_before),kinetic_after-kinetic_before,
        (jnp.any(failed)&(bucket_overflow|search_overflow|coordinate_overflow|stale)).astype(x0.dtype),required_cells.astype(x0.dtype)))
    return state._replace(x=x,v=v),diagnostics
