"""Optional 2D particle-interpolated rigid-wedge contact at the grid stage.

This adapter replaces the wedge's nodal projection for a step. It builds one
normal constraint per nearby active material point, solves the constraints by
projected Gauss-Seidel in the nodal mass metric, then lets the ordinary MPM
G2P update positions and deformation. It does not perform a post-G2P position
projection. The model is exploratory; finite tool segments, single nearest
normal per particle, and sequential PGS remain discretization choices.
"""
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import core
from .convex_penetration import polygon_penetration
from .boundaries import MovingWedgeTool, VelocityBox


class ContactMetrics(NamedTuple):
    impulse_on_workpiece: jax.Array  # constraint impulse, N s / m
    support_reaction: jax.Array      # grid momentum change - tool impulse
    tool_work: jax.Array             # J / m
    grid_delta_kinetic: jax.Array    # J / m
    active_points: jax.Array
    max_predicted_penetration: jax.Array
    max_actual_penetration: jax.Array
    max_normal_violation: jax.Array
    max_complementarity_error: jax.Array
    max_friction_bound_error: jax.Array
    max_friction_kkt_residual: jax.Array
    bucket_overflow: jax.Array
    converged: jax.Array
    sweeps_used: jax.Array


def _fixed_dofs(config, time, tool, dtype):
    """Return prescribed grid components to exclude from contact impulses."""
    n = config.n_grid + 1
    axis = jnp.arange(n, dtype=dtype) * config.dx
    coords = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    ij = jnp.stack(jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij"), axis=-1)
    fixed = jnp.zeros((coords.shape[0], 2), dtype=bool)
    if config.boundary_cells:
        for d in range(2):
            fixed = fixed.at[:, d].set(
                (ij[..., d].reshape(-1) < config.boundary_cells)
                | (ij[..., d].reshape(-1) > config.n_grid-config.boundary_cells))
    for boundary in config.boundaries:
        if boundary is tool:
            continue
        if type(boundary) is not VelocityBox:
            raise TypeError(f"particle-wedge contact currently supports static VelocityBox boundaries only; got {type(boundary).__name__}")
        lower = jnp.asarray(boundary.lower, dtype=dtype)
        upper = jnp.asarray(boundary.upper, dtype=dtype)
        active = jnp.all((coords >= lower) & (coords <= upper), axis=1)
        start = getattr(boundary, "start", 0.0)
        end = getattr(boundary, "end", np.inf)
        active = active & (time >= start) & (time <= end)
        components = getattr(boundary, "components", None)
        components = jnp.ones(2, dtype=bool) if components is None else jnp.asarray(components, dtype=bool)
        fixed = fixed | (active[:, None] & components[None, :])
    return fixed


def _contact_geometry(x, polygon):
    edges = jnp.roll(polygon, -1, axis=0) - polygon
    rel = x[:, None, :] - polygon[None, :, :]
    edge_sq = jnp.sum(edges * edges, axis=1)
    parameter = jnp.clip(jnp.sum(rel * edges[None, :, :], axis=-1)
                         / edge_sq[None, :], 0.0, 1.0)
    closest = polygon[None, :, :] + parameter[..., None] * edges[None, :, :]
    delta = x[:, None, :] - closest
    distance = jnp.linalg.norm(delta, axis=-1)
    face = jnp.argmin(distance, axis=1)
    row = jnp.arange(x.shape[0])
    nearest = closest[row, face]
    nearest_delta = delta[row, face]
    d = distance[row, face]
    edge = edges[face]
    edge_length = jnp.sqrt(edge_sq[face])
    face_normals = jnp.stack((edges[:, 1], -edges[:, 0]), axis=-1)
    face_normals /= jnp.maximum(jnp.sqrt(edge_sq)[:, None], 1e-30)
    outward_face = face_normals[face]
    parameter_selected = parameter[row, face]
    scale = jnp.maximum(jnp.max(jnp.sqrt(edge_sq)), 1e-12)
    geometry_tolerance = 8 * jnp.finfo(x.dtype).eps * scale
    parameter_tolerance = 8 * jnp.finfo(x.dtype).eps
    segment_interior = ((parameter_selected > parameter_tolerance)
                        & (parameter_selected < 1-parameter_tolerance))
    start_normal = face_normals[face] + face_normals[(face-1) % edges.shape[0]]
    end_normal = face_normals[face] + face_normals[(face+1) % edges.shape[0]]
    vertex_normal = jnp.where((parameter_selected <= .5)[:,None],start_normal,end_normal)
    vertex_normal /= jnp.maximum(jnp.linalg.norm(vertex_normal,axis=-1,keepdims=True),1e-30)
    outward_radial = nearest_delta / jnp.maximum(d[:, None], geometry_tolerance)
    cross = edges[None, :, 0] * rel[..., 1] - edges[None, :, 1] * rel[..., 0]
    signed_interior = cross / jnp.sqrt(edge_sq)[None, :]
    # Strict signed-halfspace classification avoids treating a real exterior
    # point within a fixed metric epsilon as material inside the tool.
    inside = jnp.all(signed_interior >= 0.0, axis=1)
    # Positive gap is outside; negative gap is inside the solid wedge.
    gap = jnp.where(inside, -d, d)
    # A face normal is exact and well-conditioned all along an interior
    # segment, even for points 1e-20 m from it. Radial normals are reserved
    # for a genuine exterior vertex region; at machine-scale vertex distance
    # use the normalized adjacent-face support normal.
    exterior_vertex = (~inside) & (~segment_interior)
    outward = jnp.where(exterior_vertex[:,None],outward_radial,outward_face)
    tiny_vertex = exterior_vertex & (d <= geometry_tolerance)
    outward = jnp.where(tiny_vertex[:,None],vertex_normal,outward)
    return gap, outward


def gather_projected_lambda_cache(cache, particle_ids, normals, active_count,
                                  friction, min_normal_dot=0.9):
    """Gather accepted multipliers by particle ID and project onto current disk."""
    slots = jnp.arange(particle_ids.shape[0])
    active = slots < active_count
    old = cache[particle_ids]
    alignment = jnp.sum(old[:, 2:4] * normals, axis=-1)
    usable = active & (old[:, 4] > 0.5) & (alignment >= min_normal_dot)
    lam_n = jnp.where(usable, jnp.maximum(old[:, 0], 0.0), 0.0)
    lam_t_raw = jnp.where(usable, old[:, 1], 0.0)
    lam_t = jnp.clip(lam_t_raw, -friction * lam_n, friction * lam_n)
    return lam_n, lam_t


def apply_warm_lambda_to_grid(free_grid_velocity, node_ids, weights, inv_mass,
                              normals, lam_n, lam_t, active_count):
    """Apply cached impulse only to the PGS initial iterate, not the ledger base."""
    active = jnp.arange(node_ids.shape[0]) < active_count
    tangent = jnp.stack((-normals[:, 1], normals[:, 0]), axis=-1)
    impulse = lam_n[:, None] * normals + lam_t[:, None] * tangent
    delta = weights[:, :, None] * impulse[:, None, :] * inv_mass[node_ids]
    delta = jnp.where(active[:, None, None], delta, 0.0)
    return free_grid_velocity.at[node_ids.reshape(-1)].add(
        delta.reshape(-1, free_grid_velocity.shape[-1]))


def scatter_lambda_cache(particle_count, particle_ids, normals, lam_n, lam_t,
                         active_count, dtype):
    """Build cache from final PGS multipliers; inactive IDs become zero."""
    active = jnp.arange(particle_ids.shape[0]) < active_count
    rows = jnp.concatenate((lam_n[:, None], lam_t[:, None], normals,
                            active.astype(dtype)[:, None]), axis=-1)
    rows = jnp.where(active[:, None], rows, 0.0).astype(dtype)
    return jnp.zeros((particle_count, 5), dtype=dtype).at[particle_ids].add(rows)


@partial(jax.jit, static_argnames=('capacity','sweeps','max_sweeps','residual_prefix'))
def _solve_grid_contact(grid_mass, grid_velocity, support, x, particle_active,
        polygon, tool_velocity, fixed_dofs, dt, activation_band, friction,
        capacity, sweeps, max_sweeps, velocity_tolerance, residual_prefix=128,
        lambda_cache=None):
    """PGS candidate hoisting per-step candidate stencils and coefficients.

    The point order, update formulas, sweep logic, and thresholds mirror the
    baseline. residual_prefix=128 shortens only the masked residual reduction
    when active_count<=128; otherwise it takes the full-capacity path.
    """
    gap, normal = _contact_geometry(x, polygon)
    particle_velocity = jnp.sum(support.weights[..., None]
                                 * grid_velocity[support.indices], axis=1)
    rel = particle_velocity - tool_velocity[None, :]
    vn = jnp.sum(rel * normal, axis=-1)
    predicted_gap = gap + dt * jnp.minimum(vn, 0.0)
    candidates = particle_active & (predicted_gap <= activation_band)
    count = jnp.sum(candidates.astype(jnp.int32))
    overflow = count > capacity
    active_count = jnp.minimum(count, capacity)
    ids = jnp.nonzero(candidates, size=capacity, fill_value=0)[0]

    node_free = (~fixed_dofs) & (grid_mass[:, None] > 0)
    inv_mass = jnp.where(node_free, 1 / jnp.maximum(grid_mass[:, None], 1e-30), 0.0)
    tangent = jnp.stack((-normal[:, 1], normal[:, 0]), axis=-1)
    candidate_nodes = support.indices[ids]
    candidate_weights = support.weights[ids]
    candidate_normal = normal[ids]
    candidate_tangent = tangent[ids]
    candidate_free = node_free[candidate_nodes]
    candidate_invm = inv_mass[candidate_nodes]
    candidate_wn = jax.vmap(lambda w, invm, nrm: jnp.sum(
        w[:, None]**2 * invm * nrm[None, :]**2))(
            candidate_weights, candidate_invm, candidate_normal)
    candidate_wt = jax.vmap(lambda w, invm, tan: jnp.sum(
        w[:, None]**2 * invm * tan[None, :]**2))(
            candidate_weights, candidate_invm, candidate_tangent)
    candidate_target = -gap[ids] / dt
    slots = jnp.arange(capacity)
    if lambda_cache is None:
        lambda_cache = jnp.zeros((x.shape[0], 5), dtype=x.dtype)
    lam_n, lam_t = gather_projected_lambda_cache(
        lambda_cache, ids, candidate_normal, active_count, friction)
    warm_velocity = apply_warm_lambda_to_grid(
        grid_velocity, candidate_nodes, candidate_weights, inv_mass,
        candidate_normal, lam_n, lam_t, active_count)

    def sweep_body(_, carry):
        velocity, ln, lt = carry
        def point_body(k, inner):
            velocity, ln, lt = inner
            node_ids = candidate_nodes[k]
            w = candidate_weights[k]
            nrm = candidate_normal[k]
            tan = candidate_tangent[k]
            free = candidate_free[k]
            invm = candidate_invm[k]
            wn = candidate_wn[k]
            wt = candidate_wt[k]
            vp = jnp.sum(w[:, None] * velocity[node_ids], axis=0)
            relative = vp - tool_velocity
            target = candidate_target[k]
            current_ln = ln[k]
            proposed_ln = jnp.maximum(0.0, current_ln + (target - jnp.dot(relative, nrm))
                                      / jnp.maximum(wn, 1e-30))
            dln = jnp.where(k < count, proposed_ln-current_ln, 0.0)
            dvn = w[:, None] * dln * nrm[None, :] * invm
            dvn = jnp.where(free, dvn, 0.0)
            velocity = velocity.at[node_ids].add(dvn)
            vp = jnp.sum(w[:, None] * velocity[node_ids], axis=0)
            relative = vp - tool_velocity
            max_t = friction * (current_ln + dln)
            proposed_lt = jnp.clip(lt[k] - jnp.dot(relative, tan)
                                   / jnp.maximum(wt, 1e-30), -max_t, max_t)
            dlt = jnp.where(k < count, proposed_lt-lt[k], 0.0)
            dvt = w[:, None] * dlt * tan[None, :] * invm
            dvt = jnp.where(free, dvt, 0.0)
            velocity = velocity.at[node_ids].add(dvt)
            ln = ln.at[k].set(jnp.where(k < count, current_ln+dln, ln[k]))
            lt = lt.at[k].set(jnp.where(k < count, proposed_lt, lt[k]))
            return velocity, ln, lt
        return jax.lax.fori_loop(0, active_count, point_body, (velocity, ln, lt))

    def residuals_size(velocity, ln, lt, size):
        nodes = candidate_nodes[:size]
        weights = candidate_weights[:size]
        nrm = candidate_normal[:size]
        tan = candidate_tangent[:size]
        target = candidate_target[:size]
        wt = candidate_wt[:size]
        vp = jnp.sum(weights[:, :, None] * velocity[nodes], axis=1)
        rel = vp - tool_velocity[None, :]
        vn_end = jnp.sum(rel * nrm, axis=-1)
        slot_active = slots[:size] < active_count
        normal_violation = jnp.max(jnp.where(slot_active, jnp.maximum(target-vn_end, 0.0), 0.0))
        complementarity = jnp.max(jnp.where(slot_active & (ln[:size] > 1e-15),
                                             jnp.abs(vn_end-target), 0.0))
        friction_bound = jnp.max(jnp.where(slot_active,
            jnp.maximum(jnp.abs(lt[:size])-friction*ln[:size],0.0),0.0))
        tangential = jnp.sum(rel*tan, axis=-1)
        tangent_projected = jnp.clip(lt[:size]-tangential/jnp.maximum(wt,1e-30),
                                      -friction*ln[:size],friction*ln[:size])
        friction_kkt = jnp.max(jnp.where(slot_active,
            wt*jnp.abs(lt[:size]-tangent_projected),0.0))
        return normal_violation, complementarity, friction_bound, friction_kkt

    def residuals(velocity, ln, lt):
        if residual_prefix > 0:
            return jax.lax.cond(active_count <= residual_prefix,
                lambda _: residuals_size(velocity,ln,lt,residual_prefix),
                lambda _: residuals_size(velocity,ln,lt,capacity), operand=None)
        return residuals_size(velocity,ln,lt,capacity)

    corrected, lam_n, lam_t = jax.lax.fori_loop(0, sweeps, sweep_body,
                                                (warm_velocity, lam_n, lam_t))
    def need_more(carry):
        used, velocity, ln, lt = carry
        nv, comp, fb, fk = residuals(velocity, ln, lt)
        return (used < max_sweeps) & ((jnp.maximum(nv, comp) > velocity_tolerance)
            | (fb > 1e-12) | (fk > velocity_tolerance))
    def adaptive_body(carry):
        used, velocity, ln, lt = carry
        velocity, ln, lt = sweep_body(0, (velocity, ln, lt))
        return used + 1, velocity, ln, lt
    used, corrected, lam_n, lam_t = jax.lax.while_loop(
        need_more, adaptive_body, (jnp.asarray(sweeps,jnp.int32),corrected,lam_n,lam_t))
    normal_violation, complementarity, friction_bound, friction_kkt = residuals(
        corrected,lam_n,lam_t)
    slot_active = slots < active_count
    max_pred_pen = jnp.max(jnp.where(particle_active,
        jnp.maximum(-(gap+dt*jnp.sum((jnp.sum(support.weights[...,None]
           * corrected[support.indices],axis=1)-tool_velocity[None,:])*normal,axis=-1)),0.0),0.0))
    grid_momentum_change = jnp.sum(grid_mass[:,None]*(corrected-grid_velocity),axis=0)
    impulse = jnp.sum(jnp.where(slot_active[:,None],
        lam_n[:,None]*candidate_normal+lam_t[:,None]*candidate_tangent,0.0),axis=0)
    support_reaction = grid_momentum_change-impulse
    delta_ke = .5*jnp.sum(grid_mass*(jnp.sum(corrected**2,axis=-1)
                                      -jnp.sum(grid_velocity**2,axis=-1)))
    residual = jnp.maximum(normal_violation, complementarity)
    converged = (~overflow) & (residual <= velocity_tolerance) \
                & (friction_bound <= 1e-12) & (friction_kkt <= velocity_tolerance)
    metrics = ContactMetrics(impulse, support_reaction,
        jnp.dot(impulse,tool_velocity), delta_ke, count, max_pred_pen,
        jnp.asarray(0.0,dtype=x.dtype), normal_violation, complementarity,
        friction_bound, friction_kkt, overflow, converged, used)
    next_cache = scatter_lambda_cache(x.shape[0], ids, candidate_normal,
                                      lam_n, lam_t, active_count, x.dtype)
    return corrected, metrics, next_cache


def step_with_particle_wedge_contact(state, material, config, tool, *,
                                     capacity=1024, sweeps=12,
                                     max_sweeps=None,
                                     activation_cells=0.5,
                                     gap_tolerance_cells=1e-3,
                                     velocity_tolerance=1e-6,
                                     force=None, cache=None):
    """Advance a 2D state with the wedge represented by particle constraints.

    The tool's existing `apply_grid` boundary is removed for this step so the
    PGS impulse replaces, rather than stacks with, its nodal projection.
    Other configured boundaries remain applied. `State.valid` is cleared on
    capacity overflow or unmet contact residual; the caller must checkpoint
    only the prior valid state in that case.
    """
    if config.dim != 2 or state.x.shape[-1] != 2:
        raise ValueError("particle-interpolated wedge contact currently requires dim=2")
    if not isinstance(capacity,int) or capacity < 1 or not isinstance(sweeps,int) or sweeps < 1:
        raise ValueError("capacity and sweeps must be positive integers")
    max_sweeps=sweeps if max_sweeps is None else max_sweeps
    if not isinstance(max_sweeps,int) or max_sweeps < sweeps:
        raise ValueError("max_sweeps must be an integer >= sweeps")
    if activation_cells < 0 or gap_tolerance_cells <= 0 or velocity_tolerance <= 0:
        raise ValueError("activation band must be nonnegative and gap/velocity tolerances positive")
    matches=[b for b in config.boundaries if b is tool]
    if len(matches)!=1:
        raise ValueError("the exact MovingWedgeTool object must appear once in config.boundaries")
    non_tool=tuple(b for b in config.boundaries if b is not tool)
    config_grid=replace(config,boundaries=non_tool)
    support=core.particle_stencil(state,config)
    gm,gp=core.particle_to_grid(state,material,config,support,force)
    gv=core.grid_update(gm,gp,config_grid,state.time+config.dt)
    # Gap is measured from the current particle/tool configuration; the
    # relative-velocity constraint predicts the end-of-step gap.
    polygon=tool._polygon(state.time,state.x.dtype)
    tool_velocity=tool.interval_velocity(state.time, state.time+config.dt, state.x.dtype)
    fixed=_fixed_dofs(config_grid,state.time+config.dt,tool,state.x.dtype)
    active=core.particle_activity(state,config)
    corrected,metrics,next_cache=_solve_grid_contact(gm,gv,support,state.x,active,polygon,
        tool_velocity,fixed,config.dt,activation_cells*config.dx,tool.friction,
        capacity,sweeps,max_sweeps,velocity_tolerance,lambda_cache=cache)
    updated=core.particle_update(state,material,config,support,corrected,force)
    end_polygon=tool._polygon(updated.time,updated.x.dtype)
    end_gap=-polygon_penetration(updated.x,end_polygon)
    max_actual_penetration=jnp.max(jnp.where(active,jnp.maximum(-end_gap,0.0),0.0))
    gap_tolerance=gap_tolerance_cells*config.dx
    metrics=metrics._replace(max_actual_penetration=max_actual_penetration,
        converged=metrics.converged & (max_actual_penetration<=gap_tolerance))
    updated=updated._replace(valid=updated.valid & metrics.converged & tool.interval_is_linear(state.time, updated.time))
    return updated,metrics,next_cache


def particle_wedge_step_adapter(state, material, config, force=None, *,
                                capacity=1024, sweeps=12,
                                max_sweeps=None,
                                activation_cells=0.5,
                                gap_tolerance_cells=1e-3,
                                velocity_tolerance=1e-6,
                                cache=None):
    """`examples.common.run` adapter returning state and observer metrics."""
    tools=[b for b in config.boundaries if isinstance(b,MovingWedgeTool)]
    if len(tools)!=1:
        raise ValueError("particle-wedge adapter requires exactly one MovingWedgeTool boundary")
    updated,m,next_cache=step_with_particle_wedge_contact(state,material,config,tools[0],
        capacity=capacity,sweeps=sweeps,activation_cells=activation_cells,
        max_sweeps=max_sweeps,
        gap_tolerance_cells=gap_tolerance_cells,
        velocity_tolerance=velocity_tolerance,force=force,cache=cache)
    values=jnp.stack((m.active_points.astype(state.x.dtype),
        m.max_predicted_penetration,m.max_actual_penetration,
        m.impulse_on_workpiece[0],m.impulse_on_workpiece[1],
        m.support_reaction[0],m.support_reaction[1],m.tool_work,
        m.grid_delta_kinetic,m.max_normal_violation,m.max_complementarity_error,
        m.max_friction_kkt_residual,m.bucket_overflow.astype(state.x.dtype),
        (~m.converged).astype(state.x.dtype),m.sweeps_used.astype(state.x.dtype)))
    result=(updated,m.impulse_on_workpiece,m.tool_work,values)
    return result if cache is None else (*result,next_cache)
