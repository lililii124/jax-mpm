"""Balanced frictionless contact between grid nodes and a finite-mass sphere."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SphereState(NamedTuple):
    center: jax.Array
    velocity: jax.Array
    mass: jax.Array
    radius: jax.Array


class SphereProjection(NamedTuple):
    velocity: jax.Array
    sphere_velocity: jax.Array
    impulse_on_body: jax.Array
    kinetic_loss: jax.Array
    impulse_residual: jax.Array
    normal_velocity_violation: jax.Array
    complementarity_residual: jax.Array
    iterations: jax.Array
    valid: jax.Array


@partial(jax.jit, static_argnames=('max_iterations', 'tolerance'))
def project_sphere_contact(mass, velocity, coordinates, sphere, fixed=None,
                           *, max_iterations=20, tolerance=1e-10):
    """Solve the shared sphere velocity, rather than spending its inertia per node.

    Normal impulses are nonnegative and satisfy nonpenetrating relative velocity
    on occupied nodes currently inside the sphere. The geometry is explicit;
    the normal projection is perfectly inelastic. Elastic rebound belongs to the
    deformable body's subsequent stress response. Radial impulses have no spin
    torque, so a frictionless sphere needs only translational dynamics.
    """
    if type(max_iterations) is not int or max_iterations < 1 or not 0 < tolerance < 1:
        raise ValueError('A positive iteration budget and tolerance in (0,1) are required')
    if velocity.shape != coordinates.shape or velocity.shape != (mass.shape[0], 3):
        raise ValueError('Grid mass and three-component velocity/coordinates must match')
    fixed = jnp.zeros(mass.shape, dtype=bool) if fixed is None else fixed
    if fixed.shape != mass.shape:
        raise ValueError('Fixed-node flags must match grid mass')
    radial = coordinates - sphere.center
    distance = jnp.linalg.norm(radial, axis=-1)
    normal = radial / jnp.maximum(distance, jnp.finfo(velocity.dtype).tiny)[:, None]
    occupied = mass > 0
    inside = occupied & (distance < sphere.radius)
    collision = inside & ~fixed
    weights = jnp.where(collision, mass, 0.)
    good_mass = jnp.isfinite(sphere.mass) & (sphere.mass > 0)
    body_mass = jnp.where(good_mass, sphere.mass, 1.)
    free_closing = jnp.sum((sphere.velocity-velocity) * normal, axis=-1)
    velocity_scale = jnp.max(jnp.where(collision, jnp.abs(free_closing), 0.), initial=0.)
    scale = jnp.sum(weights * jnp.maximum(jnp.abs(free_closing), 1e-12))
    target = tolerance * scale

    def evaluate(increment):
        closing = free_closing + jnp.sum(increment * normal, axis=-1)
        impulse = weights * jnp.maximum(closing, 0.)
        residual = body_mass * increment + jnp.sum(impulse[:, None] * normal, axis=0)
        active = jnp.where(closing > 0., weights, 0.)
        hessian = body_mass * jnp.eye(3, dtype=velocity.dtype) + jnp.einsum('n,ni,nj->ij', active, normal, normal)
        potential = .5 * body_mass * jnp.sum(increment**2) + .5 * jnp.sum(weights * jnp.maximum(closing, 0.)**2)
        return residual, hessian, potential

    def condition(carry):
        tool_velocity, iteration = carry
        residual, _, _ = evaluate(tool_velocity)
        return (iteration < max_iterations) & (jnp.linalg.norm(residual) > target)

    def newton(carry):
        tool_velocity, iteration = carry
        residual, hessian, potential = evaluate(tool_velocity)
        direction = jnp.linalg.solve(hessian, -residual)
        slope = jnp.dot(residual, direction)
        allowance = 64 * jnp.finfo(velocity.dtype).eps * jnp.maximum(potential, scale * jnp.maximum(velocity_scale, 1e-12))

        def backtrack(value):
            alpha, count = value
            _, _, candidate = evaluate(tool_velocity + alpha * direction)
            return (count < 16) & (candidate > potential + 1e-4 * alpha * slope + allowance)

        alpha, _ = jax.lax.while_loop(backtrack, lambda value: (value[0] * .5, value[1] + 1),
                                     (jnp.asarray(1., dtype=velocity.dtype), jnp.asarray(0)))
        return tool_velocity + alpha * direction, iteration + 1

    increment, iterations = jax.lax.while_loop(condition, newton, (jnp.zeros_like(sphere.velocity), jnp.asarray(0)))
    tool_velocity = sphere.velocity + increment
    closing = free_closing + jnp.sum(increment * normal, axis=-1)
    impulses = weights * jnp.maximum(closing, 0.)
    change = jnp.where(collision[:, None], jnp.maximum(closing, 0.)[:, None] * normal, 0.)
    corrected = velocity + change
    total = jnp.sum(impulses[:, None] * normal, axis=0)
    residual = jnp.linalg.norm(body_mass * increment + total)
    separation = jnp.sum((corrected - tool_velocity) * normal, axis=-1)
    violation = jnp.max(jnp.where(collision, jnp.maximum(-separation, 0.), 0.), initial=0.)
    complementarity = jnp.max(jnp.abs(impulses * separation), initial=0.)
    initial_ke = .5 * jnp.sum(mass * jnp.sum(velocity**2, axis=-1)) + .5 * body_mass * jnp.sum(sphere.velocity**2)
    final_ke = .5 * jnp.sum(mass * jnp.sum(corrected**2, axis=-1)) + .5 * body_mass * jnp.sum(tool_velocity**2)
    loss = .5 * jnp.sum(mass * jnp.sum(change**2, axis=-1)) + .5 * body_mass * jnp.sum(increment**2)
    energy_tolerance = 256 * jnp.finfo(velocity.dtype).eps * jnp.maximum(initial_ke, scale * jnp.maximum(velocity_scale, 1e-12))
    valid = (good_mass & jnp.isfinite(sphere.radius) & (sphere.radius > 0)
        & jnp.all(jnp.isfinite(mass)) & jnp.all(mass >= 0) & jnp.all(jnp.isfinite(velocity))
        & jnp.all(jnp.isfinite(coordinates)) & jnp.all(jnp.isfinite(tool_velocity))
        & jnp.all(jnp.isfinite(sphere.center)) & ~jnp.any(inside & fixed)
        & ~jnp.any(inside & (distance <= jnp.finfo(velocity.dtype).eps * sphere.radius))
        & (residual <= 4 * target) & (violation <= tolerance * jnp.maximum(velocity_scale, 1e-12))
        & (complementarity <= tolerance * jnp.maximum(initial_ke, scale * jnp.maximum(velocity_scale, 1e-12)))
        & (jnp.abs(initial_ke-final_ke-loss) <= energy_tolerance))
    return SphereProjection(corrected, tool_velocity, total, loss, residual, violation,
                            complementarity, iterations, valid)
