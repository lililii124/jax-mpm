"""Small-strain total-Lagrangian MPM with spectral AT2 fracture.

Reference quadrature and quadratic B-splines are shared by mechanics and damage.
The damage convention is zero intact, one broken. SI units are assumed.
"""
from typing import NamedTuple
from itertools import product

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np

from .fibre_mpm import stencil


class AT2(NamedTuple):
    lam: float
    mu: float
    gc: float
    length: float
    residual: float = 1e-8


def material(young, poisson, gc, length, residual=1e-8):
    if not np.isfinite([young, poisson, gc, length, residual]).all():
        raise ValueError('Material parameters must be finite')
    if min(young, gc, length) <= 0 or not 0 <= poisson < .5 or not 0 < residual < 1:
        raise ValueError('Require E,Gc,length > 0, 0 <= nu < .5 and 0 < residual < 1')
    return AT2(young*poisson/((1+poisson)*(1-2*poisson)),
               young/(2*(1+poisson)), gc, length, residual)


def spectral_split(strain, law):
    """Miehe spectral split; returns tensile energy and both stress branches."""
    values, vectors = jnp.linalg.eigh(.5*(strain+jnp.swapaxes(strain, -1, -2)))
    positive = jnp.maximum(values, 0)
    negative = jnp.minimum(values, 0)
    trace = jnp.sum(values, axis=-1)
    ep = jnp.einsum('...ik,...k,...jk->...ij', vectors, positive, vectors)
    en = jnp.einsum('...ik,...k,...jk->...ij', vectors, negative, vectors)
    eye = jnp.eye(3, dtype=strain.dtype)
    tensile = .5*law.lam*jnp.maximum(trace, 0)**2+law.mu*jnp.sum(positive**2, axis=-1)
    sp = law.lam*jnp.maximum(trace, 0)[..., None, None]*eye+2*law.mu*ep
    sn = law.lam*jnp.minimum(trace, 0)[..., None, None]*eye+2*law.mu*en
    return tensile, sp, sn


class Basis(NamedTuple):
    nodes: jax.Array
    weight: jax.Array
    gradient: jax.Array
    valid: jax.Array


def reference_basis(x, layout):
    """Exact B-spline derivatives, rather than APIC moment approximations."""
    support = stencil(x, jnp.zeros(len(x), dtype=jnp.int32), layout)
    q = (x-layout.origin)/layout.dx
    base = jnp.floor(q-.5)
    f = q-base
    offsets = jnp.asarray(list(product(range(3), repeat=3)))
    w = jnp.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2), axis=1)
    dw = jnp.stack((f-1.5, -2*(f-1), f-.5), axis=1)/layout.dx
    selected = w[:, offsets, jnp.arange(3)]
    derivative = dw[:, offsets, jnp.arange(3)]
    grad = jnp.stack([derivative[..., a]*selected[..., (a+1)%3]*selected[..., (a+2)%3]
                      for a in range(3)], axis=-1)
    return Basis(support.indices, support.weights, grad, support.valid)


def interpolate(d, basis):
    local = d[basis.nodes]
    return (jnp.sum(local*basis.weight, axis=1),
            jnp.einsum('pk,pki->pi', local, basis.gradient))


def scatter(values, basis, count):
    return jnp.zeros((count,)+values.shape[2:], dtype=values.dtype).at[basis.nodes].add(values)


def damage_system(history, volume, basis, count, law):
    """Weak AT2 operator with natural Gc*l*grad(d).n = 0 on free surfaces."""
    reaction = law.gc/law.length+2*(1-law.residual)*history
    rhs = scatter((2*(1-law.residual)*volume*history)[:, None]*basis.weight, basis, count)
    def operator(d):
        dp, grad = interpolate(d, basis)
        local = volume[:, None]*(reaction[:, None]*dp[:, None]*basis.weight+
                 law.gc*law.length*jnp.einsum('pi,pki->pk', grad, basis.gradient))
        return scatter(local, basis, count)
    # Absolute row-sum upper bound on the spectral radius.
    local_bound = volume[:, None]*(reaction[:, None]*basis.weight+
        law.gc*law.length*jnp.linalg.norm(basis.gradient, axis=-1)*
        jnp.sum(jnp.linalg.norm(basis.gradient, axis=-1), axis=1)[:, None])
    bound = scatter(local_bound, basis, count)
    return operator, rhs, bound


class DamageResult(NamedTuple):
    damage: jax.Array
    relative_kkt: jax.Array
    iterations: jax.Array
    converged: jax.Array


def solve_damage(history, volume, basis, previous, law, fixed, values,
                 tolerance=1e-7, max_iterations=4000, newton_steps=0):
    """Solve d_old <= d <= 1 by projected iteration and optional Newton-CG.

    `newton_steps` optionally enables a Jacobi-preconditioned active-set warm
    start. The projected method remains the default for predictable CPU cost.
    `fixed` and `values` have nodal shape. Fixed values must be compatible with
    previous damage; convergence is measured by a projected KKT residual.
    """
    operator, rhs, bound = damage_system(history, volume, basis, len(previous), law)
    step = 1/jnp.where(bound > 0, bound, 1)
    project = lambda d: jnp.where(fixed, values, jnp.clip(d, previous, 1))
    scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(rhs)), jnp.max(jnp.abs(operator(project(previous))))), jnp.finfo(previous.dtype).tiny)
    def residual_from_operator(d, applied):
        return jnp.max(jnp.abs(d-project(d-step*(applied-rhs)))/step)/scale
    def residual(d):
        return residual_from_operator(d, operator(d))
    reaction = law.gc/law.length+2*(1-law.residual)*history
    diagonal = scatter(volume[:, None]*(reaction[:, None]*basis.weight**2+
        law.gc*law.length*jnp.sum(basis.gradient**2, axis=-1)), basis, len(previous))
    def objective(d):
        return .5*jnp.dot(d, operator(d))-jnp.dot(d, rhs)
    def condition(state):
        i, d, error = state
        return (i < jnp.minimum(max_iterations, newton_steps)) & (error > tolerance)
    def body(state):
        i, d, _ = state
        gradient = operator(d)-rhs
        lower = (d-step*gradient <= previous)
        upper = (d-step*gradient >= 1)
        d = project(jnp.where(lower, previous, jnp.where(upper, 1., d)))
        gradient = operator(d)-rhs
        free = ~(fixed | lower | upper | (diagonal <= 0))
        def restricted(v):
            return jnp.where(free, operator(jnp.where(free, v, 0)), v)
        def precondition(v):
            return v/jnp.where(free, jnp.maximum(diagonal, jnp.finfo(d.dtype).tiny), 1)
        delta, _ = cg(restricted, -jnp.where(free, gradient, 0),
            M=precondition, tol=1e-10, atol=0., maxiter=300)
        direction = project(d+delta)-d
        fallback = project(d-step*gradient)-d
        direction = jnp.where(jnp.dot(gradient, direction) < 0, direction, fallback)
        slope = jnp.dot(gradient, direction)
        current = objective(d)
        def search_condition(search):
            n, alpha = search
            return (n < 25) & (objective(d+alpha*direction) > current+1e-4*alpha*slope)
        def search_body(search):
            n, alpha = search
            return n+1, .5*alpha
        _, alpha = jax.lax.while_loop(search_condition, search_body,
            (jnp.array(0), jnp.asarray(1., dtype=d.dtype)))
        updated = project(d+alpha*direction)
        return i+1, updated, residual(updated)
    initial = project(previous)
    i, d, error = jax.lax.while_loop(condition, body,
        (jnp.array(0), initial, residual(initial)))
    # Resolve degenerate changes of the active set with a globally convergent
    # projected method. Newton-CG normally supplies the converged warm start.
    def polish_condition(state):
        k, d, y, applied_d, applied_y, t, error = state
        return (k < max_iterations) & (error > tolerance)
    def polish_body(state):
        k, d, y, applied_d, applied_y, t, _ = state
        updated = project(y-step*(applied_y-rhs))
        applied_updated = operator(updated)
        tn = .5*(1+jnp.sqrt(1+4*t*t))
        beta = (t-1)/tn
        extrapolated = updated+beta*(updated-d)
        # The weak operator is linear at fixed history. Carry its extrapolate
        # to avoid a second particle gather/scatter for every projected step.
        applied_extrapolated = applied_updated+beta*(applied_updated-applied_d)
        return (k+1, updated, extrapolated, applied_updated, applied_extrapolated,
                tn, residual_from_operator(updated, applied_updated))
    applied = operator(d)
    k, d, _, _, _, _, error = jax.lax.while_loop(polish_condition, polish_body,
        (jnp.array(0), d, d, applied, applied, jnp.asarray(1., dtype=d.dtype), error))
    i = i+k
    compatible = jnp.all((~fixed) | ((values >= previous) & (values <= 1)))
    return DamageResult(d, error, i, compatible & basis.valid & (error <= tolerance))


class State(NamedTuple):
    displacement: jax.Array
    velocity: jax.Array
    strain: jax.Array
    history: jax.Array
    damage: jax.Array


def explicit_step(state, volume, density, basis, law, dt, velocity_mask,
                  velocity_value, damage_mask, damage_value, damping=0., update_damage=True):
    """Staggered damage solve then explicit PIC mechanics on the reference grid.

    Set `update_damage=False` for an explicitly lagged damage step; tensile
    history still updates at every mechanical step.
    Velocity Dirichlet data are componentwise nodal arrays. Damping is a
    nonnegative rate (1/s); reactions include inertial and damping effects.
    """
    tensile, sp, sn = spectral_split(state.strain, law)
    history = jnp.maximum(state.history, tensile)
    result = jax.lax.cond(update_damage,
        lambda _: solve_damage(history, volume, basis, state.damage, law, damage_mask, damage_value),
        lambda _: DamageResult(state.damage, jnp.asarray(0., dtype=volume.dtype),
            jnp.array(0), basis.valid), operand=None)
    dp, _ = interpolate(result.damage, basis)
    stress = ((1-law.residual)*(1-dp)**2+law.residual)[:, None, None]*sp+sn
    count = len(state.damage)
    mass = scatter((density*volume)[:, None]*basis.weight, basis, count)
    momentum = scatter((density*volume)[:, None, None]*basis.weight[..., None]*state.velocity[:, None], basis, count)
    force = scatter(-volume[:, None, None]*jnp.einsum('pij,pkj->pki', stress, basis.gradient), basis, count)
    free = (momentum+dt*force)/jnp.maximum(mass[:, None], jnp.finfo(volume.dtype).tiny)
    free = jnp.where(mass[:, None] > 0, free/(1+dt*damping), 0)
    velocity = jnp.where(velocity_mask, velocity_value, free)
    reaction = mass[:, None]*(velocity-free)/dt
    pv = jnp.sum(basis.weight[..., None]*velocity[basis.nodes], axis=1)
    gradient = jnp.einsum('pki,pkj->pij', velocity[basis.nodes], basis.gradient)
    strain = state.strain+dt*.5*(gradient+jnp.swapaxes(gradient, -1, -2))
    return State(state.displacement+dt*pv, pv, strain, history, result.damage), result, reaction



def energies(state, volume, basis, law, density):
    """Physical kinetic, elastic and regularized fracture energies (joules)."""
    tensile, _, _ = spectral_split(state.strain, law)
    total = .5*law.lam*jnp.trace(state.strain, axis1=-2, axis2=-1)**2+law.mu*jnp.sum(state.strain**2, axis=(-2, -1))
    d, grad = interpolate(state.damage, basis)
    degradation = (1-law.residual)*(1-d)**2+law.residual
    elastic = jnp.sum(volume*(degradation*tensile+total-tensile))
    fracture = .5*law.gc*jnp.sum(volume*(d*d/law.length+law.length*jnp.sum(grad*grad, axis=-1)))
    kinetic = .5*density*jnp.sum(volume*jnp.sum(state.velocity**2, axis=-1))
    return jnp.array([kinetic, elastic, fracture])
