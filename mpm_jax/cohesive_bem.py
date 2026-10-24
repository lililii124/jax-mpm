"""Axisymmetric half-space equilibrium with unilateral contact and Dugdale adhesion."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class CohesiveBEM(NamedTuple):
    effective_modulus: jax.Array
    effective_radius: jax.Array
    work_of_adhesion: jax.Array
    interaction_range: jax.Array
    edges: jax.Array
    areas: jax.Array
    profile: jax.Array
    compliance: jax.Array
    stiffness: jax.Array


class CohesiveBEMHistory(NamedTuple):
    gap: jax.Array
    indentation: jax.Array
    valid: jax.Array


class CohesiveBEMResponse(NamedTuple):
    gap: jax.Array
    displacement: jax.Array
    pressure: jax.Array
    ring_force: jax.Array
    normal_force: jax.Array
    area: jax.Array
    adhesive_area: jax.Array
    radius: jax.Array
    adhesive_radius: jax.Array
    elastic_energy: jax.Array
    cohesive_energy: jax.Array
    potential: jax.Array
    initial_potential: jax.Array
    equilibrium_residual: jax.Array
    complementarity_residual: jax.Array
    energy_increase: jax.Array
    sweeps: jax.Array
    active: jax.Array
    domain_clear: jax.Array
    converged: jax.Array
    valid: jax.Array


def annular_compliance(edges, effective_modulus, *, quadrature_order=20):
    """Host Galerkin Boussinesq compliance: mean ring displacement / ring force.

    Both ring integrals are evaluated, retaining reciprocal energy work. The
    inner radial integral is split at the logarithmic singularity r=s.
    """
    edges = np.asarray(edges, dtype=float)
    E = float(effective_modulus)
    if (edges.ndim != 1 or len(edges) < 3 or not np.all(np.isfinite(edges))
            or edges[0] != 0 or np.any(np.diff(edges) <= 0)
            or not np.isfinite(E) or E <= 0 or quadrature_order < 8):
        raise ValueError("finite increasing edges from zero, positive E*, order >=8 required")
    z, w = np.polynomial.legendre.leggauss(quadrature_order)
    areas = np.pi*np.diff(edges**2)
    C = np.zeros((len(areas), len(areas)))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        r = (z+1)*(b-a)/2+a
        wr = w*(b-a)*r/2
        for j, (aa, bb) in enumerate(zip(edges[:-1], edges[1:])):
            middle = np.clip(r, aa, bb)
            integral = np.zeros_like(r)
            for lower, upper in ((np.full_like(r, aa), middle), (middle, np.full_like(r, bb))):
                s = (z[None, :]+1)*(upper-lower)[:, None]/2+lower[:, None]
                ws = w[None, :]*(upper-lower)[:, None]*s/2
                denominator = r[:, None]+s
                agm_a = np.ones_like(s)
                agm_b = np.abs(r[:, None]-s)/denominator
                for _ in range(24):
                    agm_a, agm_b = (agm_a+agm_b)/2, np.sqrt(agm_a*agm_b)
                elliptic_k = np.pi/(2*agm_a)
                integral += np.sum(ws*elliptic_k/denominator, axis=1)
            C[i, j] = 8/E*np.sum(wr*integral)/(areas[i]*areas[j])
    C = (C+C.T)/2
    np.linalg.cholesky(C)
    return C


def cohesive_bem(*, effective_modulus, effective_radius, work_of_adhesion=0.1,
                 interaction_range, outer_radius, rings=32, quadrature_order=20):
    """Prepare one local circular half-space model on the host, outside JIT."""
    E, R, W, h, outer = map(float, (effective_modulus, effective_radius,
                                   work_of_adhesion, interaction_range, outer_radius))
    if (not np.all(np.isfinite([E, R, W, h, outer])) or min(E, R, h, outer) <= 0
            or W < 0 or not isinstance(rings, int) or rings < 4):
        raise ValueError("finite positive E*, R, range, radius; W >=0 and rings >=4 required")
    edges = np.linspace(0, outer, rings+1)
    areas = np.pi*np.diff(edges**2)
    profile = (edges[1:]**2+edges[:-1]**2)/(4*R)
    C = annular_compliance(edges, E, quadrature_order=quadrature_order)
    K = np.linalg.solve(C, np.eye(rings))
    K = (K+K.T)/2
    return CohesiveBEM(*(jnp.asarray(x) for x in (E, R, W, h, edges, areas, profile, C, K)))


def initial_cohesive_bem_history(model, indentation=0.):
    delta = jnp.asarray(indentation, dtype=model.profile.dtype)
    return CohesiveBEMHistory(jnp.maximum(0, model.profile-delta), delta, jnp.asarray(True))


def cohesive_bem_potential(gap, indentation, model):
    """Total elastic plus interface potential at independent ring gaps."""
    displacement = gap-(model.profile-indentation)
    elastic = .5*jnp.vdot(displacement, model.stiffness@displacement)
    adhesive = jnp.sum(model.areas*jnp.minimum(
        0, model.work_of_adhesion/model.interaction_range*gap-model.work_of_adhesion,
    ))
    return elastic+adhesive


def _equilibrium(gap, g0, model):
    f = model.stiffness@(gap-g0)
    pressure = f/model.areas
    sigma = model.work_of_adhesion/model.interaction_range
    contact = gap == 0
    cohesive = (~contact) & (gap < model.interaction_range) & (model.work_of_adhesion > 0)
    residual = jnp.where(contact, jnp.minimum(pressure+sigma, 0),
                         pressure+jnp.where(cohesive, sigma, 0))
    return f, pressure, contact, cohesive, residual


def update_cohesive_bem(history, indentation, model, *, max_sweeps=512, tolerance=1e-8):
    """Energy-decreasing ring-coordinate relaxation, warm-started from history.

    The selected branch is a discrete energy minimum reached by this specified
    relaxation. No unique continuum snap threshold is asserted for finite rings.
    max_sweeps and tolerance are static solver controls when jitting this function.
    """
    if max_sweeps < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive max_sweeps and tolerance required")
    delta = jnp.asarray(indentation, dtype=model.profile.dtype)
    if (delta.ndim != 0 or history.gap.shape != model.profile.shape
            or jnp.ndim(history.indentation) != 0 or jnp.ndim(history.valid) != 0):
        raise ValueError("one scalar contact required; use vmap for independent contacts")
    valid_input = (history.valid & jnp.isfinite(delta) & jnp.isfinite(history.indentation)
                   & jnp.all(jnp.isfinite(history.gap)))
    delta = jnp.where(valid_input, delta, 0)
    g0 = model.profile-delta
    gap = jnp.maximum(0, history.gap-(delta-history.indentation))
    gap = jnp.where(valid_input, gap, jnp.maximum(g0, 0))
    sigma = model.work_of_adhesion/model.interaction_range
    pressure_scale = jnp.maximum(sigma, model.effective_modulus*model.edges[-1]/model.effective_radius)
    initial_energy = cohesive_bem_potential(gap, delta, model)

    def sweep(carry):
        count, gap, _, old_energy, increase = carry
        gradient = model.stiffness@(gap-g0)

        def coordinate(i, data):
            g, grad = data
            old = g[i]
            diagonal = model.stiffness[i, i]
            bonded = jnp.clip(old-(grad[i]+sigma*model.areas[i])/diagonal,
                              0, model.interaction_range)
            open_gap = jnp.maximum(model.interaction_range, old-grad[i]/diagonal)

            def difference(new):
                d = new-old
                return (.5*diagonal*d*d+grad[i]*d
                        + model.areas[i]*jnp.minimum(0, sigma*new-model.work_of_adhesion))

            new = jnp.where(difference(bonded) <= difference(open_gap), bonded, open_gap)
            change = new-old
            return g.at[i].set(new), grad+model.stiffness[:, i]*change

        gap, _ = jax.lax.fori_loop(0, gap.shape[0], coordinate, (gap, gradient))
        residual = jnp.max(jnp.abs(_equilibrium(gap, g0, model)[-1]))/pressure_scale
        energy = cohesive_bem_potential(gap, delta, model)
        increase = jnp.maximum(increase, energy-old_energy)
        return count+1, gap, residual, energy, increase

    def unfinished(carry):
        count, _, residual, _, _ = carry
        return (count < max_sweeps) & (residual > tolerance)

    count, gap, residual, energy, increase = jax.lax.while_loop(
        unfinished, sweep, (jnp.asarray(0), gap, jnp.asarray(jnp.inf, dtype=gap.dtype),
                            initial_energy, jnp.asarray(0., dtype=gap.dtype)),
    )
    f, pressure, contact, cohesive, _ = _equilibrium(gap, g0, model)
    active = contact | cohesive
    area = jnp.sum(jnp.where(contact, model.areas, 0))
    adhesive_area = jnp.sum(jnp.where(cohesive, model.areas, 0))
    radius = jnp.max(jnp.where(contact, model.edges[1:], 0))
    c = jnp.max(jnp.where(active, model.edges[1:], 0))
    displacement = model.compliance@f
    compatibility = jnp.max(jnp.abs(gap-g0-displacement))/jnp.maximum(
        model.interaction_range, model.edges[-1]**2/model.effective_radius,
    )
    elastic = .5*jnp.vdot(f, displacement)
    cohesive_energy = jnp.sum(model.areas*jnp.minimum(0, sigma*gap-model.work_of_adhesion))
    domain_clear = jnp.all(~active[-2:])
    converged = residual <= tolerance
    eps = jnp.finfo(gap.dtype).eps
    energy_scale = jnp.maximum(jnp.abs(initial_energy),
                              model.effective_modulus*model.edges[-1]**5/model.effective_radius**2)
    valid = (valid_input & converged & domain_clear & jnp.all(jnp.isfinite(f))
             & jnp.all(gap >= 0) & (compatibility <= 256*eps*gap.shape[0])
             & (increase <= 256*eps*energy_scale))
    response = CohesiveBEMResponse(
        gap, displacement, pressure, f, jnp.sum(f), area, adhesive_area, radius, c,
        elastic, cohesive_energy, energy, initial_energy, residual, compatibility,
        increase, count, jnp.any(active), domain_clear, converged, valid,
    )
    return CohesiveBEMHistory(gap, delta, valid), response
