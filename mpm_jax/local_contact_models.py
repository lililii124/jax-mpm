"""Local circular contact selectors with explicit per-contact histories."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cohesive_bem import (
    CohesiveBEM, CohesiveBEMHistory, CohesiveBEMResponse, cohesive_bem,
    update_cohesive_bem,
)
from .fibre_contact import (
    FibreContactParameters, fibre_contact_parameters, initial_fibre_contact_history,
    update_fibre_contact,
)
from .maugis_dugdale import (
    MaugisParameters, initial_maugis_history, maugis_parameters, update_maugis,
)


MODELS = ("hertz", "jkr", "md", "self-consistent")


class SelfConsistentParameters(NamedTuple):
    effective_modulus: jax.Array
    effective_radius: jax.Array
    work_of_adhesion: jax.Array
    interaction_range: jax.Array
    base: CohesiveBEM


class SelfConsistentHistory(NamedTuple):
    active: jax.Array
    valid: jax.Array
    gap: jax.Array
    indentation: jax.Array
    previous_profile: jax.Array


class SelfConsistentResponse(NamedTuple):
    radius: jax.Array
    normal_force: jax.Array
    area: jax.Array
    potential: jax.Array
    active: jax.Array
    valid: jax.Array
    adhesive_radius: jax.Array
    adhesive_area: jax.Array
    details: CohesiveBEMResponse
    open_exact: jax.Array


def local_contact_model(model, *, effective_modulus, effective_radius,
                        work_of_adhesion=.1, interaction_range=None,
                        outer_radius=None, rings=32, quadrature_order=20):
    """Host factory; Hertz/JKR return their original parameter type unchanged."""
    if model not in MODELS:
        raise ValueError(f"Unknown local contact model: {model}")
    kwargs = dict(effective_modulus=effective_modulus, effective_radius=effective_radius,
                  work_of_adhesion=work_of_adhesion)
    if model in ("hertz", "jkr"):
        return fibre_contact_parameters(**kwargs, model=model)
    if interaction_range is None:
        raise ValueError("MD and self-consistent contact require an explicit interaction range")
    validated = maugis_parameters(**kwargs, interaction_range=interaction_range)
    if model == "md":
        return validated
    if outer_radius is None:
        raise ValueError("Self-consistent contact requires an explicit radial domain")
    E, R, W, h = map(np.asarray, validated)
    if any(not np.all(a == a.flat[0]) for a in (E, W, h)):
        raise ValueError("A shared BEM matrix requires fixed common E*, W and range")
    base = cohesive_bem(effective_modulus=float(E.flat[0]), effective_radius=float(R.flat[0]),
                        work_of_adhesion=float(W.flat[0]), interaction_range=float(h.flat[0]),
                        outer_radius=outer_radius, rings=rings, quadrature_order=quadrature_order)
    return SelfConsistentParameters(*validated, base)


def _sc_shape(parameters, shape):
    return jnp.broadcast_shapes(tuple(shape), *(jnp.shape(x) for x in parameters[:4]))


def _profile(parameters, radius):
    edges = parameters.base.edges
    return (edges[1:]**2+edges[:-1]**2)/(4*radius[..., None])


def initial_local_contact_history(parameters, shape=(), *, active=False):
    """Allocate the correct carry type; SC additionally retains every ring gap."""
    if isinstance(parameters, FibreContactParameters):
        return initial_fibre_contact_history(shape, active=active)
    if isinstance(parameters, MaugisParameters):
        shape = jnp.broadcast_shapes(tuple(shape), *(jnp.shape(x) for x in parameters))
        return initial_maugis_history(shape, active=active)
    if not isinstance(parameters, SelfConsistentParameters):
        raise TypeError("Unsupported local contact parameter type")
    shape = _sc_shape(parameters, shape)
    radius = jnp.broadcast_to(parameters.effective_radius, shape)
    profile = _profile(parameters, radius)
    return SelfConsistentHistory(
        jnp.full(shape, active, dtype=bool), jnp.ones(shape, dtype=bool),
        profile, jnp.zeros(shape, dtype=profile.dtype), profile,
    )


def _update_self_consistent(history, indentation, parameters, max_sweeps, tolerance):
    shape = _sc_shape(parameters, jnp.shape(indentation))
    shape = jnp.broadcast_shapes(shape, history.active.shape)
    rings = parameters.base.areas.shape[0]
    if (history.gap.shape != shape+(rings,) or history.previous_profile.shape != shape+(rings,)
            or history.indentation.shape != shape or history.valid.shape != shape
            or history.active.shape != shape):
        raise ValueError("SC history shape must match the full contact batch and ring count")
    delta, E, R, W, h = (jnp.broadcast_to(jnp.asarray(value), shape)
                         for value in (indentation, *parameters[:4]))
    base = parameters.base
    same_parameters = ((E == base.effective_modulus) & (W == base.work_of_adhesion)
                       & (h == base.interaction_range))
    valid_parameters = (same_parameters & jnp.isfinite(R) & (R > 0)
                        & jnp.all(jnp.isfinite(history.previous_profile), axis=-1)
                        & jnp.all(jnp.isfinite(history.gap), axis=-1)
                        & jnp.isfinite(history.indentation))
    profile = _profile(parameters, jnp.where(valid_parameters, R, base.effective_radius))
    # Preserve the old elastic displacement when the local radius changes.
    warm_gap = history.gap+profile-history.previous_profile
    old_valid = history.valid & valid_parameters

    def one(inputs):
        gap, previous_delta, valid, was_active, current_delta, current_radius, current_profile = inputs
        local = base._replace(effective_radius=current_radius, profile=current_profile)
        current_delta = jnp.asarray(current_delta, dtype=current_profile.dtype)
        old = CohesiveBEMHistory(gap, previous_delta, valid)
        free_gap = current_profile-current_delta
        open_exact = valid & ~was_active & jnp.isfinite(current_delta) & jnp.all(free_gap > local.interaction_range)

        def free(_):
            zero = jnp.asarray(0., dtype=free_gap.dtype)
            vector = jnp.zeros_like(free_gap)
            true, false = jnp.asarray(True), jnp.asarray(False)
            result = CohesiveBEMResponse(
                free_gap, vector, vector, vector, zero, zero, zero, zero, zero,
                zero, zero, zero, zero, zero, zero, zero, jnp.asarray(0),
                false, true, true, true,
            )
            return CohesiveBEMHistory(free_gap, current_delta, true), result

        def relax(_):
            return update_cohesive_bem(old, current_delta, local,
                                       max_sweeps=max_sweeps, tolerance=tolerance)

        new, result = jax.lax.cond(open_exact, free, relax, operand=None)
        return new, result, open_exact

    flat = lambda a, tail=(): jnp.reshape(a, (-1,)+tail)
    new, result, open_exact = jax.lax.map(one, (
        flat(warm_gap, (rings,)), flat(history.indentation), flat(old_valid), flat(history.active),
        flat(delta), flat(R), flat(profile, (rings,)),
    ))
    restore = lambda a: jnp.reshape(a, shape+a.shape[1:])
    new, result, open_exact = jax.tree.map(restore, (new, result, open_exact))
    valid = new.valid & valid_parameters
    result = result._replace(valid=valid)
    updated = SelfConsistentHistory(result.active, valid, new.gap, new.indentation, profile)
    response = SelfConsistentResponse(
        result.radius, result.normal_force, result.area, result.potential, result.active, valid,
        result.adhesive_radius, result.adhesive_area, result, open_exact,
    )
    return updated, response


def update_local_contact(history, indentation, parameters, *, max_sweeps=512, tolerance=1e-8):
    """Return (history, response); all responses share F/a/area/U/active/valid.

    Choose a parameter type before JIT. SC supports changing R with a fixed
    annular domain and fixed E*, W and h. Changes to those fixed parameters
    invalidate the update instead of silently rebuilding its compliance/history.
    """
    if isinstance(parameters, FibreContactParameters):
        return update_fibre_contact(history, indentation, parameters)
    if isinstance(parameters, MaugisParameters):
        return update_maugis(history, indentation, parameters)
    if isinstance(parameters, SelfConsistentParameters):
        if not isinstance(history, SelfConsistentHistory):
            raise TypeError("Self-consistent contact requires its ring-gap history")
        return _update_self_consistent(history, indentation, parameters, max_sweeps, tolerance)
    raise TypeError("Unsupported local contact parameter type")
