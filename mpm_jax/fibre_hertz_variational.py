"""Energy-conjugate local Hertz patches for TL fibre mechanics.

Uses the existing closest-segment geometry and circular equivalent curvature.
Within a fixed connected-region selection the force is the full derivative of
the local potential, including changing curvature.  Region changes remain
discrete events and require a separate energy-jump check in the process driver.
This is not an adhesive or frictional law.
"""
import jax
import jax.numpy as jnp
from .fibre_coupling import closest_segments, _curvature, section_centres
from .fibre_patch_contact import evaluate_hertz_patches


def patch_potential(centres, endpoint_sections, radii, active, effective_modulus):
    """Return per-region Hertz energies for fixed section-pair identities."""
    endpoints=centres[endpoint_sections]
    # Inactive padded slots must have nondegenerate constant geometry even
    # during autodiff; masking a NaN after evaluating a norm is insufficient.
    dummy=jnp.array([[[-1.,0.,0.],[1.,0.,0.]],[[0.,-1.,3.],[0.,1.,3.]]],dtype=centres.dtype)
    endpoints=jnp.where(active[:,None,None,None],endpoints,dummy)
    p,q=endpoints[:,0,0],endpoints[:,0,1]
    r,s=endpoints[:,1,0],endpoints[:,1,1]
    first,second,_,_,distance=closest_segments(p,q,r,s)
    normal=(second-first)/jnp.maximum(distance,1e-30)[:,None]
    tangents=endpoints[:,:,1]-endpoints[:,:,0]
    tangents=tangents/jnp.maximum(jnp.linalg.norm(tangents,axis=-1),1e-30)[...,None]
    radius,_,valid=_curvature(tangents[:,0],tangents[:,1],normal,
        jnp.where(active[:,None],radii,1.))
    indentation=jnp.maximum(jnp.sum(radii,axis=1)-distance,0.)
    potential=(8/15)*effective_modulus*jnp.sqrt(radius)*indentation**2.5
    return jnp.where(active&valid,potential,0.)


def evaluate_variational_hertz_patches(state, history, data, law, neighbours, **limits):
    """Existing admissibility checks plus the full local-potential gradient.

    ``metrics.normal_force`` remains the indentation component of the Hertz
    law; returned forces additionally contain geometric curvature terms.
    This function does not silently discard any existing contact guard.
    """
    _,updated,metrics=evaluate_hertz_patches(state,history,data,law,neighbours,**limits)
    pair_segments=jnp.maximum(metrics.pair_segments,0)
    endpoint_sections=data.segments[pair_segments]
    radii=data.fibre_radius[data.segment_fibre[pair_segments]]
    active=jax.lax.stop_gradient(metrics.active)
    def energy(x):
        centres,_=section_centres(x,state.mass,data)
        per_patch=patch_potential(centres,endpoint_sections,radii,active,
            jnp.reshape(law.effective_modulus,()))
        return jnp.sum(per_patch),per_patch
    (_,potential),gradient=jax.value_and_grad(energy,has_aux=True)(state.x)
    force=-gradient
    metrics=metrics._replace(potential=potential,
        force_residual=jnp.sum(force,axis=0),
        torque_residual=jnp.sum(jnp.cross(state.x,force),axis=0),
        valid=metrics.valid&jnp.all(jnp.isfinite(force))&jnp.all(jnp.isfinite(potential)))
    return force,updated,metrics
