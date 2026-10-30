"""Nodal-diagonal midpoint mechanics for the C2 metal discrete-work law.

This module deliberately keeps the material update from
``fragment_v6_discrete_work`` and separates the endpoint contact impulse from
the midpoint drift.  Contact candidates are rebuilt from current particle
positions and the phase-space sampler is composed with the TL nodal map;
reference-coordinate pairs are not valid for advected material nodes.
"""
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from .fragment_v6_discrete_work import update as discrete_work_update
from .fragment_multifield_v6_mpm import paired_contact
from .total_lagrangian import deformation, sample_nodal


class MetalTLState(NamedTuple):
    displacement: object
    velocity: object
    material: tuple
    time: object
    initial_momentum: object
    initial_angular_momentum: object
    initial_kinetic: object
    contact_work: object
    viscous_loss: object


class MidpointStep(NamedTuple):
    state: MetalTLState
    geometry: object
    material_result: object
    contact_impulse: object
    contact_work: object
    contact_valid: object
    force_residual: object
    fixed_reaction: object
    viscous_loss: object
    valid: object
    contact_report: object


def midpoint_geometry(displacement, velocity_old, velocity_preimpact, dt, reference):
    """Drift to q1 using the force/material endpoint velocity, before contact."""
    midpoint = 0.5 * (velocity_old + velocity_preimpact)
    following = displacement + dt * midpoint
    return (following,
            reference.positions + sample_nodal(following, reference),
            deformation(following, reference), midpoint)


def nodal_moments(displacement, velocity, reference):
    """Canonical material-node P, L, K for this inertia discretization."""
    momentum = jnp.sum(reference.nodal_mass[:, None] * velocity, axis=0)
    angular = jnp.sum(jnp.cross(reference.nodes + displacement,
                                reference.nodal_mass[:, None] * velocity), axis=0)
    kinetic = 0.5 * jnp.sum(reference.nodal_mass[:, None] * velocity**2)
    return momentum, angular, kinetic


def initial_state(reference, material_fields, velocity):
    """Bind the nonzero t0 P/L/K baselines before any force evaluation."""
    velocity = jnp.asarray(velocity, dtype=reference.positions.dtype)
    if velocity.shape != reference.nodes.shape:
        raise ValueError("Initial nodal velocity must cover every independent body node")
    displacement = jnp.zeros_like(reference.nodes)
    P, L, K = nodal_moments(displacement, velocity, reference)
    return MetalTLState(displacement, velocity, tuple(material_fields), jnp.asarray(0.),
                        P, L, K, jnp.asarray(0.), jnp.asarray(0.))


def endpoint_e0_jump(velocity_preimpact, mass, pairs, normals, fixed,
                     *, iterations=1024, tolerance=0.005):
    """Project the actual endpoint velocity with the nodal diagonal metric.

    ``pairs`` and ``normals`` are required inputs from an actual-q1 contact
    search.  This function never predicts a contact by reflecting a midpoint
    velocity.  Work is the impulse dotted with the instantaneous jump mean.
    """
    following, _, impulses, valid, closure, residual, count, multipliers = paired_contact(
        velocity_preimpact, mass, normals, pairs, fixed, iterations, tolerance,
        initial_lambda=None)
    impulse = following * mass[:, None] - velocity_preimpact * mass[:, None]
    impulse = jnp.where(fixed, 0., impulse)
    work = jnp.sum(impulse * (velocity_preimpact + following) * 0.5)
    loss = 0.5 * jnp.sum(mass[:, None] *
                         (velocity_preimpact**2 - following**2))
    # For fixed nodes the removed momentum is a fixture reaction, not a
    # contact impulse. The caller records that reaction independently.
    accepted = (valid & (closure <= tolerance) & (residual <= tolerance)
                & jnp.all(multipliers >= 0) & (work <= 1e-10 * jnp.maximum(loss, 1e-30))
                & (loss >= -1e-10 * jnp.maximum(jnp.sum(mass[:, None] * velocity_preimpact**2), 1e-30)))
    return following, impulse, work, loss, accepted, closure, residual, count


def tl_velocity_map(velocity, reference, particle_mass, E0):
    """A v = sqrt(m) [N v, (grad_X v) E0] for phase-space contact."""
    root = jnp.sqrt(jnp.asarray(particle_mass))
    sampled = sample_nodal(velocity, reference)
    gradient = jnp.einsum('pki,pkj->pij', velocity[reference.indices],
                          reference.gradients)
    affine = gradient @ jnp.asarray(E0)
    return (root[:, None, None] *
            jnp.concatenate((sampled[:, None, :], affine), axis=1))


def tl_velocity_map_transpose(values, reference, particle_mass, E0, node_count):
    """Exact A^T scatter using the same reference shapes and gradients."""
    values = jnp.asarray(values).reshape((-1, 4, 3))
    root = jnp.sqrt(jnp.asarray(particle_mass))
    weighted = values * root[:, None, None]
    sample_part = (reference.weights[..., None] * weighted[:, None, 0, :])
    E0 = jnp.asarray(E0)
    # Contract affine-coordinate impulses against grad_X v E0.
    # If A(v) = (grad_X v) E0, then A^T(y) contracts the output-column
    # index of y with the output-column index of E0. Preserve the input-row
    # index (b) so E0=I is the identity on each 3x3 affine impulse.
    affine_part = jnp.einsum('pbc,jc->pbj', weighted[:, 1:, :], E0)
    gradient_part = jnp.einsum('pkj,pcj->pkc', reference.gradients, affine_part)
    contribution = sample_part + gradient_part
    return jnp.zeros((node_count, 3), dtype=values.dtype).at[
        reference.indices].add(contribution)


def _contact_rows(chunk, reconstruction, contact, reference, particle_mass, E0):
    """Apply (J_phase A)^T to a fixed-size batch of row impulses."""
    from .fragment_v6_matrixfree_projection import physical_adjoint
    phase = jax.vmap(lambda lam: physical_adjoint(reconstruction, contact, lam))(chunk)
    return jax.vmap(lambda y: tl_velocity_map_transpose(
        y, reference, particle_mass, E0, len(reference.nodal_mass)))(phase)


_CONTACT_ROWS = jax.jit(_contact_rows)


def actual_q1_contact_jump(velocity_preimpact, reference, particle_mass, body_id,
                           x1, F1, E0, dx, fixed, *, fixed_particles=None,
                           pair_capacity=512, particle_capacity=4096,
                           tolerance=0.005, batch_size=8):
    """Compose the verified phase sampler with the TL nodal kinematic map.

    Contact candidates/normals are rebuilt from actual q1/F1 using the original
    particle-current-support sampler. Its adjoint is composed with A^T, so the
    dual Schur uses the requested diagonal material-node metric directly.
    """
    import numpy as np
    from .fragment_v6_matrixfree_projection import (
        quadratic_reconstruction, polynomial_modes, physical_contact,
        physical_adjoint, physical_sample,
    )
    from scipy.optimize import nnls

    if fixed_particles is None:
        fixed_particles = jnp.zeros((len(particle_mass),), dtype=bool)
    reconstruction = quadratic_reconstruction(
        jnp.asarray(x1), jnp.asarray(particle_mass), jnp.asarray(body_id),
        jnp.asarray(F1), jnp.asarray(E0), jnp.asarray(dx),
        jnp.asarray(fixed_particles), capacity=65536)
    modes, polynomials, mode_report = polynomial_modes(reconstruction, jnp.asarray(dx))
    contact = physical_contact(reconstruction, modes, polynomials, jnp.asarray(dx),
                               pair_capacity=pair_capacity,
                               particle_capacity=particle_capacity)
    count = int(np.asarray(contact['pair_count']))
    if not bool(np.asarray(reconstruction['valid'])) or not bool(np.asarray(mode_report['valid'])) or not bool(np.asarray(contact['valid'])):
        raise FloatingPointError('Actual-q1 contact candidate, coverage, or polynomial-mode guard failed')
    if count == 0:
        zero = jnp.zeros_like(velocity_preimpact)
        return velocity_preimpact, zero, jnp.asarray(0.), jnp.asarray(0.), dict(
            valid=True, pairs=0, initial_closing_m_s=0., final_closing_m_s=0.,
            complementarity_m_s=0., mobility_rank=0, jump_work_J=0.)
    # B rows are J_node^T = A^T J_phase^T.  Chunking bounds scratch memory;
    # only this small active-row matrix is materialized, never full P0.
    capacity = len(contact['first'])
    bmatrix = np.empty((count, len(reference.nodal_mass), 3), dtype=np.float64)
    padded_count = ((count + batch_size - 1) // batch_size) * batch_size
    for start in range(0, padded_count, batch_size):
        basis = np.zeros((batch_size, capacity), dtype=np.float64)
        valid_rows = max(0, min(batch_size, count-start))
        if valid_rows:
            basis[np.arange(valid_rows), start+np.arange(valid_rows)] = 1.
        block = np.asarray(_CONTACT_ROWS(jnp.asarray(basis), reconstruction, contact,
                                         reference, jnp.asarray(particle_mass), jnp.asarray(E0)))
        if valid_rows:
            bmatrix[start:start+valid_rows] = block[:valid_rows]
    inverse_mass = np.divide(1., np.asarray(reference.nodal_mass),
                             out=np.zeros_like(np.asarray(reference.nodal_mass)),
                             where=np.asarray(reference.nodal_mass) > 0)
    inverse_mass[np.asarray(fixed, bool).any(axis=1)] = 0.
    flat = bmatrix.reshape(count, -1)
    mobility = (flat * np.repeat(inverse_mass, 3)[None, :]) @ flat.T
    mobility = .5 * (mobility + mobility.T)
    velocity = np.asarray(velocity_preimpact)
    closing = flat @ velocity.ravel()
    before = .5 * float(np.sum(np.asarray(reference.nodal_mass)[:, None] * velocity**2))
    multiplier = np.zeros(count)
    eigenvalues, vectors = np.linalg.eigh(mobility)
    max_eigen = float(eigenvalues.max(initial=0.))
    threshold = max(count, 1) * np.finfo(float).eps * max(max_eigen, 1e-300)
    if float(eigenvalues.min(initial=0.)) < -threshold:
        raise FloatingPointError('TL nodal contact Schur matrix is not positive semidefinite')
    keep = eigenvalues > threshold
    if not keep.any() and np.max(closing, initial=0.) > tolerance:
        raise FloatingPointError('Newly active contact has zero nodal mobility')
    if np.max(closing, initial=0.) > tolerance:
        if not keep.any():
            raise FloatingPointError('Closing contact cannot be resolved in the nodal metric')
        A = np.sqrt(eigenvalues[keep])[:, None] * vectors[:, keep].T
        target = (vectors[:, keep].T @ closing) / np.sqrt(eigenvalues[keep])
        multiplier = nnls(A, target, maxiter=max(3*count, 1024))[0]
    delta = -(flat.T @ multiplier).reshape(velocity.shape) * inverse_mass[:, None]
    delta[np.asarray(fixed, bool)] = 0.
    following = velocity + delta
    final = flat @ following.ravel()
    # Independent primal-route check: apply J_phase to A v directly instead
    # of trusting values produced by J_node^T and its Schur matrix.
    pre_phase = np.asarray(physical_sample(
        reconstruction, contact,
        tl_velocity_map(jnp.asarray(velocity_preimpact), reference,
                        jnp.asarray(particle_mass), jnp.asarray(E0)).reshape(-1, 3)))[:count]
    post_phase = np.asarray(physical_sample(
        reconstruction, contact,
        tl_velocity_map(jnp.asarray(following), reference,
                        jnp.asarray(particle_mass), jnp.asarray(E0)).reshape(-1, 3)))[:count]
    route_pre_error = float(np.max(np.abs(pre_phase-closing), initial=0.))
    route_post_error = float(np.max(np.abs(post_phase-final), initial=0.))
    physical_closure = float(np.max(np.maximum(post_phase, 0.), initial=0.))
    physical_residual = np.where(multiplier > 0., np.abs(post_phase),
                                 np.maximum(post_phase, 0.))
    physical_kkt = float(np.max(physical_residual, initial=0.))
    residual = np.where(multiplier > 0., np.abs(final), np.maximum(final, 0.))
    closure = float(np.max(np.maximum(final, 0.), initial=0.))
    kkt = float(np.max(residual, initial=0.))
    impulse = delta * np.asarray(reference.nodal_mass)[:, None]
    work = float(np.sum(impulse * .5*(velocity+following)))
    loss = before - .5*float(np.sum(np.asarray(reference.nodal_mass)[:, None]*following**2))
    identity = .5*float(np.sum(np.asarray(reference.nodal_mass)[:, None]*delta**2)) + np.dot(multiplier, final)
    valid = (np.min(multiplier, initial=0.) >= 0. and closure <= tolerance and kkt <= tolerance
             and physical_closure <= tolerance and physical_kkt <= tolerance
             and np.isfinite(route_pre_error) and np.isfinite(route_post_error)
             and work <= 1e-10*max(loss, 1e-30) and loss >= -1e-10*max(before, 1e-30)
             and abs(loss + work) <= 1e-10*max(before, 1e-30))
    report = dict(valid=bool(valid), pairs=count, initial_closing_m_s=float(np.max(closing, initial=0.)),
        final_closing_m_s=closure, complementarity_m_s=kkt, mobility_rank=int(keep.sum()),
        jump_work_J=work, kinetic_loss_J=loss, impulse_identity_J=identity,
        maximum_multiplier=float(multiplier.max(initial=0.)),
        positive_multiplier_pairs=int(np.count_nonzero(multiplier > 0.)),
        resolved_normal_pairs=int(np.count_nonzero(np.linalg.norm(
            np.asarray(contact['normals']), axis=1) > 0.)),
        nodal_impulse_norm_Ns=float(np.linalg.norm(impulse)),
        newly_active_e0=bool(np.max(closing, initial=0.) > tolerance and closure <= tolerance))
    report.update(physical_route_pre_closing_m_s=float(np.max(pre_phase, initial=0.)),
        physical_route_post_closing_m_s=physical_closure,
        physical_route_complementarity_m_s=physical_kkt,
        JphaseA_vs_Jnode_pre_max_abs_m_s=route_pre_error,
        JphaseA_vs_Jnode_post_max_abs_m_s=route_post_error)
    return jnp.asarray(following), jnp.asarray(impulse), jnp.asarray(work), jnp.asarray(loss), report


@partial(jax.jit, static_argnames=('max_picard',))
def _force_material_midpoint(state, reference, material, dt, fixed, cell_length,
                             force_tolerance, *, max_picard):
    F0, tau0, ep0, cold0, fracture0, damage0, internal0 = state.material
    M = reference.nodal_mass[:, None]
    active = reference.nodal_mass > 0
    inverse = jnp.where(M > 0, 1. / jnp.where(M > 0, M, 1.), 0.)
    fixed = jnp.asarray(fixed, dtype=bool)
    v0 = jnp.where(fixed, 0., state.velocity)
    # Exact topological ownership count: only independent singleton supports
    # may use the local rank-one inverse. All other particles keep raw Picard.
    ownership = jnp.zeros((len(reference.nodal_mass),),dtype=jnp.int32).at[reference.indices].add(1)
    eligible = (damage0>=1.) & jnp.all(ownership[reference.indices]==1,axis=1)
    precondition_ids = jnp.nonzero(eligible,size=2048,fill_value=0)[0]
    precondition_slots = jnp.arange(2048)<jnp.minimum(jnp.sum(eligible),2048)


    def evaluate(vpre):
        u1, x1, F1, vm = midpoint_geometry(state.displacement, v0, vpre, dt, reference)
        # The true reference gradient makes grad(v_m) work-conjugate to F.
        W = jnp.einsum('pki,pkj->pij', vm[reference.indices], reference.gradients)
        mat = discrete_work_update(F0, tau0, ep0, cold0, fracture0, damage0,
                                   internal0, W, dt, material)
        piola = mat.piola
        # Original artificial viscosity is a separate mechanical-loss channel.
        J = jnp.linalg.det(F1)
        density = material.density / jnp.maximum(J, 1e-30)
        thermo_energy = (internal0 - cold0 - fracture0
                         - jnp.sum(tau0*tau0, axis=(-2, -1)) /
                         (4*material.shear_modulus*material.density*jnp.maximum(1-damage0, 1e-12)))
        from .materials.linear_metal import eos_state
        thermo = eos_state(jnp.linalg.det(F0), thermo_energy, material)
        divergence = jnp.trace(W @ jnp.linalg.inv(F1), axis1=-2, axis2=-1)
        from .materials.shock_metal import artificial_viscosity
        q = artificial_viscosity(density, thermo.sound_speed, divergence,
                                 cell_length, material)
        # Cauchy stress is tensile-positive, while artificial_viscosity returns
        # compression-positive pressure. Its stress is therefore -q I and its
        # work-conjugate midpoint Piola is -J_mid q F_mid^{-T}. Use this exact
        # tensor in both the force and its separate nonthermal discrete loss.
        Fmid = .5 * (F0 + F1)
        Jmid = jnp.linalg.det(Fmid)
        viscous_piola = -(Jmid*q)[:, None, None] * jnp.linalg.inv(Fmid).swapaxes(-1, -2)
        piola = piola + viscous_piola
        particle_force = -reference.volume[:, None, None] * jnp.einsum(
            'pij,pkj->pki', piola, reference.gradients)
        force = jnp.zeros_like(vpre).at[reference.indices].add(particle_force)
        vproposal = v0 + dt * inverse * force
        vproposal = jnp.where(fixed, 0., vproposal)
        delta = jnp.where(active[:, None] & ~fixed,
                          vproposal-vpre, 0.)
        scale = jnp.maximum(jnp.linalg.norm(jnp.where(active[:, None] & ~fixed,
                                                       vproposal, 0.)), 1.)
        residual = jnp.linalg.norm(delta) / scale
        work_defect = jnp.abs(jnp.sum(M*delta*vm))
        viscous_loss_increment = reference.volume * jnp.sum(
            viscous_piola * (F1-F0), axis=(-2, -1))
        # Frozen-geometry rank-one viscous Jacobian inverse. Only solver
        # iteration changes: raw force/work residuals and returned forces stay.
        ids = precondition_ids
        inds = reference.indices[ids]
        b1 = jnp.einsum('pkj,pjl->pkl',reference.gradients[ids],jnp.linalg.inv(F1[ids]))
        bm = jnp.einsum('pkj,pjl->pkl',reference.gradients[ids],jnp.linalg.inv(Fmid[ids]))
        invm = inverse[inds]
        invm = jnp.where(fixed[inds],0.,invm)
        compression = jnp.maximum(-divergence[ids],0.)
        derivative = density[ids]*(material.viscosity_linear*thermo.sound_speed[ids]*cell_length+2*material.viscosity_quadratic*cell_length**2*compression)
        derivative = jnp.where(divergence[ids]<0,derivative,0.)
        alpha = .5*dt*reference.volume[ids]*Jmid[ids]*derivative
        direction = invm*bm
        projection = jnp.sum(b1*delta[inds],axis=(1,2))
        denom = 1+alpha*jnp.sum(b1*direction,axis=(1,2))
        safe = precondition_slots & jnp.isfinite(denom) & (denom>1e-8)
        coefficient = jnp.where(safe,alpha*projection/jnp.where(safe,denom,1.),0.)
        correction = direction*coefficient[:,None,None]
        iteration_delta = delta.at[inds].add(-correction)
        viteration = vpre+iteration_delta
        return (viteration, residual, u1, x1, F1, vm, mat, force,
                jnp.sum(viscous_loss_increment), work_defect)

    work_tolerance = jnp.maximum(1e-14,1e-15*state.initial_kinetic)
    initial_result = evaluate(state.velocity)
    def pending(carry):
        _,residual,iteration,work_defect,_,_=carry
        return ((residual>force_tolerance)|(work_defect>work_tolerance))&(iteration<max_picard)
    def iterate(carry):
        velocity,_,iteration,_,previous_residual,previous_weight=carry
        result=evaluate(velocity);residual_vector=result[0]-velocity
        difference=residual_vector-previous_residual
        denominator=jnp.sum(M*difference*difference)
        numerator=jnp.sum(M*previous_residual*difference)
        weight=jnp.where(denominator>1e-60,-previous_weight*numerator/jnp.maximum(denominator,1e-60),1.)
        weight=jnp.clip(weight,.05,1.)
        return velocity+weight*residual_vector,result[1],iteration+1,result[-1],residual_vector,weight
    vpre,_,picard_iterations,_,_,_=jax.lax.while_loop(pending,iterate,
        (initial_result[0],initial_result[1],jnp.asarray(1,dtype=jnp.int32),initial_result[-1],initial_result[0]-state.velocity,jnp.asarray(1.)))
    # Re-evaluate at accepted force-balance endpoint so returned material and
    # geometry correspond to the endpoint actually used for the e=0 jump.
    result = evaluate(vpre)
    _, force_residual, u1, x1, F1, vm, mat, force, viscous_loss, _ = result
    fixed_reaction = jnp.where(fixed, M * (0. - state.velocity) - dt * force, 0.)
    return (vpre, u1, x1, F1, vm, mat, force, force_residual, viscous_loss,
            fixed_reaction, picard_iterations)


def step(state, reference, material, dt, fixed, endpoint_contact,
         *, cell_length, force_tolerance=1e-10, max_picard=24):
    """One compiled force/material drift followed by an independent e=0 jump.

    dt is a dynamic scalar input to the once-compiled midpoint kernel. Contact
    is rebuilt outside that kernel at the exact q1 and composed with its TL
    nodal map; reference-coordinate contact pairs are never used.
    """
    import numpy as np
    if max_picard < 1 or dt <= 0:
        raise ValueError("A positive step and Picard iteration cap are required")
    fixed = jnp.asarray(fixed, dtype=bool)
    (vpre, u1, x1, F1, vm, mat, force, force_residual, viscous_loss,
     fixed_reaction, picard_iterations) = _force_material_midpoint(
        state, reference, material, jnp.asarray(dt), fixed,
        jnp.asarray(cell_length), jnp.asarray(force_tolerance),
        max_picard=max_picard)
    vpost, contact_impulse, contact_work, contact_loss, contact_report = endpoint_contact(vpre, x1, F1)
    contact_valid = jnp.asarray(contact_report['valid'])
    closure = jnp.asarray(contact_report.get('final_closing_m_s', jnp.inf))
    kkt = jnp.asarray(contact_report.get('complementarity_m_s', jnp.inf))
    count = jnp.asarray(contact_report.get('pairs', 0))
    P, L, K = nodal_moments(u1, vpost, reference)
    material_valid = jnp.all(mat.valid) & jnp.all(jnp.linalg.det(F1) > 0)
    finite = (jnp.all(jnp.isfinite(vpost)) & jnp.all(jnp.isfinite(x1)) &
              jnp.all(jnp.isfinite(mat.piola)) & jnp.isfinite(contact_work))
    M=reference.nodal_mass[:,None]
    inverse=jnp.where(M>0,1./jnp.where(M>0,M,1.),0.)
    proposal=jnp.where(fixed,0.,state.velocity+dt*inverse*force)
    delta=jnp.where((M>0)&~fixed,proposal-vpre,0.)
    work_residual=jnp.abs(jnp.sum(M*delta*vm))
    work_tolerance=jnp.maximum(1e-14,1e-15*state.initial_kinetic)
    work_valid=work_residual<=work_tolerance
    contact_report.update(step_solver='mass-Aitken+rank-one-failed-q-preconditioner-v1',step_midpoint_work_residual_J=float(np.asarray(work_residual)),
        step_midpoint_work_tolerance_J=float(np.asarray(work_tolerance)),
        step_midpoint_work_valid=bool(np.asarray(work_valid)))
    contact_report.update(step_contact_valid=bool(np.asarray(contact_valid)),
        step_material_valid=bool(np.asarray(material_valid)),
        step_finite=bool(np.asarray(finite)),
        step_force_residual=float(np.asarray(force_residual)),
        step_force_tolerance=float(force_tolerance),
        step_picard_iterations=int(np.asarray(picard_iterations)))
    accepted = (picard_iterations <= max_picard) & (force_residual <= force_tolerance) & material_valid & finite & contact_valid & work_valid
    following = MetalTLState(u1, vpost, mat.state, state.time + dt,
        state.initial_momentum, state.initial_angular_momentum, state.initial_kinetic,
        state.contact_work + contact_work, state.viscous_loss + viscous_loss)
    geometry = dict(position=x1, F=F1, midpoint_velocity=vm,
                    momentum=P, angular_momentum=L, kinetic=K,
                    contact_closure=closure, contact_kkt=kkt,
                    contact_iterations=count, contact_loss=contact_loss)
    return MidpointStep(following, geometry, mat, contact_impulse, contact_work,
                        contact_valid, force_residual, fixed_reaction, viscous_loss,
                        accepted, contact_report)
