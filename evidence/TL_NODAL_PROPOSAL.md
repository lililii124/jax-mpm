# Minimal nodal-diagonal metal candidate

This is a reviewable scheme, not an executed impact or a fragment qualification. The exact-P0 implementation and all rejected preconditioner evidence remain unchanged. The proposed implementation would reuse reference-grid kinematics from `work/elastic-tl-dynamics@82d3edd`, with a separate metal-specific force/material and contact layer. It must not call the generic TL `step` or its generic `update_history` as a substitute for the already verified C2 material.

## Fresh initial condition and inertia

Start a new run at physical t0 from the existing formal case factory: a 2 mm sphere, 20 mm square plate, 0.5 mm plate thickness, 0.75 mm initial gap, 1 mm clamp strips, dx0.25 mm and particle spacing0.25×0.125×0.25 mm. Keep the original 5 km/s velocity, material parameters, EOS, damage assumptions, viscosity and free-body/fixture conditions. Regenerate and bind all geometry, mass and reference-array descriptors before computing. Do not label this as a continuation of the accepted125/172 ns particle state.

Use two independent material-node fields, one per physical body. Compact their reference support once; a body must never share a kinematic DOF with the other body. With reference shapes N and true reference gradients:

```text
M_a = sum_p m_p N_pa
q_a = X_a + u_a
x_p = sum_a N_pa q_a
F_p = sum_a q_a tensor grad_X N_pa
K = 0.5 sum_a M_a |v_a|²
P = sum_a M_a v_a
L = sum_a q_a cross (M_a v_a)
```

Particle interpolation is an observable; it must not replace the nodal inertia, kinetic energy or angular-momentum ledger. Seed the actual nonzero initial P/L and energy as immutable baselines. Rigid bodywise initial velocities give the same initial mass, P/L and K as the original physical particle initialization. There is no repeated P2G/G2P projection and no transfer-loss heat or reservoir.

All three components of the plate-edge material nodes remain fixed, with no backing. Preserve the original reference clamp definition and count corner reactions once. Any compaction capacity is a declared storage guard; an overflow stops and retains raw data.

## Force and material substep

Solve a force-only common midpoint step on the free nodes:

```text
v_m = (v_0 + v_pre)/2
q_1 = q_0 + dt v_m
F_1 = grad_X q_1
M (v_pre - v_0) = dt f_m + I_fixture
f_ma = -sum_p V0_p P_m,p grad_X N_pa
```

Evaluate the original `fragment_v6_discrete_work.update` from the same old seven material fields on every Picard iteration. Use its objective volume discrete gradient and midpoint JC stress, preserving its linear-EOS branch, damage release, cold work and surviving-fraction plastic heat. Derive the material increment from the same nodal endpoint geometry; check its consistency with the midpoint velocity gradient at the original64eps kinematic limits. Do not overwrite a failed F/history or repair energy afterward.

The material Piola work and nodal force work must be adjoints under the same reference gradients. The volume discrete gradient and deviatoric midpoint stress give symmetric P_m F_m^T, the required internal angular-momentum identity. Keep the original artificial-viscosity function and its separate mechanical-loss channel; it does not become thermal energy. Fixture impulse, torque and work come from the actual constrained nodal equations.

Retain the established distinction between a consistent Picard endpoint and its force-balance proposal. Accept only after the declared relative force-balance criterion, actual coupled work, material and kinematic gates pass. Do not restore the old overly conservative norm-work stopping budget or increase iteration caps to hide a failure.

## Endpoint e0 impact

After the force/material drift, hold q1 and the material state fixed. Build the physical contact sampler at the actual q1 and project `v_pre` with the diagonal nodal metric:

```text
v_post = argmin_v 0.5 |v - v_pre|²_M
subject to J(q1) v <=0 and the original fixed-node velocities
I_contact = M (v_post - v_pre) - I_fixture_jump
W_jump = I_contact dot (v_pre + v_post)/2
```

Keep the original nonnegative multipliers, native m/s KKT/closing limits and full coverage checks. The whole sampler and its adjoint must reproduce common rigid motion at the current contact point. Measure its physical torque using q1; record the fixture part separately. A newly active, unforced closing pair must lose its closing velocity with e0 and must not leave with the reflected e1 velocity.

There is no reflection `2*v_m-v_0` caused by a midpoint contact cone. The drift kinematic check uses `v_pre`, not the discontinuous post-impact velocity. The jump work uses its own instantaneous mean velocity; a whole-step mean is not the contact jump work. Total bookkeeping is the force/material work identity followed by the independently measured nonpositive impact work. Contact/projection loss is never used as plastic heat.

This split has a contact-time/position discretization error. Its geometric penetration/onset and dt convergence need their own fixed acceptance checks; an energy identity alone is not an accuracy certificate. Retaining the existing finite-support MPM contact search would also retain its early-contact scope and must not be called geometric surface contact.

## Admission and unresolved scope

Carry over the original material-domain, irreversibility, stress symmetry, temperature/entropy, mass, CFL, contact, actual-work, global/cumulative-energy and engineering linear/angular-momentum limits. Freeze their physical scales before examining results. Keep the original64eps kinematic diagnostics and limits. Canonical P/L are now nodal quantities of this explicitly different inertia discretization; do not silently compare interpolated particle P/L with an omitted subgrid contribution.

Reference support remains attached to the material. Damage alone does not prove that two intact regions have independent nodal support after a crack. No fracture/ballistic remapping, conservative DOF split, mesh objectivity or fragment-cloud fidelity is established. Keep failed material and every loss channel explicit; do not delete mass or invent separation for a render. The original full-plan horizon is2µs, but it is not the first qualification run.

The minimal implementation set would be a separate `mpm_jax/metal_tl_midpoint.py`, a source-bound fresh-case runner and one new-activation e0 check, leaving the shared elastic TL module unchanged. Initial checks should bind geometry and nonzero P/L/K, then verify the new activation/jump identity. Only then run a bounded fresh short window through first contact, with complete failure state and the original guards; a same-window dt/2 comparison follows only if needed for this split. No expensive full cloud or fragment media is authorized by this proposal.
