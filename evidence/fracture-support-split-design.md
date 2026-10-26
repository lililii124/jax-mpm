# Conservative TL support split at fully failed material cells

## Proposed rule

Treat the existing material quadrature cuboids (initial half-widths equal to
half the actual three sampling spacings) as the topology carriers. Do not infer
cracks from a damage threshold, a rendered surface, or spatial clustering.
At the accepted state, classify a quadrature cell as severed only when the
unchanged material law reports `fully_failed`. A severed cell keeps its mass,
current center and deformation, all seven material/history fields, and its
plastic heat. It receives an independent support component; it is never
deleted. Its constitutive traction must be unilateral (compression may remain,
tension is zero), with the eliminated tensile work charged to the model's
existing fracture/release ledger. If that ledger cannot account for the
released energy, the split is not admissible.

Construct a face-adjacency graph from the *reference* particle cuboids. Keep
face links between nonfailed cells; do not link through a fully failed cell.
Connected intact regions are separate support components. Each failed cell is
also a separate component (neighboring failed cells may be grouped only if a
validated failure-surface rule says their interface remains bonded; the
conservative default here leaves them separate and lets SAT contact carry
compression). Edge- and corner-only proximity does not create material
connectivity. This produces detached components only when the cell graph is
actually disconnected; an isolated failed cell can move independently, while
intact material paths around a crack tip remain connected.

For each particle `p`, map every reference support node `a` to a key
`(component[p], a)`. Duplicate a node for each key, initialize each copy with
the old node's same `q` and `v`, and allocate its lumped inertia by the existing
quadrature contribution `m_p N_pa`. No mass floor or post-split velocity
projection is permitted. The copied `q/v` and partition-of-unity weights make
the split an exact identity map at the instant of severing:

```text
M'_key = sum_{p in key component} m_p N_pa
sum M' v' = sum M v;  sum q' x (M'v') = sum q x (Mv)
K' = K;  x'_p = x_p;  F'_p = F_p.
```

Remap each material point's force with the same reference gradient and the
same component-specific copied indices. Therefore the force scatter remains
the exact adjoint of `F_p = sum_a q_a tensor grad_X N_pa`; retain the existing
midpoint material update, stress power, artificial-viscosity work, and plastic
heat unchanged. All material history stays attached to its original particle
ID. The mapping only changes which nodal DOFs can transmit its force.

Extend true material-domain SAT broadphase from opposite bodies to all
distinct support components. Skip same-component pairs; include cross-body
and same-body pairs between distinct components. Use the current `x_p,F_p`
parallelepipeds and the existing shared-world-point velocity row
`n dot (v_A(c)-v_B(c))`. Its transpose must scatter to the corresponding
component-specific support copies. The nonnegative e0 impulse then prevents
interpenetration without reconnecting tension. Contact work remains its own
instantaneous jump ledger.

## Admission limits and stop conditions

This is a cell-scale topology, not a mesh-objective crack method. A fully
failed quadrature point is not by itself evidence of a macroscopic crack or
fragment. A production split requires: (1) a connected-component receipt from
the reference face graph, (2) unchanged total particle mass and material
history, (3) exact nodal P/L/K equality before/after copying, (4) the existing
material force/velocity adjoint identity, (5) no tensile traction across a
severed interface, and (6) true SAT contact with its independent post-closing,
P/L and work receipts. If the failed-cell tensile energy has no existing
fracture/release account, stop rather than erase it. If the failed topology
does not disconnect intact support, do not invent separated fragments.

The finite-dimensional check in `tools/verify_tl_support_split.py` exercises
only the instantaneous support-copy invariants and a true-SAT frictionless
impact between independent rigid components. It does not validate the C2
constitutive fracture-energy release, production topology extraction, or
post-split impact trajectory.

## Accepted CP145 audit and adapter

The source-bound CP145 audit is in
`evidence/material-domain-contact/continue-cp136-midpoint-viscosity-to250/data/result/cp145-fracture-support-audit.json`.
It verifies the raw checkpoint SHA and every source hash against the producer
commit before rebuilding the original C2 reference map. On this state the
four fully failed sphere points have exactly zero deviatoric stress and
zero tensile-volume energy. Their (J\simeq0.7013) compression remains
active: EOS pressure is about23.08GPa and recoverable compression energy is
about1.262mJ per point. Their existing internal, thermal residual, cold-work,
fracture-reservoir, and recoverable-volume terms close to a maximum
1.2e-10J/kg decomposition residual. No second energy release was added.

The actual face graph has one 732-cell intact sphere component, one 25,600-cell
plate component, and four retained mass-bearing failed-cell singleton
components. Duplicating the support increases used nodes from 616,322 to
34,395 after unused zero-mass grid nodes are compacted. The serialized adapter
contains copied `q/v`, remapped incidence, masses, fixed flags, body ownership,
all seven particle material fields, cumulative ledgers, and hash-bound parent
and source metadata. The zero-time split leaves mass, P/L/K/U and thermal
residual unchanged; particle `x/F/history` are retained.

There are36 true SAT pairs between the new components in this accepted state,
with maximum pre-existing affine-domain overlap61.9µm and maximum positive
closing14.21m/s. The existing nonnegative e0 projection on the remapped
diagonal nodal mass resolves all68 total contact candidates (including32
sphere/plate pairs); 20 rows are active, route post-closing is
1.71e-12m/s, jump work is−4.46e-7J, and total P/L errors are
2.1e-17Ns/1.6e-20Nms. This overlap is recorded rather than hidden or position
corrected. The adapter is eligible for a bounded continuation using same-body
cross-component SAT; this remains a cell-scale support split, not a fragment
or mesh-objectivity claim.
