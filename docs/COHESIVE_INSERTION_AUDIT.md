# Cohesive insertion: force balance and localized loading

The moving crack implementation separates velocity fields across an explicit
polyline and inserts interfaces carrying the sampled birth traction. Prescribed
fracture energy controls separation. This audit tests two requirements that the
global energy ledger alone cannot establish: nodal force balance at insertion
and crack direction under localized loading.

## Constant-stress patch

`tools/audit_extrinsic_insertion.py` places a straight interface in a homogeneous,
area-preserving prestress field. It compares bulk and cohesive nodal forces away
from both interface ends. The intact interior has zero force to roundoff. The
reported residual is `norm(bulk + cohesive) / norm(bulk)` on the selected nodes.

The original APIC/MLS stress gradient and mass-weighted face trace leave a
19.58% residual on a grid-aligned interface. The opt-in
`consistent_gradients=True` branch uses the shape-function gradient for both
deformation and bulk force, paired with a shape-weighted, affine-complete face
trace. For aligned interfaces with segment length at most one grid spacing,
the patch residual falls to approximately 1.2e-14.

This does **not** solve general insertion balance. Moving the interface by
0.125 or 0.375 grid spacings leaves residuals of 11.74% and 23.88%, respectively.
Reducing segment length does not remove these residuals. Two-grid-spacing
segments also underresolve the interface quadrature. The remaining discrepancy
requires a cut-cell/particle quadrature or compatible discrete-traction treatment,
not a scalar rescaling of interface strength.

The optional branch is experimental; published tension animations retain their
original implementation and provenance. Their energy checks remain reproducible,
but they do not validate nodal insertion balance or crack-path prediction.

## Localized double-cantilever diagnostic

`examples/extrinsic_crack_dcb.py` defines a 60 x 6 mm plane-strain specimen,
25 mm edge precrack, 0.5 mm grid spacing and two particles per cell per direction.
Opposing grip velocities reach 1 m/s in 0.2 ms; a remote roller constrains only
axial motion. Assumed parameters are E=30 MPa, nu=0.2, density=1200 kg/m3,
strength=120 kPa, Gc=2 J/m2 and interface stiffness=1e11 Pa/m.

The legacy-gradient diagnostic with a 2-degree initial crack completed 2 ms.
It developed 3 mm of inserted interface and 2.5 mm of connected complete failure,
but turned rapidly toward the specimen edge. Reconstructed energy closes within
3.78e-14 J/m; numerical loss is 10.80% of external work. This is not a qualified
crack-path benchmark.

The consistent-gradient diagnostic starts from a horizontal crack. Before
growth, at 0.80009 ms, its 1.5 mm forward sampling region has mean tangential
stress 135.27 kPa, normal stress 98.70 kPa and shear approximately zero. The
principal-stress selector can consequently favour a transverse path even under
symmetric loading. By the next saved frame the last segment is nearly vertical.
The run subsequently failed the coupled iteration check at 1.64540 ms and
stopped. Only the 33 valid saved frames through 1.59996 ms are used for field
diagnostics; the failed state is excluded.

`mpm_jax/crack_tip.py` independently fits KI, KII and nonsingular T stress in
annuli around the tip. Before growth, KII is near zero, but KI changes from
5214 to 2260 Pa sqrt(m) as the outer radius changes from 1 to 2.5 mm. Relative
stress-fit residuals range from 37% to 75%. The nominal cohesive length
`E/(1-nu^2) * Gc/strength^2` is 4.34 mm, larger than these annuli. These fits do
not establish a valid elastic asymptotic field and are not used to steer cracks.
Positive fitted T stress alone cannot determine whether the observed kink is
physical; radius sensitivity, subcell force imbalance and the direction rule
remain confounding factors.

## Next implementation requirements

1. Enforce compatible bulk/interface quadrature for displaced and inclined cracks.
2. Replace broad principal-stress averaging with candidate-extension energy
   comparisons consistent with the cohesive law; verify on symmetric and mirrored
   loading before enabling it in demonstrations.
3. Extend the transactional step-recovery checks below to further loading paths
   and establish time-step accuracy independently of nonlinear convergence.
4. Establish spatial, temporal and insertion-spacing convergence before making
   predictive path or speed claims. Branching, fragment recontact and 3D crack
   surfaces remain separate open work.

Compact inputs, histories, patch results and tip diagnostics are retained in
[`evidence/extrinsic-insertion-audit`](../evidence/extrinsic-insertion-audit).

```bash
python tools/audit_extrinsic_insertion.py --consistent --output outputs/patch.json
python examples/extrinsic_crack_dcb.py --output outputs/dcb-consistent
python examples/extrinsic_crack_dcb.py --legacy-gradients --inclination 2 --output outputs/dcb-legacy
python tools/diagnose_extrinsic_tip.py outputs/dcb-consistent --output outputs/dcb-tip.json
```

The consistent DCB command is a failure reproducer. The diagnostic accepts its
valid prefix without treating the run as complete. Existing output directories
must be empty before a new simulation; runs never overwrite previous fields.

The patch/energy/affine checks passed five focused tests; analytical tip-field,
rotation, angle and empty-annulus checks passed three tests. These checks do not
convert the failed localized run into a successful fracture validation.

## Transactional recovery of the failed step

`mpm_jax.adaptive_crack.retry_step` retries an unaccepted coupled step with
successively halved durations. Every attempt starts from the same particle
state and irreversible cohesive history. Failed attempts contribute no work,
damage or motion. Exhaustion returns the original state and a false acceptance
flag; callers must check it before inserting segments or advancing their clock.
The duration is a runtime JAX argument, so halving does not compile a separate
solver for every step size.

The replay tool starts from saved frame 32 at 1.599955 ms and continues the
same boundary schedule to 2 ms. It rejects two attempts, temporarily reduces
the step from 224.997 ns to 56.249 ns and accepts 1,824 continuation steps.
The largest accepted nonlinear residual is 9.996e-10. The continuation completes
in 92.46 seconds on the local CPU. Its last saved state passes an independent
NumPy reconstruction of bulk, APIC kinetic and interface energies; the energy
remainder is -4.31e-14 J/m, and irreversible history is nondecreasing.

```bash
python tools/recover_extrinsic_dcb.py outputs/dcb-consistent --frame 32 --output outputs/dcb-recovery
```

The tool caps the minimum accepted step and total accepted-step count, grows
the proposed duration after 32 quiet steps, and writes only a compact report
and the last accepted field. Two focused tests verify that recovery matches a
direct step at the accepted duration and that exhaustion returns unchanged
state/history with zero work. The two existing coupled-energy tests also pass
with the runtime-duration implementation.

This repairs the observed nonlinear failure, not the earlier crack-direction
error. The original failed trajectory and its diagnostics remain preserved.
These are acceptance retries, not an estimate of temporal truncation error;
no time-convergence or crack-path claim follows from completing the replay.

## Compatible cut quadrature prototype

`tools/audit_cut_interface.py` integrates the actual quadratic MPM shape
functions on both sides of a straight crack spanning an interior rectangular
body. It partitions the body at shape-function knot lines and the physical
crack, triangulates the resulting polygons and uses 3 x 3 Duffy quadrature.
The crack trace is independently partitioned at the same knot lines and uses
three-point line quadrature. The integration weights depend only on geometry;
the interface force is never fitted to the bulk residual.

The focused check covers nine cuts on a 12-cell grid: angles 0, 0.31 and 0.68
radians, each at offsets -0.375, 0 and 0.25 grid spacings. The constant stress
has unequal normal components and nonzero shear. Both separated fields must
balance individually at nodes whose supports do not meet the outer boundary.
Their worst relative force residual is 2.89e-15. The maximum difference between
the sum of cut bulk integrals and the independently integrated intact body is
6.26e-16; face resultant and moment errors are below 7.03e-16. Partition of unity,
integrated affine reproduction and opposite interface actions are also checked.

```bash
python tools/audit_cut_interface.py --grid 12 --output outputs/cut-quadrature.json
```

This provides a verified geometric integration route to the subcell patch
defect. It is **not yet a production transfer operator**: geometry clipping
runs on the host, the crack spans the body, and the stress is constant. Finite
tips, multiple segments, evolving material domains, constitutive quadrature,
APIC transfer moments and work-conjugate face motion must be integrated before
this can replace the current point-particle solver. The existing dynamic runs
do not use this prototype.

## Finite-tip support coupling

Endpoint visibility introduces a second issue that remains even with exact
volume quadrature. For a node whose support contains a finite tip, the
particle-to-node crossing test can switch fields across a ray extending away
from that tip. This is an artificial basis jump outside the physical crack.
Integrating only the physical crack traction cannot balance that missing flux.

The optional `couple_tip_nodes=True` solver branch maps both velocity copies to
one node whenever an endpoint lies in its quadratic B-spline support. The
componentwise support radius is 1.5 grid spacings, with a floating-point tie
tolerance. Mass, momentum, bulk force and both face traces use this same map.
Coincident face contributions then cancel at coupled nodes. Supports not
containing an endpoint retain the two independent fields. The edge-crack driver
continues to extend the root's visibility segment outside the specimen.

This is a finite-resolution constraint around the tip, not an asymptotic tip
enrichment. It reduces local opening resolution and needs refinement checks.
The coupled single-step test preserves internal linear momentum, coincident
tip positions and the discrete bulk/interface/transfer work ledger.

The finite-crack audit partitions each nodal support at the physical crack,
endpoint rays and spline knots, then independently integrates the physical
face trace. Its two grid-12 cases use a horizontal crack and a 17.76-degree
crack shifted by 0.25 grid spacings. Even with this geometry quadrature, the
legacy map leaves relative interior force residuals of 6.36% and 8.93%.
Coupling endpoint supports reduces them to 3.99e-15 and 2.73e-15. Independently
assembled face resultants and moments agree with the prescribed constant
traction at roundoff. Both this patch test and the coupled-work test pass.
The [finite-crack record](../evidence/extrinsic-insertion-audit/finite-crack-quadrature.json)
retains the full comparison and source hashes.

```bash
python examples/extrinsic_crack_dcb.py --couple-tip-nodes --output outputs/dcb-tied
python tools/check_extrinsic_results.py outputs/dcb-tied --output outputs/dcb-tied-check
```

The same horizontal DCB now completes all 8,889 fixed steps and 41 saved states
through 2 ms without retries. Independent field reconstruction gives a maximum
energy remainder of 4.01e-14 J/m. The numerical loss is 6.90% of final external
work. It inserts 2 mm of cohesive interface and dissipates 0.001819 J/m, but
does not develop a connected fully failed segment. The last tip is near
(35.128, 38.310) mm, so the path still turns away from the original midplane.
This trajectory is a diagnostic, not a replacement fracture benchmark.

At the last saved state before growth (0.85004 ms), the 1.5 mm sampling region
has mean tangential stress 162.16 kPa and normal stress 35.88 kPa, with negligible
shear. The elastic-tip fit has a 67.5% relative residual. These fields continue
to expose the limitations of broad principal-stress steering; coupling the
tip nodes does not validate that direction criterion.

The production option changes the topology map only. It still uses particle
quadrature and the existing affine face interpolation, so the off-grid force
integration error discussed above remains open in dynamics. Compatible moving
domain quadrature and energy-based extension selection remain required.

## Contact-transfer diagnostics

The bending continuation separates particle-to-grid and grid-to-particle
kinetic losses. At the accepted 156.667 microsecond state, one nominal step
loses 0.002504 and 0.003299 J/m respectively. At one sixty-fourth of that
duration the first loss is unchanged and the second is 0.002323 J/m. Repeating
more transfers therefore does not make these losses vanish.

An APIC-mediated face trace was examined using a fixed-stencil G2P/P2G
round trip P and the exact force transpose P.T. It reproduces affine velocity,
preserves resultants and moments, and preserves interface power. P is a
mass-self-adjoint contraction, not an idempotent projection. However, applying
it only to surface forces breaks their compatibility with the bulk gradient:

| Patch | Original relative nodal residual | Mediated faces |
| --- | --- | --- |
| Intact uniform tensile traction | Roundoff | 14.76% |
| Fully released interface under uniform compression | 8.28e-8 | 14.68% |

Restricting mediation to fully released vertices preserves the intact patch
but still fails the compressive-contact patch. Both dynamic trials were stopped;
the solver option was removed. The small pressure-gap construction accounts for
the compressive reference's nonzero roundoff-scale geometric error. The audit
helper remains outside the solver for reproducing this rejected approach.
[`contact-transfer`](../evidence/cohesive-loading-cases/contact-transfer) contains
the measured residuals, accepted-state diagnostics and stopped-trial provenance.
The impulse norms in those diagnostics are mass norms of acceleration,
in J/(m s2); multiplying by dt2 gives the isolated impulse's kinetic energy.
They do not include cross terms between bulk and surface forces.

## JAX cut geometry

[`cut_geometry.py`](../mpm_jax/cut_geometry.py) implements convex half-plane
clipping with fixed-capacity arrays, batched bulk quadrature and outward-normal
boundary quadrature. It supports JIT and vmap; coordinates have automatic
derivatives while the active clipping topology is fixed. Capacity overflow,
clockwise input and nonconvex input are invalidated rather than silently used.
Empty or degenerate pieces have zero quadrature weights.

The three-point edge and nine-point Duffy triangle rules integrate the actual
quadratic MPM basis after pieces are split at spline knots. The compiled
grid-12 audit covers three angles and three subcell offsets with nonzero
constant shear and unequal normal stresses. All nine cases conserve area and
first moment; the largest relative nodal force residual is 3.11e-15. Independent
checks also cover batched complementary cuts and the exact area derivative
with respect to a moving cut. The
[`compiled audit`](../evidence/extrinsic-insertion-audit/jax-cut-geometry.json)
records its source hashes and timings.

```bash
python tools/audit_jax_cut_geometry.py --output outputs/jax-cut-geometry.json
```

This replaces the geometry audit's host polygon loops, not the dynamic
solver's particle integration. Production coupling still requires evolving
fragment volumes, constitutive histories, consistent APIC moments and matching
bulk/interface kinematics. Finite crack tips and curved or branching paths
also remain outside this new compiled audit.

## Conservative fragments and domain-averaged transfers

[`cut_fragments.py`](../mpm_jax/cut_fragments.py) now stores a convex reference
domain, its area and covariance, its current centroid, F, affine velocity C,
mass and intensive constitutive history. A spatial cut is pulled back through
F before clipping. Children inherit F, C and history; mass follows reference
area, and centroid velocity is shifted by C times the centroid displacement.
Empty children remain inactive slots. Small positive domains are retained.

For a current fragment volume V and centroid x_p, the quadratic basis is
averaged over the whole domain:

\[
\bar N_i=V^{-1}\int_\Omega N_i\,dV,\qquad
D_p=\sum_i\bar N_i(x_i-x_p)(x_i-x_p)^T
=\frac{\Delta x^2}{4}I+\operatorname{Cov}(\Omega).
\]

[`domain_transfer.py`](../mpm_jax/domain_transfer.py) uses these weights for
mass and affine momentum scatter. Its velocity fit uses the same D_p inverse.
Bulk force and deformation rate share the integrated basis gradient. The
current-domain covariance therefore cannot be dropped from the kinetic or
angular-momentum accounting while retaining these transfers.

With complete quadratic support, a split preserves each grid node's combined
mass and momentum. It also preserves total mass, linear momentum, angular
momentum including affine spin, and
`0.5 m (v.v + trace(C D_p C.T))`. Reference elastic energy is unchanged because
F and intensive history are inherited. These are instantaneous split and
fixed-geometry transfer properties; they are not a proof of conservation after
advecting the domains. Separate field indices let the children carry independent
velocities; putting them back on one shared field would recouple them.

The deformed-domain audit uses 256 parent domains, an isotropic contraction
and a 0.17-radian rotation, then cuts them at nine angle/offset combinations.
It applies constant compressive traction through the same knot-split geometry
used by the bulk operator. The maxima are:

| Quantity | Relative error |
| --- | --- |
| Interior bulk/contact nodal force balance | 2.37e-15 |
| Split nodal mass | 2.24e-16 |
| Split nodal momentum | 1.60e-16 |
| Split mass, momentum, angular momentum, kinetic energy and reference area | 2.56e-16 |

The [`domain-transfer record`](../evidence/extrinsic-insertion-audit/domain-transfer.json)
contains every cut, separate invariant errors, contact resultants and moments,
source hashes and timings. Quadrature is scanned in batches of 32 fragments:
25,920 bulk points per batch, without a full all-fragment quadrature array.
Domains extending outside their allocated knot tiles are rejected, not cropped.

An exactly symmetric cut initially produced two unresolved corner slivers of
3.61e-35 reference area. Clipping now aligns a plane with a vertex when their
separation is within 16 machine epsilons of the projection scale. Complementary
cuts use the same geometric convention; there is no mass-based deletion rule.
Cut-position derivatives are only claimed away from topology changes and this
roundoff alignment band.

```bash
python tools/audit_domain_transfer.py --output outputs/domain-transfer.json
```

The moving-domain bulk solver now advances F and position while recording
the signed kinetic-energy change associated with D_p. Its independent
[dynamic audit](DOMAIN_DYNAMICS.md) includes transient compression and release
of a pre-existing through-crack. The geometry term is exposed, not removed,
and the release still has substantial transfer loss. Compatible interface
motion and force scatter, and finite-tip activation with independent fragment
fields remain open. The old point-APIC kinetic ledger must not be reused for
the domain operator. These checks do not qualify dynamic crack propagation.
