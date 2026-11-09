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
