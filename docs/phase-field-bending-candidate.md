# Unpublished AT2 three-point bending qualification plan

Source: `examples/phase_field_three_point_bending.py`. Isolated manual workflow:
`.github/workflows/phase-field-three-point-bending.yml`, permitted only on
`work/phase-field-bending`. Nothing has been dispatched or published.

Geometry is a 1.6 x 0.4 x 0.16 three-dimensional beam, with a removed bottom
notch 0.16 wide and 0.16 deep. Its width exceeds the complete quadratic B-spline
support diameter 3h for every candidate grid, avoiding a shared nodal basis
across the notch faces. This wide notch is an engineering demonstration and
must not be identified as a sharp-crack benchmark. Material units are consistent
with E=1000, rho=1, nu=.2, Gc=.02; they are illustrative, not a calibrated solid.

Boundary specification: two frictionless flat shoes centered at x=.2 and 1.4
on the y=0 reference-grid row, and a prescribed downward flat punch at x=.8
on the y=.4 row. Each has half-width .06, spans thickness, and enforces a
unilateral normal displacement inequality by a velocity projection. Left shoe
central nodes anchor x,z to suppress rigid modes. Neither end is clamped.
The support gap is u_y>=0; punch gap is U_punch-u_y>=0. Normal reactions must
be nonnegative on lower shoes and nonpositive on punch. Contact is formulated
on material-attached grid rows: it does not implement finite sliding, evolving
surface normals, particle contact, or large-deformation penetration detection.

Every mechanical step uses a converged box-constrained AT2 solve, with history
maximum and d_old<=d<=1, projected KKT tolerance 1e-7. Contact prediction uses
a duplicate damage solve to match the audited mechanics force exactly while
retaining the old damage state for fracture-energy increments. The original
published core and notched-tension example remain unchanged. Cost is roughly
two damage solves per time step; optimize only after evidence supports it.

## Bounded qualification

1. Run geometry preflight only locally. Do not import JAX or ParaView under the
   current Windows commit headroom. Compile source without solver execution.
2. On an authorized isolated runner, smoke for two steps first. Check finite
   arrays, KKT, irreversibility, contact gap and reaction sign assertions.
3. Run h=.04, ell=.08, 1200 steps as the medium candidate. The workflow limits
   solver time to 90 minutes, owned-session USS to 5.5 GiB and minimum
   system-available memory to 2 GiB, with a 105 minute job timeout. Observer
   failures stop and reap the entire owned session. Source/runtime and exact
   registered workflow Git blobs are bound before and after computation. These
   are hard termination bounds, not a validated peak memory prediction. Incomplete runs have no completion summary. No renderer.
4. Assess actual load-displacement curve, notch-tip damage growth, out-of-band
   damage, support imbalance against inertia, and strain validity. Report
   kinetic/elastic ratio and run a slower loading rate before any quasi-static
   claim. A maximum-damage increase alone does not prove crack propagation.
5. Inspect cumulative P2G, G2P and damping loss, exact constraint-projection work,
   constitutive-work defect, kinetic closure and total closure. Algebraic total
   closure subtracts the independently reported defect; its smallness alone
   does not establish accurate fracture-energy balance. Compare the defect with
   input work and stored/fracture energy. Do not relabel losses or defects heat.

## Sensitivity matrix (queued plan, not executed)

First hold ell=.08 and geometry fixed: h=.04/.025/.02, with 1200/1920/2400
steps respectively. CFL dt scales with h, giving the same physical time and
ramped punch displacement. Thus ell/h=2,3.2,4. This is grid convergence at a
fixed regularization; it is not convergence to a sharp crack. Compare the
complete reaction-displacement curves, onset displacement under an explicitly
reported damage criterion, fracture energy, and geometrical advance of a
fixed damage contour from the initial notch tip. Report all resolution results;
if they disagree, retain an exploratory label and do not choose a favorable one.

Then fix h=.025 and compare ell=.08,.10,.12 (ell/h=3.2,4,4.8), 1920 steps.
This deliberately changes the regularized material response and AT2 strength,
so changes are model sensitivity, not numerical error. Keep E,nu,Gc, density,
loading history, notch and support footprints fixed. Reuse the ell=.08 run.
Six unique runs total including a half-speed medium follow-up (double steps
for equal displacement). Admit them serially after measuring first-run peak
memory and runtime; no automatic matrix dispatch. Stop work if weekly usage
reaches 90%; current pre-implementation check was 74%.

Qualification status: source candidate only. No local dynamic solve, no crack
propagation claim, no quantitative convergence or calibrated engineering claim.

## Independent existing tension-run review

The existing full1200 raw summary (run 40149430309) has max damage .95813,
max KKT 9.9991e-8 and maximum strain .04881. Its final kinetic/elastic energy
ratio is 1.109; this is dynamic, not demonstrated quasi-static fracture.
Transfer losses total 4.42358e-4 J (26.4% of exact constraint work), damping
loss is 1.90812e-4 J (11.4%), and constitutive defect is 4.60421e-5 J (2.75%).
The ~2e-18 closure residual is algebraic closure after accounting for defect.
Off-band maximum damage .439 requires inspecting localization and contour
advance before calling the output a propagating crack. The bending candidate
must pass the same physical review independently.
