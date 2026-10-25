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

## Persistent-grid follow-up candidate (source only)

`examples/phase_field_bending_persistent_grid.py` is a separate numerical model:
particles remain reference quadrature, but nodal displacement and velocity are
persistent lumped-mass Galerkin unknowns. It removes repeated PIC velocity
resets; it must not be represented as a validated fix to the original PIC run.
Original source and raw results are retained unchanged.

Raw 40154387841 has max history .0911934 J/volume; Gc/(2ell)=.125. A local
homogeneous AT2 estimate d=2H/(Gc/ell+2H) gives .422 even at the maximum H;
gradient regularization and spatial averaging suppress isolated damage further.
Actual maximum d=.166536 is plausible for limited loading, high Gc and a broad
notch, not evidence of a propagating crack. Final punch travel is .01310976.

The original audit's -2.8044e-5 J force work uses pre-contact kick velocities,
whereas the displacement/strain update uses the constrained drift. A coarse
snapshot trapezoid estimate of internal strain work is +.00118959 J (omits
first 100 steps), commensurate with final elastic .00118508 J and unlike the
old force-kick work. This diagnoses a work-definition mismatch; it does not
numerically repair the old audit. Evidence is recorded in
`.local/at2-bending-original-work-diagnosis.json`. PIC losses additionally consume
79.6% of old constraint-projection work.

The new candidate computes actual force-drift work and enforces its exact
weak identity with stress:strain_increment. Exported terms separately include
punch work, support/anchor work, contact projection/impact loss, damping loss
(default damping zero), frozen-damage strain-integration defect, kick-versus-
drift work difference, signed damage-relaxation potential change and algebraic
closure. The symplectic Euler integrator is not energy conserving; its temporal
errors must decrease under refinement. Signed damage relaxation is exposed
rather than folded into fracture energy or asserted to be physical heat.
Authoritative kinetic energy is lumped nodal, with particle interpolated
velocity only a derived field. Complete snapshots include nodal damage,
particle damage gradient, quadrature volume, nodal displacement/velocity/mass,
and local and gradient fracture energy densities for independent integration.

A NumPy no-damage 3D quadratic-B-spline beam-mode manufactured check passes:
weak force/drift identity 6.78e-21; relative maximum elastic energy errors
3.033%, 1.494%, .742% for dt,dt/2,dt/4. Moving-punch projection work/impact
closure is zero. Run `python tools/qualify_phase_field_bending_algebra.py`;
evidence: `.local/at2-bending-persistent-grid-manufactured.json`. This is an
independent algebra check, not an executed JAX kernel qualification. Before
any medium run, an isolated runner must execute the production kernel smoke
and short-displacement dt=.04/.02/.01 CFL comparison and reproduce weak work
identity/contact signs, finite histories, KKT and irreversibility gates.

Proposed stronger case: h=.025, ell=.05 (ell/h=2), Gc=.005, density=1,
speed=.12 with a linear velocity ramp .012 time units, target punch travel
.035. Preflight: 51,220 particles, dt=3e-5, 9,923 steps. This changes toughness,
length and travel deliberately; it is a new exploratory material/load case,
not a controlled comparison proving improvement. A maximum absolute strain
ceiling .1 terminates qualification; even below it small-strain validity still
needs review. No prediction that d>.9 or a crack will occur is warranted.

Bounded serial sensitivity after kernel qualification: baseline; density=.25
at identical physical displacement/time/ramp (dt=1.5e-5, 19,845 steps); ell=.075
and .10 at density=1 with all other parameters fixed; and half loading speed
with equal target displacement. Density reduction changes dynamics and does
not itself prove quasi-static behavior. Length changes regularized material
strength. Lower speed requires more steps and must meet the 25,000-step cap.
CFL refinements should use a short target travel so their matched physical
interval remains under this cap. No automatic dispatch/workflow was added.
Windows local work remains AST, NumPy algebra and geometry only; quota 76%.

## Corrected exact-boundary quadrature (supersedes earlier 51,220-point plan)

`examples/phase_field_bending_geometry.py` now segments cells exactly at beam
and notch faces. Beam bounds are [0,1.6] x [0,.4] x [0,.16]; removed notch is
[.72,.88] x [0,.16] x [0,.16]. Each segment is divided into equal cells of
width at most h/2. The particle is the actual cell centroid and its quadrature
weight is the product of actual cell widths, never an assumed (h/2)^3.
The set of material cells has exact volume .098304 on all candidate grids.
Counts for h=.04/.025/.02 are 12,288 / 53,144 / 98,304. At h=.025, cell widths
range .012 to .0124137931; thickness and notch depth remain exactly .16.
Reference cell bounds are exported with every complete raw frame. The old
51,220-point uniform-volume proposal is superseded and is unsuitable for
strict fixed-geometry convergence comparisons. Existing raw runs are unchanged.

The reference-grid support and punch rows/inequalities are preserved, including
their stated material-attached nodal approximation. Nodal sampling of shoe
footprints still changes with h, so exact volume alone does not establish
boundary-condition convergence. Natural damage boundary conditions follow the
actual quadrature boundary. Energy/force identities use true cell volumes.

Prepared `.github/workflows/phase-field-persistent-grid-smoke.yml` only: first
exact geometry and independent algebra checks, then the actual JAX two-step
kernel and three equal-displacement (.00005) CFL .04/.02/.01 short runs. The
baseline short run has 106 steps, with 211/422 at refined dt. Each solver
process has 9-minute timeout and 6-GiB virtual-address bound; workflow timeout
55 minutes. Raw source hashes, runtime versions, logs and arrays are preserved.
There is no medium/full run option, renderer, publication or dispatch. The
short-travel schedule now solves the ramp's quadratic displacement law for
its end time, avoiding overshoot from the constant-speed duration formula.

The bounded workflow now runs `tools/qualify_phase_field_bending_refinement.py`
after all three short runs. It requires matched final travel .00005 (absolute
tolerance 1e-14), finite history and complete final arrays, KKT <=1e-7, maximum
strain <=.1, damage within [0,1], exact volume within 1e-14, and cumulative
algebra/weak-work residuals below 1e-10. The absolute signed sum of the frozen
strain-integration defect and kick/drift difference must decrease by at least
20% per dt halving (absolute floor 1e-13). This is a declared admission rule,
not an observed result. Contact events can cause nonmonotonic endpoint errors;
such a result refuses promotion and must be inspected. Signed damage splitting,
actuator work, fracture energy and normalized integration errors remain exposed.
Passing this short interval does not admit a medium run or certify fracture.

Source hashes cover every Python file under `mpm_jax`, the geometry, driver,
both qualification tools and workflow. This broader closure is needed because
importing the package initializes additional material modules. The minimal new
candidate file set is those five explicit files plus `mpm_jax/phase_field.py`
and this document; existing tracked package dependencies must be present from
the exact checked-out commit. Local AST/geometry/NumPy checks passed on
2026-10-25; no production JAX smoke or refinement result is claimed.

## Required eventual crack evidence

Neither rejected GIF demonstrates a crack. Increasing maximum damage or adding
a color threshold is insufficient. Before increasing load, JAX smoke/refinement
must pass all existing KKT, irreversibility, contact and actual force/drift
identity gates. Compare signed integration/phase terms against both actuator
work and fracture energy; algebraic closure alone is not qualification.

For an eventual fracture demonstration retain unscaled displacement, exact
reference geometry, full strain/damage/gradient/nodal fields and reactions at
sufficient temporal cadence. Define and report the d>=.9 connected damaged
region attached to the initial notch boundary. Require measurable advance at
least 2h into the initially intact ligament and report length versus time;
show the through-thickness connected fraction rather than a surface-only patch.
For a near-through crack claim, require the connected path to reach the top
boundary over at least 80% of thickness; otherwise report partial damage only.
These thresholds are declared operational diagnostics, not universal material
failure laws. Check stability under h and damage-contour .8/.9/.95 sensitivity.

Measure opening from displacement differences of two tracked material strips
on opposite sides of the detected path at fixed y,z stations. Report reference
separation and its change; positive opening must exceed both interpolation
noise from the intact smoke and discretization sensitivity. Explain that a
continuous phase-field displacement field gives a regularized opening proxy,
not a literal topological separation. Render actual displaced particles with
physical void/occlusion, and label any visualization magnification explicitly.

Require a resolved reaction peak and sustained >=20% force reduction over
increasing punch travel, coincident with connected damage advance and increased
fracture energy. Force drop alone may be inertia/contact release; compare
support reactions, kinetic energy and contact gaps, and repeat a slower run
before attributing it to quasi-static fracture. If strain exceeds .1, stop this
small-strain candidate rather than force a dramatic movie. No crack threshold,
opening or force-drop criterion has been passed by this source-only work.
