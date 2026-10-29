# Active work

Updated 2026-10-30. Only a running process or a newer accepted state counts as
computational progress; an existing output directory does not.
The states below distinguish completed calculations from qualified results.
Live process status is recorded separately by the read-only local monitor.

| Work | Last established state | Next deliverable |
| --- | --- | --- |
| Three-strand forming | Full 0–20 ms animation available | Preserve the accepted animation |
| Tight dry forming | Full 0–60 ms animation available | Preserve the accepted animation |
| Triaxial weaving process | Full 24-fibre geometry reached 3.1 ms; automatic exact-state continuation is advancing toward 4.2 ms | Complete the full physical protocol and render its process |
| AT2 fracture | Zero-initial-damage path completed; user approved its GIF for publication | Preserve the accepted example; further tip-symmetry investigation cancelled |
| Anisotropic phase-field fracture | Complete load path and actual curved-crack GIF approved; fitted path angle 7.41 degrees | Preserve the accepted example |
| Notched three-point bending fracture | User approved the 27-state GIF through 0.113047 mm displacement; original schedule remains incomplete at the 2% small-strain limit | Preserve the accepted postpeak trajectory in the three-case gallery |
| Phase-field fracture with a hole | Full loading schedule, public source and 25-state GIF complete; actual empty hole with a free boundary | Preserve the completed single-mesh example in the phase-field gallery |
| Hyperelastic symmetry under blast pressure | All five 6 ms calculations and synchronized GIFs complete; sixfold refinement and rotation controls complete | Preserve the common-grid, common-scale gallery |
| Blast-loaded sandwich design | Round 2 completed and round 3 is running; baseline eccentric full MPM completed its 2 ms history and GIF; baseline uniform/central and selected eccentric histories continue | Correct the reduced model using separate training and development data, retain the best feasible design, and validate all three pressure cases before refinement and nonlinear material qualification |
| Hypervelocity sphere impact | Restart and split-state mapping repaired; host contact caching and retained converged time steps are advancing the accepted trajectory | Continue toward the debris-cloud stage, retaining physical guards and recording sustained throughput |
| Complete cutting animations | Johnson and Seo complete 61-frame GIFs replace the single-pass previews; multipass reached 272.789 microseconds in final lift toward 276.167 microseconds | Export the initial prefix through final lift after multipass completion and replace its incomplete GIFs |
| Public cutting implementation | Published plane-strain displacement-softening JC, Johnson/Seo presets, fresh/resumable three-pass drivers and ParaView renderer; actual contact state and event restart matched bitwise | Preserve the public implementation and complete production animations |

The read-only local monitor records process status, accepted steps/time, source
update times and available animations. A stalled calculation is reported as
stalled or stopped; the monitor does not silently restart a solver.

## Available forming animations

- [Three-strand forming](images/home/fibres--three-strand-forming.gif)
- [Tight dry forming](images/home/fibres--dry-three-strand-forming.gif)

## Phase-field work

The [zero-initial-damage AT2 example](AT2_SHARP_CRACK.md) completed its loading
path and was approved for publication. The explicit geometric precrack is
retained, but no physical initial damage or imposed d=1 trace is used.
The user cancelled further crack-tip asymmetry investigation. Existing
diagnostic evidence is retained; this does not establish symmetry or mesh
convergence.

Current extensions:

1. Anisotropic fracture energy, with residual and tangent derived consistently
   from the same directional energy. Compare rotated material orientations
   using the same loading and initial crack; allow the path to emerge.
2. A true hole replaces the hard-inclusion showcase. Remove interior material
   and retain a traction-free hole boundary; compute crack–hole interaction.

Both extensions retain zero physical initial damage, irreversibility, force
and damage convergence checks, and an energy ledger. Render solved fields
with ParaView; do not impose a propagated crack or alter fields for display.

## Sandwich design

The active design objective is now fixed-mass material redistribution: keep
total mass at 1.8468 kg and minimize the worst back-face peak displacement over
the three prescribed pressure cases. Geometry, thickness, clamping, pressure
and the 200 MPa skin-stress limit remain unchanged. The existing exact mean
core-density constraint preserves mass while spatial controls redistribute it.
Minimum-mass exploration is no longer the main optimization route.

The [problem specification](BLAST_SANDWICH_DESIGN.md) defines geometry, loading,
design controls, constraints, memory strategy and acceptance criteria.

The earlier reduced bases were inadequate against independent full MPM.
The current 510-mode basis adds residual modes from early full-MPM snapshots;
later time intervals remain independent checks. Eight optimization iterations
retained a feasible candidate with a 21.95% lower reduced-model objective,
0.718707 mm worst back-face peak displacement and 137.984 MPa maximum skin
stress at unchanged mass. The iteration limit was reached; optimizer
convergence and full-model design improvement have not been established.

Continue optimization in successive rounds from the best strictly feasible
controls, retaining each round's design hash and metrics. Three consecutive
rounds improving the objective by no more than 0.1% trigger a local convergence
review, not a claim of global optimality. Full-MPM constraints and improvement
must also pass before accepting the design. Validation already in progress
remains bound to its original control hash; a newer candidate does not silently
replace it. Round 2 completed with objective 0.1427671121, improving the
first-round value of 0.1446114802 by 1.2754%. Round 3 is running; its accepted
best is 0.1427383202. These are reduced-model values, not verified full-MPM
performance improvements.

A 1 microsecond versus 0.5 microsecond reduced-model comparison checks time
integration error. The original later-time validation interval remains fixed;
extending the interval must not erase a prior failed error criterion.

The independent full MPM baseline uses the same 10 mm in-plane / 1 mm
through-thickness quadrature grid as these reduced-model comparisons, all
1728 controls at their initial values, and the complete 2 ms eccentric-pressure
history. That baseline completed all 83,460 steps, with a 0.889399 mm peak
back-face displacement and 119.271 MPa peak skin stress. Its actual full-MPM
surface history has been rendered to an 84-frame GIF. The selected design is independently integrated from its virgin
state under the same eccentric pressure; only the verified design controls
are transferred from optimization. Both save full-state checkpoints every
1000 steps. Initial-design full MPM calculations for uniform and central
pressure are now also running from virgin states, with identical source,
resolution and time interval. Their selected-design comparisons remain
required. Over the common first 0.239636 ms, the independent initial and
selected eccentric histories peak at 0.846046 mm and 0.814771 mm respectively,
a provisional 3.70% reduction. The selected peak is already 13.37% above the
reduced model's 0.718707 mm full-history worst prediction. This requires model
correction with separate training and validation intervals; repeated frozen-ROM
iterations alone cannot establish an accurate optimum. Full-history and
three-case improvement remain unproven.
The persistent optimizer controller may finish round 3 on the frozen model,
then requires model correction. Selected-design steps 1000–8000 are allocated
to early-response training; steps 9000–18000 are development data and steps
19000 onward are reserved for fresh validation after the model is frozen. Subspace
projection and finite-deformation force errors must be distinguished before
claiming that additional modes resolve the discrepancy. Added snapshot modes
reduced displacement error, but the current 9.38% omitted strain energy still
fails the 5% criterion; operator enrichment has not resolved it. No further
frozen-model optimization rounds are launched until this correction is assessed.
The 5 mm production refinement and face plasticity/core crushing remain separate
requirements; neither this baseline nor a reduced-model energy ledger proves
their completion.

Parameter inversion, angular failure envelopes and dedicated material-point
validation workflows were removed in commit `187e2ef`. They are not queued.
