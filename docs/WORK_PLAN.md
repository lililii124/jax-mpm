# Active work

Updated 2026-10-30. Only a running process or a newer accepted state counts as
computational progress; an existing output directory does not.
The states below distinguish completed calculations from qualified results.
Live process status is recorded separately by the read-only local monitor.

| Work | Last established state | Next deliverable |
| --- | --- | --- |
| Three-strand forming | Full 0–20 ms animation available | Preserve the accepted animation |
| Tight dry forming | Full 0–60 ms animation available | Preserve the accepted animation |
| [Triaxial weaving process](WEAVING_PROCESS.md) | The masked contact-array implementation passed actual-state and nonzero-contact equivalence checks; production completed 6.95 ms with 3.36 GB peak USS and continues toward 8.05 ms | Complete the 47.0412 ms protocol under the physical guards and render its process |
| AT2 fracture | Zero-initial-damage path completed; user approved its GIF for publication | Preserve the accepted example; further tip-symmetry investigation cancelled |
| Anisotropic phase-field fracture | Complete load path and actual curved-crack GIF approved; fitted path angle 7.41 degrees | Preserve the accepted example |
| Notched three-point bending fracture | User approved the 27-state GIF through 0.113047 mm displacement; original schedule remains incomplete at the 2% small-strain limit | Preserve the accepted postpeak trajectory in the three-case gallery |
| Phase-field fracture with a hole | Full loading schedule, public source and 25-state GIF complete; actual empty hole with a free boundary | Preserve the completed single-mesh example in the phase-field gallery |
| Hyperelastic symmetry under blast pressure | All five 6 ms calculations and synchronized GIFs complete; sixfold refinement and rotation controls complete | Preserve the common-grid, common-scale gallery |
| Blast-loaded sandwich design | Round 9 ended partial at its soft deadline after four callback iterations; no improved callback candidate was retained and no round 10 was launched. q94934 fresh v2: uniform and central fail the 5% qualification, eccentric passes its per-case check. Physics-only basis extension is running; existing full-MPM and paired 5 mm runs continue | Seal and pin the extended basis, review steps 30000–42000 as development data, then assess the independent 71000–83000 window |
| Hypervelocity sphere impact | The conditioned volume/area contact centroid passed moving-state time-step qualification at one quarter of the original CFL step. A corrected virgin trajectory has reached material failure; the old trajectory is retired from production | Continue the corrected trajectory toward a debris cloud and complete refinement |
| Complete cutting animations | All planned runs complete; whole three-pass history 0–276.167 microseconds and full second pass 100.500–159.167 microseconds are published in eight 400×400 GIFs; both increment-rate views use a fixed logarithmic colour scale. Completed remote raw tasks were retired after verified media and compact provenance download | Preserve the completed animations and compact provenance |
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
The replacement uses 130 load-driven directions built from the initial and
selected designs' physical operators. It was frozen before reading the new
selected-design validation interval, steps 31000–42000. Maximum response error
was 1.955% and omitted strain energy 0.691%, below the unchanged 5% criteria.
The previously failed baseline interval now has a 2.509% maximum response error.
The [validation record](../evidence/blast-sandwich-reduced/union130-validation.json)
retains the original failures and distinguishes checked intervals from the
still-unqualified complete three-case response.
The frozen q94934 validation-only window (steps 30000–42000) was assessed on
all three prescribed loads without fitting or changing the 5% gates. Uniform
fails both maximum response error (8.027%) and omitted linear strain energy
(16.129%); central passes its response checks but fails omitted energy
(9.751%); eccentric passes both per-case checks (maximum response error
3.633%, omitted energy 1.634%). Reports, terminal states, source bindings and
original report hashes are preserved in the
[three-case evidence summary](../evidence/blast-sandwich-reduced/q94934-threecase-fresh-v2.json)
and its byte-identical raw reports. This frozen q94934 assessment does not
qualify the full 2 ms, three-load model or establish a candidate optimum.
The [public reproduction guide](SANDWICH_RATIONAL_REPRODUCTION.md) provides the
construction recipe and assessment commands. A cold construction from public
commit `5207438` reproduced the basis and operator/response artifact hashes
exactly, without private numeric inputs. This source reproduction is separate
from independent MPM validation.

Round 9 ended partial at its 2700-second soft deadline after four callback
iterations, with no guard and no improvement over the retained round-8 KS
objective. Its console logged a strictly feasible line-search evaluation at
0.1448407186, below the retained 0.1448417783, but no matching control vector
or evaluation record was persisted; it is therefore not a retained candidate.
The round-9 archive preserves its four callback records and terminal state.
The old-basis optimization queue remains closed. The best KS and actual-peak
controls remain separately retained, and callback versus fully evaluated
candidate provenance remains distinct. Three consecutive completed rounds
improving the objective by no more than 0.1% trigger only a local convergence
review, not a claim of global optimality. Full-MPM constraints and improvement
must pass before accepting a design. All existing candidates were reevaluated
on the corrected model:

| Design | Objective | Worst back-face peak (mm) | Peak skin stress (MPa) |
| --- | ---: | ---: | ---: |
| Uniform | 0.1911827510 | 0.947513 | 116.208 |
| Round 1 | 0.1659767800 | 0.824258 | 155.831 |
| Round 2 | 0.1635330222 | 0.811996 | 158.043 |
| Round 3 | 0.1741008079 | 0.863537 | 150.074 |
| Round 4, best accepted iterate | 0.1465874642 | 0.725186 | 161.060 |
| Round 5, best accepted iterate | 0.1458214519 | 0.719909 | 164.867 |
| Round 6, best accepted iterate | 0.1452244811 | 0.716707 | 163.119 |
| Round 7, lowest KS objective | 0.1448736541 | 0.717457 | 164.075 |
| Round 7, lowest actual sampled peak | 0.1451089835 | 0.715980 | 163.468 |
| Round 8, retained KS objective | 0.1448417783 | 0.718346 | 164.074 |

All retain the prescribed mass and satisfy the reduced-model constraints.
Round 2 supplied the corrected warm start; round 3's earlier apparent improvement
did not survive model correction. Round 4 reached its wall-time bound after
five accepted iterations, so it remains partial and does not count as a
convergence stall. Its best strictly feasible controls were
`94934d1d5792131b22756f3e2634d25708de07ade1a0ff37610881f8bb1b6ec4`.
Round 5 improved the objective another 0.523% before its soft deadline; it
also remains partial. Round 6 retained three improvements before its soft
deadline. Round 7 completed its four-iteration budget in 1191 seconds and
improved the retained KS objective by 0.242%; its iteration limit is not
optimizer convergence. Round 8 continues from that retained design.
The best actual sampled peak occurred earlier within round 7: the smoothed
objective and the actual maximum need not improve together. Both controls
are preserved in the [public candidate snapshots](../evidence/blast-sandwich-reduced/union130-round7-public-controls.json)
for separate independent verification. A [material-distribution animation](SANDWICH_FULL_MPM.md#material-redistribution)
shows ten retained designs through round 6.
These remain reduced-model predictions, not verified full-MPM improvements.

A 0.25 microsecond versus 0.125 microsecond reduced-model comparison checks time
integration error for the corrected frozen construction. The original later-time validation interval remains fixed;
extending the interval must not erase a prior failed error criterion.

The independent full MPM baseline uses the same 10 mm in-plane / 1 mm
through-thickness quadrature grid as these reduced-model comparisons, all
1728 controls at their initial values, and the complete 2 ms eccentric-pressure
history. That baseline completed all 83,460 steps, with a 0.889399 mm peak
back-face displacement and 119.271 MPa peak skin stress. Its actual full-MPM
surface history has been rendered to an 84-frame GIF. The independently
integrated round-1 selected design has now also completed all 83,460 steps
from its virgin state. Its peak back-face displacement is 0.814771 mm,
an 8.39% reduction under this eccentric load at the same mass. Peak skin
stress rises from 119.271 to 154.968 MPa, below the prescribed 200 MPa limit.
Both complete histories have matching source, resolution, pressure impulse
and duration; their ParaView animations use the same physical scale and
0–1 mm displacement colour range. The
[matched comparison](../evidence/blast-sandwich-reduced/fullmpm-eccentric-comparison.json)
records the source and downloaded output hashes.

The frozen corrected model also passed the
[completed eccentric response check](../evidence/blast-sandwich-reduced/union130-completed-eccentric-response.json):
peak error 1.164%, curve relative L2 error 1.192%, and maximum curve error
2.414% over the complete 2 ms, below the original 5% response threshold.
The later interval after step 42000 has a 2.548% maximum curve error. This
extends the response check without fitting new directions or changing the
original validation receipt. The separate
[later strain-energy check](../evidence/blast-sandwich-reduced/union130-completed-eccentric-energy.json)
also passed: the maximum omitted linear strain energy was 1.665% over all
42 saved states from step 43000 through 83460, below the unchanged 5% limit.
These results qualify this eccentric comparison at the current resolution;
they do not establish the other loads, mesh convergence or a new candidate's accuracy.

This completed comparison uses the older round-1 controls, not round 4.
The round-4 candidate has a separate virgin-state full-MPM run with its own
control hash. Initial-design uniform and central full-MPM calculations have
also completed their full 2 ms histories. The
[three-load response check](../evidence/blast-sandwich-reduced/union130-completed-initial-responses.json)
passed the original 5% threshold: maximum curve errors are 0.460%, 0.635%
and 3.730% for uniform, central and eccentric pressure. Their full-MPM peaks
are 0.566446, 0.947616 and 0.889399 mm. All completed animations are in the
[full-MPM review](SANDWICH_FULL_MPM.md).
The round-4 candidate's uniform, central and eccentric full-MPM runs remain
bound to its original controls and are advancing independently.
The original reduced model predicted 0.718707 mm for the selected worst
response and failed qualification. Its discrepancy is retained rather than
erased by the corrected model. Repeated frozen-ROM iterations alone cannot
establish an accurate optimum or complete three-case improvement.
The earlier rank-526 correction failed selected-design steps 19000–30000:
peak error 6.25%, curve error 7.80%, and omitted strain energy 14.84%.
That failure is retained; this interval became development data for the next
model, and steps 31000–42000 were reserved for its independent check.
The new two-design basis passes both the earlier intervals and that fresh check.
Its time-step sensitivity and energy balance also pass. Subsequent optimization
preserves the frozen basis, checks projected operator agreement and derivatives,
and retains only strictly feasible improvements. A changed candidate still
requires its own independent full-MPM validation.
The paired initial/round-4 eccentric refinement is now running with 5 mm
in-plane spacing, 691200 points and 85880 steps over the same 2 ms horizon.
Each starts from its own virgin state, retains the same physical controls and
fixed mass, and has a 24-hour resource budget. Grid convergence and face
plasticity/core crushing remain separate requirements; neither a coarse-grid
comparison nor a reduced-model energy ledger proves their completion.

Parameter inversion, angular failure envelopes and dedicated material-point
validation workflows were removed in commit `187e2ef`. They are not queued.
