# Active work

Updated 2026-10-29. Only a running process or a newer accepted state counts as
computational progress; an existing output directory does not.
The states below distinguish completed calculations from qualified results.
Live process status is recorded separately by the read-only local monitor.

| Work | Last established state | Next deliverable |
| --- | --- | --- |
| Three-strand forming | Full 0–20 ms animation available | Preserve the accepted animation |
| Tight dry forming | Full 0–60 ms animation available | Preserve the accepted animation |
| Triaxial weaving process | Corrected guide-work ledger passed release checks; full 24-fibre geometry reached 250 microseconds, then resumed toward 1 ms | Advance through guide release and actual contact, retaining the complete physical protocol |
| AT2 fracture | Zero-initial-damage path completed; user approved its GIF for publication | Preserve the accepted example; further tip-symmetry investigation cancelled |
| Anisotropic phase-field fracture | Directional energy implemented; the first bounded 48-by-49 run retained a prefracture path and awaits exact-source continuation | Complete the load path and render freely evolving crack deflection |
| Phase-field fracture with hard inclusions | Elastic-modulus-only inclusion was penetrated; a paired 8x-modulus / 10x-fracture-energy case is running | Render the actual resistance, arrest or deflection produced by the second phase |
| Hyperelastic symmetry under blast pressure | Sixfold 6 ms GIF approved; material-rotation and isotropic controls complete; refined sixfold calculation active | Compare square, rectangular, sixfold, eightfold and isotropic plates under identical geometry, load, camera and display scale |
| Blast-loaded sandwich design | Fixed-mass 486-mode optimization has retained feasible redistribution steps; full independent eccentric-load MPM baseline remains active | Obtain substantial redistribution, verify its response against full MPM, then add nonlinear material qualification |

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

Proceed with two extensions:

1. Anisotropic fracture energy, with residual and tangent derived consistently
   from the same directional energy. Compare rotated material orientations
   using the same loading and initial crack; allow the path to emerge.
2. Elastic hard inclusions in the matrix, with explicit stiffness contrast and
   documented fracture/interface assumptions. Compare crack propagation with
   the homogeneous matrix under the same loading.

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

The first reduced basis was inadequate: its 0.477% design improvement was
smaller than the basis error and is not a qualified optimization result.
Adding membrane and core-shear modes changes that response substantially.
The enriched 294-to-384-mode baseline comparison differs by at most about
2.2%; extreme density designs and timestep sensitivity remain under review.
At 294 modes, the admissible minimum-density design has mass 1.45314 kg and
violates the 200 MPa skin-stress limit under central and eccentric loading
(approximately 226 and 350 MPa). The constrained mass problem therefore does
not reduce to assigning the minimum density everywhere.

The independent full MPM baseline uses the same 10 mm in-plane / 1 mm
through-thickness quadrature grid as these reduced-model comparisons, all
1728 controls at their initial values, and the complete 2 ms eccentric-pressure
history. It saves accepted full-state checkpoints every 1000 steps. The
5 mm production refinement and face plasticity/core crushing remain separate
requirements; neither this baseline nor a reduced-model energy ledger proves
their completion.

Parameter inversion, angular failure envelopes and dedicated material-point
validation workflows were removed in commit `187e2ef`. They are not queued.
