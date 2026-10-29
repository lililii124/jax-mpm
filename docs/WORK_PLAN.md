# Active work

Updated 2026-10-29. Only a running process or a newer accepted state counts as
computational progress; an existing output directory does not.
The remote readback on this date found no active weaving or AT2 solver.

| Work | Last established state | Next deliverable |
| --- | --- | --- |
| Three-strand forming | Full 0–20 ms animation available | Preserve the accepted animation |
| Tight dry forming | Full 0–60 ms animation available | Preserve the accepted animation |
| Triaxial weaving process | Last accepted state 3.606330 ms; incomplete; no GIF | Resolve the continuation guard, then export the actual process |
| AT2 fracture | Current-energy tension 2400 steps / 36 ms; free-anchor bending accepted 3125 steps / 93.75 ms before a later failure; no accepted fracture result | Sharp precrack tension with irreversible energy minimization and localization/energy checks |
| Blast-loaded sandwich design | Added; implementation pending | Uniform-panel transient baseline, then constrained core-density optimization |

The read-only local monitor records process status, accepted steps/time, source
update times and available animations. A stalled calculation is reported as
stalled or stopped; the monitor does not silently restart a solver.

## Available forming animations

- [Three-strand forming](images/home/fibres--three-strand-forming.gif)
- [Tight dry forming](images/home/fibres--dry-three-strand-forming.gif)

## AT2 next steps

1. Check a stationary sharp precrack profile and its crack-surface energy.
2. Use displacement-controlled single-edge precrack tension. Converge the
   mechanical and damage subproblems using current tensile energy and the
   irreversible bound d_old <= d <= 1.
3. At fixed length scale, refine the spatial grid and compare force–displacement,
   crack advance, energy balance and damage-band widths. Assess changing the
   model length separately from mesh convergence.
4. Keep the small-strain validity limit. Choose a brittle benchmark that cracks
   within it; finite-deformation fracture requires its own formulation. Return
   to bending only after the tension benchmark passes.

The short current-energy comparison improves local energy consistency but
does not establish that the broad damage band has been fixed. The constrained
variational route is described in
[Gerasimov and De Lorenzis](https://arxiv.org/abs/1811.05334).

## Sandwich design

The [problem specification](BLAST_SANDWICH_DESIGN.md) defines geometry, loading,
design controls, constraints, memory strategy and acceptance criteria.

Parameter inversion, angular failure envelopes and dedicated material-point
validation workflows were removed in commit `187e2ef`. They are not queued.
