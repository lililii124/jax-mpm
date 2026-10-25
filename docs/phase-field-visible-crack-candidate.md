# Fresh AT2 visible-crack experiment (unrun candidate)

This isolated experiment repeats the original 28,800-particle tension solve from
zero displacement, velocity, strain, history and nodal damage, for 2,000 steps.
It does not restart the 1,200-step particle archive: that archive lacks the exact
nodal damage state. The loading duration is increased by 800 steps at unchanged
speed .15, dx .025, length .05, Gc .005, density 1, E 1000 and Poisson ratio .2.
No particle deletion, prescribed crack damage or displacement magnification is
introduced. The original two rejected GIFs remain unsuitable evidence of a crack.

The candidate workflow is `.github/workflows/phase-field-visible-crack.yml` and
accepts only `lililii124/jax-mpm`, `refs/heads/work/phase-field-visible-crack`,
matching dispatched SHA and byte-identical frozen workflow/source blobs. Publisher
must commit and select that branch before dispatch. This preparation does not
push, dispatch or publish. All output goes to `outputs/phase-field-visible-crack`.
The inherited Linux owned-session guard limits USS to 5.5 GiB, available RAM to
at least 2 GiB and runtime to 16,200 seconds; observer failures fail closed and
terminate/reap the owned session. A 2,000-step run may exceed this runtime.

`tools/audit_phase_field_visible_crack.py` writes `crack-checks.json` and exits
nonzero unless every condition passes:

- Particle d >= .99 forms one face-connected component on the reference
  quadrature lattice, inside |y-.5| <= .13, starting no farther than one particle
  spacing past x=.25, extending at least .1 past the notch and occupying at least
  80% of thickness layers. This demonstrates a finite crack extension, not full
  separation of the remaining ligament.
- Final reaction is at most 50% of the sampled peak reaction. Reaction is dynamic
  and includes inertia/damping; it is not a quasistatic force-displacement result.
- Every-step nodal irreversibility >= -1e-12 and maximum relative KKT <= 1e-7;
  final particle and nodal damage bounds, all final arrays finite.
- Exact nodal interpolation reproduces stored particle d and gradient to 1e-12,
  and quadrature fracture energy agrees with the final energy diagnostic.
- Cumulative accounting errors <= 1e-8 times max(1, work, energy), every-step
  accounting errors <= 1e-9 times that scale; transfer/damping losses nonnegative
  to 1e-10 times that scale; absolute cumulative constitutive defect <= 5% work.
  These identities check accounting. They do not establish time/mesh convergence
  or a physically negligible constitutive defect at every individual step.

Final `nodal_state.npz` contains exact reference nodal coordinates, damage
coefficients, quadrature volume and basis nodes/weights/derivatives, exact particle
fracture density, and volume-weighted recovered nodal damage gradient and fracture
density. Recovery fields are explicitly named: they are projections, not exact
pointwise derivatives of the nodal coefficients. `particles.npz` contains the
exact quadrature damage gradient and particle state. All 81 saved snapshots
include exact nodal damage and coordinates, reference and deformed particles.
These arrays improve reproducibility; an exact restart implementation is not
provided or claimed.

Only successful solve and crack admission permit ParaView installation/rendering.
The renderer retains all particles at true displacement scale with translucent
damage colors and draws the genuine d=.99 contour of an alpha=.025 tetrahedral
interpolation of saved particle damage. It rejects an empty final contour. This
interpolated isosurface is identified in the receipt; it is not the native
B-spline field and no fabricated gap is drawn. The encoder verifies every decoded
frame against its quantized PNG before admitting the GIF. A failed criterion
retains raw diagnostics and produces no qualified GIF.

Lightweight local checks: Python compilation plus
`python tests/test_visible_crack_admission.py` (two tests, three synthetic cases:
connected pass, subthreshold fail, disconnected fail). These test gate logic,
not physical fracture. No local JAX solve or ParaView rendering has been run.
Whether 2,000 steps actually produce an admissible crack remains unknown.
