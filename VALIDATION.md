# Verification and executed examples

The Windows CPU test run passed **124 tests** with Python 3.12.13 and
the PyPI release JAX 0.12.0.1. The suite covers 2D/3D transfers,
constitutive integration, formal boundary conditions, contact, elastic gradients
and interrupted-run recovery. GPU execution has not been validated.
The [test report](docs/results/tests-public.xml) records zero failures, errors
or skips. This report uses the isolated source snapshot selected for the release.
An earlier 67-test baseline also passed with the source build
JAX 0.12.1.dev20261022 and jaxlib 0.12.0.1; its separate
[report](docs/results/tests-source.xml) is retained.

## Numerical checks

- Quadratic B-spline partition, first moments and shape gradients.
- Mass, linear momentum and APIC angular momentum; affine velocity reproduction.
- Rigid rotations, stress/energy consistency, isotropic limits and rotated anisotropy.
- Unilateral walls, moving walls, Coulomb slip, component-wise prescribed velocity,
  activation intervals, translating regions with half-cosine startup and a fully
  clamped body under gravity; cyclic translation and rotation with prescribed round trips.
- Balanced two-body impulses, separation without adhesion and applied particle forces.
- Hertz scaling; independent JKR roots, stable-branch derivatives and attachment
  history; conservative pressure assignment; rigid-sphere impulse balance.
- Plastic return limits, loading/unloading, rate dependence, damage and compaction.
- Invalid deformation/support and constitutive convergence reporting.
- Checkpoint recovery reproduces uninterrupted trajectories and all history fields
  for a plastically deforming body in both failure modes, including an interruption
  between saved frames.
- Ballistic free flight, failed-particle grid/contact exclusion, retained mass and
  linear momentum, frozen history and the JC threshold transition.
- Concrete bending fixture components, prescribed displacement and reaction impulse;
  ceramic indenter/backing impulse balance and damage-region interpretation.

## Supplied constitutive traces

Fifteen compact fixtures contain four JH-2 histories (570 increments), five HJC
histories (640 increments), and six RHT histories (390 increments), generated
from the supplied compiled references.
Compression, shear reversal, damage, hydrostatic loading and unloading paths
exercise stress and selected history variables. The [manifest](tests/data/materials/manifest.json)
records source and fixture hashes; reference binaries are not needed to run the tests.

Stress comparisons use `rtol=2e-7`, `atol=1 Pa`. History tolerances are listed in
[test_reference_traces.py](tests/test_reference_traces.py). These checks apply to
the recorded parameter sets and strain paths. The HJC comparison includes
compression/unloading, reversed shear, pore collapse, tension/recompression and
hydrostatic loading. Its largest absolute stress difference over these paths
is 2.40 × 10⁻⁵ Pa; all five added regression cases pass.

[Source Final Check eligibility](docs/source_audit.json) and verification of the
JAX implementation are separate records. Selected source branches are described
in [MATERIALS.md](docs/MATERIALS.md); unsupported branches are not inferred from
the material number alone.

## Longitudinal-wave refinement

A fixed-end rod with transverse rollers starts with its first sinusoidal axial
velocity mode. The analytic solution uses the same rod length, Young's modulus,
density and final time (0.134164 s). Each cell has two particles per axis.
Errors below are RMS errors divided by the corresponding analytical amplitude.

| Grid | Particles | Velocity error | Displacement error |
|---|---:|---:|---:|
| 32³ | 12,288 | 0.0274716 | 0.0266309 |
| 48³ | 41,472 | 0.0174015 | 0.0180035 |
| 64³ | 98,304 | 0.0127027 | 0.0135669 |
| 48³, half time step | 41,472 | 0.0167466 | 0.0173536 |

Both errors decrease with grid refinement. Halving the time step at fixed grid
changes the error less than refining the grid. This study concerns the smooth
elastic rod; it does not establish convergence of damage, contact or FSI cases.
Definitions, boundary regions and final metrics are preserved in the
[run catalogue](docs/RESULTS.md).

## Finite transverse shear-wave comparison

The new transverse-wave runs use the same Neo-Hookean material, geometry,
initial transverse velocity, component constraints and particle density.
Their 51 saved states span 0.969948452 s;
the comparison uses mean tip-y displacement at matching output times without
temporal interpolation. [The comparison record](docs/results/visible-wave-comparison.json)
checks the individual reviews and their source hashes.

| Run | Grid | Time step (s) | Steps | Minimum tip y (m) | Maximum tip y (m) |
|---|---:|---:|---:|---:|---:|
| wave-grid32 | 32³ | 0.000775958762 | 1,250 | -0.09203337 | 0.09241899 |
| wave-grid48 | 48³ | 0.000524296461 | 1,850 | -0.09207041 | 0.09220094 |
| wave-grid64 | 64³ | 0.000395897327 | 2,450 | -0.09206383 | 0.09212111 |
| wave-half-dt | 48³ | 0.00026214823 | 3,700 | -0.09187538 | 0.09214416 |

| Pair | RMS difference (m) | Maximum absolute difference (m) |
|---|---:|---:|
| wave-grid32 − wave-grid64 | 0.00609678 | 0.0147161 |
| wave-grid48 − wave-grid64 | 0.002087797 | 0.004954009 |
| wave-half-dt − wave-grid48 | 0.0001498382 | 0.0003374825 |

The actual half-step/reference-step ratio is 0.5;
integer step counts preserve the common output schedule. The grid64 history
is a numerical comparator. These differences quantify sensitivity to grid
and time step; convergence order and experimental agreement have not been established.
The longitudinal-wave table above retains its original small-motion problem
and analytical reference. Its error values do not apply to these new runs.

### Prescribed finite deformation

| Case | Tracked response | Minimum | Maximum | Final | Unit | Minimum det(F) |
|---|---|---:|---:|---:|---|---:|
| Cyclic bending | tip y displacement | -8.855423e-08 | 0.1198867 | -8.855423e-08 | m | 0.7759889 |
| Simple shear | top x displacement | 0 | 0.1797375 | 0.1797375 | m | 0.6065101 |
| Cyclic torsion | tip rotation about x | -9.378083e-07 | 1.570512 | -9.378083e-07 | rad | 0.7752374 |
| Pressure-loaded panel | plate center y displacement | -0.1086131 | 0.07503472 | -0.0589359 | m | 0.6897098 |

The saved coordinates use physical geometry at scale 1. The independent
reviews reconstruct displacement and rotation, check fixed-region motion,
prescribed loading, positive determinants and constant recorded mass.
[Boundary definitions and commands](docs/VISIBLE_MOTION.md) describe the new cases.

## Spatial examples

The public [catalogue](docs/RESULTS.md) contains **56 selected completed runs**,
including five explicitly labelled coarse woven previews, covering dynamics,
contact, engineering structures, fluids and FSI. Basic compression and the
separate elastic-material cantilever comparisons remain local; their material
implementations and regression fixtures are retained in the repository.
The homepage shows 40 animations; the eight original small-motion records remain in the numerical catalogue.
Each record includes its actual particle count, steps, physical time, material
parameters, boundary regions, determinant range, damage and diagnostic CSVs.
The original 48³ suite contains approximately 17,000–119,000 particles per case.
Denser benchmark configurations use three particles per cell per axis.
The [contact guide](docs/CONTACT.md) distinguishes the geometric sphere problem
from prescribed Hertz/JKR pressure footprints and describes their verification scope.

| Dynamic response | Recorded result |
|---|---|
| Rubber ball, 44,720 particles | Centre reaches y=0.212329 m, then rises 0.239094 m by the final frame |
| Ball–beam, 88,583 particles | Ball receives 14.701633 N·s upward contact impulse and rebounds from the beam |
| Large bending, 119,944 particles | Free-end mean displacement reaches 0.138588 m at the end of the recorded interval |
| Large twist, 119,944 particles | Maximum free-end rotation 0.719098 rad (41.2013°) at 0.245 s |
| Floating block, 384,373 particles | Maximum downward heave 0.042300 m at 0.45 s; upward motion begins before the final frame |

The ball–beam sphere stays clear of the floor. Its momentum change, gravity
impulse and separately accumulated contact impulse balance within
7.11 × 10⁻¹⁵ N·s. Before floor contact, the rubber-ball centre matches the
semi-implicit discrete free-fall trajectory within 1.50 × 10⁻¹⁵ m.
The reviewed dynamics and engineering runs have constant recorded mass and positive
particle volume. The [response review](docs/results/dynamics-review.json) records
the checks and their scope; `python tools/review_dynamics.py` regenerates it.
Peak values refer to saved frames, and the large-bending maximum occurs at the
end of its time window. These checks do not establish spatial convergence.

The completed water-entry run uses **1,327,400 particles** (1,242,216 liquid and
85,184 solid) on a 72³ grid, with 27 particles per fully filled cell. It advances
2,400 steps over 0.35 s and saves 61 frames. Minimum det(F) across saved states
is 0.788163. The [record](docs/results/production--water-entry--neo-hookean/summary.json)
includes all material, time-step and boundary settings.

Only completed runs enter the catalogue. Runtime measurements exclude compilation,
checkpoint I/O and rendering; simultaneous CPU jobs make them unsuitable as
standalone performance benchmarks. The million-particle run accumulated
6,006.72 s of solver time across checkpoint continuations. File-backed output
histories avoid retaining every frame in private memory. Windows counters for
the continuation process report a 5.60 GiB peak commit and 8.16 GiB peak working
set; these figures exclude the earlier process and the renderer. The
[memory record](docs/results/production--water-entry--neo-hookean/memory.json)
states the measurement scope.

Images are generated by ParaView 6.2.0 from all exported particles. A VTK reader
check compares first/last-frame particle counts, coordinates and scalar fields
against the NPZ trajectories, allowing for float32 visualization storage.
The million-particle run passes this [readback check](docs/results/production--water-entry--neo-hookean/vtk_check.json)
at frames 0 and 60, with all 1,327,400 points present in each frame.

## Material engineering runs

The completed concrete and ceramic cases use a 64³ grid, three initial points
per cell per axis and 41 saved frames over 0.3 ms. Their geometry, fixture
definitions and prescribed motion are documented in the
[material engineering guide](docs/MATERIAL_ENGINEERING.md).
The JC round bar uses an 80³ grid and three initial points per cell per axis,
with 61 saved states over the same duration.

| Case | Particles | Steps | Minimum det(F) | Maximum damage | Recorded response |
|---|---:|---:|---:|---:|---|
| JC round-bar tension | 146,880 | 10,020 | 0.992783 | 1 | Six failed central reference layers; two load-bearing groups with a 14.8366 mm axial gap; final central section force zero |
| HJC three-point bending | 121,104 | 2,120 | 0.976602 | 0.492205 | Final midspan bottom deflection 2.25377 mm; final loading reaction 74.2952 kN |
| JH-2 spherical indentation | 118,580 | 4,080 | 0.985189 | 1 | Full 4 mm loading/unloading cycle; final tracked central surface displacement −2.57792 mm |

JC failure first appears at the 150 µs saved state. The final failed mass is
1.26343 g (4.30127%); all mass remains recorded. The stress-based central section
force peaks at 78.3999 kN at 115 µs and ends at zero. The final mean grip reaction
of 50.2499 kN includes motion of the separated pieces and is not residual
load capacity across the failed section. The
[separation review](docs/results/material-engineering--jc-tension--johnson-cook/interpretation.json)
confirms irreversible failure and a gap larger than three grid cells. These
measurements do not establish mesh convergence or an experimentally calibrated
necking and fracture response.

The HJC run never reaches `D ≥ 0.5` or the material failure flag. In the
JH-2 run, the largest sampled indenter reaction is 644.862 kN at 30 µs;
the 4 mm maximum nominal indentation occurs at 150 µs. The final reference
volume fractions are 43.0857% for `D ≥ 0.5` and 42.5004% for `D = 1`, while
the material failure flag remains false throughout. These results describe
bending and indentation with constitutive damage, without resolved fracture
surfaces or detached fragments. Reaction values use the following integration
step; their listed times identify the saved state from which they are sampled.

Each run's `interpretation.json`, generated by
`tools/review_material_engineering.py`, checks complete sampling, finite final
arrays, positive particle masses/volumes and saved determinants, and constant
recorded mass. These checks do not establish spatial convergence or experimental
agreement. The optional RHT bending configuration has not been run.

Windows memory records for [JC](docs/results/material-engineering--jc-tension--johnson-cook/memory.json),
[HJC](docs/results/material-engineering--concrete-bending--hjc/memory.json) and
[JH-2](docs/results/material-engineering--ceramic-indentation--jh2/memory.json)
cover the solver processes and their descendants; rendering, supervisors and
unrelated applications are excluded. HJC resumed after one memory-guard stop;
JC and JH-2 completed in one attempt.

## Dynamic adhesion cases

Four finite-range Dugdale interface cases use the current MPM surface
coordinates to compute the contact forces: matched nonadhesive and adhesive
pull-off, lateral release and a three-sphere array. Their independent reviews
reconstruct the gap forces and moments, compare the saved observations,
check the resolved interaction range and confirm complete final separation.

| Case | Particles | Steps | Minimum det(F) | Surface y displacement range (m) | Maximum relative mass drift |
|---|---:|---:|---:|---|---:|
| [nonadhesive-pull-off](docs/results/adhesion--nonadhesive-pull-off/interpretation.json) | 367,744 | 1,200 | 0.7943203 | -0.03225194 to 0.01010404 | 0 |
| [adhesive-pull-off](docs/results/adhesion--adhesive-pull-off/interpretation.json) | 367,744 | 1,200 | 0.7844015 | -0.03233594 to 0.01583344 | 0 |
| [lateral-release](docs/results/adhesion--lateral-release/interpretation.json) | 367,744 | 1,200 | 0.7685348 | -0.03236452 to 0.015107 | 0 |
| [three-sphere-array](docs/results/adhesion--three-sphere-array/interpretation.json) | 367,744 | 1,200 | 0.7797053 | -0.03243363 to 0.01552029 | 0 |

The [matched comparison](docs/results/adhesion-comparison.json) verifies the
same initial particle coordinates, material, grid, time step and sphere path.
Only work of adhesion and its derived traction differ. Its
[response table](docs/results/adhesion-comparison.csv) preserves the sampled histories.

| Role | Work of adhesion (J/m²) | Minimum withdrawal reaction y (N) | Final minimum gap (m) |
|---|---:|---:|---:|
| nonadhesive | 0 | 0 | 0.1419603 |
| adhesive | 12 | -13.9306 | 0.1424596 |

These are dynamic finite-layer checks with reversible reference-area adhesion
and finite penalty overlap. Analytical JKR/Maugis limits, irreversible bond
fracture, tangential friction and mesh convergence have not been established.
[The adhesion guide](docs/ADHESION_CASES.md) gives the definitions and detailed response measurements.

## Limits

Default material constants are demonstration values. Public-example adaptations
have different discretizations, constitutive laws or geometry and are not
experimental reproductions. The compressible inviscid liquid and nonadhesive
contact interface do not provide a viscous, incompressible FSI model.

Positive deformation determinants and finite states do not establish physical
accuracy. Damage has no fracture-energy regularization, and failed particles
retain mass. The default `retain` mode keeps their grid participation; optional
[ballistic failure](docs/FAILURE.md) excludes them from grid transfers and
contact starting with the following step. Debris then follows its velocity,
gravity and any applied particle force, with frozen deformation and history.
There is no crack-face contact or mesh-objective localization width. Grip
reactions after tensile separation can include vibration of the remaining
pieces and do not establish force transmission across the failed section.
Constitutive work is not recoverable elastic energy; the outputs do not claim
an exact discrete energy balance. Elastic
gradients are tested, while plasticity/damage maps are not advertised as differentiable.

## Coarse woven previews

Five full-duration coarse woven runs are archived separately from production-resolution verification.
Their original reviews remain `passed: false`; the preview gate permits only the stated resolution
limits and the compression case's unavailable inner-platen tracking sample.
Independent elastic energy, reference section force and reference bending-moment checks pass.
The [preview guide](docs/WOVEN_PREVIEWS.md) links the records and encoding provenance.

## Woven plain-weave bending

The [fine-grid bending run](docs/results/woven--plain--transverse-bending/interpretation.json) uses 345,600 material points,
4,140 steps and 61 saved states. The grid resolves the yarn minor diameter with three cells.
Peak centre deflection is 0.271474 mm; minimum J is 0.912105,
maximum saved yarn Green-strain norm is 0.056529,
and relative mass drift is zero. Independent final elastic energy, Piola section forces
and the reference bending moment agree with the recorded observations.

This dynamic, fully bonded elastic example uses illustrative material constants.
It is not a mesh-convergence or experimental validation result. The thin support and
loading patches have no full-stencil interior particle samples, so exact particle
clamping is not asserted. The [source record](docs/results/woven--plain--transverse-bending/numerical-source.json)
binds the Linux calculation to commit `035897ac34ba3327deadfa6ca839efa75768be3a`.
The archived coarse run remains available; the homepage now shows the fine-grid animation.

## Woven plain · transverse compression

The [completed run](docs/results/woven--plain--transverse-compression/interpretation.json) records 345,600 material points,
4,140 steps and 61 saved states. Minimum J is 0.962991; maximum saved yarn
Green-strain norm is 0.052852, below the prescribed 0.1 limit.
Independent final stored energy, reference section forces and bending moment pass the source-bound review.
These checks establish numerical admissibility for an illustrative dynamic elastic model;
the [model scope](docs/WOVEN_COMPOSITES.md) and convergence limitations remain applicable.

## Woven triaxial · in plane shear

The [completed run](docs/results/woven--triaxial--in-plane-shear/interpretation.json) records 345,600 material points,
4,140 steps and 61 saved states. Minimum J is 0.951967; maximum saved yarn
Green-strain norm is 0.084449, below the prescribed 0.1 limit.
Independent final stored energy, reference section forces and bending moment pass the source-bound review.
These checks establish numerical admissibility for an illustrative dynamic elastic model;
the [model scope](docs/WOVEN_COMPOSITES.md) and convergence limitations remain applicable.

Loading revision 2 prescribes ±150 µm. The retained coarse record uses ±200 µm,
so the two runs have different loading amplitudes and do not form a grid-convergence pair.

## Woven plain · in plane shear

The [completed run](docs/results/woven--plain--in-plane-shear/interpretation.json) records 345,600 material points,
4,140 steps and 61 saved states. Minimum J is 0.952719; maximum saved yarn
Green-strain norm is 0.054444, below the prescribed 0.1 limit.
Independent final stored energy, reference section forces and bending moment pass the source-bound review.
These checks establish numerical admissibility for an illustrative dynamic elastic model;
the [model scope](docs/WOVEN_COMPOSITES.md) and convergence limitations remain applicable.

## Woven plain · warp tension

The [completed run](docs/results/woven--plain--warp-tension/interpretation.json) records 345,600 material points,
4,140 steps and 61 saved states. Minimum J is 0.991945; maximum saved yarn
Green-strain norm is 0.032910, below the prescribed 0.1 limit.
Independent final stored energy, reference section forces and bending moment pass the source-bound review.
These checks establish numerical admissibility for an illustrative dynamic elastic model;
the [model scope](docs/WOVEN_COMPOSITES.md) and convergence limitations remain applicable.

## Woven triaxial · transverse bending

The [completed run](docs/results/woven--triaxial--transverse-bending/interpretation.json) records 345,600 material points,
4,140 steps and 61 saved states. Minimum J is 0.911839; maximum saved yarn
Green-strain norm is 0.071721, below the prescribed 0.1 limit.
Independent final stored energy, reference section forces and bending moment pass the source-bound review.
These checks establish numerical admissibility for an illustrative dynamic elastic model;
the [model scope](docs/WOVEN_COMPOSITES.md) and convergence limitations remain applicable.

## HJC large-displacement bending

The [completed run](docs/results/material-engineering--concrete-bending-large--hjc/interpretation.json) uses 121,104 material points, 10,500 steps and 61 saved states over 1.5 ms. Actual midspan bottom deflection reaches 16.3525 mm; 21.4378% of the reference volume reaches D ≥ 0.5. Minimum det(F) is 0.965609, recorded mass is unchanged and no particle carries a material failure flag. The strain-floor profile retains compressive response and grid participation: D=1 is constitutive damage, not a resolved crack surface or a detached fragment.

The earlier 2 mm legacy-profile run remains archived. This new loading history and constitutive profile do not form a convergence comparison with it.
