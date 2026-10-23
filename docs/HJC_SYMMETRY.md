# Concrete beam with symmetry constraints

`concrete-bending-symmetric` is a separate boundary-value problem for comparing
the support conditions of `concrete-bending-large`. It retains the same initial
particle lattice, HJC `strain-floor` material, time step, zero gravity and
12 mm loading history. The original 2 mm and large pin/roller cases remain
available and keep their definitions.

The 300 × 60 × 60 mm beam has two 18.75 mm wide bottom support patches at
x = 75 and 325 mm. Both patches constrain only vertical velocity. Two additional
planes constrain normal velocity: vx = 0 at x = 200 mm and vz = 0 at z = 200 mm.
Tangential velocities remain free on each plane. These are imposed symmetry
conditions throughout the specimen, rather than a test of unconstrained
symmetry breaking or buckling.

An even grid is required. Each symmetry plane selects exactly one node plane
using a coordinate interval of ±0.25 grid spacings; it does not clamp a thick
material slab. On the 64-cell grid the center plane index is 32, the two support
x-index sets are 11–13 and 51–53, and the centered loader includes indices
30–34. Actual selected node indices and all five boundaries are recorded in
the case definition. Support x endpoints are expanded outward by one floating
point ULP so that nominally coincident endpoint nodes are included on both sides.
This matters on the 128-cell grid; it does not change the selected 64-cell support
nodes. The old cases retain their original support endpoints.

The central loading patch constrains y velocity only. It moves down 12 mm with
0.3 ms cosine acceleration, 0.6 ms constant speed and 0.3 ms cosine deceleration,
then holds for 0.3 ms. The particle coordinates are displayed without deformation
magnification.

```sh
python -m examples.material_engineering --case concrete-bending-symmetric \
  --model hjc --grid 64 --ppc 3 --frames 60 --no-render \
  --output outputs/material-engineering
```

Each saved frame reports all five boundary reactions, actual bottom-midspan
deflection, center-of-mass displacement, and left/right reference-volume-weighted
damage statistics. Left/right averages exclude any particle layer exactly on
the reference center plane. Separate x- and z-reflection comparisons pair the
original Cartesian particle lattice and report damage MAE, RMS and maximum
differences, plus displacement-vector RMS differences. Pairing does not change
as the particles deform.

The NumPy tests apply the actual velocity-boundary code on grid nodes and check
that the full operation commutes with x and z vector reflections during startup,
loading, stopping and hold. They also detect the old pin/roller asymmetry. The
solver tests separately check initial-state equivalence and the balance between
all boundary impulses and specimen momentum. These construction checks do not
replace a completed numerical run or establish mesh convergence. Constitutive
damage remains local and unregularized; D = 1 does not imply a resolved crack or
a separated fragment.

## Recorded run

The [completed run](results/material-engineering--concrete-bending-symmetric--hjc/interpretation.json) records 121,104 material points, 10,500 steps and 61 saved states over 1.5 ms. Actual bottom-midspan deflection reaches 16.3894 mm; 20.1843% of the reference volume reaches D ≥ 0.5. Minimum det(F) is 0.985444. The [saved-frame symmetry audit](results/material-engineering--concrete-bending-symmetric--hjc/runtime-symmetry.json) records maximum damage mirror RMS differences of 4.354e-13 in x and 4.144e-13 in z, and a maximum left/right damaged-volume fraction difference of 5.551e-17; all are below the fixed 0.01 limits.

This is a separate boundary-value problem with imposed symmetry planes and two vertical-only supports. It does not establish symmetry of the original pin/roller problem, mesh convergence, or the absence of unconstrained instability. Constitutive damage retains grid participation and does not define resolved cracks or detached fragments.
