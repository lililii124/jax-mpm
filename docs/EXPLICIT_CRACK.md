# Explicit crack transfer

<img src="images/explicit-crack-mode1.gif" alt="Computed opening and propagation of a straight edge crack" width="400">

The JAX implementation in [crack_mpm.py](../mpm_jax/crack_mpm.py) separates
particle-to-node transfers on opposite sides of a geometric crack. It uses
the split-field and explicit-path ideas described by
[mpm-fracture](https://github.com/titoadibaskoro/mpm-fracture). This is an
independent implementation; source code was not copied from that repository.

The current branch supports one straight 2D crack under symmetric mode-I
loading. A link crossing the finite crack enters a separate nodal velocity
field. Opening faces exchange no momentum. Optional frictionless nodal contact
removes closing normal relative velocity with equal and opposite impulses;
it preserves tangential slip. Its proximity resolution is the grid spacing,
not a resolved surface-gap constraint.

When the growing tip passes every material point in its forward direction,
the crack has no resolved ligament ahead. Its visibility ray then continues
through exterior support nodes. Without this free-surface treatment, exterior
nodes spuriously reconnect the pieces even after the tip has crossed the body.
The physical tip and the exterior visibility extension are stored separately.

## Opening example

The plate measures 40 by 20 mm and uses plane-strain linear elasticity:
Young's modulus 3 GPa, Poisson's ratio 0.2 and density 1200 kg/m³. A 12 mm
edge crack starts at mid-height. The upper and lower grips reach opposing
2 m/s velocities with a 20 µs half-cosine ramp. The trajectory lasts 200 µs.

The tip samples normal tensile stress in a forward half-disc of radius 1.5 mm,
weighted by reference particle volume and `(1 - r²/R²)²`. Above the assumed
12 MPa strength it advances at `500 * max(1 - strength / traction, 0)` m/s,
capped at one quarter cell per time step. The radius remains fixed between
meshes. These strength and kinetic parameters are illustrative modelling
assumptions, not experimentally fitted properties or a fracture-energy law.

```bash
python examples/explicit_crack_tension.py --grid 128 --output outputs/explicit-crack
pvpython --force-offscreen-rendering tools/render_explicit_crack_paraview.py --input outputs/explicit-crack --output outputs/explicit-crack.gif
```

The renderer requires Pillow in ParaView's Python environment. It displays
particle rectangles mapped by the computed deformation gradient, at the
computed particle positions. The GIF has 41 distinct 400×400 frames, a fixed
0–60 MPa normal-stress range and unit displacement scale. There are no drawn
crack lines or deleted material points.

| Grid spacing | Crack extension | Final crack-mouth opening |
| --- | --- | --- |
| 0.625 mm | 27.844 mm | 0.7843 mm |
| 0.46875 mm | 27.888 mm | 0.7944 mm |

Both runs reach separation by the saved 100 µs state. Final openings differ
by 1.26%. A matching 80 µs loading control with an unreachable strength
threshold retains the initial crack length. Six focused tests cover finite
tip visibility, intact transfer away from cracks, opening velocity separation,
contact conservation, tensile triggering and exterior-node reconnection.
The [run evidence](../evidence/explicit-crack/mode1.json) records parameters,
source hashes and the two-grid comparison.

## Remaining validation

This demonstrates explicit separation; it does not yet establish quantitative
fracture accuracy. Peak sampled local strain reaches 4.9%, so the linear
constitutive approximation also needs checking. Two resolutions alone do not
establish convergence. The strength law has no prescribed fracture energy or
verified dissipation balance. Crack-centreline advection, stress-selected
turning, multiple cracks and three-dimensional crack surfaces are not yet
implemented. Those extensions remain active work; this example does not
replace the existing AT2 cases.
