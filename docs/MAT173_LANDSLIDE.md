# MAT173-inspired strength-reduced slope response

![Strength-reduced slope response](images/home/mat173-strength-reduced-slope-response.gif)

This 3D calculation shows the computed response of a finite-width slope after
an explicitly recorded reduction in the selected Mohr–Coulomb strength
parameters. The figure colors the connected exterior by accumulated equivalent
plastic strain. It is an illustrative numerical experiment; it is not a
complete landslide, a real-site prediction, or MAT173 validation.

## Model and loading

The soil body has 37,440 material points at 0.05 m spacing and a 33° face over
a 2 m run. A separate 2,232-point elastic foundation is anchored at grid nodes
below y=0.30 m. The two bodies exchange nonadhesive normal and Coulomb contact
impulses with interface coefficient 0.10. Lateral walls are frictionless.
Gravity ramps over 0.02 s. Both bodies start from zero stress; this is not a
geostatic initialization.

The initial self-weight state was numerically settled for 1,500 steps using a
0.995 per-step multiplier on particle velocity and affine C. The damping was
removed before the strength change. Three consecutive 100-step windows met the
recorded speed, center-of-mass drift, and reaction/support checks; the last
window had soil p95 speed 0.00327 m/s and inferred vertical boundary reaction
to soil weight 1.002. This is a settled transient state under the selected
numerical procedure, not an independently measured in-situ stress field.

At step 1,500 the material parameters changed uniformly from friction angle
38° and cohesion 10 kPa to 28° and 1 kPa. No velocity or displacement was
prescribed as the trigger. Before the first post-trigger update, 88.1% of soil
points lay outside the new yield surface; the first constitutive/contact step
remained valid. The calculation continued to step 3,101 (0.232575 s), where it
stopped at a 5% accumulated-plastic-strain limit. The maximum absolute
logarithmic strain was 4.44%.

## What the result shows

The final soil center of mass moved 1.49 mm horizontally and 6.86 mm downward
relative to the foundation. Its 4.99 mm downslope projection contains 1.25 mm
from horizontal motion and 3.74 mm from vertical settlement. The 2,200-point
basal band averaged 2.03 mm relative horizontal motion and 1.72 mm relative
motion projected down the face. These values show a small basal slip response,
but not a developed slide or runout. Plastic increments above 1% occur over
63.2% of soil points; the lower two layers also show broad plasticity rather
than a narrow shear band.

The 51-frame animation spans steps 1,501–3,101 and 0.112575–0.232575 s. Surface
cells are formed only from eight adjacent original material points on the
reference lattice; exterior faces are extracted from those connected
hexahedra. Partial cells are omitted, leaving the displayed boundary inset by
up to one particle spacing. Surface coordinates and plastic strain values are
copied from their original particles; scalar interpolation is performed by
ParaView over the connected surface without smoothing or altering the field.
Frames are rendered at 1,600 × 1,600 with FXAA and reduced to 400 × 400 using
Lanczos filtering. The fixed color range is 0–0.05 accumulated plastic strain.
The image contains no labels, coordinates, or displacement scaling.

## Constitutive scope and source status

`mpm_jax/materials/mat173.py` implements the selected constant-property,
no-joint branch described in the delivered MAT173 Fortran source: six linear
principal-stress Mohr–Coulomb inequalities, nonassociated flow, and one- to
three-surface active-set returns, plus the selected NOVOID=1 hydrostatic-tension
special case. The MPM host supplies a corotated logarithmic strain increment,
but this adapter performs a small-strain elastic return for each increment.
Finite-deformation host spin and native history semantics are not established.

The delivered source is `MAT173.f`, SHA-256
`6cfae0ae794a5f52163ed6be0e79b1a49b046943dc70d5e471390699c411689a`. Its
delivery manifest says `complex_final_check_status=NOT_VALIDATED` and
`delivery_complete=false`; native material activation and history mapping
remain pending. A single direct Fortran/JAX point comparison is recorded in
[`fortran-point-comparison.json`](results/material-engineering--mat173-strength-reduced-slope-response/fortran-point-comparison.json).
It covers one fixed-axis normal-strain symmetric two-generator edge return;
stress, plastic tensor, and accumulated plastic strain agree to near floating
point roundoff for that input. It does not establish general equivalence,
native solver acceptance, or the pending complex Final Check. Face-only,
apex, hydrostatic-tension, shear/rotation, joints, curves, native history
slots, and finite-deformation branches are not covered by this comparison.

The experiment has no pore pressure, water table, joints, erosion, progressive
weakening, or site calibration. The infinite-slope factor-of-safety estimate
in the run metadata is only a screening calculation and omits the finite
bench, foundation contact, and transient loading. Do not use this result as a
field prediction or describe it as a complete landslide simulation.

## Reproduce the numerical example

```bash
python -m examples.mat173_landslide_triggered --output outputs/mat173-strength-reduced
python -m examples.mat173_landslide_continue --parent outputs/mat173-strength-reduced --output outputs/mat173-strength-reduced-continuation
```

The checked-in GIF is a compact visualization artifact. The full particle
history, checkpoints, and intermediate frames are not included.
