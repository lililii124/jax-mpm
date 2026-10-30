# Material distribution under central pressure

This candidate search minimizes the peak back-face deflection bound under a
single central pressure pulse, keeping the sandwich mass fixed at 1.8468 kg.
It starts from a uniform core and uses reflection-symmetric controls in both
in-plane directions. It replaces the previous multi-load gallery task.

The plate is 300 × 300 mm, with two 2 mm skins, a 20 mm effective core and
15 mm clamped perimeter strips. The Gaussian pressure footprint is centred
at (150, 150) mm, with a 60 mm standard deviation in each direction. Its
area average is normalized to one. The pressure multiplier follows a
sine-squared pulse of 0.5 MPa amplitude and 0.2 ms duration; the response
window is 2 ms. The local central pressure therefore exceeds the multiplier.

The material map retains the 15 mm physical filter, core solid-fraction bounds
0.08–0.35, and a 200 MPa skin-stress bound. JAX differentiates the reduced
transient response and material-dependent mass and stiffness operators.
Only the central load contributes to the objective and stress constraint.

Run the bounded candidate search with:

```sh
python -m tools.optimize_sandwich_central_rational examples/sandwich_central_fixedmass_sqp.json
```

The configuration pins the rank-191 freeze and basis hashes. Its relative
`inputs/` paths must contain that freeze, basis and the original selected-vertex
control, material and tangent artifacts. Those selected-vertex inputs verify
the frozen operators; the optimization itself starts from uniform material.
The entry point checks a central-load directional derivative and fixed mass
before starting the two-iteration SQP budget. The output directory must be new.

This is an effective elastic-core design problem. It does not include an
explicit explosive, skin plasticity or core crushing. The frozen basis is
reused for candidate exploration; its earlier multi-load validation did not
pass every error criterion. A smaller reduced-model objective is not an
independently verified improvement. Retained designs need full-MPM comparison
with the uniform baseline before a new result gallery is published.
