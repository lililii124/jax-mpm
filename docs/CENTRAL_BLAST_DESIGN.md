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

A further bounded search starts from the saved best candidate using
`examples/sandwich_central_continue.json`. Copy the published
`evidence/blast-sandwich-reduced/central-continuation-seed.npz` to the
configuration's `inputs/continuation-controls.npz` alongside the other
pinned inputs. This starts a fresh SQP solve from that material field; it is
not a restoration of the previous solver's internal state. The configuration's
legacy `initial_material` label says uniform; the actual start is identified
by the hash in `optimization-inputs.json` and `continuation-start` outputs.
An earlier continuation stopped because its finite-difference probe was
mistakenly centred on the uniform material rather than the continuation point.
The probe now uses the same controls as the analytic gradient.

For the saved two-iteration candidate, a separate full-MPM calculation starts
from rest under the same central pressure and fixed mass:

```sh
python -m tools.run_sandwich_fullmpm examples/sandwich_central_candidate_fullmpm.json
```

This reproduces the 10 mm in-plane, 1 mm through-thickness comparison setup.
Its output must be compared with the matching uniform-material central-load
baseline; running the command does not itself establish an improvement or
mesh convergence.

For the later six-iteration continuation shown in the current gallery, use
`examples/sandwich_central_v7_fullmpm.json`. It binds a different saved control
field and therefore requires its own independent full-MPM trajectory.

After the candidate finishes, compare the complete histories' peak metrics:

```sh
python -m tools.compare_central_candidate_fullmpm evidence/blast-sandwich-reduced/union130-completed-initial-responses.json outputs/sandwich-central-candidate-fullmpm --output outputs/central-candidate-comparison.json
```

The comparison checks completion, load case, duration, physical settings,
source hashes, candidate controls and fixed mass before reporting a change.

This is an effective elastic-core design problem. It does not include an
explicit explosive, skin plasticity or core crushing. The frozen basis is
reused for candidate exploration; its earlier multi-load validation did not
pass every error criterion. A smaller reduced-model objective is not an
independently verified improvement. Retained designs need full-MPM comparison
with the uniform baseline before a new result gallery is published.
