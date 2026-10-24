# Local contact model selection

`mpm_jax.local_contact_models` selects `hertz`, `jkr`, `md` or
`self-consistent`. All updates return `(history, response)` with common
`normal_force`, `radius`, `area`, `potential`, `active` and `valid` fields.
Positive normal force denotes compression. The existing Hertz/JKR parameter,
history and response types and their update functions are preserved.

```python
from mpm_jax.local_contact_models import (
    local_contact_model, initial_local_contact_history, update_local_contact,
)
from mpm_jax.fibre_contact import normal_pair_forces

law = local_contact_model("md", effective_modulus=5.33e9,
                          effective_radius=117.5e-6, work_of_adhesion=0.1,
                          interaction_range=10e-9)
history = initial_local_contact_history(law)
history, response = update_local_contact(history, 20e-9, law)
pair = normal_pair_forces([0., 1., 0.], response)
```

`normal_pair_forces` applies equal opposite forces along the supplied normal.
For a centreline segment pair, the caller must still distribute them with
consistent position/force weights and reject unsupported geometry. A circular
equivalent radius is not the exact elliptical solution for crossing cylinders.

## History and changing curvature

MD requires an explicit finite interaction range and retains attachment history,
including its cohesive disk without intimate contact. See
[the MD equations and branch rules](MAUGIS_DUGDALE.md).

The self-consistent model requires `outer_radius` and a ring count. Its shared
compliance matrix is prepared on the host. The adapter stores per-contact
`active`, `valid`, `gap`, `indentation` and `previous_profile`; for a batch of
`N` contacts, the gap/profile arrays have shape `(N, rings)`.

All parameter types support `_replace(effective_radius=current_radius)`. For
self-consistent contact, changing radius updates the paraboloid profile while
leaving the fixed annular mesh/compliance unchanged. Before relaxation, the
adapter adds `new_profile - previous_profile` to the old gaps. This retains the
previous elastic displacement as the starting iterate. The current radius can
be an array; a compiled per-contact map preserves independent branch decisions.

An already detached contact with every undeformed ring gap strictly greater
than the interaction range has the exact solution `g = g0`, zero traction and
zero energy. The adapter selects this detached local minimum directly and
records `open_exact = true`, with zero relaxation sweeps. Active adhesive
histories always undergo relaxation. Equality at `g = h` is excluded from this
shortcut: the potential's one-sided slope permits attraction there. This
defines the detached approach branch without spending iterations on distant
contacts or silently dropping an attached metastable solution.

The shared self-consistent model keeps `E*`, `W` and interaction range fixed.
Changing any of them through `_replace` returns `valid = false`; construct a
new model and history to begin a different experiment. The radial domain does
not expand automatically. An active outer boundary or an unconverged residual
also rejects the update. Static keyword controls `max_sweeps` and `tolerance`
are passed to the [cohesive BEM solver](COHESIVE_BEM.md).

For Hertz/JKR, pass the batch shape explicitly to the history initializer, as
with the original API. MD and self-consistent histories additionally infer
batch dimensions from their parameters. Do not reuse a two-boolean Hertz/JKR
history as a self-consistent ring history.

## Standalone pull-off reference

```powershell
python -m examples.contact_pull_off --model all --output outputs/contact-reference
python -m examples.contact_pull_off --model self-consistent --rings 64 --output outputs/contact-reference-r64
```

Each selected model receives the same 121-state indentation path: −4 µm to
+2 µm and back to −4 µm. The example uses actual `lax.scan` histories, an
effective modulus of 1 MPa, radius 10 mm, `W = 0.0042441318 J/m²`, cohesive range
0.63662 µm, and a 0.4 mm BEM radial domain. These are illustrative parameters.
Hertz disables adhesion while keeping the geometry and modulus identical.

Each model writes `observables.csv`, `history.npz` and `summary.json`. Outputs
include force, contact and cohesive radii/areas, energy, attachment events and
validity. The NPZ retains every returned scalar response and, for the
self-consistent model, all ring gaps, pressures and displacements at every
saved state, plus its annular mesh and compliance. Source/file hashes and
runtime versions accompany the results.
The CSV work is saved-state trapezoidal force–displacement quadrature, not a
proof of energy conservation across snap events. Protocol fraction is a
quasistatic loading parameter, not physical dynamic time.

This example is a local model reference, not a deformable MPM specimen, and
does not generate a particle animation. The half-space compliance in MD/BEM
must not be added a second time to an already resolved MPM bulk contact.
Fibre use requires an explicit unresolved-interface interpretation; changing
curvature also introduces parameter-work terms absent from a normal-force-only
coupling. The reference results do not establish a calibrated fibre failure
envelope or experimental pull-off accuracy.
