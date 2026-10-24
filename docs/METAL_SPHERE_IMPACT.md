# Deformable metal sphere and plate

This separate numerical prototype uses the [compressible two-body solver](IMPACT_MPM.md).
Both bodies deform and use the same illustrative aluminium-like EOS and strength
parameters. There is no rigid sphere, backing plane, damage law or phase-mixture
model. The four 1 mm plate-edge strips constrain grid velocity in all three
components; corner nodes belong to the same constraint union.

The fixed request has a 2 mm sphere, a 20 × 20 × 0.5 mm plate, a 0.3 mm initial
surface gap and a sphere velocity of 5000 m/s toward the plate. Its target is
1 microsecond with 60 output intervals. Cartesian spacings are 0.1 × 0.05 × 0.1 mm;
sphere cut cells retain their sampled subcell volume and centroid. This gives
409,504 material points and two independently packed rectangular grids containing
3,952,985 nodes at 0.1 mm spacing. These are a declared allocation and requested
protocol, **not a completed or validated impact result**.

The static grid envelopes include the initial geometry and nominal free-flight
path. They are not a certified propagation bound. Invalid support, contact,
constitutive response or EOS state stops advancement. A failure preserves the
last complete accepted checkpoint and the rejected state; it does not shorten
the target time, remove mass or clip energy. Normal contact intersecting a fixed
normal degree of freedom is unsupported and rejected by the current solver.

## Records and continuation

`observables.csv` separates translational, APIC affine and total internal energy,
the boundary reservoir, nonnegative projection losses and signed coupling work.
It records each body's momentum and energy. `derived_contact_impulse` is inferred
from body momentum change minus the actual plate boundary impulse; it is not an
independent measurement of the node contact impulse or interface power.

Every checkpoint retains all `ImpactState` fields, including plastic strain,
cold work, entropy history, cumulative phase/tension flags and temperature/volume
extrema. Temperatures exported to VTK are actual EOS temperatures. Zero damage
and failed fields identify the absence of a failure model; they do not predict
that the material remains intact. `family_id` stores the real sphere/plate labels
even though the visualization archive holds one combined solver state.

Resumption requires identical source, runtime, physical plan, reference arrays,
static labels, saved clocks and trajectory-prefix hashes. The full 61-state
allocation remains unchanged by a planned stop. Partial results are marked
`checkpoint-ready`; only the final requested time sets `completed=true`.
Completion alone leaves `physical_validation_passed=false` and does not authorize
publication as a validated physical result.

```powershell
python -m examples.metal_sphere_impact --dry-run
python -m examples.metal_sphere_impact --stop-after-frame 2 --output outputs/metal-sphere-prefix
python -m examples.metal_sphere_impact --resume --output outputs/metal-sphere-prefix
```

The full-scale commands require separate numerical and resource admission; this
driver does not dispatch a job or render automatically. The independent small
control is intentionally low speed and tests transfer, saving and continuation:

```powershell
python -m examples.metal_sphere_impact --tiny-control --output outputs/metal-sphere-control
```

That control has 352 particles and remains separated. It does not validate shock
formation, impact response, fracture or a high-speed sphere/plate calculation.
Compatible global energy accounting does not establish local entropy stability;
the signed corrections and cumulative negative entropy increments must remain
visible in any later numerical assessment.
