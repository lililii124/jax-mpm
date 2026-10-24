# Triaxial fibre tension

This series studies a dry, explicitly interlaced 0°/60°/120° fibre network.
The in-plane tensile angle is measured relative to the reference family.
At 0° the load follows that family; at 30° it follows the bisector between
adjacent families. The relative directions of the three families remain fixed.

The initial angular series is 0°, 5°, 10°, 15°, 20°, 25° and 30°. The geometry
audit must establish the relevant rotation and thickness-reflected symmetry
before treating this as a reduced orientation interval.

## Initial parameters

These are adjustable demonstration assumptions, not measured fibre properties.

| Parameter | Value |
| --- | ---: |
| Fibre Young's modulus | 10 GPa |
| Poisson ratio | 0.25 |
| Density | 1500 kg/m³ |
| Nominal tensile strength | 100 MPa |
| Fracture energy | 100 J/m² |
| Complete-separation opening | 2 µm |
| Crack-band characteristic length | Background-grid spacing |
| Work of adhesion in JKR mode | 0.1 J/m² |
| Coupon length × width | 24 × 16 mm |
| End-grip length | 3 mm per end |
| Pitch | 1 mm |
| Crimp amplitude | 0.14 mm |
| Initial tightly packed series diameter | 0.235 mm |
| Required initial surface clearance | 2 µm |

The geometry module also retains its more separated 0.20 mm diameter default.
The tight series must pass the same continuous-curve exclusion audit; changing
particle sampling does not change physical packing or fracture energy.

The free gauge spans 18 fibre pitches and the width spans 16 pitches. These
length-to-pitch ratios are not counts of primitive translation cells; the legacy
metadata fields `gauge_periods` and `width_periods` store these ratios. Smaller
and larger coupons must be compared at fixed fibre dimensions and packing to
measure boundary and finite-specimen effects. A large particle count alone is
not evidence of size independence.

## Material and contact

Each fibre uses an isotropic elastic-brittle material with independent damage
history and equivalent-opening softening. The crack-band length is independent
of particles per cell. Details and the admissible length condition are in
[Isotropic brittle fibres](ISOTROPIC_BRITTLE.md).

Separate grids permit fibres to move and slide independently. A shared velocity
field would impose bonding and is unsuitable for this series. Hertz and JKR
contact are selectable. Any circular equivalent used for crossed-fibre contact
must be recorded separately from an exact elliptical-contact solution.
The current curvature calculation treats the local tangents as straight
cylinders and keeps the reference radii. It omits axial curvature of the crimped
tubes and cross-section deformation; contact-model sensitivity is required
before interpreting differences as quantitative adhesive-fibre strength.
Contact locations use polylines through section mass centroids. Curved-tube
volume weighting displaces these from geometric centreline points, so their
proxy gap also requires a separate geometry-sensitivity check.

The present material loses stiffness after complete damage. Resolved crack
faces, fragment separation and fragment recontact require an explicit additional
strategy; a particle gap alone is not evidence that these have been resolved.

## Measurements

For every angle, retain the macroscopic stress–strain curve, first material-point
failure, peak tensile load, failed volume and remaining load-bearing paths.
The initial envelope uses peak nominal tensile stress and its associated gauge
strain; first failure and subsequent loss of the load-bearing connection remain
separate events.

Checks include single-fibre softening work, independent-field motion, contact
force balance, boundary reactions, mass and momentum, positive deformation
Jacobians, grid and particle sensitivity, and specimen-size sensitivity.
Only reviewed results receive ParaView animations and public gallery entries.
The component implementations do not imply that an angular failure envelope
has already been computed.

Cross-job continuation retains the original reference geometry, full particle
history, contact history and accumulated fixture quantities. Planned stops do
not shorten the physical target. A separate final job reviews the completed
history and renders it with ParaView.

The small continuation comparison uses a fixed relative tolerance of 2e-10
on quantities grouped by physical units. Position errors use grid spacing;
near-identity stress and driving-strain comparisons also include binary64
roundoff budgets scaled by elastic stiffness and stretch. These are engineering
budgets, not certified eigensolver error bounds. Discrete states, event times
and static labels remain exact. Both sequences must independently pass record
and full saved-history reviews. This small test remains undamaged with inactive
contact; it does not establish active JKR snap or crack continuation accuracy.

Nominal tensile stress uses the rectangular reference envelope area: specimen
width multiplied by twice the crimp amplitude plus the fibre diameter. This
includes voids and differs from a sum of fibre cross-sections. The record audit
reports this area, the per-step dynamic force maximum, prescribed strain at
that instant, and first-failure and sampled connectivity observations separately:

```powershell
python tools/audit_triaxial_records.py outputs/triaxial-checks/smoke/theta-00/hertz
```

This audit checks record identity, time, reactions and impulses, mass, momentum,
energy decomposition and fixture errors. It does not reconstruct the full
particle history or replace convergence studies. The completed Hertz smoke
check has a kinetic-to-elastic energy ratio about 0.65 at its largest saved
interval-average tensile force; it is a dynamics integration check, not a
quasi-static strength measurement.

The optional `--grid-storage tube` retains the same reference particles,
quadratic stencils, grid spacing, material and loading as `dense`. It stores
only nodes in a padded tube around each fibre. Missing stencil support causes
an explicit invalid state rather than dropping its mass or force. `plan.json`
retains the dense allocation upper bound; `layout.json` records the actual
active-node count and mapping identity. Solver speed and memory require fresh
same-source probes and are not inferred from the node count alone.
