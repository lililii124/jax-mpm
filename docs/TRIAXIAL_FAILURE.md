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

The 18-period free gauge and 16-period width are starting dimensions. Smaller
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
