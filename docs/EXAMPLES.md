# Example guide

All spatial suite cases are 3D and use SI units. Coordinates below are in a
unit cube. Boundary conditions act on grid nodes; wall/contact location has
grid-scale error. The free surface has zero applied traction.

| Case | Geometry and load | Boundary conditions | Main observable |
|---|---|---|---|
| `wave` | Rod x=0.125–0.875; initial sinusoidal axial velocity | Fixed axial endpoints; transverse rollers | Velocity/displacement error against the longitudinal eigenmode |
| `cantilever` | Beam x=0.15–0.8; quadratic transverse initial velocity | Clamp x≤0.2 | Free-end displacement |
| `torsion` | Same beam; initial twist velocity about x | Clamp x≤0.2 | Free-end motion and energy |
| `shear` | Block y=0.25–0.75; upper region moves at 0.05 m/s | Lower region fixed; upper velocity prescribed | Upper-half shear displacement |
| `compression` | Cube 0.35–0.65; upper platen moves downward | Rough fixed lower platen; prescribed upper velocity | Displacement, constitutive work, plastic strain, damage |
| `surface-load` | Panel y=0.45–0.55; 20 Pa pressure with 0.05 s cosine ramp | Four edges clamped | Deflection, load work, net boundary impulse |
| `head-on` | Two elastic blocks; opposite axial velocities | Free exterior; frictionless normal contact | Contact impulse and total momentum |
| `oblique` | Two blocks with axial and tangential initial velocities | Free exterior; Coulomb coefficient 0.3 | Contact impulse and tangential response |
| `sliding` | Block initially on floor with horizontal velocity | Floor y=0.125; Coulomb coefficient 0.3 | Mean horizontal speed and support impulse |
| `rebound` | Elastic block falls under gravity | Frictionless floor y=0.125 | Lowest particle height and energy |
| `dam-break` | Water column x=0.125–0.4, y=0.125–0.65 | Six slip tank walls at 0.125/0.875 | Front position and free-surface height |
| `sloshing` | Shallow layer with initial horizontal velocity | Same tank | Surface height, momentum, energy |
| `fsi` | Water column encounters an elastic obstacle | Tank walls; obstacle anchored below y=0.18 | Interface impulse and obstacle-top displacement |
| `water-entry` | Elastic block drops onto a liquid layer | Tank walls; free block | Interface impulse and water-surface response |
| `elastic-gate` | Water column against a thick elastic panel | Tank walls; panel clamped above y=0.65 | Lower-tip displacement and fluid front |
| `floating-block` | Density ratio 0.5; initially 25% submerged block | Slip tank walls; free elastic body | Heave, vertical speed, surface height |
| `rubber-ball` | Filled elastic sphere, radius 0.15 m, under gravity | Frictionless floor y=0.125 | Centre height and rebound |
| `ball-beam` | Elastic sphere falls onto an elastic beam | Both beam ends clamped; two-body normal contact | Sphere height, beam displacement, contact impulse |
| `large-bending` | Cantilever with 0.6 m/s initial tip velocity | Clamp x≤0.2 | Large transverse displacement |
| `large-twist` | Cantilever with amplified initial twist velocity | Clamp x≤0.2 | End rotation and energy |
| `perforated-panel` | Nine-hole elastic panel; 3 kPa, 0.08 s smooth pressure pulse | Four edges clamped; reference-face dead traction | Central displacement and applied work |
| `cellular-cushion` | Falling flat mass; two skins joined by sixteen elastic columns | Cushion base clamped; two-body friction coefficient 0.2 | Mass deceleration, cushion displacement, contact impulse |
| `baffled-tank` | Three perforated elastic baffles and a hydrostatic liquid layer | Slip tank walls; baffle bases clamped; 6 m/s² harmonic base acceleration | Surface elevation and baffle displacement |
| `unbaffled-tank` | Same tank, liquid height and acceleration without baffles | Slip tank walls | Surface elevation and liquid centre of mass |

Tank tops are included as protective walls. `elastic-gate`, `floating-block`,
`baffled-tank` and `unbaffled-tank` use a hydrostatic pressure initialization;
the other fluid cases start
unstressed, so their transient includes settling.
The bulk modulus is 0.2 MPa: these are compressible-flow demonstrations, not
calibrated water experiments. FSI uses separate grids and nonadhesive contact,
not a viscous no-slip interface.

## Resolution and duration

```bash
python -m examples.suite --case wave --grid 32 --no-render --output outputs/grid32
python -m examples.suite --case wave --grid 48 --no-render --output outputs/grid48
python -m examples.suite --case wave --grid 64 --no-render --output outputs/grid64
python -m examples.suite --case wave --grid 48 --dt-factor 0.5 --no-render --output outputs/half_dt
```

The initial acoustic time step is based on 0.15 dx/c and rounded down so that
all output intervals have equal step counts. Fluid/solid pairs share a time
step; the obstacle's wave speed remains within the explicit stability margin.
`--dt-factor` can reduce this step. Nonlinear tangent wave speeds can differ
from the initial estimate.

Default durations: 0.15 s for beam/shear/panel cases, 0.24 s for block collision,
0.25 s for sliding, 0.36 s for rebound, and 0.35 s for fluids/FSI. Compression
uses one initial acoustic transit time through the specimen; this is a dynamic
compression test, not a quasi-static stress–strain identification test.

## Materials and output

```bash
python -m examples.suite --case cantilever --model all-elastic
python -m examples.suite --case compression --model all-plastic
python -m examples.material_paths --model rht --path all
```

`all-plastic` includes mixed and isotropic bilinear hardening separately, fixed-
temperature plasticity, Steinberg–Guinan, JC, JH-2, HJC, and RHT. Material-point
paths are homogeneous constitutive checks and have no spatial grid. Their
stress/strain CSVs complement the larger spatial examples.

`summary.json` records all material parameters, grid settings, BCs, physical
duration, particle count, device, and solver time. `end: null` in a velocity BC
means no deactivation time. `diagnostics.csv` is per body; `observables.csv`
contains scenario metrics and the net boundary impulse inferred from the
global momentum balance. The latter includes all supports together.

`strain_energy` is available for elastic/volumetric laws. `constitutive_work`
is accumulated stress work for incremental materials, not recoverable elastic
energy. Applied-force work uses trapezoidal particle velocities. APIC affine
kinetic energy and a complete contact/boundary work budget are not included.
Do not interpret these columns as an exact discrete energy balance.

The RHT return map is substantially slower than the simpler models on CPU.
Use the material-point command for constitutive investigations before launching
a large spatial run.

## Public example adaptations

The [SPHinXsys elastic gate](https://www.sphinxsys.org/html/examples/example2_elasticGate.html)
and [LS-DYNA gate](https://lsdyna.ansys.com/dam5/) motivate `elastic-gate`.
This implementation extrudes the geometry into 3D and uses a compressible liquid,
a Neo-Hookean solid and grid contact. Dimensions and parameters differ from the
published benchmark; agreement with its experimental displacement has not been tested.

`floating-block` follows the free-body/tank setting in the
[DualSPHysics test catalogue](https://github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases).
It starts from a heave imbalance in still water, with no wavemaker. The block
is deformable and its density ratio is 0.5.

`rubber-ball` and `ball-beam` exercise gravity, large deformation and contact.
The [LS-DYNA ball example](https://www.dynaexamples.com/introduction/intro-by-j.-day/misc/ball)
uses a pressurized shell; the present example uses a filled elastic sphere.
`large-bending` and `large-twist` use the finite-strain loading ideas in
[JAX-FEM hyperelasticity](https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html),
with dynamic release and beam geometry instead of its static cube problem.
These adaptations are independent implementations, not imported solver decks.

## Dense runs and recovery

```bash
python -m examples.suite --case elastic-gate --grid 48 --ppc 3 --frames 40 --no-render --resume --output outputs/benchmarks
python -m examples.suite --case water-entry --grid 72 --ppc 3 --frames 60 --no-render --resume --output outputs/production
```

The second configuration contains 1,327,400 particles and advances 2,400 steps
over 0.35 s. Check the [run catalogue](RESULTS.md) for execution status and measured
results. Configuration size alone does not certify a completed run.

`frames_data/` stores every output frame. Two alternating checkpoint generations
store the full state, including constitutive history and accumulated impulses.
An atomic `checkpoint.json` selects the last complete generation. `--resume`
requires matching geometry, material, boundary and integration settings. Resume
with the same solver revision: the configuration signature does not hash source
files. A checkpoint is not an accuracy or convergence certificate.

Rendering can be performed separately after completion; see [RENDERING.md](RENDERING.md).

## Engineering configurations

```bash
python -m examples.suite --case perforated-panel --grid 48 --ppc 3 --frames 50 --no-render --resume --output outputs/engineering
python -m examples.suite --case cellular-cushion --grid 48 --ppc 3 --frames 50 --no-render --resume --output outputs/engineering
python -m examples.suite --case baffled-tank --grid 48 --ppc 3 --frames 50 --no-render --resume --output outputs/engineering
python -m examples.suite --case unbaffled-tank --grid 48 --ppc 3 --frames 50 --no-render --resume --output outputs/engineering
```

The perforated panel spans 0.68 × 0.68 m, has thickness 0.08 m and nine
0.05 m radius through-holes. Its 0.35 s response includes both the pressure
pulse and free vibration. The nominal traction acts in the fixed negative-y
direction on the reference upper face; it is not a follower pressure.

The cushion has sixteen 0.04 m radius columns connecting two skins. A separate
elastic platen falls through a 0.18 m initial gap under gravity. The 0.34 s
run records platen velocity and pad deformation. It models elastic cushioning;
there is no strut self-contact or material fracture model in this example.

The tank cases run for 0.7 s under 1.5 Hz horizontal base acceleration with a
0.1 s smooth ramp. Computation uses the translating tank frame, applying the
inertial force `-mass * acceleration` to both fluid and structure. The three
baffles are 0.05 m thick and have rectangular openings. The comparison keeps
tank dimensions and liquid height fixed; the baffles displace some liquid,
so the two cases have different liquid masses. Wall pressure and response
are grid-resolution dependent.

These configurations exercise engineering geometry, loading and observables.
Their demonstration parameters are not calibrated component designs or certified
load ratings. Full dimensions, load definitions and material values are written
into each run's `summary.json`.
