# Material engineering examples

These 3D cases use physical dimensions in SI units and prescribed mechanical
boundaries. They cover metal tension, concrete bending and ceramic indentation.
Their specimen dimensions and loading histories are demonstration configurations.

[Pressure-pulse panels](PRESSURE_PULSE_PANELS.md) add four-edge-clamped JC,
HJC, JH-2 and RHT plates under prescribed follower pressure.

## Run

```bash
python -m examples.material_engineering --case jc-tension --grid 80 --ppc 3 --frames 60 --no-render --resume
python -m examples.material_engineering --case concrete-bending --model hjc --grid 64 --ppc 3 --frames 40 --no-render --resume
python -m examples.material_engineering --case ceramic-indentation --model jh2 --grid 64 --ppc 3 --frames 40 --no-render --resume
```

Results go to `outputs/material-engineering/<case>/<model>`. `--ppc` is the
initial sampling count per cell per axis. The time step is derived from
`0.15 * dx / c` and rounded down to equal output intervals; increasing `--frames`
changes output sampling and this rounding. Gravity is omitted over these short
dynamic loading intervals. Unconstrained exterior faces have zero applied
traction. Boundary geometry is resolved on grid nodes.

`summary.json` records the geometry, material values and loading assumptions.
`observables.csv` contains the quantities described below. Saved frames can be
rendered with [ParaView](RENDERING.md); omit `--no-render` to render during the run.
After completion, `python tools/review_material_engineering.py` checks the saved
states and observables and writes each run's `interpretation.json`.

## JC smooth round-bar tension

The bar lies along x, has a nominal diameter of 15 mm and a total length of
60 mm. Its two 10 mm end regions are gripped, leaving a 40 mm gauge length.
The computational domain is a 100 mm cube. Each grip prescribes only axial
velocity, allowing transverse contraction.

The grips move apart with a 100 µs half-cosine speed ramp followed by constant
speed, reaching 18 mm total extension at 300 µs. Each plateau speed is
36 m/s, giving a nominal gauge strain rate of 1,800 s⁻¹ and a final
engineering strain of 0.45. These are dynamic loading conditions; local strain
rates change during wave propagation and necking.

A smooth 0.5% radial imperfection favours localization near the centre:

```text
R(x) = R0 * [1 - 0.005 exp(-((x - xc) / (Lg / 6))²)]
```

Here `R0 = 7.5 mm`, `xc = 50 mm` and `Lg = 40 mm`. The initial particle
volumes and masses scale with the squared radial profile. There is no cut,
notch or prescribed initial damage. Dynamic localization can produce several
separated groups; the imperfection does not enforce a single neck.

### Material values and sources

The original AA2024-T351 JC constants are reproduced from D. R. Lesuer's
[FAA report, Table 6, printed p. 17 (PDF p. 29)](https://ntlrepository.blob.core.windows.net/lib/17000/17400/17488/PB2001101864.pdf).

| Parameter | Value |
|---|---:|
| A, B | 265 MPa, 426 MPa |
| n, C, m | 0.34, 0.015, 1.0 |
| D1, D2, D3 | 0.13, 0.13, −1.5 |
| D4, D5 | 0.011, 0.0 |
| Reference strain rate | 1 s⁻¹ |
| Reference temperature | 298 K |

The report's equations (1)–(5), printed p. 2 (PDF p. 14), define the strength,
temperature normalization and accumulated failure law. Its printed p. 18
(PDF p. 30) identifies the original fit as six torsion tests spanning
0.088–123 s⁻¹. The present tensile rate exceeds that range. Its Table 8 gives
a different 2024-T3 fit and is not used here.

[Kaiser Aluminum's 2024 rod and bar data sheet](https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1022/Kaiser_Aluminum_2024_Rod_and_Bar.pdf)
provides typical `E = 73.1 GPa` for T4/T351 on p. 1, and `ρ = 2770 kg/m³`,
`cp = 875 J/(kg K)` and a 502–638 °C melting interval on p. 2. The example
uses the 502 °C solidus, `Tm = 775.15 K`. The quoted heat capacity is measured
at 100 °C; holding it constant is an approximation. Poisson's ratio `ν = 0.33`
and the plastic-work heat fraction `β = 0.9` are assumptions. These choices
and the published JC constants do not establish a calibration for this
specimen's necking or fracture response.

The failure law uses tensile mean stress as positive:

```text
η = tr(σ) / (3 q)
εf = [D1 + D2 exp(D3 η)] [1 + D4 ln(max(ε̇p / ε̇0, 1))] [1 + D5 T*]
D(n+1) = min(1, D(n) + Δεp / εf)
```

The implementation holds the rate factor at its reference value below `ε̇0`.
Uniaxial tension has `η = 1/3` and this convention uses `D3 = −1.5`.
For comparison, the [Abaqus JC dynamic-failure documentation](https://abaqus-docs.mit.edu/2017/English/SIMACAEMATRefMap/simamat-c-johnsoncook.htm)
uses compressive pressure `p = −tr(σ)/3` and reverses the sign of its third
failure coefficient. A pressure-based coefficient must be converted before
use with this implementation.

### Failure and observables

The solver's optional `ballistic` failure mode is selected for this example.
The JC threshold clears the particle stress and affine velocity. From the next
step, failed points cease participating in grid transfers; their mass remains
in freely moving debris. This is a numerical
removal of load-carrying capacity. It includes neither crack-face contact nor
fracture-energy regularization, so the localization width and post-peak
response depend on resolution.

`mean_grip_tensile_force_N` is half the difference between the axial forces on
the left and right grips; `nominal_grip_stress_Pa` divides this load by `π R0²`.
Grip reactions are sampled from the next integration step's grid velocity
correction, at `boundary_sample_time_s`. After separation, the remaining
halves can vibrate against their grips: a nonzero grip reaction does not
measure force transmitted across the broken section.

`central_material_section_force_N` estimates the axial force across the
initially central material section from its current Cauchy stress:

```text
Fx(section) = Σactive e_x · σ · cofactor(F) · e_X * V0 / ΔX
```

`ΔX` is the initial axial layer spacing, `V0` the particle reference volume,
and `cofactor(F) = det(F) F⁻ᵀ`. Two central layers are averaged when the centre
falls between them. This stress-based quantity is evaluated at the saved state
time; grip reactions use the following step. The minimum absolute gauge-section
force is recorded separately. Inertia permits the force to vary along the bar.

Additional observables include plastic strain, damage, failed mass, total mass
error and the smallest load-bearing section area ratio. The latter uses
`det(F) / |F e_x|` for active material points grouped by their initial axial
section. `failed_gauge_sections` counts sections with no active points.
`load_bearing_axial_gap_m` is the largest gap between sorted active x coordinates.
`support_separated_components` counts groups separated by gaps greater than
`3 dx`; `gap_exceeds_grid_support` reports whether there is more than one such
group. This conservative axial test excludes shared quadratic support across
the detected gaps. It does not reconstruct three-dimensional fragment topology
or establish convergence to a single fracture location.

### Recorded response

The completed 80³-grid run uses 146,880 particles and 10,020 steps over 0.3 ms,
with 61 saved states. Minimum `det(F)` is 0.992783 and recorded mass is constant.
The first saved material failure occurs at 150 µs. At the end, six central
reference layers contain no active particles, and two load-bearing groups are
separated by a 14.8366 mm axial gap, exceeding the 3.75 mm grid-support threshold.
Failed material retains 1.26343 g, or 4.30127% of the total mass, as ballistic debris.

The central material-section force reaches 78.3999 kN at the 115 µs saved state
and is zero at the end. The final mean grip force is still 50.2499 kN because
the separated portions continue to move and vibrate against their grips; it
does not indicate force transmission across the failed section.
The [run record](results/material-engineering--jc-tension--johnson-cook/summary.json)
and [independent review](results/material-engineering--jc-tension--johnson-cook/interpretation.json)
preserve these measurements and the separation checks. This single resolution
does not establish convergence of the localization width or fracture response.

## HJC dynamic three-point bending

The beam is 300 × 60 × 60 mm along x, y and z in a 400 mm cubic domain.
Support centres are 250 mm apart. Each bottom support patch is 18.75 mm wide:
the left constrains x and y, while the right constrains y and permits axial
sliding. A small patch at the left support's mid-width constrains z to remove
lateral rigid motion. These are bilateral grid constraints representing
fixtures; their finite widths affect local stress.

A 25 mm wide central loading patch prescribes downward y velocity while
allowing x/z motion. A 75 µs half-cosine startup reaches 7.619 m/s, producing
2 mm displacement over 0.3 ms. Inertia and constitutive rate effects remain
active. HJC uses the repository's material defaults; neither the material nor
the loading is calibrated to a specific beam experiment.

Observables include the loading and individual support reactions, prescribed
motion, mean deflection of the initially bottom midspan layer, maximum downward
displacement and vertical momentum. Forces are reported **on the fixture**;
the specimen receives the opposite force. Reactions need not balance at each
instant because the beam accelerates. Sampling uses the next integration step,
identified by `boundary_sample_time_s`.

The recorded HJC run uses 121,104 particles and 2,120 steps over 0.3 ms. Its
minimum `det(F)` is 0.976602 and maximum damage is 0.492205. At the final frame,
midspan bottom deflection is 2.25377 mm and the loading reaction is 74.2952 kN.
No particle reaches the material failure flag or the `D ≥ 0.5` reporting
threshold. This run records deformation and partial damage, with no demonstrated
beam fracture.


### Large-displacement strain-floor profile

`concrete-bending-large` uses the same beam and supports with an explicit `strain-floor` HJC profile. Half-cosine velocity ramps accelerate and decelerate the moving patch over 0.3 ms each, with 0.6 ms at constant speed between them. This prescribes a 12 mm stroke over 1.2 ms, followed by a 0.3 ms hold. The original `concrete-bending` recipe and its default legacy profile remain reproducible.

```bash
python -m examples.material_engineering --case concrete-bending-large --model hjc --grid 64 --ppc 3 --frames 60 --no-render
```

The [completed run](results/material-engineering--concrete-bending-large--hjc/interpretation.json) uses 121,104 material points, 10,500 steps and 61 saved states over 1.5 ms. Actual midspan bottom deflection reaches 16.3525 mm; 21.4378% of the reference volume reaches D ≥ 0.5. Minimum det(F) is 0.965609, recorded mass is unchanged and no particle carries a material failure flag. The strain-floor profile retains compressive response and grid participation: D=1 is constitutive damage, not a resolved crack surface or a detached fragment. The actual deflection can exceed the imposed stroke during dynamic motion. These observations do not establish experimental calibration or mesh convergence.

RHT can use the same geometry and boundaries:

```bash
python -m examples.material_engineering --case concrete-bending --model rht --grid 64 --ppc 3 --frames 40 --no-render --resume
```

This is an available configuration; an RHT bending run has not yet been
completed. Its constitutive integration is substantially more expensive on CPU.

## JH-2 spherical indentation

The [wide-tile variant](CERAMIC_INDENTATION.md) uses a 180 mm tile and a
37.5 mm sphere radius. The geometry and completed measurements below describe
the original `ceramic-indentation` case.

A 100 × 25 × 100 mm tile lies on a bonded rigid backing in a 250 mm cubic
domain. The backing fixes all velocity components at `y ≤ 75 mm`. A rigid
sphere of radius 75 mm starts tangent to the upper face at `y = 100 mm`, with
initial centre `(125, 175, 125) mm`. Contact is frictionless and unilateral
at grid nodes.

The sphere completes an indentation and withdrawal cycle in 0.3 ms:

```text
δ(t) = 4 mm * [1 - cos(2π t / 0.3 ms)] / 2
```

Maximum nominal indentation occurs at 0.15 ms; peak speed is 41.89 m/s.
The tile uses the repository's JH-2 defaults. This dynamic loading history
does not constitute a quasi-static hardness test. The recorded `sqrt(R δ)`
contact-radius estimate describes spatial resolution only; no Hertz solution
agreement is claimed for the damaged ceramic.

Observables include nominal indentation, indenter speed, backing and indenter
reactions, mean displacement of the initially central upper surface, maximum
downward displacement and vertical momentum. Reaction signs and sampling
follow the bending convention above.

The completed run contains 118,580 particles and advances 4,080 steps, saving
41 frames through the full loading/unloading cycle. Minimum `det(F)` is
0.985189. The largest sampled indenter reaction is 644.862 kN at the 30 µs
frame; maximum nominal indentation is 4 mm at 150 µs. At 300 µs the sphere has
returned to its initial position, while the tracked central surface remains
2.57792 mm below its initial elevation.

Damage first reaches `D = 1` at the 22.5 µs saved frame. At the end, 43.0857%
of the reference volume has `D ≥ 0.5` and 42.5004% has `D = 1`. The material
failure flag remains false for every particle. These are constitutive damage
values in a retained continuum, with no demonstrated tile fragmentation.
The offline review confirms finite final arrays, positive saved determinants,
complete sampling and unchanged recorded mass.

## Interpreting brittle damage

Both brittle examples retain material points in the continuum (`failure_mode`
is `retain`). They record the maximum damage and reference-volume-weighted
mean damage, the reference volume with `D ≥ 0.5`, the fully damaged volume,
and the current bounds of points above that threshold. Empty damage regions
have null bounds. The `material_failed_reference_volume_fraction` follows the
material's failure flag separately.

`D = 1` is a local constitutive state; it does not identify a resolved crack
surface or detached fragment. There is no fracture-energy regularization.
Damage extent, local contact response and post-peak load histories require
spatial and time-step refinement before quantitative interpretation. See the
[material scopes](MATERIALS.md) for the underlying constitutive implementations.
