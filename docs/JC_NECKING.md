# Steel round-bar necking

`jc-necking` uses the material constants, circular-arc waist and prescribed loading in [SPHinXsys PR 1127](https://github.com/Xiangyu-Hu/SPHinXsys/pull/1127), pinned to commit `c3c1556ec86f9ebc6f0697149dff052702d20a75`. The PR was open and had no review comments when checked. It is a numerical implementation reference, not experimental validation. The older AA2024 `jc-tension` case and its recorded results remain available as a different problem.

## Geometry and loading

The rod is 6.4 mm long with a nominal radius of 0.8 mm. Its central radius is 0.75 mm. Two tangent circular arcs of radius 2.825 mm connect each half of the waist to the straight shaft over 0.75 mm. The profile has continuous position and tangent, with no initial crack or damage. This is a deliberately waisted specimen; it is not a nominally uniform bar with a numerical imperfection.

The axial coordinate is x, with the specimen centred at `(5, 5, 5)` mm in a 10 mm cubic domain. Regions more than 2.8 mm from the axial centre form the grips. The two grips ramp to opposite axial speeds of 10 m/s over 20 microseconds, using the PR's half-cosine velocity ramp. Their transverse velocities are zero. At 150 microseconds the imposed extension is 2.8 mm, or 50% of the initial 5.6 mm distance between the grip faces. Gravity is zero.

The JAX implementation applies these velocities to translating grid boxes. SPHinXsys constrains reference-labelled grip particles and resets their positions. These are different discrete boundary treatments. The JAX observer measures position and velocity errors for material points at least 1.5 grid cells inside each grip face, whose complete quadratic support remains constrained throughout grid crossing. Axial positions use end-step velocity integration, so the startup ramp has a position error of order `speed * dt`. Each nominal grip face is 2.05 mm from the end of the waist transition; quadratic grid support smooths the discrete boundary over adjacent cells.

## Material and integration

| Quantity | Value |
| --- | ---: |
| Density | 7830 kg/m³ |
| Young's modulus / Poisson ratio | 200 GPa / 0.29 |
| A / B / n | 792 MPa / 510 MPa / 0.26 |
| C / reference rate / m | 0.014 / 1 s⁻¹ / 1.03 |
| Reference / melting temperature | 298 K / 1793 K |
| Heat capacity / plastic-work heat fraction | 477 J/(kg K) / 1 |
| D1–D5 | 0.05, 3.44, −2.12, 0.002, 0.61 |

The [PR's constitutive source](https://github.com/lililii124/SPHinXsys/blob/c3c1556ec86f9ebc6f0697149dff052702d20a75/src/shared/materials/johnson_cook_solid.hpp) and the JAX model share tension-positive stress triaxiality, the density-pressure law `p = K (1/J − 1)`, a logarithmic corotated strain increment and complete stress removal at damage one. Their integration choices differ:

| Update | JAX | SPHinXsys PR |
| --- | --- | --- |
| Flow stress | Implicit radial return using the current plastic strain and rate | Explicit hardening tangent using the preceding plastic rate |
| Rate in fracture strain | Current equivalent plastic strain increment divided by dt | Current equivalent deviatoric total strain increment divided by dt |
| Thermal factor | Preceding temperature | Preceding temperature |
| Heating density | Current density | Reference density |

The JAX core is unchanged. Matching constants and loading does not imply identical finite-step stress histories. The material-point test independently solves the JAX return equation and checks its plastic work, temperature and damage, while making the different PR first-step rate convention explicit. Neither solver comparison nor a regression test establishes agreement with a physical tensile test.

Failed particles retain their mass and last velocity, cease participating in grid transfers and carry no stress. Their deformation and material history freeze while positions continue ballistically. Crack-face contact, fracture-energy regularization and a post-initiation softening law are absent. Local failure remains sensitive to the grid and loading history.

## Discretization and commands

Particles lie on a uniform Cartesian lattice clipped by the analytical outline. All particles have the same reference volume and mass. Coordinates are not compressed to produce the waist. To retain reflection symmetry and identical spacing on every axis, the requested particle density is rounded up to an even number of points across the nominal diameter; the axial count is four times that number. Metadata records both requested and effective particles per grid cell.

| Grid / requested ppc | Diameter in grid cells | Uniform particle spacing | Material points | Effective ppc |
| --- | ---: | ---: | ---: | ---: |
| 50 / 1 | 8 | 200 µm | 1,648 | 1 |
| 80 / 3 | 12.8 | 40 µm | 199,224 | 3.125 |
| 100 / 2 | 16 | 50 µm | 102,320 | 2 |
| 100 / 3 | 16 | 33.3333 µm | 341,584 | 3 |

The 50/1 geometry reproduces the PR's coarse point count. A 50 µm lattice also reproduces its 102,320-point geometry. Particle spacing and MPM grid resolution are separate: increasing ppc alone does not refine the displacement field.

Activate the project environment and run from the repository root:

```powershell
python -m examples.jc_necking --grid 100 --ppc 2 --frames 60 --cfl 0.15 --no-render --resume
```

Results go to `outputs/material-engineering/jc-necking/johnson-cook`. `python -m examples.material_engineering --case jc-necking --grid 100 --ppc 2` is also available. The dedicated command accepts `--end-time` for a shorter diagnostic run and `--output` for its separate directory. The loading ramp is unchanged by a shorter observation window. Full reference loading lasts 150 microseconds.

At CFL 0.15 the 80/3 cycle requires 46,320 steps; both grid-100 options require 57,900 steps. Measure short-run cost and memory before starting a full cycle. A larger CFL is an explicit option, not an established accuracy result; it requires a time-step comparison. No full-cycle result is asserted by this setup description.

## Recorded evidence

The observer records grip reactions, the central material-section force, empty load-bearing sections, active-particle gaps, damage, plastic strain, temperature, mass, kinetic energy, plastic work and stress work. Section force integrates `sigma * cofactor(F) * e_x` with each particle's reference volume divided by the actual lattice spacing. Grip reactions include imposed acceleration and subsequent vibration; they need not vanish when the central section separates.

Geometric necking is measured independently from particle positions. Each initial axial layer supplies a transverse RMS radius about its current centroid. The central ratio uses every original layer particle and therefore includes ballistic mass after failure. The minimum intact-gauge ratio uses only layers with no failed particles. Neither is a post-failure load-bearing area. The area derived from F freezes for failed particles and must not be used to claim continued neck evolution.

Saved particle positions, failed flags and scalar histories support independent checks of localization and separation. Final loss of central force and a gap exceeding three grid cells demonstrate loss of shared quadratic grid support; they do not establish a mesh-independent crack surface. Energy quantities and sampled actuator power are diagnostics, not a complete energy closure.

ParaView renders true particle coordinates without geometric amplification, axes or annotations. Silver boxes represent the prescribed grip regions. Failed mass remains visible as grey point markers, not as a reconstructed fracture surface.
