# Crossed-fibre bending and pull-off

`examples.fibre_pull_off` couples two separate MPM fibres through the selected `hertz`, `jkr`, `md`, or `self-consistent` local contact model. Both end regions of the lower fibre are fixed. Both end regions of the upper fibre follow the same vertical displacement. The fibres have independent grids and exchange the computed normal contact force; they share neither grid velocity nor a prescribed pressure footprint.

This is an illustrative dynamic experiment. Its material and adhesion parameters are assumptions, not an experimental calibration. It does not change the material or loading used in the triaxial angle study.

## Geometry and loading

| Quantity | Default |
|---|---:|
| Length / diameter of each fibre | 64 / 4 mm |
| Initial surface gap | 50 µm |
| Fully constrained length at each end | 5 mm |
| Grid spacing / particle spacing | 0.4 / 0.2 mm |
| Particles / packed grid nodes | 202,240 / 380,250 |
| Young's modulus / Poisson ratio / density | 2 MPa / 0.25 / 1,000 kg/m³ |
| Tensile strength / fracture energy | 0.3 MPa / 100 J/m² |
| Adhesion work / interaction range | 1.2 J/m² / 20 µm |
| Self-consistent outer radius / rings | 1 mm / 32 |
| Duration / steps / saved states | 30 ms / 24,495 / 61 |

Particles lie on a uniform Cartesian lattice clipped by the circular cross section. Each retains volume `spacing³`. The sampled cross-sectional area exceeds the analytic circle area by 0.586%; the plan records both values. Reference section centroids coincide with the straight axes by symmetry.

The upper grips move down 0.3 mm over 10 ms, hold until 12 ms, rise to 1.5 mm above their starting position by 25 ms, and hold until 30 ms. Each moving interval uses a half-cosine displacement. At a time step, the prescribed velocity equals the exact displacement increment divided by that step's duration. All nodes supporting a tagged grip particle are constrained in three components, so the numerical constraint extends beyond the nominal 5 mm particle region by its grid support. There is no backing plate or gravity.

All four models use this same geometry, material and fixture protocol. Hertz has no attractive force. JKR, MD and the self-consistent model use the recorded adhesion work. MD and the self-consistent model also use the finite interaction range. See [local contact models](LOCAL_CONTACT_MODELS.md) for their different branch definitions.

## What the coupling resolves

The particle grids resolve overall fibre bending and stretching. Contact uses current section centroids and segment closest points. The local circular-equivalent model represents indentation of the cross sections below the resolved contact-patch scale; it is not an additional pressure load applied to a resolved half-space. Forces are distributed through the transpose of the centroid and segment interpolation, preserving pair force and torque.

This is a beam-scale/local-indentation approximation, not a rigorous separation of all continuum deformation scales. The circular replacement is not the full elliptical solution for crossed cylinders. The reference radius is fixed, axial surface curvature is omitted, and the effective circular radius can change as the fibres turn. Its omitted parameter-work contribution and adhesive snap prevent a claim of global contact-energy conservation. The output records actual `contact_radius_m`, `effective_radius_m`, and their step maxima so these limitations can be assessed. No friction or tangential dissipation is added.

The neighbour list has an 8 mm coverage skin. Every step checks the actual nonrigid motion bound, missing close segment pairs, active-history relocation, end contact and material validity. Finite-range models capture virgin approaches at gaps up to the interaction range; an existing attachment retains its history during withdrawal. A failed search or local solve stops the run. Damage onset also stops this particular elastic demonstration instead of silently changing it into a fracture experiment.

## Run and resume

Inspect the geometry and step budget without importing JAX:

```powershell
python -m examples.fibre_pull_off --model all --dry-run
```

Measure a prefix before scheduling a complete run. The stopping frame is a runtime option; it does not shorten the physical protocol or reset contact history:

```powershell
python -m examples.fibre_pull_off --model hertz --stop-after-frame 2
python -m examples.fibre_pull_off --model hertz --resume
python -m examples.fibre_pull_off --model md
```

The same commands accept `jkr` and `self-consistent`. `--dx`, `--spacing`, `--frames` and `--rings` allow explicit resolution studies. A coarse run remains a discretization study; increasing particle density does not increase the grid's mechanical resolution. Use a new output root when changing physical or numerical parameters.

Each checkpoint contains every MPM and local-contact history field, the cached current particle contact force, accumulated impulses and work, and the original self-consistent compliance table. Resume requires the same source and runtime, all previously saved arrays, matching frame hashes, unchanged static particle labels and the same physical plan. A planned stop reports `checkpoint-ready`, not completion. If a step fails between output frames, `rejected_checkpoint.npz` records the last accepted partial-step state and its clock, separately from `last_saved_frame`; its `complete_saved_frame` flag is false. It is diagnostic evidence and cannot resume automatically.

After a complete run, use the existing ParaView pipeline at true scale:

```powershell
python -m mpm_jax.visualization outputs/fibre-pull-off/hertz --gif --field family_id
python tools/encode_video.py outputs/fibre-pull-off/hertz
```

## Recorded evidence

`observables.csv` records the contact force, gap, contact radius, potential, central deflection of each fibre, prescribed and measured grip motion, and each fibre's contact and fixture impulses. Contact force and history correspond to the saved positions. Fixture forces are separately labelled as saved-interval averages and last-step impulse divided by the time step.

`diagnostics.csv` checks each fibre's linear momentum against its independently accumulated contact and boundary impulses. Pair force and torque residuals, positive volume, mass, and damage are retained. Kinetic energy includes the quadratic APIC affine term. Actual fixture work and the grid boundary kinetic-energy correction are separate quantities; the contact force-displacement work is also recorded. These measurements do not establish quasistatic equilibrium or calibrated pull-off strength.

The small CI tests exercise actual MPM motion and force transfer for all four models, plus complete versus checkpointed self-consistent evolution. Full-period numerical, resolution and final ParaView checks are required before treating a run as a completed demonstration.
