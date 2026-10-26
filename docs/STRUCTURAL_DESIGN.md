# Differentiable structural material-layout optimisation

`examples/structural_design.py` differentiates a 3D
total-Lagrangian material-point simulation. It redistributes solid material
volume along a metre-scale cantilever while keeping the reference particle
locations, support topology, and material-point stencil fixed. This is a
continuous material-mixture design inside a fixed envelope; it does not move
the exterior surface or claim a geometric thickness change.

## Physical model and design constraints

The continuum is a compressible Neo-Hookean solid (`E = 200 kPa`, `nu = 0.25`)
with a base density of `1000 kg/m^3`. The local design field is a smooth
ten-control axial solid-fraction field. At each material point its scale is
applied consistently to both reference quadrature volume and particle mass.
This is the linear `p = 1` Voigt material-mixture assumption: local stiffness
and mass both scale with solid fraction. The total solid fraction is exactly
0.75 by normalization, so total solid volume and mass remain constant; the
unconstrained analytical bounds on the local scale are exactly
`0.75 exp(±0.56)` (about `[0.4284, 1.3130]`). Admissible designs also enforce
the physical local solid-fraction constraint `0 < fraction <= 1`; the optimizer
skips infeasible steps and final validation checks both design and production
meshes. This avoids a void residual stiffness, density/stiffness mismatch, and
zero-mass points. No SIMP penalty or geometric shape derivative is implied.

The material positions form a 3D solid beam. A finite-width reference-node
clamp prescribes zero displacement, velocity, and acceleration. Two
design-independent, fixed-resultant loads act on orthogonal exterior tip
patches (the y and z cases). Each fixed 8 N resultant rises with a 20 ms raised
cosine ramp, remains applied through the hold interval, and unloads smoothly
over the final 20 ms of a shared 0.24 s horizon. The objective is the mean
squared tip-region displacement over this full dynamic history and both load
cases, normalized by 40 mm. The horizon spans about one quarter of the
Euler-Bernoulli first-bending period for this beam and material.
It therefore rewards response reduction under the same external force and
does not trade a weaker load or lower mass for a smaller response.

The AD path is `jax.jit(jax.value_and_grad(...))` through the repository's
`mpm_jax.total_lagrangian.step`, with `lax.scan` for time integration,
12-step rematerialized chunks for reverse-mode memory, and sequential
checkpointed `lax.map` over two load cases so both full histories are not held
in memory at once. A central finite difference along the initial AD gradient direction
is recorded in the result JSON. The coarse design solve uses `dt = 5e-4 s` on
the 24-cubed grid (CFL about 0.186). It is followed by independent forward
checks on a 64-cubed background grid (about 67,700 material points) at
`dt = 2e-4 s` (CFL about 0.198), both over the same 0.24 s physical horizon.
The refined run captures an initial state and 20 particle-sampled frames at
60-step intervals for each design and load case. It uses the refined particle
cloud and interpolated continuous design field, not the coarse trajectory.

## Run

Use Linux with JAX installed; CPU is selected by default. A typical
resource-limited invocation is:

```bash
python -m examples.structural_design \
  --design-grid 24 --production-grid 64 --iterations 5 \
  --time-horizon 0.24 --design-dt 5e-4 --production-dt 2e-4 \
  --output outputs/structural-design
```

The result directory contains `summary.json`, `design-history.csv` (per-step
objective, bounds, solid fraction and mass), a compressed particle trajectory
archive, and ParaView-readable PVD/VTU state series for
the initial and optimized layouts under both loads. `summary.json` records the
mass/volume constraint check, design bounds, directional AD/finite-difference
comparison, objective values on the optimization and production meshes, and
per-load tip responses. The CLI prints the same JSON to stdout.

The benchmark was produced with 21 accepted Adam updates followed by a uniform
control-amplitude feasibility projection (`beta = 0.9670227802309797`). The
replay bundle in [`structural-design-evidence/`](structural-design-evidence/)
includes the raw preprojection history, raw feasible-forward summary, task
status records, and ParaView frame audit. To replay the recorded design and
rerun both forward checks, use:

```bash
python -m examples.structural_design \
  --design-grid 24 --production-grid 64 --iterations 0 \
  --time-horizon 0.24 --design-dt 5e-4 --production-dt 2e-4 \
  --resume-history docs/structural-design-evidence/optimization-history-preprojection.csv \
  --project-resume-amplitude 0.9670227802309797 \
  --output outputs/structural-design-replay
```

`optimization-history-preprojection.csv` has rows 0–21 from the original
21-update run. The CLI reconstructs the saved controls and applies the same
projection; zero additional updates make this an exact recorded-design replay,
followed by fresh differentiability and independent forward checks. The
shorter `--iterations 5` command above is a fresh run and does not reproduce
the benchmark.

Every optimizer update atomically checkpoints the control field and both Adam
moments in `optimizer-state.npz`, and rewrites the CSV history. Continue from
that checkpoint with `--resume-state PATH`; if only an older CSV is available,
`--resume-history PATH` recomputes the saved gradients and reconstructs Adam's
moments before taking additional steps. Resume with the same design grid,
time-step and horizon; the checkpoint records and validates that model
signature.

`--design-grid` changes only the differentiation mesh; `--production-grid`
sets the independent confirmation mesh and defaults to 64. For a fast systems
qualification without design updates, `--skip-design` retains the uniform
0.75 field. Reduced production grids are useful during development but should
not be described as the default production result.

Create ParaView 6.2 OSMesa animations from the actual production-particle
coordinates after the simulation. The default y-load output compares uniform
and optimized layouts; its 800 x 400 GIF places two 400 x 400 views side by
side with one shared camera and material scale. It also writes one 400 x 400
GIF for each layout:

```bash
pvpython --force-offscreen-rendering tools/render_structural_design_pv.py \
  --input outputs/structural-design \
  --output outputs/structural-design/paraview-gif --case y
```

Every GIF frame fetches and verifies the exact saved particle coordinates
through ParaView. The original displacement scale is 1.0; the animation uses
all 21 forward-confirmation frames and has no axes, labels, or overlays. The
moving particle clouds use `volume_ratio` for the material-layout color. The
PVD series, VTK states, JSON summary, CSV optimization history, and GIFs form
the reviewable numerical and visual deliverables. Pass `--case z` to render
the other load direction or `--case both` to render both.

## Production comparison

These ParaView 6.2 animations use the final 64-cubed forward analyses. Every
GIF contains the 21 actual saved frames and uses undeformed-scale coordinates.
Each initial/optimized pair shares a camera and material-fraction color scale.

| Load case | Uniform reference | Optimized material layout |
| --- | --- | --- |
| 8 N y-direction | ![Uniform y-load response](assets/structural-y-initial.gif) | ![Optimized y-load response](assets/structural-y-optimized.gif) |

The recorded optimization used 21 Adam updates and reduced the coarse dynamic
objective from `0.0815990` to `0.0571938` before projection, then to
`0.0574304` after the final feasibility projection scaled the controls by
`0.9670228`. A separate 64-cubed, 67,704-
particle forward analysis over the same 0.24 s horizon reduced the two-load
objective from `0.0829364` to `0.0592507` (28.56%). The local solid fractions
were `0.611–0.995` on the design mesh and `0.614–0.9995` on the production
mesh; each mesh retained the `0.75` mean. Total mass was `23.76 kg` to
roundoff. Both production cases remained valid, with minimum Jacobian
determinants `0.9835` and `0.9861`. The final central directional
finite-difference check differed from the autodiff derivative by `1.51e-7`
relative. This is a finite-budget feasible design, not a claim of convergence
or a global optimum.

The machine-generated summary retains its run command verbatim; the replay
command above documents the feasible design path from the archived history.
Job status JSON records successful exits for optimization, forward
confirmation, and ParaView rendering. The render manifest checks all 21 fetched
frames at deformation scale 1.0 and records source summary and trajectory SHA-256.
