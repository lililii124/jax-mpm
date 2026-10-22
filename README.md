# jax-mpm

Two- and three-dimensional material point mechanics in JAX: quadratic B-splines,
APIC transfers, explicit dynamics, constitutive history, and two-body contact.

| Contact | Fluid–solid interaction |
|---|---|
| ![Oblique contact](docs/images/oblique.png) | ![Fluid–solid interaction](docs/images/fsi.png) |

[Gallery](docs/GALLERY.md) · [Examples](docs/EXAMPLES.md) · [Verification](VALIDATION.md)

## Installation

Python 3.12 or later:

```bash
git clone https://github.com/lililii124/jax-mpm.git
cd jax-mpm
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` on Windows, then install:

```bash
python -m pip install -e ".[test]"
```

On Windows, `setup.ps1` creates `.venv312`. Its `-PythonPath` and `-JaxSource`
options select an interpreter and an optional local JAX source archive.

## Examples

```bash
python -m examples.suite --list
python -m examples.suite --case oblique --gif
python -m examples.suite --case fsi --gif
python -m examples.suite --case elastic-gate --grid 48 --ppc 3 --gif
python -m examples.suite --case cantilever --model all-elastic
python -m examples.suite --case compression --model all-plastic
python -m examples.material_paths --model all
```

The spatial suite defaults to a **48³ grid and 2 particles per cell per axis**
(up to 8 per cell). Depending on geometry, these cases contain roughly
17,000–119,000 particles. `--grid`, `--ppc`, and `--dt-factor` control resolution;
`--frames` changes output frequency while preserving the physical duration.
`--case all` runs the 20 spatial scenarios with their default materials.
For denser sampling use `--ppc 3` (27 particles per cell). The water-entry case
at `--grid 72 --ppc 3` contains **1,327,400 particles**.

| Group | Cases |
|---|---|
| Elastic dynamics | Longitudinal wave, cantilever vibration, torsional release, large bending/twist |
| Prescribed loading | Simple shear, platen compression, pressure-loaded clamped panel |
| Contact | Head-on collision, oblique collision, frictional sliding, rebound, rubber-ball drop, ball–beam contact |
| Fluids | Dam break, sloshing |
| Fluid–solid interaction | Clamped obstacle, elastic-block water entry, top-clamped gate, floating block |
| Material points | Compression/unload, hydrostatic compression, shear reversal, confined shear |

Each spatial run saves particle trajectories and final states as NPZ, diagnostics
and observables as CSV, a JSON definition/result summary, and a final image.
`--gif` adds an animation; `--no-render` skips rendering. ParaView renders
every particle, with a fixed camera and no axes or annotations. Compressed
VTK files and a ParaView state accompany the images. Install ParaView and put
`pvpython` on PATH, or set `PARAVIEW_PYTHON` to its executable. See
[the rendering guide](docs/RENDERING.md).

Long runs save a checkpoint after every output frame:

```bash
python -m examples.suite --case water-entry --grid 72 --ppc 3 --frames 60 --no-render --resume --output outputs/production
python -m mpm_jax.visualization outputs/production/water-entry/neo-hookean --gif
```

Repeat the first command to resume an interrupted run. `status.json` records
progress; `summary.json` is written after the full duration completes.

The original 2D examples and material-gradient check remain available:

```bash
python -m mpm_jax --scene elastic --gif
python -m mpm_jax --scene dam-break --dim 3 --gif
python -m examples.gradient
```

See [the example guide](docs/EXAMPLES.md) for geometry, boundary conditions,
observables, and refinement commands.

## Materials

| Family | Implementations |
|---|---|
| Linear elasticity | Isotropic, orthotropic, transverse isotropic; rotated material axes |
| Hyperelasticity | Compressible Neo-Hookean, selected Blatz–Ko rubber branch |
| Metal plasticity | Mixed/isotropic bilinear J2, prescribed-temperature plasticity, Johnson–Cook, selected Steinberg–Guinan branch |
| Brittle materials | JH-2, HJC, three-surface RHT with p-alpha compaction |
| Fluid | Compressible inviscid volumetric law |

The incremental models use three-dimensional stress/history tensors. Defaults
are demonstration parameters. Model conventions, source scopes, and omitted
branches are recorded in [docs/MATERIALS.md](docs/MATERIALS.md).

## Python API

```python
from mpm_jax import Config, advance, block, fixed_box, neo_hookean, validate_state

config = Config(
    dim=3, n_grid=48, dt=2e-4, material="neo-hookean",
    gravity=(0., 0., 0.), boundary_cells=0,
    boundaries=(fixed_box((0., 0., 0.), (.2, 1., 1.)),),
)
material = neo_hookean(young=2e4, poisson=.2)
state = block(config, (.15, .4, .35), (.8, .6, .65), density=1000.)
state = advance(state, material, config, steps=1000)
validate_state(state, config)
```

`PlaneWall` supplies unilateral slip/friction walls, including constant wall
velocity. `VelocityBox` supplies prescribed velocity components in a physical
coordinate region with an activation interval. `fixed_box` is its zero-velocity
form. `step(..., force=...)` accepts an `(N, dim)` array of particle forces in N.
`coupled_step` advances two bodies on separate grids with balanced normal and
Coulomb impulses. See `examples/scenarios.py` for complete setups.

## Numerical scope

- Dense background grid on `[0, L]^dim`; 2D particle volumes assume unit thickness.
- Stress is evaluated before P2G; deformation/history are updated after G2P.
- Incremental constitutive integration uses a polar rotation and logarithmic
  strain increment. Linear elasticity assumes small strain and small rotation.
- The fluid is compressible and inviscid. The interface treatment is nonadhesive
  normal contact; there is no incompressible pressure solve, cavitation,
  turbulence model, or thermal conduction.
- Failed particles retain mass and grid participation. There is no crack-surface
  reconstruction, particle deletion, or fracture-energy regularization.
- Elastic material gradients are tested in 2D and 3D. Plasticity/damage return
  maps and their adaptive loops are not advertised as differentiable.
- Invalid deformation, missing grid support, nonfinite states, and failed
  constitutive convergence are reported. Time-step stability and spatial
  convergence remain problem dependent.

## Verification

```bash
python -m pytest -q
```

Tests cover transfers, conservation, elasticity, constitutive limits, boundary
conditions, contact, gradients, and ten supplied material-point reference traces.
[VALIDATION.md](VALIDATION.md) records quantitative checks;
[the run catalogue](docs/RESULTS.md) lists completed cases.
GPU execution has not been validated.

## License

[AGPL-3.0-or-later](LICENSE). The earlier solver and Taichi-derived transfer code
retain their Apache-2.0 attribution in [NOTICE](NOTICE) and `third_party/`.
Sources are listed in [REFERENCES.md](REFERENCES.md).
