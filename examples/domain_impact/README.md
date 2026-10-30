# C2 domain-impact reproduction

This command runs a source-bound snapshot of the qualified virgin C2 metal
impact solver. The snapshot is isolated under `vendor/`; it does not replace or
import the public `mpm_jax` implementation. Run from the repository root:

```console
python examples/domain_impact/run.py --help
```

For a fresh formal run, use a new output directory:

```console
python examples/domain_impact/run.py --output outputs/domain-impact --end-time 1.6e-7 --max-seconds 1800
```

For a source-bound continuation, pass the accepted checkpoint and its bridge
receipt using the option names shown by `--help` (`--resume-checkpoint` and
`--resume-bridge`). Choose a new output directory for each run. The formal
case is the 5 km/s, 2 mm metal sphere against the 20 mm plate with the
qualified material and contact guards. `run.json` records the executing CFL
coefficient as `dt_cfl = 0.025` and `dt_cfl_limit = 0.025`; it records
`original_cfl_scale = 0.25` for the ratio to the original 0.1 coefficient.
The executed formula is `dt_cfl * 0.00025 / max(signal)` and is unchanged.

## Isolated source layout

- `vendor/run_metal_tl_fresh.py`: formal fresh-run and continuation driver.
- `vendor/mpm_jax/`: solver, material, contact, and fracture-support modules.
- `vendor/examples/`: formal impact case and geometry modules.
- `vendor/tools/`: source-bound physical-audit helpers.
- `source-map.json`: origin SHA-256 and destination for every copied source.
- `cfl-metadata-erratum.json`: corrected run-metadata semantics, separately
  bound to the unchanged actual-state replay receipt.

The implementation files retain their source imports. `run.py` puts `vendor/`
first on `sys.path`, so those imports resolve within this snapshot. Numerical
function ASTs are unchanged. The vendored `mpm_jax/__init__.py` is deliberately
minimal: the original package initializer eagerly re-exported the general MPM
API and pulled in unrelated modules. Its import-time statements define/import
objects; they do not advance a state or apply forces. The isolated run imports
only the direct solver modules it uses.

## Environment

The source uses JAX with 64-bit mode, NumPy, and SciPy. Install the repository
with its fracture extra, then run the command above. GPU is not required for
the CPU replay path. Large production runs need substantial host memory and
runtime; `--help` describes bounded run and output options.

The public snapshot contains the complete import closure for the formal
runner. It can produce solver outputs from a fresh run or accepted checkpoint.
ParaView rendering is a separate post-processing stage: the GIF/VTK conversion
tools and production result files are not bundled in this solver package.

## Source equivalence check

The port was replayed for one accepted step from the actual w01 bridge state
(CP1331) using the production CP1332 timestep. This verifies a real-state
single-step source match; it does not claim a full-trajectory replay. The original source tree and the
vendored snapshot both passed the solver's material, contact, finite-state,
midpoint-work, and admission checks. All emitted state, material, geometry,
contact-pair, contact-point, and normal arrays were bitwise identical. The
force residual and cumulative energy, linear-momentum, angular-momentum, and
contact-pair metrics matched the recorded production CP1332 row. The receipt
and source hashes are in `one-step-replay-receipt.json` and `source-map.json`.
