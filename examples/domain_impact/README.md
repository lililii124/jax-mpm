# C2 domain-impact reproduction

This command runs a source-bound snapshot of the qualified virgin C2 metal
impact solver. The snapshot is isolated under `vendor/`; it does not replace or
import the public `mpm_jax` implementation. Run from the repository root:

```console
python examples/domain_impact/run.py --help
```

For a complete formal horizon, every window targets the same 1 µs endpoint and
has a 3500 s solver-time bound. A run can stop earlier with an accepted
checkpoint; use the capsule tool to prepare the next window. Give every window
a new output directory:

```console
python examples/domain_impact/run.py --output outputs/domain-impact/w01/result --end-time 1e-6 --max-seconds 3500
python tools/domain_impact/prepare_resume.py --run-dir outputs/domain-impact/w01/result --output outputs/domain-impact/w02/resume
python examples/domain_impact/run.py --output outputs/domain-impact/w02/result --end-time 1e-6 --max-seconds 3500 --resume-checkpoint outputs/domain-impact/w02/resume/resume-checkpoint.npz --resume-bridge outputs/domain-impact/w02/resume/resume-bridge.json
```

If `outputs/domain-impact/w02/resume/capsule-manifest.json` reports
`"support_split": true`, use this full continuation command instead:

```console
python examples/domain_impact/run.py --output outputs/domain-impact/w02/result --end-time 1e-6 --max-seconds 3500 --resume-checkpoint outputs/domain-impact/w02/resume/resume-checkpoint.npz --resume-bridge outputs/domain-impact/w02/resume/resume-bridge.json --resume-parent-checkpoint outputs/domain-impact/w02/resume/resume-parent-checkpoint.npz --resume-support-audit outputs/domain-impact/w02/resume/resume-support-audit.json
```

Repeat the same two commands with fresh `w03`, `w04`, … paths whenever a
bounded window ends before 1 µs. For example, prepare `w03` from `w02` with:

```console
python tools/domain_impact/prepare_resume.py --run-dir outputs/domain-impact/w02/result --previous-bridge outputs/domain-impact/w02/resume/resume-bridge.json --output outputs/domain-impact/w03/resume
```

Then run `w03` with its newly built checkpoint and bridge, adding the parent
and support-audit options if its capsule manifest says `"support_split": true`.
The `--source-root` defaults to this public solver snapshot; a different source
tree is rejected unless every file hash matches the accepted run's
`run.json`. Stop when `run.json` reports the requested endpoint; a timeout
checkpoint is resumable but is not a completed 1 µs trajectory. The capsule
builder verifies the accepted step/time, mandatory contact and e0 receipts,
and (when present) the split parent, audit and duplicated-support checkpoint
bindings. It copies the accepted state byte-for-byte and trims JSONL diagnostics
after that accepted step. It does not run or modify the solver.

The formal case is the 5 km/s, 2 mm metal sphere against the 20 mm plate with
the qualified material and contact guards. `run.json` records the executing
CFL coefficient as `dt_cfl = 0.025` and `dt_cfl_limit = 0.025`; it records
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
- `../../tools/domain_impact/prepare_resume.py`: accepted-checkpoint capsule
  builder; `../../tools/domain_impact/resume-capsule-port-receipt.json` records
  its bytewise comparison against a real production handoff.

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
