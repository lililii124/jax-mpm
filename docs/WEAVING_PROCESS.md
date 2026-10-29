# Triaxial weaving process

The process driver advances three fibre families with end fixtures and
releasable normal bearings. Fibre deformation follows total-Lagrangian MPM;
the machine prescribes fixture motion and bearing constraints. Each fibre uses
isotropic elastic-brittle material with energy-based softening.

The reference grid retains every node touched by the original quadratic
particle stencils. Bearing forces use mass-weighted RATTLE projections; their
work is the constraint impulse dotted with the corresponding midpoint velocity.
Time steps end at fixture and bearing-release events.

Fibre contact uses a frictionless circular equivalent-curvature Hertz
potential. JAX differentiates the potential with respect to particle positions,
including changes in local curvature. Changes in connected contact regions have
a separate energy-jump check. Adhesive laws used in the
[fibre contact comparisons](FIBRE_CONTACT.md) are separate from this process.

## Run and resume

Enable JAX 64-bit arithmetic before starting Python:

```bash
export JAX_ENABLE_X64=true
```

In PowerShell, use `$env:JAX_ENABLE_X64 = "true"` instead. From the repository root:

```bash
python -m examples.triaxial_weaving_tl_process \
  --output outputs/weaving-1100us --per-family 8 --dt 2.4e-7 \
  --stop-time .0011 --max-steps 5000 --frame-every 200
python -m examples.triaxial_weaving_tl_process \
  --output outputs/weaving-2200us --per-family 8 --dt 2.4e-7 \
  --stop-time .0022 --max-steps 5000 --frame-every 200 \
  --resume-from outputs/weaving-1100us
```

Each command writes to a new directory. Resume verifies the checkpoint hash,
source, material, reference layout, runtime and physical-plan bindings. It
restores the full state, material history and cumulative work and impulses.
Keep the same numerical environment for this strict public restart path.

For the entire eight-fibres-per-family protocol:

```bash
python -m examples.triaxial_weaving_tl_process \
  --output outputs/weaving-full --per-family 8 --dt 2.4e-7 \
  --max-steps 250000 --frame-every 200
```

This configuration has 24 fibres and a 47.0412 ms protocol. The bounded commands
above cover shorter intervals; reaching their step limit does not complete the
full process. Check `summary.json`: `completed=true` requires the full physical
duration and release of every bearing. The end fixtures remain at their final
machine positions.

## Saved results and current evidence

Each directory contains `plan.json`, `binding.json`, `observations.jsonl`,
particle frames, `checkpoint.npz` and `summary.json`. The observations include
actual time steps, guide reactions, kinetic and elastic energy, contact energy,
work remainder, momentum errors and contact coverage.

The [accepted-prefix record](../evidence/weaving-tl-process/accepted-prefix.json)
records the completed 5.3 ms trajectory and its source hashes. Its terminal
state still has 200 active bearings and no active contact patches. The full
weaving and contact sequence remains in progress; this prefix does not establish
full-process completion or convergence. The same source files are provided here.
