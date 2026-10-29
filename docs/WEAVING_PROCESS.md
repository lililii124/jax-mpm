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

## Contact coverage buckets

The host still builds and validates the complete real coverage graph. The
driver then pads only the arrays passed into JAX, using a power-of-two storage
bucket and a shared boolean mask. Real pairs, seed IDs, graph edges and history
indices keep their order. Dummy graph rows contain `-1`; dummy gaps are finite,
and every contact selection and rejection flag applies the mask. This avoids
compiling another contact program for each new coverage-list length.

The [bounded JSON recipe](../examples/weaving_tl_buckets.json) supplies the same
eight fibres per family, requested time step and 200-step frame stride:

```bash
export JAX_ENABLE_X64=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m examples.triaxial_weaving_tl_process \
  --config examples/weaving_tl_buckets.json
```

Explicit command-line arguments override the recipe's `run` values. Its
environment and resource sections are records, not automatic enforcement;
set the environment before starting Python and enforce the four-CPU,
16 GiB container / 12 GiB process-tree USS budget externally. This recipe ends
at 1.1 ms. Continue with a new output directory and a later `--stop-time`, using
the same source and `--resume-from` as above.

The [execution-equivalence record](../evidence/weaving-tl-process/bucket-execution-equivalence.json)
compares 200 actual steps from a clean 6.592 ms checkpoint. All per-step state,
material/contact history and contact metrics match bitwise; energy, work and
linear/angular-momentum observations match exactly. Real coverage lengths
154, 167 and 186 exercise a diagnostic tight first bucket followed by bucket256.
Separate fixtures rigidly translate one actual saved fibre to create nonzero
contact energy and force. Full forces and JAX gradients, active/failed/end/
same-family/unseeded flags, and dummy zero energy/gradient match between the
original and bucket implementations. These translated states are rejected by
the physical guards and are not production trajectory samples.

The [public function-port record](../evidence/weaving-tl-process/bucket-public-function-port.json)
binds this port to the executed functions: the contact module is byte-identical,
and the bucket and driver contact functions have identical Python syntax trees.
The public geometry/configuration wrapper differs from the frozen production
entry. It retains strict source-bound restarts and does not silently import
checkpoints produced by the older source or the frozen production capsule.

## Saved results and current evidence

Each directory contains `plan.json`, `binding.json`, `observations.jsonl`,
particle frames, `checkpoint.npz` and `summary.json`. The observations include
actual time steps, guide reactions, kinetic and elastic energy, contact energy,
work remainder, momentum errors and contact coverage.

The [5.3 ms record](../evidence/weaving-tl-process/accepted-prefix.json) preserves
the earlier trajectory and source hashes. A later
[5.85 ms checkpoint and Linux handoff](../evidence/weaving-tl-process/accepted-prefix-5850us.json)
completed cleanly and restored every state and history field bitwise. That
checkpoint still has 200 active bearings and no active contact patches.
The following Linux window reached 6.344 ms before exceeding its 6 GiB USS
guard; it wrote no final checkpoint. The subsequent recovery retained the same
physical state, sources and time step, and completed a clean
[6.4 ms checkpoint](../evidence/weaving-tl-process/accepted-prefix-6400us.json)
under a 16 GiB container / 12 GiB USS budget. After the bucket qualification,
production migrated at a clean 6.776 ms boundary with the old checkpoint retained.
The first bucket window completed a clean
[6.95 ms checkpoint](../evidence/weaving-tl-process/accepted-prefix-6950us.json):
nine real coverage lengths from 260 to 349 reused bucket512, with peak USS
3,362,148,352 bytes. The queue continues through bounded 1.1 ms windows toward
47.0412 ms under the same resource and physical guards. The 6.95 ms boundary
still has 200 active bearings and no active contact patches. These records do
not establish completed weaving, a successful contact trajectory or convergence.

## Render saved particle states

The renderer reads explicit output directories containing `plan.json` and actual
`frame_*.npz` archives. It does not run the solver or interpolate saved states.
With ParaView's `pvpython` available:

~~~bash
pvpython --force-offscreen-rendering tools/render_weaving_paraview.py \
  --input outputs/weaving-1100us outputs/weaving-2200us \
  --output outputs/weaving-render --frames 13 --supersample 3
~~~

Use ordinary Python with the same arguments and `--validate-only` to check inputs
before rendering. Adjacent windows may repeat a boundary state; repeated
step/time pairs must have identical required saved arrays. Conflicting copies,
non-increasing times or steps, and incompatible physical or numerical plans
are rejected.

Geometry uses each particle's saved position and deformation gradient:
`x + F * reference_offset`, with unit deformation scale. Reference offsets come
from actual straight-cord sampling: equal-area annular sectors and the saved
longitudinal station spacing. Analytic sector volumes are checked against saved
`reference_volume`. Curved boundaries use eight chord segments by default; for the
current four/eight-particle rings this understates sector volume by at most
0.642 percent. Increase `--arc-segments` for a finer display tessellation.

Fibre IDs are grouped into three families: blue, orange and green. Only particles
marked failed are hidden. The camera remains fixed across the selected actual
frames, with no axes or text. Default images render at 1200 square pixels and
downsample to 400 square pixels with threefold supersampling. Outputs include
a GIF, PNGs, input hashes, camera settings and geometry checks. Rendering a
bounded prefix does not establish completed weaving, successful contact, or
convergence.
