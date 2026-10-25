# RHT active return-map WIP

Prepared only; no JAX execution, server job, real-state106 comparison or speedup
claim has occurred. Base b7318ac retains original0c99377 physics.

mpm_jax/materials/rht.py keeps the original return body unchanged as
_return_branch_full. N<=256 takes that path. Larger calls compute the full
zero-multiplier initial update once, select only its exact yielding mask,
solve static256-point batches and scatter to the full initial result. Scalar
and size1 broadcasts remain unchanged; leading-N context/material/dt arrays
are gathered. Padding gathers index0 with zero multiplier and writes only to
out-of-range N using mode=drop. No arithmetic blend is used.

The original60 growth iterations,56 bisections,64 safeguarded-root cap and
all physical tolerances remain. AST comparison verified the full reference,
_at_multiplier, _safeguarded_newton, _single_step, full-domain energy iteration
and adaptive subdivision bodies unchanged. Roots are pointwise; their global
any only extends loops while already-done point roots remain frozen.

Run tools/check_rht_active_return.py --output NEW.json in the assigned pinned
Linux environment. It has five narrowly targeted fixtures: original small-N,
sparse padding, dense-enough cross-batch final padding for stages1/2, and
no-yield compaction. It checks all History fields, booleans, non-yielding full
initial updates and point0 preservation. These tests have NOT run yet.

Remaining risks: shape-dependent floating rounding/valid flags; extra local
hardening work in copied padding point0; dense-yielding batches may cost more
than the original vector path. No tolerance or global state mask may be
relaxed. The real105->106 source-state comparison, followed by a bounded
runtime measurement including the109 subdivision threshold, remains required
before using this optimization in dynamics.
