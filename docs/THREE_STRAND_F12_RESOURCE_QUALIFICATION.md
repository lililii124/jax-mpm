Standard Linux exact replay qualification, separate from physical admission

The original f10->f12 failed-run solver monitor reports peak process-tree USS
2,603,061,248 bytes (2.424 GiB), 502 complete samples and 509.473 s until its
physical/numerical validity rejection. Its minimum available RAM was
13,362,851,840 bytes. The admitted f10 receipt records peak USS 2,740,887,552
bytes (2.553 GiB). These measurements do not prove that a new diagnostic JIT fits
a smaller runner. Actual CPU count/utilization was not retained; original source
sets OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=1, which are settings, not a CPU
benchmark.

The new standard-linux-qualification profile requires Linux, >=2 CPUs in process
affinity and >=5 GiB currently available RAM. Its 4 GiB tree-USS ceiling is about
1.57 times the larger measured original peak. Its continuously sampled 1 GiB
system-available reserve explicitly protects the runner outside the child tree.
The probe aborts on either bound, monitor failure or 720 s wall time. It samples
every0.2 s; a fast OS OOM remains possible between samples and must be reported as
resource failure, never physical rejection. The overall Actions job remains20 min.
The old large-original 8CPU/12GiB profile remains the default for standalone calls.
Windows is rejected by the qualification profile before JAX import.

This is an experiment to qualify standard runner resources, not a claim that they
are already qualified. No constitutive parameter, dt, grip motion, contact law,
coverage/extent criterion, energy gate or original numerical function changes.
At most32 actual exact steps are attempted from the original f12/step3500 bytes.
No failure within this bound means the predicate remains undetermined. The raw
artifact is diagnostic-only and remains never-resumable, regardless of outcome.

Publisher layout: copy the Python and original-pins.json files from this directory
into tools/three_strand_exact_probe/, then copy three-strand-exact-probe.yml to
.github/workflows/. Register those exact workflow bytes also as
.github/workflows/engineering-previews.yml on fixed feature ref
refs/heads/work/three-strand-f12-exact-probe in lililii124/jax-mpm. Review/commit
them together. bind_controller.py verifies each actual Git blob and file bytes,
including registered override equality, fixed repository/ref and workflow path.
The workflow admits the controller
only when selected commit == workflow source SHA == checkout HEAD, records all
controller file hashes, and separately checks out original numerical commit
253e11d7762a6914c3b5f225d1e646d8622c548f. It verifies19 original source file hashes,
the pinned raw manifest/full archive/each member/checkpoint, exact recovered runtime,
and valid checkpoint f12/step3500. It rechecks original source/runtime after the
probe. The complete downloaded original diagnostic delivery and generated scalar
evidence are uploaded together on success/failure. No point of this workflow calls
production restore/admission, writes a continuation checkpoint, or publishes media.

Guard hardening: launch in an owned Linux session with subreaper enabled; use two
matching complete session/PID-creation-time inventories for each USS sum, including
the controller once. Never skip a disappearing live member and accept a partial
sum. Inventory failure aborts. Final cleanup always kills the owned group even
after leader exit, checks remaining session members and reaps adopted children
within a20 s deadline. Cleanup failure yields nonzero status and explicit evidence.
