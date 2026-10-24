# JC refinement continuation

`jc-continuation.yml` advances an admitted 341,584-particle refinement by ten
saved-frame intervals. The physical problem remains grid 100, ppc 3, 60 output
intervals, 57,900 steps and 150 microseconds, with the original material, waist,
grips and CFL 0.15. Partial endpoints 20, 30, 40 and 50 correspond to 50, 75,
100 and 125 microseconds. Frame 60 requires a separate complete-result audit;
this controller does not bypass the fracture or publication checks.

The workflow has two source trees. The controller is checked out at the
dispatched commit. Numerical modules are imported exclusively from
`08d83a051a9764a5a218fd33e184f3291eb3dd1c`, the source of the initial 25-microsecond
run. The original producer identity, full plan and numerical signature remain
unchanged. Every tracked source byte is checked against the original manifest
and fixed Git tree before and after execution. The new controller has its own
source record and receipt chain; its commit is never substituted for the old
solver commit.

The dispatch inputs are the exact previous cache key, independently checked
receipt SHA256, previous frame and next frame. A cache miss, wrong receipt,
altered source, incomplete prefix or invalid numerical state stops the job.
There is no fallback to a fresh solve. The parent cache is retained unchanged
inside the job, while the working result receives new output. Previous
`frames_data` files and the initial plan must remain byte-for-byte identical.
The latest complete State and History arrays and every saved frame are audited
before and after advancing. Plastic strain and material-section force are
independently reconstructed from the latest checkpoint; intermediate values
are original observer records because those frames do not store F or plastic
strain. Positive-weight axial grid-support cuts are reported separately from
material failure. Events first appearing in the same saved interval do not
establish substep ordering.

The first job recorded Python, operating-system family, machine architecture
and the versions of JAX, jaxlib, NumPy, SciPy and psutil. Those recorded values
are pinned and checked. CPU identity is recorded for each job; a CPU change
does not relax reference-array or plan checks. Dependencies not recorded by
the first job cannot be retroactively certified. Reference arrays are rebuilt
by the unchanged constructor and must match every original array descriptor;
any floating-point sampling difference is rejected, not silently substituted.
The continuation also records the complete installed package set, excluding
packaging tools and the separately pinned project source. Subsequent
continuations pin and verify that additional lock.

The process-tree guard limits USS to 12 GiB and stops below 2 GiB system
available memory. A child solve has at most 265 minutes and receives SIGINT
before a forced timeout. The complete carry is limited to 1 GB and contains
the latest checkpoint, every original saved-prefix NPZ, metadata and receipt
chain. Full-size mmap arrays are reconstructed by the original `common.run`
and are not cached. No partial state file is presented as a usable checkpoint.
A confirmed cache produces a records-only artifact of at most 8 MiB; a failed
handoff prioritizes complete diagnostic state within 192 MiB.

Implementing this controller and passing its small tests do not establish
cross-job numerical equivalence. An actual admitted continuation will record
that it was executed, while equivalence to an uninterrupted refined solve
remains a separate question. Partial jobs produce no rendering, no completed
fracture claim and no homepage entry.
