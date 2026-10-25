# RHT continuation profile

The original frame-three state contains 266,295 particles at 5 microseconds.
Its following 35-step batch exceeded three hours. The warm-up had already
finished, and the measured process stayed below its memory ceiling.

This diagnostic restores the same geometry, materials, constraints and state.
It dispatches individual existing coupled steps and records the RHT EOS energy
iterations, adaptive subdivision attempts and scalar root iteration counts.
Removing the trace statements and their explicit temporary restores the
original material source bytes. No constitutive tolerance, failure rule or
particle subset is changed. Trace callbacks and single-step dispatch affect
timing, so these measurements locate the expensive stage rather than establish
a production speedup or a new accepted penetration result.

The full original multipart archive is verified before extracting its state and
source. The probe is bounded to 1,200 seconds and 3 GiB USS, with a cgroup-aware
available-memory check. It retains the trace and last returned state on failure.
The next optimization must follow the measured bottleneck; global maxima and
global validity reductions are currently only candidate causes.
