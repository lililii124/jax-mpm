# Matrix-free JAX projection status

These are algebraic checks of one saved input, not additional trajectory advancement. The input is accepted22 at171.951403430ns (SHA3579600d…530809); the same midpoint geometry and golden projection are preserved in the source-bound raw artifact. The separately accepted physical step23 remains the original1d2fda7 run at172.640726878ns (CPa11ab92a…d726bc).

The native operator reproduces the original quadratic reconstruction, physical centroid/affine contact and polynomial modes. For26,336 particles and146 contact rows, the native-contact check passes against the original sampler: relative sample/adjoint/mobility differences1.22e-14/2.08e-16/1.73e-15. The full QP differs from the saved golden velocity by9.85e-12 relative; original-unit KKT residual is9.36e-8m/s and physical torque is about1e-22kgm²/s. Contact storage retains every actual row and all540 adjacent particles.

| Kernel/source | Preparation | Compile | QP solve | Peak USS |
|---|---:|---:|---:|---:|
| Dynamic-array JAX, reference LDL (`853f228`) |38.34s |1.52s |78.10s |3.95GB |
| Compact exact triangular factor (`b1cd02c`) |31.55s |6.02s |61.45s |2.85GB |
| Native tensor reconstruction (`7cc4277`) |15.85s |6.67s |58.21s |1.47GB |
| Native contact and modes (`8b5b946`) |18.65s |6.55s |49.42s |1.43GB |

The first compilation failure under the9GiB USS guard is retained. Large closed-over arrays caused it; dynamic operator arguments resolved that resource issue without changing equations. Exact sparse LDL preparation remains outside JAX in the passing variants. These timings do not establish an affordable production impact step.

Two bounded attempts removed the external factor. Both stopped after the single saved physical-impulse projection; neither attempted the QP, material update or dynamics:

| Native preconditioner |64-iteration normal residual | Combined-state relative difference | Actual work difference | Outcome |
|---|---:|---:|---:|---|
| Column scaling plus exact body-mode deflation |3.49e-4 |8.67e-6 |+2.03e-5J |Rejected in9.17s |
|4³ local blocks,8-color SSOR |4.46e-6 |3.82e-6 |−3.20e-7J |Rejected in147.17s |

SSOR used682 blocks and the existing1e-8 shift only in its preconditioner; objective regularization was zero. Its right-factor transpose defect was4.65e-18, but physical accuracy was insufficient and one projection took139.58s. No iteration cap or original physical gate was relaxed. No failed projection was used to advance a state or to create heat. Further long runs on this path are not justified by these results.

There is also an independent time-integration caveat: a newly active midpoint velocity constraint can turn an initial closing speed+1 into midpoint0 and reflected endpoint−1. Endpoint e0 projection then leaves the opening velocity unchanged, so initial feasibility in the old cone does not prove e0 restitution for new rows. Passing the frozen convex QP does not certify that activation law. A future nodal-diagonal method should separate force/material drift from instantaneous impact work, or resolve the activation event explicitly.

The full raw states and rejected results remain on the isolated Linux tasks. JSON receipts here preserve source commits, resource outcomes and actual numerical failures. No fragment media or long production continuation was generated.
