Temporal sensitivity workflow for publisher review
=================================================

The admitted125–250 ns frozen baseline has102 adaptive steps. The following
experiment restarts independently from the authenticated125 ns carry at CFL
0.1,0.05,0.025. It preserves the existing geometry, particles, material,1024
contact iterations,0.005 m/s tolerance,125 ns topology epoch, and numerical
reservoir accounting. Each cell runs both original and instrumented mechanics
and requires all35 carries bitwise equal. It does not change transfer methods.

Source preparation (no mechanics, works on Windows):

```text
python -m tools.run_metal_v6_dt_convergence --parent <restored>/prefix-proof --admission tools/metal_v6_energy_probe125_admission.json --source-admission <new-source-admission.json> --output <unused> --prepare-source-admission
```

The publisher must review/pin this manifest in the dispatched source, restore the
whole authenticated seed using its existing verified transport, and install the
exact producer runtime. Reuse its existing single-job concurrency group and
quota observer: latest reported usage76, stop90. No new parallel numerical jobs.
The runner itself has no quota API; the publisher must refuse dispatch if the
whole timeout plus packaging reserve would cross that limit.

Linux numerical step, first on125–135 ns:

Before any mechanics, execute inside the exact overlaid runtime-source tree:

```text
python -m tools.smoke_metal_v6_dt_runtime --runtime-source <runtime-source> --source-admission <reviewed-source-admission.json> --frozen-admission tools/metal_v6_energy_probe125_admission.json --output <fresh-smoke.json> --numerical-imports
```

This imports every runtime dependency without advancing mechanics, rejects project
imports from outside that tree and controller-chain dependencies, verifies frozen
and overlay hashes, and freezes every Python source before/after import. Run it
under the publisher's existing bounded numerical process monitor. The guard is
self-contained in tools/metal_v6_process_guard.py; no recovery-controller imports
are needed. The runner requires this numerical smoke receipt and rejects source
changes between smoke and execution. A local nonnumerical smoke is insufficient.

```text
python -m tools.run_metal_v6_dt_convergence --parent <restored>/prefix-proof --admission tools/metal_v6_energy_probe125_admission.json --source-admission <reviewed-source-admission.json> --smoke-receipt <fresh-smoke.json> --output <fresh-output> --stop-ns 135 --wall-seconds 700
```

Only after acceptance and a renewed publisher quota check, repeat at250 ns in a
fresh directory. The runner executes cells sequentially in separate process
sessions and enforces aggregate wall time,6 GiB owned-session USS,1.75 GiB raw
output, and3 GiB disk reserve. It kills the owned session on failure and preserves
partial logs/states plus monitor.json. The workflow must upload whole raw output
on failure using the existing bounded transport; do not delete failed evidence.
Set the outer step timeout above the requested aggregate wall cap plus cleanup,
and reserve packaging time in the job timeout. The dedicated publisher is
tools/publish_metal_v6_dt_convergence.py and its template is
.github/workflows/metal-v6-dt-convergence.yml. The template permits only125–135 ns.
Publication and dispatch require the publisher's renewed quota check and a
byte-identical registered workflow; this document dispatches no jobs.

The comparison validates channel trace sums and interval coverage and reports
all nine signed/absolute channels, measured adaptive step counts/dt ranges,
successive channel differences and changes in x,v,C,F,damage. It describes
whether successive differences shrink, without asserting a Richardson order or
a converged limit for nonsmooth contact/damage. Smaller CFL can increase APIC
transfer loss by introducing more projections; this is an empirical question.
No candidate is automatically admitted because its numerical reservoir is lower.

Spatial or particle refinement is deliberately not run from125 ns: the saved
constitutive histories, APIC affine momenta, contact carry and reference-bond
topology cannot be conservatively refined by an authenticated mapping currently
available here. Grid-only refinement with unchanged particles also changes
particles per cell and the represented affine kinetic energy dx²||C||²/8; it is
not a clean joint convergence test. A defensible spatial/particle study must
restart from the same analytic0 ns geometry and independently validate mass,
linear/angular momentum, energy, boundary/contact and reference topology at each
resolution. This exceeds the bounded authenticated125 ns restart scope.

Neither the runner nor comparison permits production restart, validates fragments,
weakens contact/thermodynamic gates, or produces geometry/media.
