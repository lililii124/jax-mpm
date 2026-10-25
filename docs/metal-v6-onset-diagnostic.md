Bounded onset experiment, 125–250 ns
==================================

This experiment attributes the frozen v6 trajectory. It makes no velocity,
constitutive, contact, geometry, damage, or transfer repair. Its output cannot
serve as a production checkpoint or a validated fragment cloud.

The existing solver already uses APIC. The NumPy analytical benchmark reproduces
translation, rigid rotation and a general affine field with mass, linear and
affine angular momentum and kinetic energy errors below 1e-12. PIC loses angular
momentum and energy in the rotation test. The reported FLIP identity covers a
zero-force grid increment only; it provides no evidence for contact or stress.

Next cloud experiment, on the exact producer Python/JAX runtime and restored
seed, after the publisher pins these isolated files in its monitored job:

```text
python -m examples.metal_fragment_v6_onset_probe --parent <seed>/prefix-proof --admission tools/metal_v6_energy_probe125_admission.json --stop-ns 135 --max-steps 512 --output <fresh>/onset135
```

If that passes, repeat with `--stop-ns 250` and a new output directory, within the
existing USS, wall, disk and quota limits. Do not run the full impact locally.
The frozen parent/source hashes and exact producer runtime remain mandatory.
The existing probe125 cloud launcher does not copy the new modules automatically;
its source inventory, file copying and admission must be updated before dispatch.

Admission requires the actual adaptive counts to agree, all 35 final carries to
match bitwise, valid endpoint and contact residuals <=0.005 m/s, positive recorded
time steps covering the requested interval, unchanged mass arrays and sources,
and the nine-channel reservoir identities within the existing energy tolerance.
It does not prescribe eight steps or infer adaptive counts from frame spacing.
`adaptive-trace.npz` stores every actual dt plus nine signed channels, while
`equivalence.json` retains signed and absolute aggregate channels.

`physical-audit.json` includes APIC affine angular momentum, linear momentum minus
the stored boundary impulse, and the original reference-bond topology with
irreversible failure/opening. The frozen carry contains no boundary torque:
angular momentum change is reported explicitly as informational. A future repair
needs measured boundary/contact torque and a matching analytical/contact test
before claiming angular conservation or production admission. Components alone
do not validate fragments, and this procedure creates no surface geometry.

Local verification: exact saved125 ns preflight passed without JAX advancement;
three NumPy analytical/topology tests passed. No 125–250 ns mechanics have been
executed by this change. No repair is admitted.
