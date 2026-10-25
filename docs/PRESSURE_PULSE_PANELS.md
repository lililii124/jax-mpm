# Clamped material panels under a pressure pulse

`examples.pressure_pulse_panels` supplies JC, HJC, JH-2 and RHT presets through
the existing `Case` and `examples.common.run` JAX integration/checkpoint path.
The default is a 180 × 12 × 180 mm plate in a 240 mm computational domain,
grid 128 and two particles per cell per axis: approximately 480,000 particles.
Four 10 mm edge strips are fixed in all velocity components through the plate
thickness, leaving a 160 × 160 mm free span. There is no backing plate or
bottom support. Gravity is omitted over the short dynamic interval.

The assumed mechanical input is a spatially uniform positive pressure
`p(t)=p0 sin²(pi t/T)` for `0<t<T`, zero afterward. Defaults use
T=80 µs and a 200 µs observation interval, with material-specific assumed
engineering severity:

| Preset | Peak pressure | I=p0 T/2 | I/(rho ×12 mm) |
|---|---:|---:|---:|
| JC aluminum | 50 MPa | 2000 Pa·s | 60.17 m/s |
| HJC concrete | 15 MPa | 600 Pa·s | 20.49 m/s |
| JH-2 ceramic | 80 MPa | 3200 Pa·s | 106.67 m/s |
| RHT concrete | 15 MPa | 600 Pa·s | 21.61 m/s |

The last column is an unconstrained flat-sheet impulse/areal-mass scale,
not a prediction of the clamped panel's deformation or speed. These loads
provide appreciable dynamic bending/plasticity opportunities; the actual
response and admissibility require calculation. `--pressure` overrides p0.
This is a prescribed equivalent
load, without explosive chemistry, JWL, surrounding gas or blast-flow coupling.
The setup is not experimentally or IJIE calibrated.

The loaded surface is the topmost reference material-particle layer (+Y).
Its discrete area is `A0=V0/dY`, using the actual seeded layer thickness;
the weights sum to 0.0324 m². Newton forces are
`f=-p(t) A0 cross(F e_z,F e_x)=-p(t) A0 cofactor(F)e_y`.
This follower traction uses the current surface normal and area. At F=I the
JC peak resultant is 1.62 MN toward −Y, with continuous reference impulse
64.8 N·s; the other presets scale with their tabulated pressure impulse.
The metadata also records the actual left-end sampled reference impulse for
the selected dt. Other unrestrained faces are traction-free. The original
material face remains the loaded face through local damage; no resolved
crack-surface, erosion or fragment-pressure model is added.

## Run and recover

From a source checkout with Python/JAX installed:

```bash
python -m examples.pressure_pulse_panels --model jc --grid 128 --ppc 2 --stop-after-steps 20
python -m examples.pressure_pulse_panels --model jc --grid 128 --ppc 2 --resume --stop-after-steps 40
```

Choose `--model hjc`, `jh2` or `rht` for the other presets, with separate output
directories selected automatically. Omit the stop flag only when deliberately
requesting the full physical protocol. Stops are multiples of 20; resuming uses
the same output, material, source hashes, loading and full time plan. The step
is at most the original 0.1 dx/c estimate and rounded down to fit that plan.
Use bounded Linux resources for these defaults. No rendering is requested.

JC reuses `examples.material_engineering.tensile_aluminum` and its documented
AA2024-T351 parameters and thermal assumptions. Its existing volume law is
`p=K(1/J−1)`, linear in relative density increment; plastic work heats the
material with the existing heat fraction, and normalized thermal softening is
clipped to [0,1]. This is not a claim of equivalence to the separate 5 km/s
metal solver. HJC, JH-2 and RHT retain the public `material_model` demonstration
factories, original EOS, return maps and validity gates. HJC uses `legacy-v3`;
profile scopes and material evidence are in [MATERIALS.md](MATERIALS.md).
Material parameter sources are reused; the new loading is an explicit assumption.

`case.json` preserves geometry, loading, source hashes and the original plan.
The standard atomic checkpoints preserve all particle/material history fields.
Observations record actual applied work/impulse, pressure force and area,
central deflection/velocity, clamp motion and material energy, plastic strain,
damage, temperature, pressure and compaction. History energy/work fields are
model-specific; APIC affine kinetic and the full constraint/contact budget are
not combined into a conservation certificate. Numerical losses are not heat.
Local softening has the existing mesh dependence and failed mass is retained.

The targeted `tools.check_pressure_pulse_loading` checks only discrete area,
time impulse and rotation of the follower direction. Twenty-step large-scale
runs are resource prefixes, not completed pressure histories or physical/
convergence validation. Full execution, response review and refinement remain
separate work. No pressure-panel GIF is published for these prefixes.
