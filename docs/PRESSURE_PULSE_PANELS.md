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
convergence validation.

## Completed JC pressure history

The JC preset has completed 6680 steps through 200 microseconds. Its 335 saved
states preserve the prescribed four-edge clamps, positive deformation
determinants and valid material history. Mass is constant at 1.076976 kg;
peak displacement is recorded in the [result summary](results/completed--pressure-panel-jc/summary.json).
The maximum damage is 0.304875 and temperature is 307.483 K. This is a bending
and plasticity example, without material separation or an experimental
accuracy claim. The [GIF](images/home/completed--pressure-panel-jc.gif) uses
61 actual saved states at their unscaled positions, coloured by displacement.

The original run wrapper exited after its final checkpoint while compressing
a duplicate combined trajectory: mapped history pages exceeded the 6 GiB
private-memory limit. The final full checkpoint and all 335 frame files passed
an independent read-only review. Original failed-wrapper flags are retained
in `evidence/pressure-jc-complete`, alongside its source bridge, diagnostics,
frame hashes and review script. This establishes completion of the numerical
protocol, without claiming a complete APIC energy certificate or convergence.

Pressure examples now retain individual frame NPZ files and omit the redundant
combined archive. `examples.common.run` keeps archive writing enabled by
default for other examples through its `write_trajectory_archive` argument.
When both archive writing and rendering are disabled, the runner keeps one
transient frame buffer instead of loading prior particle histories into a
full-course cache; resume still validates every saved frame and checkpoint.
Enabling either archive writing or rendering retains full-course history.
The output change does not alter solver steps, material parameters or loading.
The RHT full-history result remains in progress.
## Completed HJC pressure history

The HJC demonstration preset completed 7000 steps through 200 microseconds.
Its terminal checkpoint is frame 350 and the saved course contains 351 states.
The independent read-only audit verified the terminal checkpoint and every
actual frame, including finite particle fields, positive mass and volume,
positive deformation determinants, and valid material history. Mass remained
0.948672 kg; J ranged from 0.981127 to 1.057205; maximum damage was 0.613042,
maximum plastic strain was 0.026484, and maximum temperature was 293.15 K.
The observer reports zero motion at fully supported clamps. Particles in the
broader nominal clamp strips moved by up to 2.51e-5 m. The [result summary](results/completed--pressure-panel-hjc/summary.json)
and [GIF](images/home/completed--pressure-panel-hjc.gif) retain these scoped
measurements. The 400x400 GIF contains 61 distinct actual saved frames with
the original displacement scale and no axes or text.

This run uses the repository's HJC `legacy-v3` demonstration preset; it is not
MAT111-compatible. It is not a material validation, an experimental comparison,
a spatial or temporal convergence result, or a complete discrete energy
certificate. The original completed status and summary are preserved. The
retained `segment.json` still describes the preceding 6820-step continuation;
the final checkpoint manifest and terminal review independently establish
the 7000-step endpoint. Compact review evidence and per-frame hashes are in
`evidence/pressure-hjc-complete`; binary checkpoints and source NPZ archives
are not published.


## Additional pressure severities: JC and HJC backside damage

These two completed runs use stronger assumed pressure severities than the
50 MPa JC and 15 MPa HJC examples above. They retain the same 180 × 12 × 180 mm
panel, 128 grid, two particles per cell per axis, four edge clamps, 80 µs
sine-squared pulse and 200 µs observation interval. The pressure is applied to
the +Y reference material face. These values are engineering assumptions, not
experimental pressure calibration.

| Material | Peak pressure | Steps | Minimum J | Course-wide particle damage maximum | Backside damage maximum across 61 rendered saved frames |
|---|---:|---:|---:|---:|---:|
| JC aluminum | 75 MPa | 6,680 | 0.976690 | 0.678250 | 0.514463 |
| HJC legacy-v3 | 25 MPa | 7,000 | 0.968503 | 1.000000 | 1.000000 |

All saved material histories were valid and every saved deformation
determinant was positive. Each 400 × 400 GIF shows the minus-Y backside,
opposite the loaded face, at a fixed damage range of 0–1. It samples 61 actual
saved states; the connected material-point surface is rendered at 1600 × 1600
with FXAA, then reduced with Lanczos. The damage values are the saved particle
values interpolated over connected surface faces; there is no damage smoothing.

The course-wide damage maximum includes all particles and is not a surface
measurement. Backside maxima above refer only to the 61 rendered samples. The
[JC summary](results/completed--pressure-panel-jc-75mpa-damage/summary.json),
[HJC summary](results/completed--pressure-panel-hjc-25mpa-damage/summary.json),
per-frame source hashes and surface checks are included with each result.
The binary particle histories and PNG intermediates are not published; the
compact render records retain hashes for the selected source NPZ frames. These
runs do not add crack-face topology, erosion, fragment pressure or explosive
flow coupling. They are not experimental validations, and spatial/temporal
convergence has not been established.

## Completed JH-2 pressure history

The JH-2 plate completed 8460 steps through 200 microseconds under an assumed
80 MPa, 80-microsecond sine-squared pressure pulse. The terminal checkpoint
and all 424 saved states passed the independent audit. Maximum displacement
was 20.133 mm and maximum damage reached 1; mass remained 0.972 kg.
The observer reports zero motion at fully supported clamps. Particles in the
broader nominal clamp strips moved by up to 0.429 mm.

The [result](results/completed--pressure-panel-jh2/summary.json) and
[61-frame GIF](images/home/completed--pressure-panel-jh2.gif) use actual saved
coordinates. This is an idealized pressure-load example, without coupled
explosive flow or explicit crack surfaces. Experimental accuracy and full
space/time convergence are not established. Original records and frame hashes
are retained in `evidence/pressure-jh2-complete`.
