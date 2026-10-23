# Finite deformation examples

`examples.visible_motion` defines a separate set of loading cases for examining motion at the actual geometric scale. The original small-amplitude benchmarks and their validation records remain separate. The cases below are definitions; measured deformation and refinement errors must come from completed runs.

All five cases use three-dimensional compressible neo-Hookean elasticity with Young's modulus 20 kPa, Poisson's ratio 0.2, density 1000 kg/m³ and zero gravity. Lengths are in metres and times in seconds. The background grid covers `[0, 1]³`. Formal runs use three particles per cell in each direction and 50 output intervals.

## Geometry and loading

| Case | Initial body bounds | Constraints and loading | Duration |
| --- | --- | --- | --- |
| Transverse wave | `(0.125, 0.35, 0.375)` to `(0.875, 0.55, 0.625)` | All grid nodes have `vx = vz = 0`; nodes with `x ≤ 0.175` have all velocity components fixed. Initially, `vy = 0.6 sin(π clip((X − 0.175)/0.7, 0, 1)/2)` m/s. The right end is free in y. | `4 × 0.7 / sqrt(μ/ρ)`, about 0.970 s |
| Cantilever | `(0.15, 0.4, 0.35)` to `(0.8, 0.6, 0.65)` | Nodes with `x ≤ 0.2` are fixed. A translating reference box `(0.72, 0.3, 0.3)` to `(1, 0.7, 0.7)` prescribes only y velocity, raising its position by 0.12 m at 0.6 s and returning at 1.2 s. Its x and z velocity components remain free. | 1.4 s |
| Surface load | `(0.2, 0.45, 0.2)` to `(0.8, 0.55, 0.8)` | Four edge strips are fixed: `x ≤ 0.25`, `x ≥ 0.75`, `z ≤ 0.25` or `z ≥ 0.75`. The reference top face receives a downward nominal traction with a peak of 1 kPa: 0.3 s rise, 0.3 s hold, 0.3 s fall and 0.3 s rest. | 1.2 s |
| Shear | `(0.3, 0.25, 0.3)` to `(0.7, 0.75, 0.7)` | Nodes with `y ≤ 0.28` are fixed. A translating reference box `(0, 0.72, 0)` to `(1, 1, 1)` prescribes velocity `(0.5, 0, 0)` m/s after a 0.08 s half-cosine ramp. Its prescribed translation reaches 0.18 m at the final time. | 0.4 s |
| Torsion | `(0.15, 0.4, 0.35)` to `(0.8, 0.6, 0.65)` | Nodes with `x ≤ 0.2` are fixed. The reference box `(0.73, 0.3, 0.3)` to `(1, 0.7, 0.7)` rotates around the x axis through `(0, 0.5, 0.5)`, reaching 90° at 0.5 s and returning at 1.0 s. All three velocity components in the rotating box follow its rigid motion. | 1.2 s |

The cantilever and torsion drives use a half-cosine displacement cycle, `q(t) = qmax [1 − cos(2πt/T)]/2` for `0 ≤ t ≤ T`. After the cycle, the driven velocity components remain fixed at zero in the original box. They are not released into free vibration. The shear drive continues moving through its final recorded time.

The surface-load force is integrated over the **reference** top-face area of 0.36 m² and acts in the fixed negative y direction. Its rise and fall are half-cosine ramps. This is a prescribed nominal load, not a pressure that follows the deformed surface normal. The fixed edge strips are bilateral grid velocity constraints.

## Wave comparison

The wave runs use grid sizes 32, 48 and 64, plus a grid-48 run with half the nominal time-step factor. Geometry, material, particle sampling density, initial velocity, boundary conditions and final time are shared. The default step limit is `dt ≤ 0.12 dx / cp`, where `cp = sqrt((K + 4μ/3)/ρ)`; each run divides its final time into equal output intervals and an integer number of steps per interval. The half-step variant uses a factor of 0.5 before this rounding.

This is a finite-amplitude transverse initial mode with a fixed left section and a free right end. It is not the previous two-ended longitudinal standing-wave benchmark. Comparisons should use the newly measured displacement history, deformation gradient, support constraints and grid/time-step differences. Previous axial-wave error values do not apply.

After all four formal wave runs have passed their individual reviews, compare their saved tip-displacement histories:

```powershell
python tools/review_wave_comparison.py outputs/visible-motion
```

The tool writes `outputs/visible-motion/wave_comparison.json`. It checks each `interpretation.json`, its check flags and source-file SHA values before comparing the complete material, geometry, loading and boundary definitions. It requires matching output schedules, allowing at most `1e-12` s of accumulated floating-point time difference, and records that difference. It reads only JSON and CSV files.

The report includes each tip-y displacement history, its positive and negative peaks and RMS value, the grid-32 and grid-48 RMS differences from grid-64, and the grid-48 half-step difference from the base run. RMS values use all equally spaced saved frames, including both endpoints, without interpolation. Whether the grid difference decreases is reported as an observation; grid-64 is a numerical comparator and the report does not establish an analytical error or a convergence order. The half-step comparison alone does not establish a temporal convergence trend.

## Observations

Every run saves maximum and RMS displacement, minimum and maximum `J = det(F)`, and fixed-region displacement and speed. Fixed-region checks use only initial particles whose complete quadratic B-spline support lies inside a fixed strip; the count and selection rule are saved in `definition`. A coarse smoke run can have no such particles, in which case the associated observations are null.

Additional fields track free-end y displacement and the constrained x/z motion for the wave; top translation for shear; driven-end translation for the cantilever; signed end rotation around x for torsion; and centre deflection, prescribed load and force-assignment error for the surface-load case. Prescribed targets and measured particle responses are recorded separately.

## Run

From the repository root in PowerShell:

```powershell
foreach ($grid in 32, 48, 64) {
    .\.venv-public\Scripts\python.exe -m examples.visible_motion --case wave --grid $grid --ppc 3 --frames 50 --no-render --resume
}
.\.venv-public\Scripts\python.exe -m examples.visible_motion --case wave --grid 48 --dt-factor 0.5 --ppc 3 --frames 50 --no-render --resume
foreach ($case in 'cantilever', 'surface-load', 'shear', 'torsion') {
    .\.venv-public\Scripts\python.exe -m examples.visible_motion --case $case --grid 48 --ppc 3 --frames 50 --no-render --resume
}
```

Outputs are under `outputs/visible-motion/<case>/neo-hookean`. The four wave names are `wave-grid32`, `wave-grid48`, `wave-grid64` and `wave-half-dt`. Run these jobs sequentially within available memory.

Rendering uses ParaView, physical particle coordinates and a geometric scale of one. Speed and displacement magnitude are calculated from the saved states; a fixed colour range is shared by all frames of each animation. Axes, annotations and colour-bar text are hidden. No geometric displacement multiplier is applied.
