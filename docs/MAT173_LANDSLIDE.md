# MAT173-inspired toe excavation

<table width="100%">
<tr>
<td width="33%" align="center"><img src="images/home/mat173-toe--plastic-activity.gif" alt="Plastic activity" width="100%"><br>Plastic activity</td>
<td width="33%" align="center"><img src="images/home/mat173-toe--shear-softening.gif" alt="Shear softening" width="100%"><br>Shear softening</td>
<td width="33%" align="center"><img src="images/home/mat173-toe--displacement.gif" alt="Displacement" width="100%"><br>Displacement</td>
</tr>
</table>

A homogeneous slope settles under self-weight, then loses its toe through actual
particle removal. The animations show the next 0.09 s with actual deformation.
This completed exploratory calculation shows localization and motion over a short
interval; it does not establish a through-going slip surface or developed runout.

## Setup and trigger

A strictly 2D grid assumes unit thickness. Soil tensors are embedded in 3D for
plane strain (Fzz = 1, no off-plane shear). The 4 m domain has 160 cells per
side, grid spacing 0.025 m, soil spacing 0.0125 m, time step 0.00003 s, and no F-bar.
Soil occupies x = 0.2–3.0 m above y = 0.5 m, with a crest at y = 1.8 m up to
x = 1.0 m and a descending face of slope 1.3/2 (about 33 degrees).
The elastic foundation spans x = 0.2–3.3 m, y = 0.2–0.5 m, with 0.025 m
particle spacing, E = 50 MPa, nu = 0.28, and density 2500 kg/m³.
Foundation nodes at y <= 0.3 m are fixed; both bodies have a frictionless left
wall at x = 0.2 m. Nonadhesive soil–foundation contact has Coulomb coefficient 0.7.

Both bodies start stress-free. Gravity ramps to 9.81 m/s² over 0.02 s;
settling multiplies particle velocity and affine C by 0.995 per step.
Three consecutive 100-step windows must pass the recorded speed, drift, net-force,
kinetic-energy, and support thresholds, followed by 100 undamped verification
steps. The run passed at step 2100 (0.063 s): soil p95 speed 0.002955 m/s and
contact support/soil weight 1.00572. This is numerical balance, not an in-situ measurement.

Every soil particle with reference x >= 2.2 m is then removed over the full
locally occupied height, leaving a vertical face about 0.52 m high. Removed
mass is 374.34375 kg/m and reference area 0.20796875 m². Particle mass, volume,
and histories are removed together; retained histories are preserved without reset.
The trigger imposes no velocity, weak stripe, or global strength change.
The calculation continues for 3000 undamped steps.

## Material and plastic mechanisms

| Soil parameter | Value |
| --- | --- |
| Shear modulus / Poisson ratio / density | 5.86 MPa / 0.28 / 1800 kg/m³ |
| Peak / residual friction | 38 / 20 degrees |
| Peak / residual cohesion | 1000 / 0 Pa |
| Peak / residual dilation | 2 / 0 degrees |
| Rankine tensile strength T0 | 0 Pa |
| Characteristic length L / fracture slip Dc | 0.0125 / 0.0005 m |

The selected Mohr–Coulomb active-set return includes a simultaneous Rankine
cutoff. Softening is active during settling and excavation. The shear driver κs
sums positive MC surface multipliers weighted by sqrt(2/3) times the deviatoric
flow-direction norm; this specifies a selected multi-face corner definition.
D = clip(κs L / Dc, 0, 1) interpolates peak to residual shear strengths.
Rankine tensile plasticity accumulates independently from total plastic trace
minus the MC contribution; it is diagnostic and neither drives D nor reduces T0.
Ambiguous zero-stress partitions are explicitly recorded in the result JSON.

At residual strength κs freezes, while total equivalent plastic strain can keep
accumulating through residual shear or tensile flow. D = 1 is saturated shear
softening, not particle deletion. The primary image therefore uses total plastic
increment rather than κs increment. These selected extensions and the corotated
logarithmic-strain MPM host are not a validated native MAT173 implementation.

## Images and limits

The 31 frames span steps 2100–5100 (absolute time 0.063–0.153 s), every
100 steps (0.003 s). All images use the same fixed parallel camera and a
blue-to-red scale. Plastic activity is total equivalent plastic strain increment
over the preceding window (initial frame zero), range 0–0.00602356018;
shear softening is D, range 0–1; displacement magnitude is relative to the
post-excavation initial coordinates, range 0–0.01322654882 m. The terminal plastic
increment maximum 0.00629276591 slightly exceeds the fixed color ceiling.
Each particle owns an independent reference-volume square transformed by actual
Fxy, with scale 1 and no smoothing, gap filling, or neighbor-ID mesh. Positive
areas and area = reference volume × det(Fxy) are checked; no failed flags occur.

All saved windows passed finite-state, positive-J, and constitutive-validity guards.
Limits are maximum absolute total Hencky strain 0.05 during balance / 0.2 after
excavation, incremental strain 0.000625, and soil elastic principal strain from
compliance 0.0035. Terminal maxima are 0.0725101, 0.0000553258, and 0.00163620,
with minimum J = 0.998382. Duration completion is not equilibrium: terminal
soil p95 speed is still 0.16183 m/s and the balance criteria no longer pass.
There is no pore pressure, groundwater, joints, erosion, site calibration,
spatial-convergence study, mesh-objective band width, or certified energy closure.
The example supports qualitative inspection, not engineering safety assessment.

## Reproduce

From an installed checkout (the numerical run can be expensive):

```bash
python -m examples.mat173_toe_excavation --dry-run
python -m examples.mat173_toe_excavation --output outputs/mat173-toe
pvpython tools/render_mat173_toe.py --input outputs/mat173-toe --output outputs/mat173-toe-render
```

The renderer requires ParaView's pvpython with NumPy and Pillow. It reads example
frames/checkpoints directly, writes three GIFs, and records a render receipt.
It uses the original camera, LUT, fixed ranges, 800-to-400 Lanczos reduction,
and 100 ms GIF timing. Library/platform differences can affect exact hashes.
The [model](results/mat173-separated-mechanism-toe-v1/model.json),
[summary](results/mat173-separated-mechanism-toe-v1/summary.json),
[ledger](results/mat173-separated-mechanism-toe-v1/excavation-ledger.json), and
[provenance](results/mat173-separated-mechanism-toe-v1/provenance.json) retain
parameters, thresholds, diagnostics, and verified source/frame/GIF hashes.
Full histories, checkpoints, and intermediate images are not checked in.
The earlier [3D strength-reduced results](results/material-engineering--mat173-strength-reduced-slope-response/summary.json)
and [animation](images/home/mat173-strength-reduced-slope-response.gif) remain
historical references with a different setup and trigger.

The earlier [gravity-driven slope response](MAT173_GRAVITY_SLOPE.md) retains
the whole toe and uses a uniform strength path instead of excavation.
