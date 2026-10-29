# Gravity-driven slope response

<table width="100%">
<tr>
<td width="50%" align="center"><img src="images/home/mat173-gravity--plastic-activity.gif" alt="Plastic activity" width="100%"><br>Plastic activity</td>
<td width="50%" align="center"><img src="images/home/mat173-gravity--shear-softening.gif" alt="Shear softening" width="100%"><br>Shear softening</td>
</tr>
</table>

This earlier strict 2D plane-strain example keeps the complete slope geometry:
there is no toe excavation or particle removal. Motion develops under gravity
after a prescribed uniform strength path and continued strain softening. It is
the unexcavated slope example, not an untouched slope that fails with constant
material properties.

The source calculation first balances self-weight, then changes friction and
cohesion from 38 degrees / 10 kPa to 28 degrees / 0 Pa over 1250 steps
(0.0375 s). Subsequent plastic softening reduces friction toward 20 degrees,
with residual cohesion 0 Pa, dilation 2 to 0 degrees, characteristic length
0.0125 m, and fracture slip 0.0005 m. No sliding velocity is imposed.

The 61-frame, 400×400 animations show saved steps 4950–12450, absolute time
0.1485–0.3735 s. Plastic activity is the saved-window increment of total
equivalent plastic strain (first frame zero), fixed range 0–0.00486393618.
Shear softening is D, fixed range 0–1. D=1 is residual strength, not particle
deletion. Geometry uses actual particle coordinates, without deformation
scaling or smoothing. This historical rendering uses reference-neighbor quads
with per-frame orientation/open-edge checks and particle fallback, rather than
the independent affine particle domains used by the current excavation example.

The last accepted state is valid and finite. The next attempted window exceeded
the selected 0.2 absolute Hencky-strain guard; the animation ends at the last
accepted state, with maximum 0.19897034. It is an exploratory response, not a
completed runout, site prediction, engineering verification, or proof of a
mesh-independent shear band.

The [summary](results/mat173-gravity-slope-response/summary.json) records the
strength path, stopping condition, rendering and checkpoint checks; the
[provenance](results/mat173-gravity-slope-response/provenance.json) records
frame/checkpoint hashes and the original GIF hashes. These existing animations
were retained without rerunning the calculation or rendering. The full particle
histories and checkpoints are not included in the repository.

See also the [toe-excavation example](MAT173_LANDSLIDE.md), which uses actual
particle removal and a different shear/tensile history definition.
