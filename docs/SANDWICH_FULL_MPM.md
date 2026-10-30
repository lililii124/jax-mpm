# Sandwich plate under pressure pulses

The response animations show complete 2 ms independent MPM trajectories,
with actual deformation and a common 0–1 mm back-face displacement scale.

## Uniform material distribution

<table width="100%">
<tr>
<td width="33%" align="center"><img src="images/sandwich-fullmpm/initial-uniform.gif" alt="Uniform pressure" width="100%"><br>Uniform pressure</td>
<td width="33%" align="center"><img src="images/sandwich-fullmpm/initial-central.gif" alt="Central pressure" width="100%"><br>Central pressure</td>
<td width="33%" align="center"><img src="images/sandwich-fullmpm/initial-eccentric.gif" alt="Eccentric pressure" width="100%"><br>Eccentric pressure</td>
</tr>
</table>

The back-face peaks are 0.566446, 0.947616 and 0.889399 mm, respectively.
The frozen reduced model's maximum curve errors are 0.460%, 0.635% and
3.730%, below the unchanged 5% response criterion.
[Three-load response record](../evidence/blast-sandwich-reduced/union130-completed-initial-responses.json).

## Fixed-mass redistribution under eccentric pressure

<table width="100%">
<tr>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/initial-eccentric.gif" alt="Initial material distribution" width="100%"><br>Initial distribution</td>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/selected-eccentric.gif" alt="Round-one material redistribution" width="100%"><br>Round-one redistribution</td>
</tr>
</table>

At the same 1.8468 kg mass, redistribution reduces the peak from 0.889399 to
0.814771 mm (8.39%). Peak skin stress rises from 119.271 to 154.968 MPa,
below the prescribed 200 MPa limit.
[Matched independent comparison](../evidence/blast-sandwich-reduced/fullmpm-eccentric-comparison.json).

This comparison uses the earlier round-one design at 10 mm in-plane and
1 mm through-thickness spacing. Newer optimization candidates have separate
three-load verification runs; 5 mm refinement is also running. The model uses
elastic skins and an effective elastic core, without face plasticity or core
crushing. See the [problem definition](BLAST_SANDWICH_DESIGN.md) and
[media provenance](../evidence/blast-sandwich-reduced/fullmpm-completed-media.json).

## Material redistribution

<table width="100%">
<tr>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/material-redistribution.gif" alt="Ten retained fixed-mass material distributions" width="100%"></td>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/design-metrics.gif" alt="Peak deflection and skin stress for the same ten designs" width="100%"></td>
</tr>
</table>

Ten retained designs through round six, at a constant 1.8468 kg mass.
Blue to red represents core solid fraction from 0.08 to 0.35; the skins are
hidden to expose the core. These are discrete design states, not a deformation
animation. Full-MPM verification of the newer candidates is ongoing.
The adjacent plots use the same design order and frame durations. They show
the largest sampled back-face deflection and skin stress across all three
loads in the frozen reduced model. The dashed line is the 200 MPa skin-stress
limit. [Metric data and provenance](../evidence/blast-sandwich-reduced/material-design-history/rom-metrics.json).

The [control snapshots](../evidence/blast-sandwich-reduced/material-design-history/controls.npz)
and their [provenance](../evidence/blast-sandwich-reduced/material-design-history/controls.json)
reproduce the fields without rerunning optimization:

```sh
python -m tools.export_sandwich_design_history evidence/blast-sandwich-reduced/material-design-history/controls.npz outputs/sandwich-design-fields
pvpython --force-offscreen-rendering tools/render_sandwich_design_history.py outputs/sandwich-design-fields outputs/sandwich-design-render
python -m tools.render_sandwich_design_metrics evidence/blast-sandwich-reduced/material-design-history/rom-metrics.json outputs/sandwich-design-metrics
```

The metric renderer uses Matplotlib, included in `pip install -e ".[design]"`.

## Through-thickness material sections

<table width="100%">
<tr>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/material-section-xz.gif" alt="X–Z core section at y equals 150 mm" width="100%"><br>X–Z section</td>
<td width="50%" align="center"><img src="images/sandwich-fullmpm/material-section-yz.gif" alt="Y–Z core section at x equals 150 mm" width="100%"><br>Y–Z section</td>
</tr>
</table>

Each image shows the full 300 mm span above and a 92 mm central crop below,
with the same magnification in the horizontal and thickness directions.
Grey layers are the actual 2 mm skins. Colours show core solid fraction,
using the same 0.08–0.35 scale as the plan view.
[Section provenance](../evidence/blast-sandwich-reduced/material-design-history/sections-render.json).

The three-load objective includes a pressure footprint centred at
(210, 105) mm, offset from the plate centre (150, 150) mm. Consequently,
the loading set is not reflection-symmetric and the material controls are
not constrained to be symmetric. This explains why symmetry is not required;
it does not replace the independent response checks.

```sh
pvpython --force-offscreen-rendering tools/render_sandwich_design_sections.py outputs/sandwich-design-fields outputs/sandwich-design-sections
```
