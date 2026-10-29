# Sandwich plate under pressure pulses

These ParaView animations show complete 2 ms independent MPM trajectories.
Each uses actual deformation and a common 0–1 mm back-face displacement scale.

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
