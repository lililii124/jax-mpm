# Ti-6Al-4V cutting

Snapshots of orthogonal cutting with Johnson-Cook plasticity, thermal softening
and damage. The multiple-pass sequence includes cutting, lift, return and a
deeper second pass. Fully failed particles are hidden in the rendered views.

| View | Rendered interval (µs) | State |
|---|---:|---|
| Multiple passes | 16.607–107.731 | Continuing |
| Second pass close-up | 100.500–107.731 | Continuing |
| Johnson parameters | 0–26.674 | Planned run complete |
| Seo parameters | 0–14.560 | Continuing |

Johnson and Seo use different end times here; these animations are not a
matched-time material comparison. The multiple-pass views use saved deformation
gradients; the single-pass views use saved particle centres. Colours remain
fixed within each sequence. No displacement scaling is applied.

The exact production drivers are not yet part of the public examples.
These are numerical previews, not an experimental validation of
segmented-chip formation.

## Animations

### Multiple passes

<table width="100%">
<tr>
<td width="25%" align="center"><img src="images/home/cutting--multipass--equivalent-plastic-strain.gif" alt="Multiple passes · Equivalent plastic strain" width="100%"><br><a href="results/cutting--multipass/summary.json">Equivalent plastic strain</a></td>
<td width="25%" align="center"><img src="images/home/cutting--multipass--temperature.gif" alt="Multiple passes · Temperature" width="100%"><br><a href="results/cutting--multipass/summary.json">Temperature</a></td>
<td width="25%" align="center"><img src="images/home/cutting--multipass--damage.gif" alt="Multiple passes · Damage" width="100%"><br><a href="results/cutting--multipass/summary.json">Damage</a></td>
<td width="25%" align="center"><img src="images/home/cutting--multipass--plastic-increment-rate.gif" alt="Multiple passes · Plastic strain rate" width="100%"><br><a href="results/cutting--multipass/summary.json">Plastic strain rate</a></td>
</tr>
</table>

### Second pass close-up

<table width="100%">
<tr>
<td width="25%" align="center"><img src="images/home/cutting--second-cut-closeup--equivalent-plastic-strain.gif" alt="Second pass close-up · Equivalent plastic strain" width="100%"><br><a href="results/cutting--second-cut-closeup/summary.json">Equivalent plastic strain</a></td>
<td width="25%" align="center"><img src="images/home/cutting--second-cut-closeup--temperature.gif" alt="Second pass close-up · Temperature" width="100%"><br><a href="results/cutting--second-cut-closeup/summary.json">Temperature</a></td>
<td width="25%" align="center"><img src="images/home/cutting--second-cut-closeup--damage.gif" alt="Second pass close-up · Damage" width="100%"><br><a href="results/cutting--second-cut-closeup/summary.json">Damage</a></td>
<td width="25%" align="center"><img src="images/home/cutting--second-cut-closeup--plastic-increment-rate.gif" alt="Second pass close-up · Plastic strain rate" width="100%"><br><a href="results/cutting--second-cut-closeup/summary.json">Plastic strain rate</a></td>
</tr>
</table>

### Johnson parameters

<table width="100%">
<tr>
<td width="25%" align="center"><img src="images/home/cutting--johnson--equivalent-plastic-strain.gif" alt="Johnson parameters · Equivalent plastic strain" width="100%"><br><a href="results/cutting--johnson/summary.json">Equivalent plastic strain</a></td>
<td width="25%" align="center"><img src="images/home/cutting--johnson--temperature.gif" alt="Johnson parameters · Temperature" width="100%"><br><a href="results/cutting--johnson/summary.json">Temperature</a></td>
<td width="25%" align="center"><img src="images/home/cutting--johnson--damage.gif" alt="Johnson parameters · Damage" width="100%"><br><a href="results/cutting--johnson/summary.json">Damage</a></td>
<td width="25%" align="center"><img src="images/home/cutting--johnson--plastic-increment-rate.gif" alt="Johnson parameters · Plastic strain rate" width="100%"><br><a href="results/cutting--johnson/summary.json">Plastic strain rate</a></td>
</tr>
</table>

### Seo parameters

<table width="100%">
<tr>
<td width="25%" align="center"><img src="images/home/cutting--seo--equivalent-plastic-strain.gif" alt="Seo parameters · Equivalent plastic strain" width="100%"><br><a href="results/cutting--seo/summary.json">Equivalent plastic strain</a></td>
<td width="25%" align="center"><img src="images/home/cutting--seo--temperature.gif" alt="Seo parameters · Temperature" width="100%"><br><a href="results/cutting--seo/summary.json">Temperature</a></td>
<td width="25%" align="center"><img src="images/home/cutting--seo--damage.gif" alt="Seo parameters · Damage" width="100%"><br><a href="results/cutting--seo/summary.json">Damage</a></td>
<td width="25%" align="center"><img src="images/home/cutting--seo--plastic-increment-rate.gif" alt="Seo parameters · Plastic strain rate" width="100%"><br><a href="results/cutting--seo/summary.json">Plastic strain rate</a></td>
</tr>
</table>
