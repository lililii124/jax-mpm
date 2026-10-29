# Ti-6Al-4V cutting

Snapshots of orthogonal cutting with Johnson-Cook plasticity, thermal softening
and damage. The multiple-pass sequence includes cutting, lift, return and a
deeper second pass. Fully failed particles are hidden in the rendered views.

| View | Rendered interval (µs) | State |
|---|---:|---|
| Multiple passes | 16.607–107.731 | Continuing |
| Second pass close-up | 100.500–107.731 | Continuing |
| Johnson parameters | 0–26.674 | Planned run complete |
| Seo parameters | 0–32.004 | Planned run complete |

Johnson and Seo use different end times here; these animations are not a
matched-time material comparison. The multiple-pass views use saved deformation
gradients; the single-pass views use saved particle centres. Colours remain
fixed within each sequence. No displacement scaling is applied.

Public drivers now support both flow-stress parameter groups and a fresh
three-pass trajectory. Existing rendered sequences retain their original
source/checkpoint provenance; the new entry points do not retroactively change it.
These are numerical previews, not an experimental validation of
segmented-chip formation.

## Public runs

Run from the repository root with JAX 64-bit enabled. Both material presets use
the current `johnson_cook_displacement` law. `johnson` uses A=862.5 MPa,
B=331.2 MPa, n=0.34, C=0.012, m=0.8; `seo` uses A=997.9 MPa,
B=653.1 MPa, n=0.45, C=0.0198, m=0.7. Elastic, thermal and JC initiation
parameters are shared explicitly in `examples/cutting_presets.py`. This compares
parameter groups, rather than reproducing every assumption of the cited paper.

```bash
python -m examples.orthogonal_cutting_ti64_2d --material-preset seo \
  --grid 256 --ppc 1 --cfl .08 --friction .3 --cut-depth 1e-5 \
  --tool-tip-radius 3e-6 --tool-tip-arc-segments 12 \
  --jc-damage --fast-jc-return --particle-grid-contact \
  --failed-wedge-swept-contact --particle-contact-max-sweeps 1024 \
  --particle-contact-warm-start --steps 1969500 --substeps 500 \
  --output outputs/cutting-seo
python -m examples.orthogonal_cutting_multipass --material-preset johnson \
  --output outputs/cutting-three-pass
```

For the Johnson single-pass endpoint shown above, use `--material-preset johnson`
and `--steps 1641500` with the same grid, contact and output settings.

The multipass default starts from an unstressed, undamaged material block at
actual time zero; no private seed or `/work` directory is required. The checked-in
`examples/cutting_three_pass_path.json` specifies all three cuts, lifts, returns
and the final lift. Its default nominal dt is the original CFL .08 value.
`--dt-factor 4` is an explicit exploratory option and does not establish baseline
accuracy. The displacement-to-failure default is 2 grid cells and is not a
calibrated fracture energy.

Append `--resume` with the same output directory to resume a public multipass
checkpoint. The signature binds material values, geometry, contact settings,
softening, path, nominal dt and cache-reset protocol. State/history and contact
cache are retained across events and restarts; the cache resets at physical-time
boundaries 500 baseline steps apart. Actual accumulated time is saved, and final
substeps end at velocity events without snapping the state clock. Legacy private
checkpoints require an explicit conversion and are rejected by this public
restart route. `--max-output-windows N` provides a checkpoint-ready bounded run;
it does not mark the full trajectory complete.

Migration checks include one real-contact v50 checkpoint step: the frozen and
public kernels produce bitwise identical common state/history arrays, contact
cache, impulse and work. Four additional public diagnostic arrays remain finite
zeros in this JC branch. A separate short g256/p1 baseline-dt cut-to-lift test
checks actual velocity-event endpoints and a restart after the first event;
state/history/cache and the retained event ledger match the uninterrupted run.
Receipts are under `evidence/cutting-public-integration/`. These bounded checks
do not establish long-run convergence or experimental validity.

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
