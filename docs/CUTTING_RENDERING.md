# Render public cutting output

Use ParaView's `pvpython` with NumPy and Pillow available in that Python
environment. From the repository root:

```bash
pvpython --force-offscreen-rendering tools/render_cutting_paraview.py \
  --input outputs/cutting-three-pass --output outputs/cutting-three-pass-gifs
```

The same command accepts the public single-pass output directory. It exports
`equivalent-plastic-strain.gif`, `temperature.gif`, `damage.gif` and
`constitutive-plastic-strain-rate.gif`, each 400×400 pixels, plus final PNGs,
sampled PNG frames and a `render.json` receipt. The default renders up to 61
sampled states and adds states nearest saved multipass velocity events. It
renders only accepted archives up to the checkpoint pointer; a partial run is
shown at its actual saved endpoint.

The default geometry uses actual saved particle centres. Failed particles are
hidden only in the view; damage alone never removes particles, and the input
archives are unchanged. Rendering uses 1200×1200 screenshots, square point
sprites and Lanczos reduction to reduce grid moiré. Every field keeps one range
throughout the sequence. The fourth field is the actual saved constitutive
`history.plastic_rate`, rather than a difference between selected frames.

Actual physical time and virtual tool tip come from saved arrays or the public
single-pass checkpoint's frame-indexed observations. The renderer does not
derive a clock from frame numbers or use the old polygon-vertex `tool_tip` as
the virtual tip. The prescribed public wedge length defaults to 256 µm;
`--tool-length` must match any independently modified model.

Use `--geometry affine` only for outputs containing actual saved `F` and
`reference_volume`. This view transforms each public block's actual reference
particle domain by the saved in-plane deformation gradient. The block spacing
is recovered from saved geometry and initialization counts, and its area is
checked against saved reference volume and F. Missing or inconsistent inputs
raise an error; the renderer never supplies synthetic deformation gradients.

Useful options:

```bash
# Validate all required selected-frame inputs without ParaView.
python tools/render_cutting_paraview.py --validate-only \
  --input outputs/cutting-three-pass --output outputs/cutting-input-check

# A camera-only close-up; times and lengths are SI units.
pvpython --force-offscreen-rendering tools/render_cutting_paraview.py \
  --input outputs/cutting-three-pass --output outputs/cutting-second-pass-gifs \
  --start 0.0001005 --end 0.00015916666666666667 --geometry affine \
  --follow-tool --follow-offset 0.000012 0.000005 --parallel-scale 0.000055
```

`--ranges ranges.json` accepts explicit fixed ranges keyed by `plastic_strain`,
`temperature_K`, `damage` or `plastic_rate`, for example to compare material
groups on the same scale. `render.json` records ranges, actual times, saved
poses, input hashes, field checks and camera settings. It does not assert that
the original planned calculation has finished or that the model is validated.
