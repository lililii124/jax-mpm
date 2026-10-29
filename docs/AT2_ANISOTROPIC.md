# Anisotropic AT2 crack deflection

The zero-initial-damage single-edge crack specimen uses the same mechanical
energy and irreversible solve as [the isotropic example](AT2_SHARP_CRACK.md).
Its crack-density energy is

`Gc/2 * (d^2/ell + ell grad(d).A.grad(d))`,

with `A = 9 t tensor t + n tensor n`, where `t` is oriented at 25 degrees and
`n` is its normal. The residual and tangent derive from this same energy.
No propagated crack nodes or unit-damage boundary values are imposed.
The preferred material direction is not a prescribed crack trajectory.

The completed load path reaches a displacement of 75 micrometres. The
connected damage band spans the remaining 0.625 m ligament; its fitted angle
is 7.41 degrees and its maximum measured center offset is 0.0656 m. The final
work-energy discrepancy is 0.169%. This is a single-grid demonstration, not a
mesh-convergence study. The first bounded run was continued from its accepted
checkpoint using the exact original numerical source.

```bash
python -m pip install -e ".[fracture]"
python -m examples.at2_material_case_benchmark --case anisotropic --angle 25 --anisotropy-ratio 9 --displacement .000075 --increments 24 --output outputs/at2-anisotropic
pvpython --force-offscreen-rendering tools/render_at2_material_case_paraview.py --input outputs/at2-anisotropic --output outputs/at2-anisotropic/render --gif-name at2-anisotropic.gif --size 400
python tools/gif_review.py outputs/at2-anisotropic/render/at2-anisotropic.gif --open
```

ParaView displays the actual saved damage with a fixed range of zero to one,
without rotating, mirroring or averaging the field. The 25 accepted states
encode into 23 distinct GIF frames with their original duration retained.

[Load history](results/at2--anisotropic-crack/history.csv) ·
[Path measurement](results/at2--anisotropic-crack/connected-path-measurement.json) ·
[Run summary](results/at2--anisotropic-crack/summary.json)
