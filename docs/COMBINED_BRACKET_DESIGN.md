# Three-load perforated bracket sizing with JAX

`examples/combined_bracket_design.py` differentiates a three-dimensional,
total-Lagrangian material-point solve through
`mpm_jax.total_lagrangian.step`. The example sizes a perforated bracket under
axial tension, transverse bending, and torsion. It keeps the reference particle
coordinates and the material-point stencil fixed while varying the widths of
the two load-carrying ligaments.

This is bounded member sizing with a continuous material mixture. It is not a
free-form topology or exterior-shape optimization. The visualization extracts
the `phi = 0.5` isosurface of that mixture for display; the rendered surface is
not evidence of a validated binary or manufacturable topology.

## Geometry, material, and constraints

The reference envelope is `0.8 × 0.44 × 0.24 m`. It contains a through-hole
with radius `65 mm`, a fixed full-material annular boss of outer radius
`170 mm`, a full-material clamp pad, and a full-material load pad. The boss
bridges the upper and lower ligaments around the hole. Their five Gaussian-RBF
width controls each map bounded latent variables to `60–120 mm` local widths;
the `60 mm` lower bound is the explicit minimum member size. The overlapping
RBF map also avoids abrupt control-point steps.

The constitutive law is compressible Neo-Hookean elasticity (`E = 200 kPa`,
`nu = 0.25`, base density `1000 kg/m³`). A `p = 1` Voigt mixture scales
reference quadrature volume, particle mass, and stiffness with the same solid
fraction `phi`. The `0.02` residual fraction is reported separately from solid
and gray regions. The topology and exact hole are fixed.

Three design-independent loads are applied through the full load pad: `100 N`
axial tension, `100 N` transverse bending, and a pure `15 N·m` torsional
couple. The objective is the mean, across these cases, of full-horizon
time-averaged squared displacement on the load-end tip region, normalized by
`1e-4 m²`. The `60 ms` history uses `dt = 0.2 ms` and 300 MPM steps.

The optimizer enforces a maximum solid volume fraction of `0.75`, the local
width bounds, valid MPM states, and an unnormalized nested `p = 24` stress
upper bound of `20 kPa`. The stress aggregate uses the Cauchy von Mises stress
and `phi` weighting so the residual floor cannot hide a solid-particle hot
spot; actual solid, gray, and floor maxima are also reported. The `20 kPa`
limit is an illustrative `E × 10%` nominal-strain assumption, not a calibrated
material strength.

The AD path uses `jax.jit(jax.value_and_grad(...))`, `lax.scan` for all three
histories, and sequential load-case evaluation. A central directional finite
difference is checked before optimization.

## Run

```bash
python -m examples.combined_bracket_design \
  --grid 24 --horizon 0.06 --dt 0.0002 \
  --iterations 21 --learning-rate 0.04 \
  --output outputs/combined-bracket
```

The optimizer writes its `design-history.csv`, Adam checkpoint, and JSON
summary after every accepted or rejected update. For a refined forward run,
pass the ten `controls` from `optimization-summary.json`:

```bash
python -m examples.combined_bracket_forward \
  --grid 48 --horizon 0.06 --dt 0.0002 \
  --controls <five-upper-then-five-lower-latent-values> \
  --output outputs/combined-bracket-forward
```

The forward CLI evaluates the initial and optimized designs on the same
`48³` background grid and exports 21 actual bending positions per design.
Render them through ParaView 6.2 OSMesa:

```bash
pvpython --force-offscreen-rendering tools/render_combined_bracket_pv.py \
  --input outputs/combined-bracket-forward \
  --output outputs/combined-bracket-forward/paraview
```

The renderer extracts a continuous surface from the regular MPM neighbour
grid, uses the same camera and displacement scale for both designs, renders at
`800 × 800`, and downsamples each animation to `400 × 400`. The geometry uses
the actual displacement scale (`1.0`), and the GIFs have no axes or labels.

## Recorded result

The archived run requested 21 Adam updates; 18 were accepted before the
projected line search stalled near the volume limit. It is a feasible
finite-budget result, not a convergence or global-optimum claim.

| Quantity | Initial | Optimized |
|---|---:|---:|
| Grid-24 normalized response | 0.41102 | 0.37092 |
| Grid-48 normalized response | 0.32026 | 0.29757 |
| Grid-48 solid volume fraction | 0.71784 | 0.73889 |
| Grid-48 mass | 60.643 kg | 62.421 kg |
| Grid-48 `p = 24` stress upper bound | 16.271 kPa | 15.988 kPa |
| Grid-48 maximum solid stress | 8.022 kPa | 8.191 kPa |
| Continuous ligament width range | 90.0 mm | 99.215–99.400 mm |

The refined response fell by `7.084%` under the same three external loads.
The design added `2.93%` mass as its volume fraction rose toward the `0.75`
cap, so this is not an equal-mass comparison. Both refined designs passed the
volume, width, stress, reference-grid, and all-step validity checks. The
directional AD/FD relative error on the design grid was `6.59e-8`.

The GIFs below show the initial and optimized bending histories with a shared
camera, color range, and true deformation scale:

| Initial | Optimized |
|---|---|
| ![Initial bracket bending](images/home/jax-design--combined-bracket-initial.gif) | ![Optimized bracket bending](images/home/jax-design--combined-bracket-optimized.gif) |

Machine-readable summaries, all 21 design-history rows, the Adam state, the
refined particle trajectory, source manifests, and rendering audits are in
[`docs/results/jax-design--perforated-combined-load-bracket/`](results/jax-design--perforated-combined-load-bracket/).
