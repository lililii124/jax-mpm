# Minimum-material cantilever candidate

This candidate uses the same 50 kN end load, fixed support, illustrative aluminum material and 2 mm / 100 MPa limits described in the [strength-constrained cantilever](JAX_STRENGTH_CANTILEVER.md). The search minimizes material volume using NLopt MMA and JAX derivatives through the three-dimensional MPM equilibrium solve.

The density model uses SIMP exponent 3, 0.001E residual stiffness, a two-cell Gaussian filter and a smooth projection with beta 12. The clamp, load pad and a narrow central spine are fixed as solid. A mean gray-density measure is constrained to 0.15. The stress constraint used by MMA is density-relaxed; the unrelaxed stress and disconnected components are checked separately. Consequently the search is local and restricted by the fixed spine, and is not a proof of a minimum.

The selected layout is evaluation 21. It was a trial evaluation, not an accepted continuous feasible iterate. Its thresholded solid was independently re-solved with **zero stiffness in the void**, removing unsupported grid nodes and retaining the full load and clamp. Both thresholded geometries have one six-connected component.

| Background grid | Solid fraction | Mean pad deflection | Maximum von Mises stress | Relative residual |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 62.65% | 0.829 mm | 62.99 MPa | 1.14e-9 |
| 64 | 62.11% | 0.865 mm | 72.36 MPa | 1.06e-9 |

The finer density is a trilinear reconstruction of the coarse projected field before thresholding, so the voxel geometries differ. The fine solid uses about 37.9% less material than the full beam. The calculations are small-strain elastic and do not establish manufacturing robustness or nonlinear strength.

The GIF shows 22 actual MMA evaluations, including infeasible trials. Its final frame shows the independently checked fine-grid binary solid. The [saved density fields and audits](results/jax-design--mma-cantilever/) permit the two solid checks to be repeated:

```bash
python tools/audit_min_volume_cantilever.py \
  --input docs/results/jax-design--mma-cantilever/fine-density.npz \
  --grid 64 --threshold 0.5 --output outputs/mma-cantilever-zero-void
```

To reproduce the evaluation sequence, install NLopt and run:

```bash
python -m pip install nlopt==2.10.0
JAX_ENABLE_X64=true python tools/run_mma_minimum_material.py \
  --grid 48 --maxeval 22 \
  --initial-density docs/results/jax-design--strength-cantilever/best-feasible-checkpoint.npz \
  --output outputs/mma-cantilever
```
