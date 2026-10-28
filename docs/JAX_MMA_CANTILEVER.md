# Minimum-material cantilever candidate

The cantilever carries a 50 kN end load with the fixed support, illustrative aluminum material and 2 mm / 100 MPa limits described in the [strength-constrained cantilever](JAX_STRENGTH_CANTILEVER.md). NLopt MMA minimizes material volume using JAX derivatives through the three-dimensional MPM equilibrium solve.

The density model uses SIMP exponent 3, 0.001E residual stiffness, a two-cell Gaussian filter and a smooth projection with beta 12. The clamp, load pad and a narrow central spine are fixed as solid. A mean gray-density measure is constrained to 0.15. MMA uses a density-relaxed stress constraint; constituent stress and disconnected components are checked separately. The fixed spine restricts the search, and this result is not a proof of a minimum.

The selected layout is evaluation 47. The continuous search did not establish feasibility under the unrelaxed stress check. Its thresholded solid was independently re-solved with **zero stiffness in the void**, eliminating unsupported grid nodes while retaining the full load and clamp. Both checked solids have one six-connected component.

| Background grid | Solid fraction | Mean pad deflection | Maximum von Mises stress | Relative residual |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 58.95% | 0.931 mm | 62.82 MPa | 1.14e-09 |
| 64 | 58.40% | 1.000 mm | 69.74 MPa | 1.10e-09 |

The finer density is a trilinear reconstruction of the coarse projected field before thresholding. The two voxel geometries therefore differ; these checks do not establish mesh convergence. The fine solid uses 41.6% less material than the full beam. These small-strain elastic calculations do not establish manufacturing robustness, buckling resistance or nonlinear strength.

The GIF shows 48 actual MMA evaluations, including infeasible trials, followed by the independently checked fine-grid solid. The [saved density fields and audits](results/jax-design--mma-cantilever/) allow the solid checks to be repeated:

```bash
python tools/audit_min_volume_cantilever.py --input docs/results/jax-design--mma-cantilever/fine-density.npz --grid 64 --threshold 0.5 --output outputs/mma-cantilever-zero-void
```

To reproduce the evaluation sequence:

```bash
python -m pip install nlopt==2.10.0
JAX_ENABLE_X64=true python tools/run_mma_minimum_material.py --grid 48 --maxeval 48 --initial-density docs/results/jax-design--strength-cantilever/best-feasible-checkpoint.npz --output outputs/mma-cantilever
```
