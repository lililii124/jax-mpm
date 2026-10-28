# Strength-constrained cantilever candidate

This 3D linear-elastic cantilever uses the same geometry and boundary conditions as the [classical beam comparison](JAX_CLASSIC_CANTILEVER.md), with a 50 kN downward force on the end pad. The illustrative material has E = 70 GPa, Poisson ratio 0.30 and density 2700 kg/m³. The design limits are 2 mm mean pad deflection and 100 MPa von Mises stress.

The candidate comes from four accepted JAX MPM updates at fixed 65% continuous density, warm-started from a 70% candidate. The objective combines compliance and penalties for deflection and stress, with SIMP exponent 3 and a 0.001E residual stiffness. The GIF shows those four actual density updates. This is a feasible candidate, not a proof of the minimum material amount.

The density field was thresholded and solved again with **zero stiffness in the void**. A separate sparse MPM equilibrium solve eliminated unsupported nodes and retained the full load and clamp. The two grids have different thresholded voxel geometries.

| Threshold | Background grid | Solid fraction | Mean pad deflection | Maximum von Mises stress | Relative residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 48 | 70.11% | 0.612 mm | 61.84 MPa | 1.27e-9 |
| 0.50 | 64 | 68.87% | 0.766 mm | 65.21 MPa | 1.14e-9 |
| 0.54 | 48 | 63.35% | 0.770 mm | 61.64 MPa | 1.24e-9 |
| 0.54 | 64 | 63.98% | 1.233 mm | 83.62 MPa | 1.06e-9 |
| 0.55 | 64 | 62.64% | 1.438 mm | 99.99 MPa | 1.10e-9 |
| 0.60 | 64 | 56.45% | 3.960 mm | 225.70 MPa | 1.08e-9 |

At threshold 0.54, the grid-64 solid is one six-connected component and uses about 36.0% less material than the full rectangular beam. At 0.55 the peak stress is close to its limit; 0.60 fails both constraints. These are small-strain elastic calculations with assumed limits, not a manufacturing or failure certification. The thresholded geometry and stress concentration remain mesh-dependent.

The saved field and detailed equilibrium records are in [the results directory](results/jax-design--strength-cantilever/). Re-run the zero-void check with:

```bash
python tools/audit_min_volume_cantilever.py \
  --input docs/results/jax-design--strength-cantilever/refined-optimized.npz \
  --grid 64 --threshold 0.54 --output outputs/strength-cantilever-zero-void
```

To reproduce the four continuous updates from the saved 70% warm start:

```bash
JAX_ENABLE_X64=true python tools/run_min_volume_cantilever_topopt.py \
  --grid 48 --validation-grid 64 --outer-bisections 0 --inner-iterations 4 \
  --cg-maxiter 2400 --cg-tol 1e-9 --volume-lower 0.65 \
  --initial-q docs/results/jax-design--strength-cantilever/warmstart-vf070-feasible.npz \
  --output outputs/strength-cantilever
```

The archived candidate and the zero-void audits are the results shown above. A separate MMA minimum-material search is in progress.
