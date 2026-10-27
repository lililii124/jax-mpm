# JAX classic cantilever material layout

This example minimizes static compliance in a 3D cantilever using a JAX implicit equilibrium gradient and the repository's quadratic B-spline MPM material-point shape gradients. It uses a fixed particle lattice with a continuous SIMP density field. The optimizer is not a transient spring surrogate.

## Model and optimization

The beam measures 0.75 x 0.25 x 0.1875 m. The left face is clamped, with two cell layers fixed as non-design material. A fixed-total 100 N resultant acts through a finite central pad on the right face; the pad is also non-design material. Each solve uses the same force and moment. The isotropic linear-elastic constants (E=70 GPa, nu=0.30, density=2700 kg/m3) are illustrative aluminum assumptions.

The coarse optimization uses a 36 x 12 x 9 cell beam and 3,612 voxel-density controls. The objective is compliance, with volume fraction 0.38, SIMP exponent p=3, residual modulus 0.001E, a radius-2-cell Gaussian spatial filter and projection beta values 1.5, 3, 6 and 12. JAX `value_and_grad` differentiates through an implicit `custom_linear_solve`; the AD directional check differs from finite differences by 3.06e-9 relative. Forty-eight updates were accepted. The result is a budgeted design, not a converged optimum.

The coarse continuous-field compliance changed from 1.93515e-3 to 2.96635e-4 N m. At the 64-grid independent forward resolution (48 x 16 x 12 beam cells, 73,728 quadrature points), initial and optimized continuous fields each have 36.0703 kg and volume fraction 0.38; compliance changed from 1.81456e-3 to 2.94366e-4 N m, an 83.78% reduction at the same mass. The fixed load and support are identical. The phi=0.50 thresholded field is 37.67% solid and one six-connected component. Its minimum longitudinal section is 12 cells on this grid; this is only a voxel audit, not a minimum feature-size or manufacturing certification. Removing one-cell openings deletes 14.75% of thresholded solid.

## Classical beam-theory comparison

The model spans 0.75 m from the fixed left beam face at global x=0.125 m to the right load face at x=0.875 m. Its y-z rectangular section is 0.25 x 0.1875 m. The 100 N downward resultant is distributed by a 2 x 2 face quadrature over a centered 0.125 x 0.08333 m non-design pad (equivalent average traction 9.6 kPa). The force/moment audit reports a global-origin moment (50, 0, -87.5) N m; shifting the origin to the clamp section centroid gives (0, 0, -75) N m = r x F. Thus, the far-field loading matches a downward free-end centroid force, while the local stress field near the finite pad remains three-dimensional.

For a fully solid, uniform, linear-elastic beam with the model's E=70 GPa, nu=0.30, and the same load, Euler-Bernoulli gives a bending deflection of 0.823 micrometres and compliance 8.23e-5 N m. Since L/H=3, transverse shear is not negligible: the rectangular-section Timoshenko estimate adds 0.0713 micrometres of shear deflection (5/6 shear factor), for 0.894 micrometres total and 8.94e-5 N m compliance. These are analytical full-solid references; the Timoshenko result is the more appropriate slenderness check.

We also solved the **same full-solid three-dimensional MPM beam** at three grids, bypassing the density projection and setting every material-point density to one. Geometry, 70 GPa / nu=0.30 material, fixed face, and the original 100 N pad transfer are identical to the published model. The compliance/F measure is the force-weighted mean displacement of the loaded pad:

| Background grid | Beam cells | Material points | Mean pad deflection | Difference from Timoshenko | CG residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 24 x 8 x 6 | 9,216 | 0.9299 micrometres | +4.00% | 9.78e-11 |
| 48 | 36 x 12 x 9 | 31,104 | 0.9265 micrometres | +3.62% | 8.94e-11 |
| 64 | 48 x 16 x 12 | 73,728 | 0.9191 micrometres | +2.79% | 9.81e-11 |

The 32-to-64 compliance change is -1.16%; the last 48-to-64 refinement changes it by -0.795%. The grid-64 MPM response is 11.70% above Euler-Bernoulli and 2.79% above Timoshenko. Force and moment balance errors stay below 1.2e-11 on all grids. This is a matched-material comparison. The remaining difference from one-dimensional beam theory combines 3D elasticity and end-region effects from the finite load pad and fixed face with MPM particle/stencil discretization; these contributions are not separated by this study. The observed grid changes measure mesh sensitivity, not a pure error estimate.

The refined MPM initial and optimized cases are not homogeneous beams: both have 0.38 mean material volume fraction, fully solid non-design end pads, and an ersatz SIMP stiffness law. Their load-weighted pad responses are compliance/F = 18.15 micrometres initially and 2.944 micrometres after optimization. Relative to the full-solid Timoshenko compliance, these are 20.29 and 3.29 times larger. Those ratios compare different material layouts and mass; they are not discretization errors or validation against a uniform beam. The optimized field must not be treated as a uniform cantilever.

The analysis uses two reproducible commands. The first runs the actual JAX MPM equilibrium at the three full-solid grids; the second checks the audited load and merges the beam formulas and mesh study:

```bash
JAX_ENABLE_X64=true python tools/run_full_solid_cantilever_mpm.py \
  --grids 32 48 64 --cg-maxiter 6000 --cg-tol 1e-10 \
  --output outputs/full-solid-beam-comparison.json
python tools/classic_cantilever_beam_reference.py \
  --full-solid-mpm outputs/full-solid-beam-comparison.json \
  --output outputs/beam-theory-reference.json
```

The [machine-readable beam reference](results/jax-design--classic-cantilever/beam-theory-reference.json) contains formulas, source hash, force/moment checks, full-solid MPM residuals and grid comparisons, and the separate 38%-SIMP response ratios. Raw full-solid MPM output and its task binding are recorded in [the mesh study](results/jax-design--classic-cantilever/full-solid-mpm-mesh-study.json) and [source manifest](results/jax-design--classic-cantilever/source-manifest.json).

## Visuals

The 49-frame phi=0.20 GIF shows the continuous density field during optimization. A separate phi=0.35 history and a phi=0.50 binary-cell surface are included. The initial projected field is below phi=0.35 in the design region, so the first phi=0.35 frame shows only the clamp and load pads; density appears as the same design field is updated. The binary view is made from exposed faces of the 0/1 thresholded cells without cell-to-point smoothing. It retains the same camera and shows a connected voxel topology.

Both contour views are visualization choices. The optimization and the conventional thresholded equilibrium use ersatz void with modulus 0.001E. The thresholded voxel image is not a zero-void finite-element certification. No deformation is drawn in these design-history GIFs.

## Exact zero-void check and limits

A separate audit assembles the thresholded MPM material-point stiffness with void stiffness and inertia exactly zero, removes only unsupported grid nodes, retains all 108 loaded nodes, and solves the 25,785 free DOFs using SciPy sparse LU after exact symmetric diagonal equilibration. The matrix residual is 1.81e-12 and force/moment balance errors are 1.97e-12 and 1.50e-12. This is not the JAX optimization solve. These algebraic checks alone do not certify a stable physical displacement field.

The exact zero-void active-particle field is solver-order sensitive: a second sparse-LU ordering changes its RMS by 28.4% relative to the first equilibrated result; scaled CG stopped at a relative residual of 3.40e-6, above the 1e-8 audit gate. Therefore the zero-void physical stability check is **not accepted**. The affine-local aggregation sweep retains all applied loads and preserves affine coordinates to roundoff, but adds constraints and changes the discrete space. Its maximum aggregation distance, force/moment projection error, weak-constraint residual, and original-operator residual are recorded for each grid in [the sweep summary](results/jax-design--classic-cantilever/affine-local-grid-sweep/summary.json). These constrained solves are sensitivity diagnostics, not equivalent zero-void solutions. Do not render their nodal values as validated physical deformation. The filter radius and section counts do not establish minimum manufacturable features, and the 48-update run does not claim convergence.

## Reproduction

```bash
python -m examples.classic_cantilever_topopt \
  --grid 48 --iterations 12 --cg-maxiter 1200 --forward-grid 64 \
  --output outputs/classic-cantilever

python -m examples.classic_cantilever_topopt \
  --grid 48 --forward-grid 64 \
  --true-void-input outputs/classic-cantilever/refined-optimized.npz \
  --true-void-method scipy-sparse-equilibrated-direct \
  --output outputs/classic-cantilever-true-void
```

The optional independent SciPy solve is available with `pip install -e ".[forming]"`. See [the run summary](results/jax-design--classic-cantilever/summary.json), [raw optimization record](results/jax-design--classic-cantilever/optimization-summary.json), [gradient check](results/jax-design--classic-cantilever/gradient-and-solver-check.json), [zero-void audit](results/jax-design--classic-cantilever/true-void-audit.json), and [affine-local grid sensitivity](results/jax-design--classic-cantilever/affine-local-grid-sweep/summary.json).
