# Differentiable MPM material inverse example

`python -m examples.inverse_material --output results/inverse-demo` estimates
Young's modulus and Poisson's ratio from two synthetic sensor records using
the repository's 3D reference-grid total-Lagrangian MPM solver. The constitutive
law is compressible neo-Hookean in its smooth invariant form; the example does
not use an SVD-based polar decomposition. Density is fixed at 1050 kg/m^3.

The target signals are deliberately labeled synthetic in the result JSON and
NPZ metadata. The script generates them by running this same MPM implementation
with disclosed parameters (`E=1.25 MPa`, `nu=0.31`), optionally adds seeded
Gaussian displacement noise, then fits those signals. The two inverse loads
drive opposite x-face motion along x and y. Twelve interior material points
(configurable from 4 to 12) act as sensors. A separate holdout uses different
load directions, amplitudes, period, sensor points, duration, grid and particle
quadrature; its truth and inferred trajectories are saved for comparison.

The inverse path is actual MPM time integration: quadratic B-spline reference
grid transfers and internal forces, nodal lumped inertia, prescribed material
node boundary motion and velocity-Verlet stepping. `jax.lax.scan` advances the
trajectory, `jax.checkpoint` rematerializes each step in reverse mode,
`jax.vmap` batches the two loading experiments, and a jitted
`jax.value_and_grad` drives bounded Adam optimization. E and nu are represented
by sigmoid transforms inside `E=[0.2,3.0] MPa` and `nu=[0.12,0.46]`. A central
finite-difference directional derivative is recorded once against autodiff.

## Resolution and memory

The default inverse grid is 14 cells per edge with 3 particles per cell per
axis. The centered cube occupies 40% of the box edge, giving 17^3 = 4,913
particles. This modest grid keeps gradient iterations practical and is not a
mesh-convergence result. The independent holdout defaults to 24 cells and 4
particles per cell per axis: 38^3 = 54,872 particles. For that holdout, the
reference stencil stores 27 indices, weights and 3D gradients per particle;
those arrays alone require roughly 51 MiB in float64/int32 before solver state,
JAX executable buffers, outputs or allocator overhead. The inverse run uses a
5e-5 s step and 100 steps by default; the holdout uses 120 steps. Runtime and
peak memory depend on the backend and are recorded only as measured if the
runtime reports them; the script does not invent a device memory measurement.

Use `--grid`, `--ppc`, `--steps`, `--forward-grid`, and `--forward-ppc` to
change resolution. Increasing particle count has cubic cost in this example;
doubling `ppc` increases the particle count about eightfold. The default
forward holdout is in the requested engineering scale of roughly 50k–100k
particles. Lower `--forward-grid` or `--forward-ppc` for machines with less
memory. No claim is made that the holdout is experimentally independent: it is
a numerical generalization check against the same disclosed constitutive law.

## Recorded run

The recorded run in `results/inverse-material-synthetic-20261027/` used the
defaults above and completed on the CPU JAX image in 515 s (4 CPUs, 12 GiB
container memory limit, 1.77 GiB peak process USS). It estimated E=1.289681 MPa
and nu=0.302586 from synthetic truth E=1.25 MPa and nu=0.31, giving relative
errors of 3.17% and 2.39%. Loss fell from 2.51045e-9 to 5.73013e-12. The one
directional autodiff derivative and central finite difference differed by
1.03e-9 relative. The inverse used 4,913 particles; the distinct holdout used
54,872 particles and returned a 2.97e-6 m component RMSE. That is 2.30% of
the truth holdout's 1.294e-4 m component RMS displacement, or 0.43% of its
0.695 mm peak vector displacement. Reported holdout errors compare inferred
and truth sensor positions; normalization uses displacement after subtracting
the initial sensor locations. `result.json` records the exact source SHA-256
used for the run; `source_used_inverse_material.py` preserves that executed
version. The saved trajectories and metrics are numerical evidence for this
single synthetic case, not experimental validation or a convergence study.
An audit using that exact source snapshot and the same JAX image confirmed that
the inverse clamps fixed 162 nodes at each end, carrying 4.25016 kg of lumped
nodal mass per end; the holdout clamps 338 nodes per end with positive mass.
The run's initial sensor displacement peaked at 0.639 mm. The raw `result.json`
has a zero clamp-node count because its reporting expression selected only
empty node zero; `run_audit_receipt.json` records the correction from the actual
`make_loads` masks and mass sums. This reporting error did not affect the MPM
boundary arrays, which were also checked through the nonzero sensor motion.

## Artifacts

Each invocation requires a new output directory. The CLI writes:

* `result.json`: parameters, relative errors, initial/final losses, one
  directional gradient check, holdout response error, actual resolutions and
  SHA-256 hashes of the source files used.
* `loss_history.csv`: per-iteration objective, parameters and gradient norm.
* `checkpoint.npz`: the latest Adam iterate and best-so-far parameters.
* `observations_synthetic.npz`: inverse observations, truth parameters, sensor
  coordinates and load directions.
* `holdout_responses.npz`: truth and inferred holdout displacement histories.

The saved run directory also contains post-run evidence files:

* `holdout_metrics.json`: postprocessed displacement-normalized holdout error;
  this distinguishes displacement response from the sensors' absolute position.
* `run_audit_receipt.json`: exact executed source hash, resource status, and
  same-image verification of the actual material clamp masks and masses.

The checkpoint is a reproducibility record, not yet a resume interface.
