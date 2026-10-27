# Orthotropic MPM inverse evidence (2026-10-27)

## Scope

This is a synthetic 3D identification run for the released MAT002 orthotropic
St. Venant–Kirchhoff energy in the reference-grid total-Lagrangian MPM solver.
Density is fixed at 1050 kg/m³. The training archive is fixed-seed synthetic
data with 0.3% channel noise; it is not experimental data. No damage or plastic
return map is involved.

The training specimen is 0.40 × 0.30 × 0.25 m. It uses eight full MPM loading
histories (three axial tensions, three independent shears, three-point bending,
and x torsion), 12 displacement sensors, reaction force and moment, 200 steps
at 40 μs (8 ms). Truth used grid 48 / ppc 3 (89,784 particles); fitting used
the **same 48-cell background grid** / ppc 2 (26,448 particles). Both have
19.2, 14.4, and 12 background cells across the specimen axes. Thus this
recorded fit changes particle quadrature, not background-grid resolution.
The fixture masks use the same fixed 47.5 mm physical clamp band; all support
and driven masks were checked for overlap and their actual mass was recorded.

Training archive SHA-256:
`022e915e098b3212439167c0496c3e555a58318a4d337971d2deb802ee2b4269`.
The exact frozen archive is included as `training_synthetic.npz` (979,207 bytes); its SHA-256 is listed below and matches the run receipt.

## Recorded fit

The optimizer started from the disclosed INITIAL_ENGINEERING constants and
ran Adam through 25 iterations, continuing from atomic iteration checkpoints.
The training loss decreased from 2111.979 to a best value of 151.339. Evaluating
the true constants on the fit grid/particle quadrature gives a loss of 142.265.
That floor is the fixed synthetic discretization/noise mismatch; fitting nearly
reaches it. It does not imply all nine engineering constants were recovered.

| Parameter | Truth | Best fit | Absolute relative error |
|---|---:|---:|---:|
| E1 | 1.200 MPa | 1.258 MPa | 4.9% |
| E2 | 0.800 MPa | 0.734 MPa | 8.2% |
| E3 | 0.600 MPa | 0.585 MPa | 2.6% |
| nu12 | 0.220 | 0.250 | 13.6% |
| nu13 | 0.180 | 0.223 | 23.9% |
| nu23 | 0.250 | 0.330 | 32.1% |
| G12 | 0.420 MPa | 0.473 MPa | 12.7% |
| G13 | 0.340 MPa | 0.360 MPa | 5.9% |
| G23 | 0.260 MPa | 0.226 MPa | 13.3% |

The noise-whitened local response Jacobian is full rank (9/9), condition number
4.40. This local rank does not establish accurate or global parameter recovery;
the Poisson ratios remain biased, especially nu23.

## Independent holdout

The fit was checked against a separate 0.36 × 0.28 × 0.25 m block and mixed
axial/torsion loading, with disjoint sensors and a different time step (32 μs)
and 250 steps. The true and inferred responses were produced by full MPM scans.
The holdout uses grid 48 / ppc 3 (74,880 particles), and it was not included in
the objective. Normalized RMSE relative to true-response RMS is 5.26% for
sensor displacement, 5.12% for reaction force, and 3.23% for reaction moment.
The saved `holdout_synthetic.npz` contains the computed histories; the plots
show sampled channels directly from those histories without temporal
interpolation. No particle snapshots/VTK frames were saved for ParaView, so no
particle-motion animation is claimed.

## Final-parameter directional derivative check

A separate audit evaluated the exact best checkpoint at grid 48 / ppc 2 against
the same frozen training archive and objective. With epsilon 2e-4, autodiff
(direction dot gradient) was −669.343604811397; the central finite difference
was −669.3436236923844; relative error was 2.82 × 10⁻⁸ (threshold 5 × 10⁻³).
The direction, fit-result/checkpoint/data hashes, and software versions are in
`gradient_check.json`.

## Reproduction and provenance

- Numerical fit source: `examples/inverse_orthotropic_svk.py` SHA-256
  `12b34673780ee69247660f705a0fbe6ca34ece8f27ead9bacca92241e5253929`. The
  current source includes post-run metadata wording for particle-quadrature
  differences and persists the existing derivative audit in checkpoint receipts;
  these changes do not alter the forward model, objective, or recorded run. The
  recorded result also includes explicit grid and ppc values that establish the
  same-background-grid comparison. Current source SHA-256 is recorded in
  `manifest.json`.
- Frozen training observations: `training_synthetic.npz`, SHA-256
  `022e915e098b3212439167c0496c3e555a58318a4d337971d2deb802ee2b4269`.
- Final fit result: `result.json`; SHA-256
  `63278383197e515201c573a7ec79a832fb78c6e408d35b96d1c52a911d9dd212`.
- Best-checkpoint container: `jaxmpm-j3-inverse-grid48-ppc2-20261027-r8`,
  cpuset 52–55, 4 CPUs, 12 GiB memory limit, 9 GiB USS guard, 1800 s wall
  guard. It completed in 733.86 s, peak USS 2.33 GiB, with no guard trigger.
- Directional check container: `jaxmpm-j3-final-gradfd-grid48-ppc2-20261027`,
  same resource limits; completed in 243.70 s, peak USS 2.28 GiB.
- Both used image `sha256:8390e590627e7859db7933d92c4c39546e0f5b8c7a2e9c194145ec6858e84890`,
  JAX 0.12.0.1 and NumPy 2.5.4.
- `loss_history.csv`, `checkpoint.npz`, both remote status receipts, and both
  training and holdout synthetic histories are included. Their SHA-256 digests
  are recorded in `manifest.json`.
- To regenerate the figures, install Matplotlib and run
  `python tools/plot_orthotropic_inverse_evidence.py --result evidence/orthotropic_inverse_20261027/result.json --loss-history evidence/orthotropic_inverse_20261027/loss_history.csv --holdout evidence/orthotropic_inverse_20261027/holdout_synthetic.npz --output-dir evidence/orthotropic_inverse_20261027`.
