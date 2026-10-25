# Local grid refinement

`mpm_jax.hierarchical_grid` implements a fixed, two-level, three-dimensional
quadratic THB-spline basis and APIC transfer operators. Refinement faces must
coincide with coarse spline knots, and particles must retain complete background
support. Time integration and contact are not yet connected to these operators.

The fine Greville lattice is shifted by half a fine cell. Coarse functions are
truncated using the quadratic refinement mask. Both levels contribute to one
grid; the stencil contains at most 54 entries per particle.

The APIC moment is `D = sum(w * d * d.T)`, and the carried affine moment is
`B = C D`. `D` varies through the transition. The physical velocity gradient
uses the analytical basis derivatives. Internal forces use the same derivatives,
so grid internal power equals negative material stress power. The uniform-grid
`4/h²` substitution is not used in the transition.

## First application

A 160 × 160 × 10 mm elastic plate is clamped along four 5 mm edge strips, with
no backing. A 40 mm hemisphere is to indent the centre by 4 mm. Coarse and fine
spacings are 2.5 and 1.25 mm. The central 60 × 60 mm particle region is refined
through the thickness; the basis refinement box includes its support padding.

The mixed layout contains 260,096 particles, compared with 1,048,576 for uniform
fine sampling. Both represent the same volume and mass. These counts alone do
not establish accuracy or a performance advantage. The next comparison uses
identical constitutive and transfer formulations on coarse, fine and hierarchical
grids, measuring force–displacement, centre deflection, contact radius, energy,
interface wave reflection, memory and run time.

## Checks

```powershell
python -m pytest tests/test_hierarchical_grid.py -q
python tools/check_hierarchical_transfer.py --output outputs/thb-transfer
```

Six targeted tests cover partition of unity, nonnegative weights, linear
precision, derivative consistency, variable moments, affine reproduction,
frozen-position transfer conservation and internal virtual work.

The [plate-layout probe](results/hierarchical-transfer.json) passed on Windows
CPU with JAX 0.12.0.1 and float64.
One stencil evaluation took 0.504 s; one transfer/force check took 0.706 s,
excluding compilation. The separately monitored process peaked at 2.14 GiB USS.
The maximum returned velocity error was 7.55e-15 m/s, and the internal-power
difference was 1.35e-14 W. These are single-call measurements, not full-step or
steady-state timings. Dynamic advection, contact and indentation accuracy remain
to be checked.

The basis design follows local-refinement work by
[Sun et al.](https://doi.org/10.1002/nme.6312) and
[THB-spline MPM](https://doi.org/10.1016/j.compgeo.2021.104097).
The affine moment representation follows
[APIC](https://doi.org/10.1145/2766996).
