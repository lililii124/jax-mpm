# Local grid refinement

`mpm_jax.hierarchical_grid` implements a fixed, two-level, three-dimensional
quadratic THB-spline basis and APIC transfer operators. Refinement faces must
coincide with coarse spline knots, and particles must retain complete background
support. An experimental explicit integrator is available in
`mpm_jax.hierarchical_dynamics`; engineering contact is not yet qualified.

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
steady-state timings.

## Dynamic prototype

The neo-Hookean prototype carries `B` through particle advection and reports the
resulting change in moment-dependent kinetic energy separately. Its energy
record also separates particle/grid transfer, applied work, constraint
projection, constitutive increment and force-update terms. Four targeted checks
cover moving particles across both refinement interfaces, momentum balance,
fixed constraints and energy bookkeeping.

```powershell
python examples/hierarchical_wave.py --resolution coarse --output outputs/wave-coarse
python examples/hierarchical_wave.py --resolution fine --output outputs/wave-fine
python examples/hierarchical_wave.py --resolution hierarchical --output outputs/wave-hierarchical
```

The [first pulse comparison](results/hierarchical-wave-first-comparison.json)
uses a 6 mm Gaussian width, 0.02 m/s amplitude and 650 µs duration. Each run
completed 600 steps with finite states, positive deformation determinants and
the same time step. Velocity errors against the linear small-amplitude solution
were 48.52%, 26.68% and 37.49% for coarse, fine and hierarchical grids. The fine
reference fails the 5% target; this resolution pair is not qualified for the
pulse. Small reflected tails and closed energy records do not establish an
accurate transmitted pulse. Spatial refinement and a qualified fine reference
are required before engineering use of the prototype.

The [0.5 mm uniform reference](results/hierarchical-wave-refined-reference.json)
completed 1,500 steps for the same pulse. Independent checks of the full archive,
source hashes, every recorded step and the final field reproduce a 5.83%
velocity error, down from 26.68%. This still exceeds the unchanged 5% target.
Total energy decreases by 0.51%; the Gaussian peak is 98.54% of its reference
amplitude. A best-fit position shift of -0.276 mm accounts for part of the error,
but even the shifted shape has 3.64% error. No shift is applied to the acceptance
metric. Peak USS is 4.22 GiB and the bounded Linux run takes 1,051 seconds.
Further work must resolve propagation error before a hierarchical engineering
comparison is accepted.

Both grids accept inclusive integer `node_bounds` for removing empty background
storage. All retained material points still require complete basis support.
Cropping checks compare the retained mass, momentum and internal forces against
the full grid. This storage option does not change the spline spacing or count
as local refinement by itself.

The pulse CLI also accepts `--coarse-cells` and `--transverse-ppc`. Axial sampling
remains two points per cell. A separate uniform-grid check confirms that reducing
transverse sampling from two to one does not alter this constrained plane-strain
mode. This equivalence does not apply to general three-dimensional deformation.

The basis design follows local-refinement work by
[Sun et al.](https://doi.org/10.1002/nme.6312) and
[THB-spline MPM](https://doi.org/10.1016/j.compgeo.2021.104097).
The affine moment representation follows
[APIC](https://doi.org/10.1145/2766996).
