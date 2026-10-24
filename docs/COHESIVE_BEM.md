# Self-consistent axisymmetric cohesive contact

`mpm_jax.cohesive_bem` solves the elastic displacement and the contact/cohesive
traction together. It is a local axisymmetric half-space model with annular
elements, not a boundary-element solution of an entire fibre network or an MPM
plate. The initial profile is a paraboloid; no Hertz or JKR pressure footprint is
prescribed.

The annular compliance is a Galerkin discretization of the Boussinesq kernel
`1/(pi E* distance)`. A ring force produces the mean displacement of every ring.
Both radial integrals are evaluated, the logarithmic singularity is split, and
the resulting reciprocal matrix is checked positive definite on construction.
The dense matrix is small: 16 or 32 rings are initial local discretizations;
contact-edge resolution must be assessed separately.

Let `g0 = profile - delta`, `g` be the current mean gaps, `C` the compliance,
`f = C^-1 (g-g0)` the ring forces, and `Ai` the ring areas. The solved potential is

```
U(g, delta) = 0.5 (g-g0)^T C^-1 (g-g0) + sum Ai phi(gi)
phi(g) = min(sigma g - W, 0),  g >= 0
sigma = W / h
```

At equilibrium:

| Region | Gap | Mean pressure |
|---|---|---|
| Contact | `g = 0` | `p >= -sigma` |
| Cohesive | `0 < g < h` | `p = -sigma` |
| Open | `g >= h` | `p = 0` |

Each coordinate update exactly minimizes the two possible gap intervals and
cannot increase the discrete potential. The previous gap/displacement provides
the warm start. This defines an energy-decreasing relaxation branch, not a
universal continuum snap criterion. The finite ring discretization gives
staircase changes in the cohesive radius; force error need not decrease
monotonically with every mesh refinement. The tests compare specified states
at 32 and 64 rings with the independently solved Maugis contact, and separately
check the Hertz limit, compliance quadrature, equilibrium and energy work.

```python
from mpm_jax.cohesive_bem import (
    cohesive_bem, initial_cohesive_bem_history, update_cohesive_bem,
)

model = cohesive_bem(effective_modulus=1e6, effective_radius=0.01,
                    work_of_adhesion=0.0042441318,
                    interaction_range=0.6366198e-6,
                    outer_radius=0.4e-3, rings=32)
history = initial_cohesive_bem_history(model)
history, response = update_cohesive_bem(history, 1e-6, model)
```

These are illustrative parameters. Matrix preparation is host-side NumPy; the
relaxation is JAX and accepts `jit`. Use `vmap` for independent contacts. Keep
`max_sweeps` and `tolerance` static under JIT. The response exposes ring gaps,
pressures, displacement, energies, force, areas, iteration count, normalized
equilibrium residual, compatibility residual and energy increase. Failure to
converge or an active outer boundary makes `valid = false`; the last iterate is
diagnostic output, not an accepted equilibrium. The outer two rings must be
open. Increasing the radial domain and resolving the contact/cohesive edges
remain necessary spatial checks.

The total force equals the partial derivative of the energy with respect to
indentation at fixed equilibrium gaps. The iterative branch selector is not a
differentiable constitutive map across active-set changes; reverse-mode AD
through the convergence loop is not provided. Positive pressure is compressive,
negative pressure attractive. No tangential friction or irreversible cohesive
damage is present.
The work derivative holds at fixed geometry, compliance and cohesive parameters.

This model includes half-space elastic compliance. It may represent an explicitly
unresolved local interface in a fibre model, or a standalone pull-off reference.
Adding it as a second bulk compliance to an already resolved MPM contact is not
a self-consistent coupled model. A future MPM surface relaxation would instead
need to use the actual MPM displacement response and a separately checked
equilibrium residual.

The governing half-space/cohesive equations follow
[Bazrafshan et al. (2017), Numerical method for the adhesive normal contact analysis based on a Dugdale approximation](https://doi.org/10.1016/j.triboint.2017.04.001).
The ring energy-coordinate solver here is independently implemented; it is not
that paper's conjugate-gradient algorithm. The energy formulation is also
consistent with the elastic-plus-interface construction in the official
[ContactEngineering/Adhesion systems](https://github.com/ContactEngineering/Adhesion/blob/master/Adhesion/System/Systems.py).
