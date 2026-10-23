# Failed particles

`Config(failure_mode="retain")` is the default and preserves the original grid
transfers. A material can set `history.failed` and remove its stress while the
particle still contributes mass and momentum to the grid.

`Config(failure_mode="ballistic")` removes failed particles from grid mass,
momentum, stress transfer and two-body contact. Their mass and reference volume
remain in the state. This is a numerical deactivation rule, not a crack-surface
or fragment-contact model.

```python
config = Config(dim=3, material="johnson-cook", failure_mode="ballistic")
material = johnson_cook(failure=(D1, D2, D3, D4, D5))
```

The parameters must use the stress and strain-rate conventions described below.
They are not supplied by the grid-deactivation option.

## Time integration

A particle participates in the step that first sets `history.failed`. It keeps
that step's position, velocity, deformation and constitutive history. Its affine
velocity matrix is cleared. Starting with the next step, it contributes nothing
to the grid and follows

\[
v^{n+1}=v^n+\Delta t\,(g+f_p/m_p),\qquad
x^{n+1}=x^n+\Delta t\,v^{n+1}.
\]

Here `force` is the applied force on each particle, in newtons. Both `step` and
`coupled_step` apply it to failed particles directly. Their deformation gradient
and every constitutive-history field remain frozen. Grid walls, grips and
friction do not act on them, and they can leave the grid domain.

Inactive coordinates are replaced by safe coordinates before constructing their
stencils; their weights and gradients are zero. Active particles must retain
their full quadratic support inside the grid. Finite values, positive mass and
volume, and positive deformation determinants remain required for every
particle, including debris.

`State` and `History` keep their existing fields, so saved particle arrays need no
new mask. The mask is reconstructed from `history.failed` and the serialized
`Config.failure_mode`. Resume with matching configuration and solver revision.

## Diagnostics and interpretation

The original mass, momentum and kinetic-energy diagnostics include all
particles. Ballistic runs additionally report `active_particles`,
`debris_particles`, `active_mass`, `debris_mass`, and `active_current_volume`.
The original `current_volume` also includes the frozen volume of failed particles.
Particle count or colour alone does not establish that a section has separated.

For a tensile specimen, inspect loss of the unfailed load-bearing section,
separation of the remaining material and the grip-force response. The two sides
still use the same MPM grid. While their shape-function supports overlap, they
can remain numerically coupled even after intervening particles fail. Resolving
an arbitrarily narrow crack requires additional kinematic fields or a dedicated
fracture/contact method.

The present option has no crack-face contact, fragment recontact, cohesive law,
fracture-energy regularization or mesh-objective localization width. Deactivation
also discards the affine part of particle motion; it does not preserve its spin
or provide a complete energy balance. Frozen history is retained for inspection,
not evolved as a material model for the debris. Failure strain, separation time
and post-failure response can depend on grid spacing, particle spacing and time
step.

## Johnson–Cook convention

The existing constitutive law accumulates

\[
D_{n+1}=\min(1,D_n+\Delta\bar\epsilon^p/\epsilon_f),\qquad
\epsilon_f=(D_1+D_2e^{D_3\eta})
\left[1+D_4\log\max(\dot{\bar\epsilon}^p/\dot\epsilon_0,1)\right]
(1+D_5T^*).
\]

It sets `failed` at `D >= 1` and clears the complete stress tensor. Here
`eta = mean_tensile_stress / von_Mises_stress = -pressure / q`; pressure is
positive in compression. The rate factor is held at its reference value below
the reference rate. The damage law does not itself add gradual stiffness
degradation before this threshold. The cumulative criterion and its separable
triaxiality, rate and temperature factors follow the form documented in the
[Sandia LAMÉ manual](https://www.sandia.gov/files/sierra/SM_LAME_5_20/main/models/failure/jc.html).

Coefficient signs must be checked when transferring data. The
[Abaqus Johnson–Cook documentation](https://abaqus.uclouvain.be/English/SIMACAEMATRefMap/simamat-c-johnsoncook.htm)
uses a compression-positive pressure ratio and explicitly notes the opposite
`D3` sign relative to the original form. Its element-removal choices are not
equivalent to the mass-retaining ballistic rule here. The original reference is
[Johnson and Cook (1985)](https://doi.org/10.1016/0013-7944(85)90052-9).

## Checks

`tests/test_failure.py` checks forced and gravitational free flight, preservation
of mass and linear momentum, frozen history, checkpoint round trips, zero grid
transfer from failed particles, active-support rejection, exclusion from two-body
contact, unchanged default behaviour, and a constitutive JC threshold crossing
followed by ballistic motion. These are implementation checks, not experimental
fracture validation.
