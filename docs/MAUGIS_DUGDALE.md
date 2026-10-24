# Maugis–Dugdale local contact

`mpm_jax.maugis_dugdale` solves a circular, frictionless elastic half-space
contact with constant cohesive traction `sigma = W / h`. It determines the
intimate-contact radius `a` and the outer cohesive radius `c` together with the
force. The `a = 0, c > 0` branch is retained during separation. This differs
from the existing explicitly coupled finite-range surface traction in
`adhesion.py`, which does not solve the Maugis sphere problem.

The constitutive inputs are effective modulus `E*`, effective radius `R`, work
of adhesion `W` in J/m² and interaction range `h` in metres. Two isotropic bodies
have `1/E* = (1-nu1²)/E1 + (1-nu2²)/E2`. Positive indentation and force denote
compression. The default `W = 0.1 J/m²` is an adjustable example value, not a
measured material property.

```python
from mpm_jax.maugis_dugdale import (
    maugis_parameters, initial_maugis_history, update_maugis,
)

p = maugis_parameters(effective_modulus=5.33e9, effective_radius=117.5e-6,
                     work_of_adhesion=0.1, interaction_range=10e-9)
history = initial_maugis_history()
history, response = update_maugis(history, indentation=20e-9, parameters=p)
```

Construction validates parameters on the host. The update uses JAX throughout,
supports scalar or broadcast arrays, and can run under `jit` and `lax.scan`.
`response.valid` must be checked before accepting an update. `phase` is 0 for
detachment, 1 for a cohesive disk without intimate contact, and 2 for intimate
contact with a surrounding cohesive annulus.

## Equations and branch selection

Set `K = 4E*/3`, `L = (pi W R²/K)^(1/3)`,
`lambda = 2 sigma R/(K L)`, `A = a/L`, `C = c/L`,
`D = delta R/L²`, and `P = F/(pi W R)`.
For `q = a/sqrt(c²-a²)`, write `B = sqrt(C²-A²)`,
`theta = atan2(1,q)` and `H = sqrt(1+q²)`. The cohesive-energy condition is

```
alpha B² + beta B = 1
alpha = lambda/2 [q + (1-q²) theta]
beta  = 4 lambda²/3 [theta - 1/(H+q)]
A = q B; C = H B
D = A² - 4 lambda B/3
P = A³ - lambda [A B + (A²+B²) theta]
```

The positive quadratic root avoids a nested solve for `a` and `c`. A bracketed
scalar solve then selects the largest-radius branch with `dD/dq > 0`. Large-q
series avoid cancellation in the energy condition. The response records both
equation residuals and the stability decision.

For the noncontact cohesive disk, set `b = 8 lambda/(3 pi)`:

```
D = C²/2 - b C - 2/(pi lambda)
P = -pi lambda C²/2
central_gap = (L²/R) [-D - 4 lambda C/3]
```

The stable disk root is `C >= b`. At sufficiently small lambda, intimate
contact vanishes before this branch loses stability. Its displacement-control
fold is `Doff = -2/(pi lambda) - b²/2`. At larger lambda the contacting branch
has its own fold. A detached contact attaches when the undeformed gap reaches
`h`; an attached contact follows its stable branch until the appropriate fold.
An eight-machine-epsilon budget in dimensionless displacement treats rounding
at the fold as detachment. This is a floating-point endpoint convention, not a
physical increase in the adhesive range.

`W = 0` uses Hertz directly. Finite-lambda tests approach the JKR and DMT
**contact-branch** formulas at fixed contact radius. These comparisons do not
identify load-controlled pull-off with the displacement-controlled fold.

## Energy and derivatives

On the noncontact branch the potential is

```
U = -8 sigma² c³/(3 E*) - pi W c²
    + pi sigma c⁴/(4 R) - pi sigma delta c².
```

The contacting-branch potential continues from `a = 0` by integrating `F ddelta`
with 64-point Gauss–Legendre quadrature. Tests check this derivative against the
force and independently integrate the analytic pressure. Gradients of the
stable root use implicit differentiation. Derivatives are not defined at snap
events, the `a = 0` transition, or exactly `W = 0`; parameter fitting must stay
within a smooth branch. Snap energy changes are not frictional dissipation.
The work relation assumes fixed `E*`, `R`, `W` and `h`; changing a local curvature
or material parameter introduces additional energy-work terms.

Both the parabolic geometry and the linear elastic half-space approximation
require a contact/cohesive patch small relative to body curvature and dimensions.
Very small lambda can produce a large cohesive disk, invalidating that geometric
approximation even when the algebraic solve succeeds. A circular equivalent
radius for crossing fibres is a local approximation, not the full elliptical
crossed-cylinder solution. Applying this half-space force directly to an MPM
body whose same local compliance is already resolved would count compliance
twice; integration must define the unresolved interface compliance explicitly.

## Sources

- D. Maugis (1992), [Adhesion of spheres: The JKR-DMT transition using a Dugdale model](https://doi.org/10.1016/0021-9797(92)90285-T).
- K. L. Johnson (1997), [Adhesion and friction between a smooth elastic spherical asperity and a plane surface](https://bdml.stanford.edu/twiki/pub/Rise/AdhesionModels/JohnsonAdhesionModel.pdf), equations 2.6–2.14.
- W. Deng and H. Kesari (2019), [Depth-dependent hysteresis in adhesive elastic contacts at large surface roughness](https://doi.org/10.1038/s41598-018-38212-z), equation 3, including the noncontact extension attributed to Kim, McMeeking and Johnson (1998).
- [ContactEngineering/Adhesion Maugis reference solution](https://github.com/ContactEngineering/Adhesion/blob/master/Adhesion/ReferenceSolutions/MaugisDugdale.py), an independent MIT-licensed implementation used to check nondimensional conventions. The present parameterization and solver are independently implemented.
