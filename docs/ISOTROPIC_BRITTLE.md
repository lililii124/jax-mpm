# Isotropic brittle fibre material

`mpm_jax.materials.isotropic_brittle` provides a separate material-point model
for tensile fracture demonstrations. It has its own `BrittleHistory`; it does
not alter the existing solver's material registry or plasticity history.
Parameters are illustrative, not measured fibre data.

## Elasticity and stress convention

For a positive-determinant deformation gradient, use its right polar
decomposition `F = R U`, Biot strain `e = U - I`, and

```text
T0 = lambda tr(e) I + 2 mu e
psi0 = lambda/2 tr(e)^2 + mu e:e
P = (1-D) R T0
tau = P F^T
sigma = tau / det(F)
psi = (1-D) psi0
```

`psi` is stored energy per reference volume and `P` is its derivative with
respect to `F` at fixed damage. `tau` is Kirchhoff stress for a transfer that
uses reference particle volume; `sigma` is Cauchy stress. Rigid rotation changes
the stress orientation without changing damage or stored energy.

The elastic law extends isotropic linear elasticity to finite rotations. It
does not supply calibrated finite-strain behaviour. The failure strength `ft`
is a **nominal stress on the reference area**, not a Cauchy-stress threshold.
For free-side uniaxial extension `eps`, lateral stretches are `1-nu*eps` and
the nominal stress is `(1-D)*E*eps`; the Cauchy stress divides this by
`(1-nu*eps)^2`.

## Tensile damage and crack-band length

The driving strain is the positive maximum principal undamaged Biot stress
divided by `E`. History stores its irreversible maximum `kappa`. Define

```text
delta = h kappa
delta0 = h ft/E
deltaf = 2 Gf/ft
```

Damage is zero below `delta0`, one at or above `deltaf`, and between them

```text
D = deltaf (delta-delta0) / [delta (deltaf-delta0)]
```

Here `delta` is an equivalent total band extension. The maximum equivalent
opening is `w = D delta`; it is a scalar diagnostic, not a resolved crack
surface. In monotonic free-side uniaxial loading the softening relation is
`Pxx = ft*(1-w/deltaf)`. Its area from `w=0` to `deltaf` is `Gf`. Integrating
nominal stress over total band extension from the initially unstressed state
to complete failure gives the same energy, including released pre-peak elastic
energy. Unloading and reloading below the previous maximum use the damaged
elastic stiffness and do not heal.

The constructor requires `deltaf > delta0`, equivalently
`h < 2 E Gf/ft^2`. An unresolved band is rejected rather than changing its
strength or fracture energy. `h` is a reference localization-band length,
specified explicitly as a scalar or one value per particle. It must not be
silently replaced by the particle spacing when changing particles per cell.
Using the grid spacing for `h` is a discretization choice requiring structural
refinement checks; the material-point energy check alone does not establish
mesh-independent fracture paths.

Virgin states with all principal undamaged stresses nonpositive do not
initiate damage. Once damaged, **all elastic stiffness is reduced**, including
compression and shear. At `D=1` this material carries no stress in any mode;
it does not recover compressive stiffness on closure. Crack-face contact,
fragment separation and particle mass handling belong to the spatial solver.
No viscosity, plasticity, fatigue, friction or nonlocal interaction is included.

The thermodynamic damage force is `psi0 >= 0`, so an irreversible damage
increment has nonnegative dissipation. A general multiaxial path's accumulated
dissipation cannot be inferred from `kappa` alone. Compute it from actual
stress work and stored energy; the module does not invent a cumulative energy
history from a uniaxial formula.

## API and demonstration values

```python
from mpm_jax.materials.isotropic_brittle import (
    initial_brittle_history, isotropic_brittle, update_brittle,
)

material = isotropic_brittle(
    young=10e9, poisson=0.25, density=1500,
    tensile_strength=100e6, fracture_energy=100,
    characteristic_length=25e-6,
)
history = initial_brittle_history(particle_count, dtype=F.dtype)
history, tau = update_brittle(history, F, material)
```

These values give `deltaf=2 micrometres`, maximum admissible `h=200
micrometres`, and onset strain `0.01`. At `h=25 micrometres`, onset band
extension is `0.25 micrometres` and complete-failure equivalent strain is
`0.08`. The update is rate independent and accepts no `dt`. It is a pure JAX
function and can be JIT-compiled or used in `lax.scan`. History fields are
`kappa`, `damage`, `failed`, and `valid`; no plastic-strain slot is reused.
An inverted, singular or nonfinite `F` marks the history invalid. The caller
must reject that state. A failure flag alone does not delete a material point.

`brittle_energy`, `brittle_first_piola`, `brittle_kirchhoff_stress`, and
`brittle_cauchy_stress` evaluate a fixed history. Differentiating the stored
energy must hold damage fixed; differentiating through a damage update is not
the elastic stress law. Spectral operations have the usual nonsmooth maximum
at tied principal tensile stresses; no smooth damage tangent is claimed.

## Checks and sources

```powershell
python -m pytest tests/test_isotropic_brittle.py -q
```

The checks cover actual nominal and Cauchy uniaxial stress, fracture energy at
three band lengths, opening softening, unload/reload cycles, compression,
failure retention, energy derivatives, rigid rotation, reflection, monotonic
step partition and invalid deformations. They do not validate a fibre-scale
MPM simulation or an experimental material.

The crack-band length/strength/energy construction follows the framework in
Bažant and Oh, [*Crack band theory for fracture of concrete*, Materials and
Structures 16, 155–177 (1983)](https://link.springer.com/article/10.1007/BF02486267).
The traction–opening interpretation originates in Hillerborg, Modéer and
Petersson, [*Analysis of crack formation and crack growth in concrete by means
of fracture mechanics and finite elements*, Cement and Concrete Research 6,
773–781 (1976)](https://doi.org/10.1016/0008-8846(76)90007-7).
Those sources support fracture-energy regularization, not the illustrative
fibre constants or a claim that this isotropic model reproduces their complete
constitutive formulations.
