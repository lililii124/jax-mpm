# Orthotropic finite-strain solid

`orthotropic-svk` implements the supplied MAT002 solid kernel's St. Venant–Kirchhoff law:

```
E = (FᵀF − I)/2
S = C:E
τ = F S Fᵀ
Ψ = (E:S)/2
```

`τ` is Kirchhoff stress and `Ψ` is energy per reference volume. The existing `orthotropic` factory supplies the stiffness and fixed reference material axes. Public Voigt order is `xx, yy, zz, yz, xz, xy`, with engineering shear strains. The original `linear`, `orthotropic` and `transverse-isotropic` branches retain their small-strain, small-rotation formulation.

Run a three-dimensional, freely released prestrained block:

```sh
python -m examples.elastic_models --model orthotropic-svk --grid 128 --frames 1 --substeps 20 --output outputs/mat002-svk
```

The example uses demonstration elastic constants in Pa and initial stretches `(1.12, .96, .98)`, zero gravity and no imposed traction or clamp. It is a constitutive demonstration, not a calibrated material experiment.

The targeted check compares 64 float64 stresses with compiled native-kernel outputs, checks `∂Ψ/∂F · Fᵀ = τ` and superposed-rotation objectivity, then advances the real grid128 block for 20 steps and records full initial/final states, positive determinants, mass, momentum and energy:

```sh
python -m tools.check_mat002_svk --reference evidence/mat002-svk/native-reference.json --output outputs/mat002-svk-check
```

The native reference is bound to MAT002 source SHA256 `1c7bd28b31df78c3eb5905e9f59cf53a0e880f5a5cc710b89be5678cf2d933fb`; private Fortran source is not included. Its native Voigt order was converted to the public order. Native stress comparison covers identity material axes and superposed spatial rotations; it does not certify a rotated material-axis adapter.

The [single-run receipt](../evidence/mat002-svk/review.json) records a maximum native stress error of `7.14e-16` relative to the stress scale. The 456,533-particle release retained 27kg mass, ended with minimum `J=.9224`, and changed total kinetic plus strain energy from 17.5343J to 17.0026J (`−3.03%`) over 4ms. Its momentum error was `1.12e-14 kg m/s`. These are results of this discrete prefix; the energy change is retained as numerical error.

This check does not establish spatial or temporal convergence, long-time energy conservation, or stability under severe compression. St. Venant–Kirchhoff elasticity is a finite-strain hyperelastic model with a restricted useful deformation range.
