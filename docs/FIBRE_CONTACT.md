# Local fibre contact law

`mpm_jax.fibre_contact` supplies Hertz and displacement-controlled JKR laws
with independent attachment history. It is a **circular, equivalent-curvature
contact approximation**. The caller supplies the current gap, normal and
effective radius. The module does not solve the full elliptical contact of
crossed cylinders, determine curvature, detect fibre intersections, or
distribute contact tractions over the particle surfaces.

For two isotropic elastic materials,

```text
1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
```

For spherical surfaces `1/R = 1/R1 + 1/R2`; that identity alone is not an
elliptical-cylinder solution. A fibre application must state its reduction of
the two principal relative curvatures to `R_eff`, and check its range of
validity. Applying these laws to moving centreline sections represents
unresolved local contact compliance. It does not validate the fully resolved
three-dimensional pressure distribution or remove possible overlap between
continuum and reduced contact compliance.

## Force, energy and branch history

Indentation `delta` is positive in compression (`delta=-gap`). Hertz gives

```text
a = sqrt(R max(delta,0))
F = 4 E* a^3 / (3 R)
```

Its contact is active only at positive indentation. For positive work of
adhesion `W`, the JKR branch is

```text
delta = a^2/R - sqrt(2 pi W a/E*)
F = 4 E* a^3/(3 R) - sqrt(8 pi W E* a^3)
ac^3 = pi W R^2/(8 E*)
delta_c = -3 ac^2/R
```

The attached stable branch has `a>ac`. A fixed 64-iteration bracketed solve
selects its largest radius. A detached pair attaches on reaching `delta>=0`.
An attached pair remains attached while `delta>delta_c`, and detaches at or
below `delta_c`. A detached pair at a negative indentation exerts no force,
even when an attached equilibrium also exists there. There is no attraction
between previously separated surfaces at a finite range. `W=0` returns the
same outputs as Hertz, including its history transitions.

The comparison uses the threshold computed in that update and returned as
`critical_indentation`. Separately evaluated eager and JIT thresholds may
differ by floating-point rounding. A path intended to cross the instability
must cross it, rather than rely on bitwise equality to a threshold evaluated
in a different compiled expression. No distance tolerance or shifted physical
detachment threshold is applied.

This is displacement control. The often quoted load-controlled JKR pull-off
force `-3*pi*W*R/2` is not the detachment rule here. At zero indentation the
attached branch has `a^3=2*pi*W*R^2/E*` and `F=-4*pi*W*R/3`; approaching the
displacement instability gives `F=-5*pi*W*R/6`.

The potential at independent radius and indentation is

```text
U(a,delta) = E* a delta^2 - (2/3) E* a^3 delta/R
             + (1/5) E* a^5/R^2 - pi W a^2
```

At fixed `E*`, `R` and `W`, the attached branch has `dU/da=0`, `d²U/da²>0`,
and `dU/ddelta=F`. If those parameters evolve with the fibre geometry or
material damage, their additional energy derivatives are not included in
the normal force. Hertz's
stored elastic energy is `2*F*delta/5`. Detached contact has zero potential.
The potential can jump at attachment and detachment; those releases must be
accounted for by a spatial solver's energy budget. The branch switch alone
does not model acoustic radiation, viscous relaxation or frictional loss.
Derivatives use the implicit branch equation within smooth attachment;
there is no finite tangent at the instability and no derivative through the
discrete attachment event is claimed.

## API

```python
from mpm_jax.fibre_contact import (
    effective_modulus, fibre_contact_parameters,
    initial_fibre_contact_history, update_fibre_contact, normal_pair_forces,
)

law = fibre_contact_parameters(
    effective_modulus=effective_modulus(E1, nu1, E2, nu2),
    effective_radius=R_eff,
    work_of_adhesion=0.1,
    model="jkr",  # or "hertz"
)
history = initial_fibre_contact_history(shape=(pair_count,))
history, response = update_fibre_contact(history, -current_gap, law)
forces = normal_pair_forces(current_normal_from_first_to_second, response)
```

All quantities are SI. The default `W=0.1 J/m²` is an adjustable demonstration
assumption, not measured fibre adhesion. `E*` and `R_eff` must be positive;
`W` must be nonnegative. Construction performs host validation. Updates accept
scalar or broadcast batched JAX arrays and can be JIT-compiled. Changing
geometry may update `law._replace(effective_radius=current_radius)`; the
caller remains responsible for the chosen curvature model.

History fields are `active` and `valid`. Response fields are `radius`,
`normal_force`, `area`, `potential`, `active`, `snap_in`, `snap_out`,
`critical_radius`, `critical_indentation`, and `valid`. Invalid numeric input
latches `valid=False`; callers must reject it. `snap_in/out` record attachment
transitions; Hertz transitions are continuous rather than finite jumps.

The normal points from the first body to the second. The returned pair has
`first=-F*n`, `second=+F*n` and `valid`, with `n` normalized internally. Thus
positive force repels, negative force attracts, and there is no tangential
traction. Nonfinite or zero normals are invalid. Equal opposite resultants
conserve force; a coupling layer must additionally choose application points
and interpolation weights that conserve moment. It must update pair geometry
and keep history attached to physical pair identities when candidates change.
It must also decide how fractured material participates in contact; this law
does not restore the stiffness of failed fibres.

## Verification and references

```powershell
python -m pytest tests/test_fibre_contact.py -q
```

The tests compare the stable radius against an independent quartic solve,
check the energy derivatives and implicit radius tangent, test hysteretic
attachment/detachment, verify the exact Hertz limit, and check vector force
balance and rotation. They are contact-law checks, not validation of a woven
assembly or an elliptical contact model.

The spherical adhesion theory is from Johnson, Kendall and Roberts,
[*Surface energy and the contact of elastic solids*, Proceedings of the Royal
Society A 324, 301–313 (1971)](https://doi.org/10.1098/rspa.1971.0141).
Johnson's [*Adhesion and friction between a smooth elastic spherical asperity
and a plane surface* (1997), Section 2](https://bdml.stanford.edu/twiki/pub/Rise/AdhesionModels/JohnsonAdhesionModel.pdf)
sets out the Hertz, JKR and finite-range adhesion distinctions. The branch
limits and potential derivatives above follow directly from the stated
spherical equations. No Maugis–Dugdale transition, experimental calibration,
rolling resistance or friction law is claimed.
