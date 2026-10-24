# Compressible two-body prototype

`mpm_jax.impact_mpm` couples two independently deformable bodies to the
[shock-metal material](SHOCK_METAL.md). It uses separate APIC grid fields and
frictionless finite-mass contact at matching global nodes. This is an explicit
numerical prototype, not a validated hypervelocity-impact model.

The stress used in both force and work is
`tau_dev - J * (pressure + artificial_viscosity) * I`. The compressive artificial
viscosity enters that work once. Total specific internal energy includes EOS
energy, stored deviatoric elastic energy and cold work. The strength return
repartitions this total; it does not add plastic heat a second time.

Grid kinetic energy is balanced against stress and impulse work evaluated at the
midpoint of the actual old and final grid velocities. The interface uses one
mass-weighted midpoint velocity for equal and opposite body energy exchanges.
Nonnegative instantaneous contact projection loss and **signed numerical
coupling work** are recorded separately. The latter is not physical heat. Their
sum, distributed using the opposite nodal mass fraction, supplies the compatible
particle allocation. The same separation applies to fixed-grid projection and
the boundary energy reservoir. APIC transfer variance losses are nonnegative and
enter particle internal energy once.

Particle positions and deformation gradients use the post-contact velocity
field. The Cayley deformation/spin update provides objective rotation; the
frozen-stress spatial update and thermal partition remain first order in time.
Adaptive steps use the EOS longitudinal speed, particle and affine transport,
and viscosity restriction. Invalid support, unresolved occupied interface
normals, invalid EOS states and failed energy/momentum checks stop advancement.
No mass deletion or energy clipping repairs an invalid state.

Closing contact whose normal intersects a fixed grid degree of freedom is
currently unsupported and rejected. The free two-mass projection cannot enforce
that additional constraint; re-clamping alone would leave a closing velocity.
Purely tangential fixed components do not cause this rejection.

## Numerical limits and checks

Exact total-energy accounting does not prove a local entropy inequality. The
state retains initial and minimum particle entropy and the cumulative sum of
negative entropy increments. Contact projection losses and signed corrections
are also retained by body. Per-particle phase/tension flags accumulate over the
whole advance, together with maximum temperature and minimum/maximum volume
ratio; returning to an unflagged final state does not erase an earlier excursion.
Smooth reversible paths can have a finite thermal
truncation error, and contact-startup heating can remain concentrated in a
shrinking layer as the grid is refined.

The focused tests cover transfer losses, grid stress-power duality, individual
body interface work, Galilean changes of frame, rigid spin, boundary accounting,
carry validation and smooth-path time refinement. A symmetric, strength-free
planar collision at ±500 m/s compares the pressure, volume ratio and specific
energy against the material's linear Hugoniot at three grid spacings. It records
entropy histories and signed work terms, not only final total energy. These
controls do not validate the separate 5000 m/s sphere-and-plate geometry.

The EOS uses a single compressible phase with an expansion extrapolation and
temperature-based strength softening. Phase and tension extrapolation flags are
reported. Melting thermodynamics, latent heat, vaporization, cavitation, thermal
conduction and material failure are not resolved. A debris-impact run needs its
own domain, temperature, phase, time-step and spatial checks before any result
can be interpreted physically.

```python
from mpm_jax.impact_mpm import ImpactConfig, build_grid, make_state, advance_to
from mpm_jax.materials.shock_metal import shock_metal

material = shock_metal()
grid = build_grid(x, body_id, dx=dx, reference_bounds=bounds)
state = make_state(x, reference_volume, body_id, material, grid, v=velocity)
state, steps = advance_to(state, material, grid, ImpactConfig(), output_time)
```

Enable JAX float64 before constructing the state. `body_id` must contain both
0 and 1; the present prototype uses one material parameter set for both bodies.
The returned validity flag must be checked before saving or continuing a run.
`validate_state` checks the registered shapes and float64 fields before a host
resumes a saved state; broadcasting a shorter history is not accepted.
