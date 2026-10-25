# Three-strand forming candidate

This new case starts with three parallel straight stress-free cylinders. The
right material collar rotates one turn while both collars develop 0.5% axial
extension of the free span. The free material moves only through APIC momentum
transfer and constitutive evolution. No intermediate or final helix is prescribed.

Contact is standard separate-field frictionless nodal normal contact, not Hertz,
JKR, adhesion, or a resolved surface pressure. Mass-gradient normals follow the
existing contact.py convention. Three unilateral constraints share the nodal
velocities and are solved by dual projected Gauss-Seidel to a closing/KKT
residual of 1e-8 m/s, at most 64 sweeps. Failure stops the step. Prescribed nodal
fields have zero inverse mass in the projection; equal/opposite pair impulses
are balanced by an explicit fixture counter-impulse at constrained fields.

The finite-increment rotation velocity is (Q-I)x/dt. Full quadratic reproduction
then maps a fully constrained particle by Q and updates F by Q, avoiding explicit
Euler radial expansion. Material collars are fixed reference particle masks,
not moving geometric selections. Their full occupied stencil is prescribed.

The geometry is L=24 mm, R=1 mm, initial orbit=1.4 mm, particle spacing=0.1 mm,
grid h=0.2 mm, 227520 particles. E=50 kPa, nu=0.25, rho=1000 kg/m3 are illustrative
elastic parameters; crack-band history is retained but any damage rejects the
forming step. Duration 20 ms, 5164 steps, CFL=0.15. The maximum commanded angular
speed is about 331 rad/s; outer collar speed about 0.79 m/s versus longitudinal
wave speed 7.75 m/s. This is a dynamic process, not certified quasistatic forming.

Source closure and pure NumPy geometry planning pass. JAX kernel tests and actual
64-step timing/geometry prefix are pending assigned Linux resources. No full-turn
trajectory, resolution convergence, nonpenetration certification, or media exists.
The saved section-centre circular gap is explicitly a proxy; nodal closing
residual is not a sub-grid surface gap. Grid support, J>0, finite constitutive
history, zero damage and converged simultaneous contact are step-level guards.

Run the small kernel check with python -m tools.check_fibre_forming_contact, then
python -m tools.run_three_strand_forming --output OUTPUT --stop-steps 64.
The runner refuses Windows physical allocation and caps process-tree USS at 5 GiB
and wall time at 900 s. A later segment must use --resume and the identical source
plan signature. Only completed sampled states replace the checkpoint.

Energy outputs distinguish imposed fixture work, initial fixture projection loss
and contact projection loss. APIC transfer and constitutive time integration are
not a conservative total-energy certificate. No such certificate is claimed.

Stage 2 is planned as actual slow viscous MPM filling of a cylindrical mold with
the accepted formed fibres held fixed as obstacles. A finite inlet reservoir can
avoid dynamic JAX particle-count changes. This future approximation will be
labelled fixed-obstacle filling, not curing or complete fluid-structure interaction.
No prefabricated cylinder fade or manufactured fibre motion is permitted.
