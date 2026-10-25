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

The actual trajectory now reaches step5164 at20ms, with final imposed angle
2pi and collar-position error5.30e-16m. The original integrator supplies0->1280;
the boundary-aware midpoint candidate continues that exact checkpoint to5164.
The original64-sweep contact QP remains unchanged; bounded exact active-set
fallback and unique shared-node packing are described in the energy review.
All52 source files, saved NPZ CRCs/fields, parent mass/volume/IDs, material
validity and diagnostic elastic energy were checked on Linux. A63-snapshot
400x400 fixed-camera GIF contains only actual nonuniform saved states; every
PNG is distinct and first/last differ. No interior positions are interpolated.

Final J is[.61244,1.27523], principal stretches[.61185,1.55921], damage zero.
J<.8 affects.3046% of particles; max stretch>1.3 affects.3147%. This remains an
illustrative compressible finite-elastic dynamic example, not calibrated
manufacturing or demonstrated time/space convergence. The actual point/cell
geometry guards find no foreign-centre cell intrusion at the saved resolution;
this is not a smooth-surface or sub-grid nonpenetration certificate. Grid
support, J>0, finite history, zero damage and converged contact remain guards.
The separate dry tight candidate and resin filling are not this baseline.

Run the small kernel check with python -m tools.check_fibre_forming_contact, then
python -m tools.run_three_strand_forming --output OUTPUT --stop-steps 64.
The runner refuses Windows physical allocation and caps process-tree USS at 5 GiB
and wall time at 900 s. A later segment must use --resume and the identical source
plan signature. Only completed sampled states replace the checkpoint.

Energy outputs distinguish imposed fixture work, initial fixture projection loss
and contact projection loss. APIC transfer and constitutive time integration are
not a conservative total-energy certificate. No such certificate is claimed.

Stage2 is an independent experimental actual viscous filling process with the
accepted formed fibres frozen as obstacles. Preallocated particle slots become
active only at the physical inlet; they are not preplaced in a final cylinder.
Short inlet-flow pilots and a particle-wall event correction are under review,
not a completed filling result. This is fixed-obstacle filling, not curing or
complete fluid-structure interaction. No prefabricated cylinder fade is used.
