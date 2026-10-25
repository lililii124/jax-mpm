# AT2 tension localization repair: source candidate

The previous 2800-step PIC case is rejected as a localized-crack demonstration.
Its d>=0.95 region occupies 15.19% of material volume; the native d>=0.99
mid-thickness profile is about 0.143 wide. Preserve that raw evidence.

New implementation: `examples/phase_field_tension_persistent_grid.py` with
`examples/phase_field_tension_geometry.py`. This uses the previously checked
persistent nodal mechanics and material-point integration, not repeated PIC
velocity transfers. It is a new boundary/initial-value problem, not yet validated.

Plate: 0.6 x 0.6 x 0.025, E=1000, nu=0.2, rho=1, Gc=0.0002,
ell=0.0375, h=0.0125 (ell/h=3), eight integration particles per full grid cell.
Full material volume=0.009, 36864 particles. No geometric void is removed.
The edge precrack ends at x=0.3 on y=0.3. Because the native quadratic basis
needs three unit coefficient rows to attain d=1 on the centerline, its seed is
explicitly |y-0.3|<=h and x<=0.3. Its coefficient width is 2h, not an exact
zero-thickness trace. A zero-history minimum-energy initial solve creates its
diffuse tail. Seed energy and masks are exported. Initial damage is never
counted as crack growth. No claim of equivalence to a sharp geometric slit.

Normal displacement is prescribed on bottom/top exterior support rows. Other
transverse degrees of freedom are free except Ax, Az, Bz on two distinct
bottom nodes. These three pins remove residual rigid translations/rotation;
the boundary masks are exported. No damping. No unilateral contact.

Dimensional strength scale: 3sqrt(3)/16 sqrt(E Gc/ell)=0.75, characteristic
strain sqrt(Gc/(3Eell))=0.001333. The spectral split changes the homogeneous
peak slightly; these are scales, not a notch initiation prediction. Axial wave
transit is H/sqrt((lambda+2mu)/rho)=0.018. A linear velocity ramp over 0.06s
to 0.02 gives target displacement 0.0018 at t=0.12 (8000 nominal steps,
dt=0.000015). Nominal strain is 0.003; local max strain is capped at 0.02.
Rate is not yet quasistatic: it must be checked against the half-speed case.

First bounded segment: 1200 steps, t=0.018, end displacement=0.000054.
This probes propagation of the initial elastic loading wave and validates
precrack initialization, reactions, free contraction and energy bookkeeping;
it is not expected or required to crack. Keep raw initial and final checkpoints,
frames, step work, KKT, source/runtime hashes, measured wall/USS/RSS. Process
40min, job45min, 6GiB fail closed. First execute actual-JAX 4 versus2+2 parity
and changed-binding rejection on the same mesh before the1200 segment.

Before continuation: all states finite; nodal irreversibility >=-1e-12;
relative KKT<=1e-7; boundary displacement error<=1e-10; weak force/strain-work
identity checked; report signed and absolute work, potential splitting and
projection bookkeeping separately. Algebraic closure alone is not convergence.
Inspect physical energy residual vs absolute work and step refinement; stop
if an unresolved residual exceeds 5%, or KE/potential remains above 5% in a
claimed quasistatic loading interval. The first wave-transit segment can be
dynamic and cannot qualify the whole ramp as quasistatic.

Required controls at meaningful matched displacement: fixed ell with h=.00625
(ell/h=6, thinner discrete seed), time step .02 vs .04 CFL, and half speed.
Do not launch all controls before measured capacity. A refined mesh has 294912
particles and needs its own bounded memory pilot. If memory is insufficient,
report that convergence remains unestablished; do not change ell with h.
After fixed-ell qualification, compare ell=.025 at resolved h to distinguish
material length sensitivity from grid sensitivity, with all changes recorded.

Localization audit uses all thresholds .5,.8,.9,.95,.99, true volume fractions,
notch-connected component, actual normal widths in each x/z column, full native
profiles, reaction/displacement and damage/energy histories. Compare extension
outside the initial seed and its initial diffuse tail. No single threshold or
connectivity gate qualifies localization. A growing nearly-unit plateau many
ell wide is a failure requiring model/solver diagnosis, not a rendering choice.
No GIF until this physical evidence is reviewed.
