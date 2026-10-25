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


## First measured segment and next bounded continuation

Official run40175008288, sourcecf66f924c14e438787181b5eb956db4d1d938c9a,
completed1200. Whole ZIP SHA256
64196b8e97bf16cdfca33fe43bf0031d95a1b439b356faadf96f1e9aabb55ba9.
Independent recheck:39 exact parity fields; actual checkpoint4 bytes identical;
all source hashes match. Wall275.58s, USS1.502GB, RSS1.522GB, measured step
median0.23467s/max0.261924s. No newly threshold-crossing particle volume for
.5/.8/.9/.95/.99. Initial fracture energy2.038729319e-6 is excluded from
physical energy increment; total energy residual/absolute work=-0.02635%.
KE/elastic=1.533: accepted only as an explicit dynamic first loading-wave case.

Next unique segment is step1200 to3600, t=0.054, displacement0.000486;
checkpoint1200 SHA256
2547ab6cbb66701665369ef0a65041e16ec15602143c7f2a20d754cbfe3522d1.
Producer, geometry, all material/integrator sources, BC and loading remain
byte-identical. Full parent ZIP/source binding is verified before stepping.
Actual checkpoint carry remains strict. Only recomputed elastic density uses
an independently evaluated strain-scaled roundoff bound; all other exports
remain exact. Process40min/job45min and6GiB guard remain. There is no crack
claim and no media publication. Dynamic initiation may be accepted as dynamic
if subsequent stability, time-step and fixed-ell localization checks pass;
quasistatic labeling is optional and requires the stated rate/KE evidence.


## Preserved global-strain guard failure and explicit observer correction

Run40175930114 stopped at3274 because four quadrature points exceeded the
old global2% strain cap. All four were initially d0=1, remain d=1, and lie
inside the fixed precrack. Last accepted checkpoint3200 SHA256:
2f6a9154fa19d47204272904918d40e7f160f94438b14d07a54ff27bb9038a7c.
Official ZIP183481959 bytes, SHA256:
6095095ba0d5c8f1e1357db635a9022619eccfcc651f1b34d4c55319196dc8c4.
The failure is preserved, not reclassified as a completed3600 run. No newly
threshold-crossing volume exists at d>=.8/.9/.95/.99 through3274.

The classical variational model uses bulk small-strain elasticity outside the
crack and continuous phase fields to approximate displacement jumps
([Bourdin, Francfort, Marigo2000](https://ms.mcmaster.ca/~bourdinb/downloads/Bourdin-Francfort-EtAl-2000a.pdf),
p798 and pp799¨C801). A jump regularized inside an already broken band is not
an intact-material strain. This supports a partitioned interpretation, not
arbitrary relaxation of material validity or a claim of finite-strain accuracy.

Observer policy initial-unit-seed-jump-v1 exempts only the immutable376
initial quadrature points with d0>=1-1e-12 inside the original coefficient
seed, never later damage. All other points keep the2% strain cap. Every step
also checks polar rotation<=1degree, detF>.5, max particle displacement/H<=1%,
seed residual elastic energy/total<=.1%, and integrated absolute difference
between Green-strain and linear-strain degraded elastic energies outside the
mask<=2%. Seed damage must remain unit within1e-10. These limits are explicit
case-specific engineering checks, not universal constants from the paper.
Global strain and the parallel old-cap outcome remain reported every step.

At3274: nonmask max strain1.689%, rotation0.08754degree, min detF0.99977,
seed residual energy fraction1.23e-6, nonmask absolute geometric elastic
energy difference0.08364%, max displacement/H0.06807%. The new checks pass
on preserved raw data. The physical equations and all advance-function ASTs
remain unchanged. A new source/policy binding preserves every old checkpoint
carry array and physical time exactly; it is explicitly not the old binding.
Only3200¡ú3600 is authorized here. Snapshot3274 is retained for strict physical
array comparison with the old guard failure state. No large-load extension or
media follows automatically. Bandwidth/time-step/mesh checks remain necessary.
