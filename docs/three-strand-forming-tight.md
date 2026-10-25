# Independent dry tight forming candidate

This F8 case is separate from baseline F6 forming and subsequent resin filling.
Three initially straight 24 mm fibres of radius 1 mm start at orbit
1.174124516 mm: their analytic cylindrical surface gap is 33.6433 micrometres.
The independent three-field frictionless unilateral contact model and its
mass-metric QP tolerances are retained. No adhesive or Hertz claim is made.

Only the collars rotate/translate. Interior positions evolve through MPM;
analytic helices never generate animation positions. Rotation takes 60 ms with
5 ms speed ramps. Free material length is 21 mm. Prescribed free axial span is
sqrt((ell*(1+eps(t)))²-(angle*orbit)²), with eps=0 until 5 ms, then a smooth ramp
to 3% at 15 ms. Final span is 20.333056873 mm: the two collars together feed
0.666943127 mm inward while retaining actual axial tensile extension along
the target arc.

This tensile ramp revises the initial zero-arc-tension design. The first 320
actual accepted steps end at 1.549136679 ms, before the new term is nonzero.
Continuation requires an explicit future-plan adoption flag and rejects an
adoption at or after 5 ms, changed geometry/material/grid or an already revised
protocol. Original source, checkpoint and plan remain intact. The new term is
identically zero throughout accepted history; no prior physical step is replayed.

A thin-rod estimate screens the tension choice: EI is about 3.93e-8 N m²,
terminal curvature about 100 m⁻¹ and outward bending term about .37 N/m;
3% tensile strain gives EA*eps about 4.71 mN, or T*kappa about .47 N/m inward.
This is a design estimate, not a predicted contact force or dynamic result.
It avoids relying on a zero-tension helix to remain tightly pressed together.

The grid is .25 mm and particle spacing .125 mm, with 119808 particles and
12394 planned steps. An actual 64-step pilot and continuation to 320 passed
source/CRC/finite/material/geometry and explicit energy audits. At 320, J is
[.99125,1.00875], stretches [.98709,1.01303], damage zero, and the interval
remainder is .4313% of cumulative absolute fixture work. This is only an early
1.549 ms prefix. Major free-length three-pair contact has not been established.

Every step records 64 physical axial bins of pair normal force, plus actual
particle/local-cell geometry checks. Final qualification requires sustained
contact force and grid-resolved small gaps over most of the free span for all
three pairs, with collar/energy/J/penetration checks. Positive roundoff-scale
nodal impulses alone do not establish contact coverage. No rendering-radius
change, camera trick, prescribed helix or hidden parameter change can satisfy
that requirement. No tight-forming GIF is ready for publication yet.
