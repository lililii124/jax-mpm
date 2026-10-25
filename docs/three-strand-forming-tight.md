# Independent dry tight forming candidate

This F8 case is separate from baseline F6 forming and subsequent resin filling.
Three initially straight 24 mm fibres of radius 1 mm start at orbit
1.174124516 mm: their analytic cylindrical surface gap is 33.6433 micrometres.
The independent three-field frictionless unilateral contact model and its
mass-metric QP tolerances are retained. No adhesive or Hertz claim is made.

Only the collars rotate/translate. Interior positions evolve through MPM;
analytic helices never generate animation positions. Rotation takes 60 ms with
5 ms speed ramps. Free material length is 21 mm. Prescribed free axial span is
sqrt((ell*(1+eps(t)))¬≤-(angle*orbit)¬≤), with eps=0 until 5 ms, then a smooth ramp
to 3% at 15 ms. Final span is 20.333056873 mm: the two collars together feed
0.666943127 mm inward while retaining actual axial tensile extension along
the target arc.

This tensile ramp revises the initial zero-arc-tension design. The first 320
actual accepted steps end at 1.549136679 ms, before the new term is nonzero.
Continuation requires an explicit future-plan adoption flag and rejects an
adoption at or after 5 ms, changed geometry/material/grid or an already revised
protocol. Original source, checkpoint and plan remain intact. The new term is
identically zero throughout accepted history; no prior physical step is replayed.

A thin-rod estimate screens the tension choice: EI is about 3.93e-8 N m¬≤,
terminal curvature about 100 m‚Åª¬π and outward bending term about .37 N/m;
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


The audited dry process reached12394 actual steps/60ms and a complete imposed
2pi end rotation, using119808 points at0.125mm spacing and a0.25mm grid.
Initial orbit radius is1.174124516mm (analytic initial gap33.643micrometres).
Three-percent future arc tension ramps5®C15ms; final free span20.333056873mm.
The baseline uses227520 points,0.1mm spacing/0.2mm grid,1.4mm orbit and a20ms
rotation with105micrometre axial extension. Both use the same material and
frictionless three-field nodal mass-metric contact. This dry case changes real
geometry/feed/time discretization; it is not a camera or display-radius variant.

Fresh bounded CLI (requires the existing forming dependencies):

```sh
python -m examples.three_strand_forming_tight --output outputs/tight-000512 --steps 512
python -m examples.three_strand_forming_tight --output outputs/tight-001024 --steps 512 --parent outputs/tight-000512
```

Continue with new output directories and the last accepted parent; each call
accepts1®C512 actual steps.12394 is the planned end (24x512+106 from a fresh run).
The authenticated existing trajectory began64+256 steps, adopted the future
3percent tension law at320 before its5ms onset, and then continued without
replaying that identical physical prefix. Source5d2e414 and the full checkpoint
SHA06ff311f05f4e06bee30c08fb75353babf68fbfce9d9e9b3439e2adbb11366c5
identify the delivered trajectory.

Actual end-state J spans.75867®C1.20579 and principal stretch.66377®C1.41179;
damage remains0 and the six directed all-candidate cell-intrusion checks are0.
Three actual normal forces are15.639/15.456/15.436mN. Joint force-and-distance
free-span coverage is75.93/57.41/57.41percent, below the independent all-pairs
70percent candidate screen. Thus a complete rotation is demonstrated; certified
tight contact over that stronger coverage criterion remains open.

All12394 step rows give fixture work116.863831microJ, fixture/contact loss
.754151microJ, final K .013206microJ, U110.907371microJ and remainder5.189104microJ
(4.4403percent of cumulative absolute fixture work). The last298-step deceleration
interval alone has32.38percent remainder relative to its small.72335microJ work.
The ledger identity closes to1.58e-19J; this does not establish energy/time/space
convergence. Particle-cell geometry checks are resolution-scale diagnostics.
The61-frame private GIF uses actual frames16®C12394, fixed400x400 camera and
three fixed colours, with no position interpolation. Rendering particle radius
is.62 times the actual point spacing; it does not change mechanical contact.

The release closure contains only the dry example/boundary, default-preserving
optional boundary/contact-bin hooks, snapshot/coverage reviewers and private
renderers. It includes no fluid transport, viscosity, pressure or resin source.
Detailed receipts are in three-strand-tight-dry-evidence.json.
