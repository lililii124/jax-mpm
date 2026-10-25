# Dry three-strand forming

This independent dry process completes a real60ms end-driven turn. Three
initially parallel elastic fibres form a three-colour helix through MPM dynamics,
axial feed and frictionless three-field nodal contact. No interior helix or frame
positions are prescribed. There is no resin or adhesion in this case.

The [publication summary](results/fibres--dry-three-strand-forming/summary.json)
links the original media, saved-state review and all12394 step ledgers. The61
actual snapshots span step16 (0.07746ms) through12394 (60ms), rendered in
ParaView at400x400 with the original forming camera. The complete prescribed
rotation is verified as2pi; stronger tight-contact qualification remains open.

Parameters are L24mm, R1mm, initial orbit1.174124516mm (initial analytic surface
gap33.643micrometres),119808 points at0.125mm spacing and0.25mm background grid.
Material is E50kPa, nu.25 and density1000kg/m3, with damage remaining0.
Rotation ramps over5ms;3percent future arc tension ramps5-15ms. Axial feed ends
at a20.333056873mm free span. The original forming baseline uses1.4mm orbit,
227520 points at0.1mm spacing/0.2mm grid and a20ms turn with105micrometres axial
extension. This is a different physical forming protocol, with the same camera.

The final normal forces are15.639/15.456/15.436mN. Joint force-and-distance
coverage is75.93/57.41/57.41percent of free-span bins; the all-pairs70percent
screen remains false. Final J is.75867-1.20579 and principal stretch.66377-1.41179.
Six directed cell-intrusion screens are0; these grid-scale tests do not certify
exact smooth-surface nonpenetration or manufacturing accuracy.

Whole-trajectory fixture work is116.863831microJ; fixture/contact loss is
.754151microJ, final K .013206microJ and U110.907371microJ. The energy remainder
is5.189104microJ, or4.4403percent of cumulative absolute fixture work. The last
298-step deceleration interval alone has32.38percent remainder relative to its
small.72335microJ work. Ledger closure is not energy/time/space convergence.
Production eligibility, manufacturing accuracy, time/space convergence and the
stronger tight-contact screen are all explicitly false in the publication.

Run bounded actual segments (existing forming optional dependencies required):

```sh
python -m examples.three_strand_forming_tight --output outputs/tight-000512 --steps 512
python -m examples.three_strand_forming_tight --output outputs/tight-001024 --steps 512 --parent outputs/tight-000512
```

Each call accepts1-512 steps and a new output folder. Continue from the accepted
parent to12394 (24x512+106 for a fresh run). The archived trajectory used an
identical early64+256-step prefix, then adopted the future tension law at320
before its5ms onset. Frozen numerical source is5d2e414; final checkpoint SHA is
06ff311f05f4e06bee30c08fb75353babf68fbfce9d9e9b3439e2adbb11366c5.
The optional boundary/bin hooks preserve the original baseline defaults.
