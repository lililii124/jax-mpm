# Original process and isolated consistency candidate

Original numerical source: 7b6ea1020b1b431c210e588c6bd0e42dfa8d542d.
The original process completed step1280, t=4.957397 ms, right collar84.457 deg.
It has not completed a turn, established tight contact, or produced qualified media.
J is0.895708..1.107069; principal stretch0.859274..1.170046; no damage.
Three nearest actual particle-centre distances are0.390020/0.407250/0.408249 mm.
These distances are not certified continuous surface gaps.

The original1280 checkpoint SHA256 is
6e4b05244cc2ac205d4f2f7f7e31033fbf7221511c473e3f3ab2b77e0b8d351d.
It and all saved original particle states are preserved. The original step512
checkpoint SHA256 iscccab42578978939a737c6b58d3a959c4d75bf9cac171938495d99e7df2a0111.
The initial kernel check and actual64-step prefix both passed bounded admission.

At1280: W=26.419424 microjoules, K=15.519702, U=10.614350.
Thus actual W-K-U is positive0.285372 microjoules: stored plus kinetic energy
has NOT exceeded fixture input. A prior report of numerical net gain requires
qualification: W-Lboundary-Lcontact-K-U=-3.231646 microjoules is a split ledger
remainder after subtracting projection terms, not proof of physical energy
creation above external work. The stress/boundary split contributes a signed
compensating term; its projection loss cannot be independently identified as
all irreversible physical loss while omitting that term.

Only steps513..1280 have full stepwise work records:768 consecutive rows,
sum(abs(deltaW))=14.87879772453 microjoules and max(abs(deltaW))=20.8439480542
nanojoules. Earlier stepwise absolute/max work remains unknown. Saved interval
absolute work is explicitly only a lower bound for that earlier part.

Two authorized single-checkpoint diagnostics used read-only copies and did not
advance the accepted trajectory. At1280 the original hypothetical next step has
P2G/G2P losses0.080193/0.188089 nJ, force deltaK=-8.077563 nJ and deltaU=11.433950 nJ.
Its work-conjugate S=dt*sum(V0*tau_old:C_post)=11.348869 nJ matches minus nodal
stress impulse work to2.65e-23 J. Material endpoint defect deltaU-S is0.085081 nJ;
force velocity incompatibility deltaKforce+S is3.271306 nJ. The latter contributes
97.5 percent of their3.356386 nJ sum. Contact projection work matches its mass
metric to1.27e-21 J. This local evidence identifies the force/Dirichlet/kinematic
split; it does not establish whole-window energy or time convergence.

The isolated candidate in fibre_forming_consistent.py first imposes collar
velocity, then applies stress and contact with explicit fixture counter-impulses.
It advects x/F with the average admissible beginning/end grid velocity while
retaining final velocity/APIC C. Its ledger uses the SAME average velocity for
nodal force work and configuration work, records signed contact work, and leaves
the material finite-increment defect explicit. It introduces no heat, reservoir,
energy renormalization, changed material, force range or loading.

One actual candidate step1280->1281 gives a positive remainder0.183588 nJ,
grid work identity error1.20e-22 J and material endpoint defect0.084694 nJ.
This is an improvement in split/time compatibility, not proof the candidate is
physically more accurate or stable for a whole process. Midpoint position
advection adds angular transport truncation:6.60e-14 kg m^2/s in that step.
A subsequent bounded64-step candidate window measures this term and actual
angular/fixture torque balance explicitly. Exact angular conservation is not
claimed, and the old trajectory is not overwritten or mixed with the candidate.


Actual step2163 stopped because one shared node failed the unchanged PGS64
KKT gate(3.13647e-5m/s versus1e-8). The full rejected state is preserved. A
bounded8-active-set fallback solves the identical mass metric QP only on
unconverged nodes, rejects rank-deficient active sets and capacity overflow,
and never discards small positive masses. Actual16-node KKT, independent
whitened-primal SLSQP, fixture work, rank, objectivity, padding and capacity
certificates passed. Recovery2161->2497 completed with6 exact node solves.

Actual step2881 contact-operator comparison found2095 shared nodes among253313.
Packing every node with at least two positive material masses preserves the
complete QP; remaining nodes have no contact constraints. Padding has zero mass
and scatter uses out-of-bounds drop, with explicit16384 shared-node capacity.
Full/packed projection both take42 sweeps and have the same9.032348e-9m/s
residual; velocity differs at most1.39e-16m/s, impulses5.17e-25Ns, fixture0.
One post-compilation contact-operator execution took1.9355s versus0.11639s;
this is not a whole-step or whole-trajectory speed claim.
