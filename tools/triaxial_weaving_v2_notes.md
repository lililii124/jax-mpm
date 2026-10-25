# Independent process v2: rate-limited flexible cords

Version1 last valid CP16600 remains retained. Its first invalid state16685 had log stretch.1201528 at free incoming+60 particle2231/section4, outside three-section endcaps. The extreme direction aligned.9909 with the reference cord axis; axial velocity6.028m/s. No contact occurred, search nonrigid displacement remained below skin, and the complete quadratic grid stencil was covered. End-to-end length stayed fixed but local axial waves caused12% extension near the pulling end. With cL89.44m/s,2v/c≈13.5% is consistent with the failure. The numerical guard is not raised.

V2 starts from zero mechanics/independent source binding. Keep E10MPa/nu.25/rho1500, circular Hertz,12% log-stretch cap and all end/region/radius/coverage/stencil guards. Each half-cosine axial feed lasts max(3.6ms,pi*actual travel/(2*3m/s)); therefore each incoming cord has peak speed<=3m/s even as the repeated-cell station gets larger. The reflected elastic-bar estimate falls to6.708%, a scale estimate rather than proof. Existing opening/closing commands remain explicit and must be checked against actual material heights.

The3-cord v2 protocol is9.187286ms/136956 nominal steps atCFL.08. This is not the final24–36-cord deliverable. Current hash implementation is source-only until actual-JAX stencil equivalence and a real64-step comparison with retained dense fields run under an allocated budget. The first64 steps lie in the unchanged opening command, so the retained v1 dense64 fields provide an independent physical comparison without a duplicate dense simulation. No movie is a weave qualification before actual material crossing/closing and contact/energy audits.

## Measured24-yarn qualification and source-only batch version

Actual24-yarn184224-point CP64->664 completed in372.335s, peakUSS
2495635456B. Warm200-step blocks111.998--114.237s, CPU/wall1.83--1.89.
No contact yet, D=0, maxlogstretch.01499864, minimumJ.98996729,
CFL.0808121. CP664 SHA256:
4cffbcd67db49bd891e1777e322387bc881a41c0218967cedd25fff265367755.
This is opening/resource evidence, not completed interlacing.

Reducing fixed packed capacity4194304->2097152 in a read-only32-step
same-CP64 diagnostic improved median throughput only1.17245x. Downloaded
17-field final states:16 fields exact, boundary-impulse reduction differed
by5.03609e-27kg m/s; physical x/v/F/C/stress/history exact. Full physical
stencil nodes, weights and offsets exact. First comparator incorrectly
required packed local node indices to match; its failure is retained.

The serial24-yarn protocol requires4405164steps/.295507s, about29days at
.56s/step. It is unsuitable as the final execution plan. The new source-only
`triaxial_weaving_batch_geometry.py` groups each incoming family into one
compatible batch: every member has the same over/under sign against each
installed cord. It commands only genuine end fixtures. Vertical alignment,
axial feed and closing are all limited to3m/s peak; this also avoids the old
high parked-yarn alignment speeds. Fresh24-yarn duration.0450412054s,
8stages vs64, unchanged184224 real material points. No actual batch run yet.
This is a new process version, never a continuation of oldCP664. Actual
contact, crossing order, strain/work and targeted CFL checks remain required.

## Actual fresh batch CFL diagnostic (73129d1)

One bounded actualJAX container completed both fresh24-yarn184224-point
cases: CFL.08/360steps and.24/120steps at t=24.1495341569977microseconds.
Wall347.057s, peakUSS2620424192B. Alloriginal strain/material/contact/
coverage guards passed; additional explicit case CFLceiling.25 passed.
Both cases conserved mass/volume/IDs, D=0, histories irreversible, fixture
speed<=3m/s and griperror<1e-10m. Maxlogstretch.00175357/.00185653;
energy-minus-machine-work divided byabsolute machinework4.969%/3.045%.

**The predeclared5% relative time-consistency target FAILED.** xincrement
L2 difference3.476%, v3.690%, F-I7.816%, kineticenergy1.901%, elasticenergy
10.125%, actualfixturework1.837%. Do not qualify.24 for continuation or
claim global time convergence. This is only initial opening, no contact or
woven topology. Raw/audit:
`.local/triaxial-weaving-v3-batch-cfl-short-audit/independent-equal-time-audit.json`.
Inputcapsule SHA2569fa6cfe77b1898768dbecc2aa9da370f6ffc3cd4e9706b5af404e9916d81c365.

The read-only24-yarn sameCP66432-step material-spectrum guard comparison
completed90.783s/peakUSS2891595776B. Same2097152node layout. Replacing
separate SVD with the existing material `_spectrum(F)` stretches for the
same12%logstretch condition yielded1.14274x median speedup. Downloaded17
state/historyfields are allbitexact; both finalNPZ SHA256:
373e696ad257ad46a57d3b1da6259caca604800eea9bc52cc3fda309dc57c182.
This does not qualify larger time steps. Production batch driver is still
unchanged; optimization has only been measured in the diagnostic tool.

## Same-world-grid fibre decomposition review

ActualCP36016step comparison: all84same-family actual world-stencil support
intersections are empty; same reference world grid, weights and offsets.
Occupiednodes1589330unchanged. Localcapacity2621440vs4194304global.
Parallel8worker wall.156566s/step vs.588773global,3.76054x faster,
CPU/wall7.779vs2.028. Full23-field downloaded states were independently
compared. The original relative-net-work1e-12gate failed and is retained;
complete evidence was exported before the final old-gate failure.

Engineering1e-12comparison passes for v/c,tau/E,F,dt*C,impulse/(Mc),and
work/(EV), including full material/contacthistory and work/CFLaccumulators.
E=1e7Pa,V=3.65962e-8m3,EV=.365962J;EV/|Wgrid|=1.994e8, while
Wabs/|Wmachine|=1. There is no netpositive/negativework cancellation here;
small-strain material stress subtracts O(E)terms, and normalizing by a tiny
work quantity magnifies its floating-point effect. This is not bitexact and
not a general contact-window proof. Fullsource/output reports are retained
in local audit directories; compact numeric review is committed alongside.

Actual continuation uses16workers then8workers in a source-bound fresh
execution fork of the existing fineCP360, preserving every input carry bit.
Empty real contact coverage plus absolute particle path<skin/2 certifies
contact-free parallel blocks. A real coverage pair switches to the original
global every-dtHertz/contact-history/APIC operator. Fixed-IDreassembly and
original strain/material/CFL/geometry guards remain. LargerCFL.24 is still
rejected by the separate predeclared time-sensitivity test.
