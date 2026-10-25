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
