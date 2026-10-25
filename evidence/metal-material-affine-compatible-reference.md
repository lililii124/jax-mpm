# Compatible material-affine reference (not an impact admission)

The three original geometric-DELTA runs and their raw ZIPs are unchanged.
Their grid/physical stress-work mismatch approaches a nonzero power. Reducing
the time step cannot resolve that mismatch.

Use material affine velocity W=Fdot and fixed reference moment D0. At the
authenticated seed set D0=F^-1 Dgrid F^-T and W=C F, so the original kinetic
energy, linear momentum and angular momentum agree exactly. Thereafter:

* Lphysical=W F^-1.
* Ctransfer=W D0 F^T Dgrid^-1.
* K=sum m (v.v + tr(W D0 W^T))/2.
* Free motion advances x by dt v and F by dt W, keeping v and W constant.

For grid velocity u use ONE reconstruction G u=(Iu,A(u)F). Its kinetic metric
H=diag(m I,m D0) defines the consistent mass G^T H G and the orthogonal
particle projection. The stress covector is (0,-V tau F^-T); its grid force
is the exact adjoint G^T of that same reconstruction. Thus stress work is
identical in grid and physical variables rather than balanced by a reservoir.
This is a different evolving affine kinetic metric from the old fixed-grid
C norm. Both metrics must be reported; the old metric must not be silently
substituted or its difference thermalized.

Grid mass can have null modes. The dense reference records a backward-error
rank and adds only a nullspace gauge reproducing rigid velocities, which is
needed for momentum-preserving common-node contact. Contact is an instantaneous
energy-metric projection on the SAME reconstructed velocity. It changes neither
F nor internal energy. Its kinetic loss, the stress kick's midpoint work, and
the following contact loss form one exact work decomposition.

`tools/prove_metal_material_affine.py` checks arbitrary nonaffine fields, stress
virtual work, ballistic affine motion and a singular two-body contact fixture.
The fixture has 42 null grid modes; its analytic contact loss is 3.64 J and
contact/stress/contact work closes to 1.22e-16 J. This is a small reference,
not integration with the full constitutive return.

`tools/prove_metal_material_affine_fixed125.py` bounds a sparse least-squares
projection of the authenticated original 125 ns state without advancing time.
At 1024 iterations its loss is 2.598245e-6 J out of 135.473222 J; the variance
identity residual is 6.56e-13 J, linear momentum changes by at most 9.16e-15,
and angular momentum by at most 4.42e-18. However, LSMR reaches its iteration
limit. A separate 4096-iteration diagnostic also reaches its limit. Neither
is a certified consistent-mass solve; the reports explicitly deny production
admission. Small ledger residual alone does not bound a near-null-mode error.

Before a real impact continuation, implement and certify a scalable rank-aware
consistent-mass inverse and rigid gauge, extend contact to its metric, and
couple the original material return with compatible time-discrete stress work.
Retain all original material, contact, conservation and validity gates. No
impact, public media or long run is admitted by these reference results.
