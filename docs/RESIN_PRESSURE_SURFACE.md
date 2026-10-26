# Resin free-surface pressure support

The corrected MAC-face inlet run reaches a valid checkpoint at step 1952.
At step 1966, the former smoothed cell-mass threshold selects cell (10,16,17)
with no particle-owned volume sample and no mass on its positive z face.
Its interpolated cell mass is 0.31153 times rho*h^3. The nearest particle
centre is 294.59 micrometres from the cell centre on a 250 micrometre grid.
A replay of the last 14 steps reproduces every saved pressure input array
exactly; this is a domain-selection failure, not pressure-solver stagnation.

The pressure-domain variant excludes unsupported threshold-only cells as
free-surface cells. Every particle-owned cell remains mandatory. A missing
face in an owned cell still rejects the step; no face mass is invented, fixed
velocity changed, or unsupported particle transported. The pressure solve,
residual, stationarity, momentum and work checks remain unchanged.

This changes the free-surface discretization from the former failed step
onward. Earlier valid domains are identical because all their selected cells
have complete face support. The new source records how many threshold cells
are excluded. It does not establish spatial convergence or complete filling.

`python -m tools.verify_resin_pressure_surface` checks the occupied-cell
negative control and an independent six-face analytic pressure projection.

The bounded continuation reaches step 2048 with the original pressure and
transport checks. All state arrays immediately before the former step 1966
rejection match the old source exactly. At most one empty threshold cell is
excluded during this interval. The largest pressure relative residual is
9.961e-8 against the original 1e-7 bound; stationarity is 6.628e-11 against
1e-10. The cumulative transport log-J range is [-2.181e-10, 2.465e-10].
The final checkpoint SHA256 is
`cb03eccfd1c5730a4724f1395ab062bd75aee4fa4c06046ccf788408dc26728d`.
Source-bound receipts and every step's observables are in
`evidence/resin-supported-pressure-2048`.
