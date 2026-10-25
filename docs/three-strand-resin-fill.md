# Independent resin fill candidate

This is an isolated source candidate, not a finished filling result. Forming
remains unchanged. `examples/three_strand_resin_fill.py` accepts a valid completed
actual forming checkpoint and its plan, freezes those fibres as obstacles, and
introduces resin particles exclusively at a physical inlet. No fluid particles
are preplaced throughout a final cylinder. No solidification or full FSI is
modeled.

Candidate fluid: density 1100 kg/m³, Newtonian viscosity 0.05 Pa s, grid
0.25 mm, particle spacing 0.2 mm, dt 0.04 ms. The cylindrical mold has radius
3.2 mm and x limits −1 to 25 mm. An inlet of radius 2.5 mm prescribes a parabolic
profile with maximum speed 0.2 m/s; the opposite port is open to zero gauge
pressure. Nominal mean-speed Reynolds number is 11 and advective CFL is .032.
The conservative explicit symmetric-viscosity number 12 μ dt/(ρ h²) is .349, below the kernel rejection threshold .5. These are design values pending the geometry/pilot.

The APIC transfer is staggered by velocity component. The pressure projection
uses actual face masses without an inertia floor, a D/Dᵀ adjoint pair, and
fixed-face counter impulses. Empty cells use zero gauge pressure; fluid pressure
cells currently use particle cell mass > .25 ρh³. This free-surface approximation
requires volume and refinement assessment. Boundary cells are voxelized from
actual F-advected quadrature cells, with a bounding-sphere query that retains all
candidate cells. The resulting staircase obstacle has grid-scale uncertainty;
it is not a smooth reconstructed fibre surface.

Every source birth and outlet removal records mass and generalized APIC kinetic
energy. The step separately records transfer losses, prescribed-boundary work
and projection loss, explicit viscous work/loss, pressure work/loss, and the
exact algebraic energy identity. This identity alone does not certify temporal
or spatial convergence. Solid-cell intrusion, incomplete support, pressure solve
failure, mass mismatch and excessive advective CFL reject a candidate state.

One bounded Linux algebra check (`resin-operator-check`) passed in 10.08 s with
203 MB peak USS, executing zero filling steps. D/Dᵀ adjoint error was zero;
rigid-rotation strain 5.42e−16; viscous power adjoint error 1.69e−21 W; pressure
relative residual 6.32e−9, divergence 2.95e−6 s⁻¹ and work identity 1.55e−25 J;
affine APIC velocity/gradient errors 6.94e−18 / 4.45e−14 and mass error
6.35e−22 kg. The full source/obstacle/inlet flow has not been run. A 64-step
physical pilot and measured memory/throughput are required before long filling.


The main tensor path is JAX, not a NumPy reference solver: staggered APIC scatter
and gather, generalized kinetic energy, symmetric-gradient Newtonian forces,
D/D-transpose pressure projection with JAX CG, particle v/C/x/time updates, and
bounded voxel-face TOI events all execute inside a jitted step. TOI uses actual
linear-flight intersection time, e=0 normal impulse and unchanged tangential
velocity/affine C; stationary-wall work is zero and positive collision loss is
separate from grid/viscous losses. Normalized-grid representable side selection
addresses world-coordinate cancellation at x=0; it creates no finite safety gap.

Host NumPy/SciPy constructs static fixed-fibre voxel geometry and inlet birth
coordinates once. Python schedules bounded segments, activates inlet slots,
removes measured outlet particles, maintains scalar source/sink ledgers and
writes/checks arrays. These host controls do not perform MAC/APIC/viscous/
pressure/time integration. The one-phase free-surface approximation assumes
ambient air pressure and does not model gas trapping, wetting or curing.

Original actual flow reached160steps/6.4ms, then rejected step168 because one
particle crossed a fixed voxel face. The rejected state is preserved. A directed
face/edge/corner and physical zero-face check passed; continuation of the same
accepted160 state through the eight necessary steps captured one real collision
and passed the unchanged penetration guard. Original particle11 overshot the
wall by .46778micrometres at vx=.05013m/s; TOI sets vx=0, retains tangential
velocity and accounts1.10573e-11J collision loss. Corrected boundary position is
-2.22e-19m, a representable floating side of the x=0 face. This is an event
constraint, not arbitrary position projection. Subsequent actual flow reaches
30.72ms/768steps with7547 active particles; pressure-mask/particle-volume ratio
is1.06856, down from1.567 at the first thin inlet layer. These are bounded
experimental results, not a completed filling or convergence certificate.

The original quadratic-transport trajectory subsequently reached5376steps
(215.04ms), but an actual bulk-density and occupied-volume audit rejected it
for filling qualification. It remains a private reference, including its media.
An incremental RT0 characteristic-transport candidate starts from its explicitly
hashed5376 state. A particle-owned/reachable pressure-domain closure includes
only open cells with complete positive free-face mass support; fixed walls remain
fixed and no mass is manufactured. The fine mass-metric D/gradient system and
residual admission remain unchanged. Symmetric Galerkin aggregation supplies an
SPD preconditioner. A16-step real interval reached5392 with accumulated incremental
logJ between -1.016e-10 and9.172e-11, no solid intrusion and conserved source mass.
That result does not repair or certify the preceding density distribution.

A later actual step5420 failed the original400-iteration pressure execution
budget. The residual7.445e-7 failed the unchanged1e-7 admission and its rejected
state was retained. The next bounded candidate uses800 standard PCG iterations
maximum with actual counts recorded; it changes solver work, not physical
parameters or error tolerance. Its last persisted parent is5408. Recovery uses
a read-only view of the original checkpoint and the ledger truncated at that
checkpoint, retaining the original rejection journal separately. Only after
necessary incremental validation may a single clean corrected fill begin from
the same physical empty-fluid initial state. Neither numeric completion nor an
incremental logJ result by itself proves a filled, volume-qualified composite.
