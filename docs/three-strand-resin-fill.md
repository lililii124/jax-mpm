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
