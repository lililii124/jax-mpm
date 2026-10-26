# HJC beam fixture geometry

The MAT111 beam keeps its mechanical fixture dimensions fixed when the
background grid changes. Its support patches extend to y = 0.1515625 m;
the translating loading patch starts at y = 0.206875–0.2225 m. The support
span, patch widths and prescribed 12 mm loading history are unchanged.
These dimensions reproduce the original grid-64 setup exactly.

Previously, the support depth and loading-patch depth scaled with grid
spacing. A grid comparison therefore also changed the boundary-value
problem. Grids below 32 are now rejected because they cannot consistently
resolve the loading-patch depth. The two symmetry constraints still select
one central node plane each, representing fixed physical symmetry planes.

For a spatial comparison with the same initial particle spacing, use
`concrete-bending-mat111` with grid/particles-per-cell pairs 64/3 and 96/2.
Fixing the fixture dimensions is a prerequisite for this comparison; it
does not establish convergence. Grid-phase sensitivity and APIC projection
losses still require assessment. Existing frozen numerical results retain
their original source and boundary definitions.
