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

## Comparison through 1 ms

The grid-64/ppc-3 and grid-96/ppc-2 runs start from identical particle
positions, masses and volumes. They use 175 and 263 steps per saved frame,
respectively, so this comparison changes both spatial and temporal resolution.
The prescribed loading lasts 1.5 ms; this record covers the first 1 ms.

| Observable at 1 ms | Grid 64 | Grid 96 |
| --- | ---: | ---: |
| Midspan bottom deflection (mm) | 13.7996 | 13.8497 |
| Reference-volume mean damage | 0.0354340 | 0.0426605 |
| Fully damaged reference-volume fraction | 0.00128815 | 0.00982627 |
| Damage mirror RMS, x | 2.53e-13 | 7.55e-13 |

The endpoint deflection differs by 0.363%, while mean damage differs by
20.4%. The maximum deflection difference over the saved window is 0.0718 mm.
Symmetry is retained, but damage remains resolution sensitive. These results
do not establish mesh or time convergence. The saved loading reaction is a
next-step boundary probe with a different timestep on each grid; it cannot
serve as an exact reaction or work balance.

[Framewise comparison](../evidence/hjc-grid-comparison/observables.csv),
[summary](../evidence/hjc-grid-comparison/comparison.json), and
[source provenance](../evidence/hjc-grid-comparison/provenance.json).
