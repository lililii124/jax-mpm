# Edge-notch geometry comparisons

<table width="100%"><tr>
<td width="50%" align="center"><img src="images/home/fracture--offset-tension.gif" alt="Accepted prefix of offset-notch tension" width="100%"><br>Offset notch</td>
<td width="50%" align="center"><img src="images/home/fracture--inclined-tension.gif" alt="Accepted prefix of inclined-notch tension" width="100%"><br>Inclined notch</td>
</tr></table>

These are stopped development runs, not completed loading histories. Both
were stopped during contact-driven time-step reduction; they are not included
in the homepage gallery. The animations contain every saved accepted frame.

| Case | Last saved time / requested time | Saved frames | Connected complete failure | Numerical loss / work |
| --- | --- | --- | --- | --- |
| Offset notch | 72.5 / 120 microseconds | 30 | 3.0 mm | 12.35% |
| Inclined notch | 67.5 / 120 microseconds | 28 | 3.0 mm | 6.40% |

Independent reconstruction checks the stored energies, stress, irreversible
cohesive history and work ledger. Peak sampled compressive penalty pressures
are 305 MPa and 221 MPa respectively. At the saved offset state, the nominal
21.09 ns trial step fails its iteration criterion; a 1/64 step passes. Its
P2G loss remains 0.000759 J/m before either trial. Reducing the step therefore
does not remove the repeated transfer loss. The first trial is a diagnostic
rejection and is not included in the accepted trajectory.

Inputs, stopped-run records, checks and render provenance are retained in
[`additional-geometries`](../evidence/cohesive-loading-cases/additional-geometries).
Compatible contact-force transfer remains the prerequisite for resuming these
cases. Their completed-prefix energy bookkeeping does not establish accuracy.

The offset and inclined-notch cases use the same 40 x 20 mm plate, material,
grid and prescribed grip velocities as the existing edge-notched tension case.
Only the initial crack geometry changes. Lengths below are in millimetres;
coordinates refer to the 60 x 60 mm computational domain.

| Case | Initial crack endpoints | Initial length | Orientation |
| --- | --- | --- | --- |
| Reference tension | (10, 30) to (22, 30) | 12 mm | Horizontal, centred |
| Offset notch | (10, 26) to (22, 26) | 12 mm | Horizontal, 4 mm below centre |
| Inclined notch | (10, 26) to (20.3923, 32) | 12 mm | 30 degrees above horizontal |

The upper and lower grips move at +2.5 and -2.5 m/s vertically, respectively,
with horizontal grip motion fixed and a 20 microsecond startup ramp. The
duration is 120 microseconds. The 128-cell grid has 0.46875 mm spacing, with
two initial material points per cell per direction. The plane-strain bulk is
neo-Hookean: E=3 GPa, Poisson's ratio=0.2 and density=1200 kg/m3. Cohesive
strength is 12 MPa, fracture energy is 200 J/m2 and stiffness is 1e13 Pa/m.

Only the initial crack is prescribed. Subsequent potential segments are
inserted from the computed stress and develop cohesive softening. The
25 mm insertion cap, 0.5 mm segment spacing and 1.5 mm stress-sampling radius
are identical across these tension cases. The cap is a geometric limit,
not a prediction of physical crack arrest.

These examples use the existing point-particle cohesive solver. The new
domain-integrated bulk operator is undergoing separate verification and is
not silently substituted here. Insertion can run ahead of the connected
fully failed interface, and temporarily detached failure can occur. The
animations illustrate computed response; they do not establish crack-front
speed, mesh convergence or predictive crack paths.

```bash
python examples/extrinsic_crack_cases.py offset --output outputs/fracture-cases-offset
python examples/extrinsic_crack_cases.py inclined --output outputs/fracture-cases-inclined
python tools/check_extrinsic_results.py outputs/fracture-cases-offset outputs/fracture-cases-inclined --allow-partial --output outputs/fracture-geometries-check
pvpython tools/render_explicit_crack_paraview.py --input outputs/fracture-cases-offset --output outputs/fracture-cases-offset.gif --field principal_stress --allow-partial
pvpython tools/render_explicit_crack_paraview.py --input outputs/fracture-cases-inclined --output outputs/fracture-cases-inclined.gif --field principal_stress --allow-partial
```

Both ParaView views use actual positions and deformation, a 400 x 400 frame,
the same camera and a 0–60 MPa maximum-principal-stress colour scale. No
propagated crack line or artificial displacement magnification is added.
GIF encoding merges identical initial images: the offset animation has 29
encoded frames and the inclined animation has 27. All 30 and 28 saved states,
respectively, were rendered. Irregular localized response near the crack is
retained for diagnosis, not interpreted as a validated fragmentation pattern.

Further cases are a crack approaching a free hole and interacting cracks.
The latter requires multiple independent crack paths and compatible field
connectivity; it is not currently supported by this single-path runner.
