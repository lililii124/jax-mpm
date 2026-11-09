# Cohesive cracks under different loads

<table width="100%">
<tr>
<td width="33%" align="center"><img src="images/home/fracture--notched-tension.gif" alt="Edge-notched plate under tension" width="100%"><br>Notched tension</td>
<td width="33%" align="center"><img src="images/home/fracture--mixed-loading.gif" alt="Combined opening and shear" width="100%"><br>Mixed loading</td>
<td width="33%" align="center"><img src="images/home/fracture--three-point-bending.gif" alt="Three-point bending development result" width="100%"><br>Three-point bending</td>
</tr>
</table>

These plane-strain examples use a neo-Hookean bulk and moving cohesive faces.
The assumed parameters are E=3 GPa, Poisson's ratio 0.2, density=1200 kg/m3,
initiation strength=12 MPa, fracture energy=200 J/m2 and interface
stiffness=1e13 Pa/m. All prescribed velocities use a 20 microsecond startup ramp.

| Case | Specimen and initial crack | Boundary conditions | Duration |
| --- | --- | --- | --- |
| Edge-notched tension | 40 x 20 mm plate; 12 mm horizontal edge crack | Upper/lower grips move vertically at +2.5/-2.5 m/s; horizontal grip motion is fixed | 120 microseconds |
| Mixed opening and shear | Same plate and initial crack | Upper grip moves at (2, 1.5) m/s; lower grip at (-2, -1.5) m/s | 120 microseconds |
| Three-point bending | 60 x 15 mm beam; 6 mm vertical bottom crack at midspan | Lower supports are 50 mm apart; the left support fixes both components and the right only vertical motion; the central upper strip moves downward at 6 m/s | 160 microseconds |

The two plate calculations use a 0.469 mm grid; the beam uses a 0.5 mm grid.
There are two initial material points per cell per direction. No initial bulk
damage is imposed. New cohesive segments are 0.5 mm long, sampled stress uses
a 1.5 mm radius, and tip-support nodes are coupled across the two fields.

Insertion caps are 25 mm for tension, 9 mm for mixed loading and 7.5 mm for
bending. They delimit these demonstrations; they do not model natural arrest
or guarantee complete separation of the specimen. The principal-stress direction
rule and point-particle cut integration remain experimental. These examples
do not establish predictive crack paths, front speeds or mesh convergence.

All three trajectories completed their prescribed duration, with independent
reconstruction of the 49 saved states. Their 400 x 400 ParaView animations use
the same 0–60 MPa maximum-principal-stress scale. Identical adjacent frames
collapse in GIF encoding: tension and mixed loading have 48 animated frames,
and bending has 49; the physical history remains complete.

| Case | Connected complete interface failure | Numerical loss / final external work | Largest reconstructed energy remainder |
| --- | --- | --- | --- |
| Tension | 24.5 mm | 9.31% | 1.97e-11 J/m |
| Mixed loading | 8.5 mm | 5.36% | 1.60e-10 J/m |
| Bending | 6.5 mm | 26.46% | 7.96e-11 J/m |

Numerical loss is accounted for separately from fracture dissipation. The
bending result's late-time numerical loss is substantial; it is retained here
for development review and is not promoted to the homepage gallery. Completing
an energy ledger does not validate its dynamic response. The two plate cases
also develop temporarily disconnected failed segments, so their inserted-tip
motion must not be interpreted as a validated physical crack-front speed.
Inputs, histories, field checks and render provenance are in
[`evidence/cohesive-loading-cases`](../evidence/cohesive-loading-cases).

## Face support near empty grid nodes

The first versions stopped at the accepted-duration floor. An exact replay of
the bending state at 85.64993 microseconds identified a face-support nodal mass
of 7.52e-17 kg/m. The face trace's largest inverse-mass sum was 1.35e14 m/kg;
the original step converged only at one five-hundred-and-twelfth of its nominal
duration.

The revised face trace excludes nodes below 1e-8 of the largest mass in its
own support, then solves the same affine-reproduction conditions on the remaining
nodes. The mass and material points are retained. Face motion and force scatter
use the same resulting operator, preserving conjugate power, resultant and
moment. Insufficient remaining support still invalidates the step.

At the exact stopped state, this changes the minimum used face-node mass to
5.43e-9 kg/m and the largest inverse-mass sum to 3.46e5 m/kg. The nominal step
then converges in five iterations, with residual 1.44e-11. Thresholds 1e-10,
1e-8 and 1e-6 produce identical particle positions, velocities, deformation,
affine velocities and cohesive history for that step. This is a local sensitivity
check, not a trajectory-convergence result. The existing off-grid quadrature
limitation is not removed by this support rule.

```bash
python examples/extrinsic_crack_cases.py tension --output outputs/fracture-tension
python examples/extrinsic_crack_cases.py mixed --output outputs/fracture-mixed
python examples/extrinsic_crack_cases.py bending --output outputs/fracture-bending
python tools/check_extrinsic_results.py outputs/fracture-tension outputs/fracture-mixed outputs/fracture-bending --output outputs/fracture-check
```

The runner retries rejected steps from unchanged state and irreversible
history, limits the smallest accepted duration and total step count, and
records accepted work separately from rejected attempts. The ParaView renderer
uses computed material-domain positions and maximum principal stress without
displacement amplification or a drawn crack line.
