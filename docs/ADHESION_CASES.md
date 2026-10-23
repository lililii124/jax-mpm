# Deformation-coupled adhesion cases

These four cases use a finite-range Dugdale normal interface coupled to a deformable MPM layer. The current material-surface positions determine the gap and contact force at every explicit step. They complement the analytical Hertz/JKR routines and the prescribed JKR footprint in [CONTACT.md](CONTACT.md). The footprint applies a specified load; the cases here calculate their forces from the evolving gap.

## Shared model

The initial soft layer occupies `(0.14, 0.15, 0.14)` to `(0.86, 0.38, 0.86)` m. It uses compressible neo-Hookean elasticity with Young's modulus 12 kPa, Poisson's ratio 0.25 and density 1000 kg/m³. Gravity is zero. All velocity components are fixed at grid nodes with `y ≤ 0.15` m; the remaining faces carry no imposed traction apart from the contact forces on the sampled top surface.

Default production settings are grid 48, three particles per cell per axis, 367,744 material points and 10,816 top-surface quadrature points. The 0.85 s cycle has 50 output intervals, including 51 saved states. The explicit step limit includes both the bulk wave speed and the surface penalty indicator `k A0 dt²/m ≤ 0.09`. This local indicator is not a proof of nonlinear stability.

Every indenter is a prescribed rigid sphere of radius 0.16 m. The interaction range is 0.025 m and the repulsive penalty stiffness is `1.2 × 10⁶ Pa/m`. The adhesive cases use a work of adhesion `w = 12 J/m²`, hence a nominal attractive traction magnitude `w/h = 480 Pa`. These are illustrative continuum-interface parameters. The deliberately resolved 25 mm range does not represent a molecular separation or a fitted material interface.

Surface points are the top layer of particle centres, initially half a particle spacing below the nominal upper face. Their fixed reference areas sum to 0.5184 m². At the default resolution the interaction range spans about 3.61 surface spacings. No deformation-dependent offset or current-area rescaling is applied to these points.

For each current surface point `x` and sphere centre `c`, the signed gap is `g = |x − c| − R`. The interface potential per reference area is

```text
ψ(g) = 0.5 k g² − w     g < 0
       (w/h) g − w      0 ≤ g < h
       0               g ≥ h
```

The particle force is `−A0 ψ'(g) (x − c)/|x − c|`. Compression permits finite penalty overlap; separation within the interaction range produces attraction. Outside the range, force and potential are zero. The force is the position gradient of this discrete potential away from its corners at `g = 0` and `g = h`. The law is reversible and supplies no tangential friction or irreversible bond damage. Model parameters are static in the JAX interface; parameter differentiation is not asserted.

## Four questions

| Case | Prescribed motion | Comparison purpose |
| --- | --- | --- |
| `nonadhesive-pull-off` | One sphere centred at `x = z = 0.5` m: approach, indent, hold, retract fully, hold clear. `w = 0`. | Repulsive-contact baseline: no adhesive area or attractive interface force. |
| `adhesive-pull-off` | Exactly the same geometry, initial state, path, grid, stiffness and time step as the baseline. `w = 12 J/m²`. | Isolate the effect of finite-range attraction on force, contact area and separation. |
| `lateral-release` | At `z = 0.5` m, indent at `x = 0.38`, sweep laterally to `x = 0.55` at the same centre height, then lift while moving to `x = 0.65`. | Observe a migrating contact patch and oblique pull-off. Horizontal resultant forces arise from the radial normal forces; this is not a friction experiment. |
| `three-sphere-array` | Three separated spheres indent at different times, translate laterally and retract completely. | Observe interacting deformation fields and separate force histories under multiple prescribed indenters. |

The first two cases share `comparison_pair = single-sphere-normal-pull-off-v1`. Only the work of adhesion and its derived nominal traction change. Their roles are `nonadhesive` and `adhesive`; the other cases have no comparison-pair identifier.

### Single-sphere schedule

Let `ys` denote the initial sampled surface height. The sphere's initial centre height is `ys + R + 0.045` m, the indentation height is `ys + R − 0.035` m, and the final centre height is `ys + R + 0.145` m. Each segment uses a half-cosine position interpolation with zero endpoint velocity.

| Time | Normal pair | Lateral release |
| --- | --- | --- |
| 0–0.06 s | Hold clear | Hold clear at `x = 0.38` |
| 0.06–0.25 s | Indent | Indent at `x = 0.38` |
| 0.25–0.42 s | Hold at the indentation height | Sweep from `x = 0.38` to `0.55` |
| 0.42–0.64 s | Retract vertically | Lift and move from `x = 0.55` to `0.65` |
| 0.64–0.85 s | Hold clear | Hold clear |

The array uses initial `(x, z)` centres `(0.32, 0.28)`, `(0.68, 0.28)` and `(0.50, 0.70)` m, stage delays of 0, 0.10 and 0.20 s, and x translations of −0.025, +0.025 and +0.065 m. Its path geometry keeps the rigid spheres apart. Actual final separation from the deforming layer and penalty overlap must be checked from the saved states.

## Quantities and review

Each saved frame records per-sphere position, velocity, reaction force and moment, compressed and adhesive reference areas, active point counts, minimum gap, maximum surface and whole-body penalty overlap, contact potential and actuator power. Reference areas are quadrature sums, not measured current contact areas.

The observer also evaluates the same sphere and interface law on the **initial undeformed** surface. This diagnostic force is never applied to the MPM body. The three `sphere_i_undeformed_surface_reaction_*_N` columns and `sphere_i_deformation_feedback_force_difference_N` make the change caused by surface deformation explicit.

Whole-body fields include surface displacement extrema, force and moment reciprocity residuals and base force. Sphere reactions are sampled at the output time. The base force is inferred from the next grid boundary impulse with the same contact forces included; `base_boundary_sample_time_s` records that timing distinction.

The independent review reconstructs the gap law from saved particle positions and sphere paths. It records source hashes and checks finite and positive-volume states, mass, force and moment reciprocity, resolved interaction range, limited penalty overlap, surface/bulk consistency and complete final separation. The nonadhesive baseline must have no attractive interface contribution; the adhesive cases must exhibit a tensile resultant during withdrawal before becoming force-free. Comparing a deformed surface with a frozen reference surface measures the effect of deformation on the applied interface forces.

The layer is finite and dynamic. A force–displacement loop or net actuator work can include kinetic energy, elastic energy and numerical losses; it is not by itself a measurement of adhesive fracture energy or frictional dissipation. Mesh convergence, molecular-scale contact physics, analytical JKR/MD limits and experimental calibration are outside this suite's demonstrated scope.

## Run

Activate the installed project environment, then run cases sequentially from the repository root:

```powershell
foreach ($case in 'nonadhesive-pull-off', 'adhesive-pull-off', 'lateral-release', 'three-sphere-array') {
    python -m examples.adhesive_suite --case $case --grid 48 --ppc 3 --frames 50 --no-render --resume
}
```

Results are written to `outputs/adhesion/<case>`. Review each completed run with `python tools/review_adhesion.py outputs/adhesion/<case>`. Rendering uses ParaView at the actual geometric scale, without axes or annotations. Rigid-sphere CAD represents prescribed geometry and is not counted as material points. The original `python -m examples.adhesive_array` command remains available and writes the same three-sphere loading definition to its original `dugdale-array` directory.

The suite's rigid-sphere overlays use opacity 0.45 so that the actual soft surface remains visible through them. This is a display setting only: it changes neither the interface model nor particle positions, forces or counts. The matched pair uses a common displacement colour range, camera bounds and opacity derived from both complete datasets; the other cases may use their own automatic ranges. See [the rendering controls](RENDERING.md#controlled-visual-comparisons) for the range checks and recorded display parameters.

After both normal pull-off cases pass their reviews, generate the matched force, area and displacement comparison:

```powershell
python tools/review_adhesion_comparison.py outputs/adhesion
```

This writes `adhesion_comparison.json` and `adhesion_comparison.csv`. The comparison verifies the shared initial particle coordinates, material, grid, time step and tool motion before reporting the measured differences.

<!-- ADHESION_RESULTS_START -->
## Recorded calculations

The four runs below completed their prescribed cycles and passed independent saved-state reviews. Forces are reactions on the rigid spheres; positive y is upward. Areas are reference-quadrature measures.

| Case | Total sphere reaction y range (N) | Peak summed contact / adhesive area (m²) | Surface y displacement range (mm) | Minimum final gap (mm) |
| --- | ---: | ---: | ---: | ---: |
| [Nonadhesive pull-off](results/adhesion--nonadhesive-pull-off/interpretation.json) | 0 to 64.2766 | 0.0239645 / 0 | -32.2519 to 10.104 | 141.96 |
| [Adhesive pull-off](results/adhesion--adhesive-pull-off/interpretation.json) | -13.9306 to 55.7875 | 0.0262651 / 0.0379598 | -32.3359 to 15.8334 | 142.46 |
| [Lateral sweep and release](results/adhesion--lateral-release/interpretation.json) | -13.6451 to 49.2562 | 0.0257858 / 0.0372888 | -32.3645 to 15.107 | 144.928 |
| [Three-sphere array](results/adhesion--three-sphere-array/interpretation.json) | -13.6146 to 131.38 | 0.0729 / 0.110333 | -32.4336 to 15.5203 | 141.231 |

For multiple spheres, each area's reported peak is the maximum over time of the sum of the individual reference areas; it is not a contact-union measurement. All spheres finish with zero interaction force and a gap greater than the interaction range plus one maximum particle spacing.

Particle counts: 367,744; step counts: 1,200; saved frames: 51. The minimum sampled J across the four runs is 0.768535.

### Matched pull-off pair

For the matched path, the saved instantaneous reaction difference (adhesive minus nonadhesive) ranges from -13.9694 to 0 N. The difference between the two peak compressive reactions is -8.48909 N; those peaks can occur at different times. The corresponding signed change in peak reference contact area is 0.00230059 m², and the change in minimum surface y displacement is -0.0839939 mm.

| Role | Minimum reaction during withdrawal (N) | Tensile withdrawal samples | Last active / first permanently separated saved time (s) |
| --- | ---: | ---: | ---: |
| Nonadhesive | 0 | 0 | 0.476 / 0.493 |
| Adhesive | -13.9306 | 4 | 0.51 / 0.527 |

The two times bracket final separation at the saved-frame resolution. See the [controlled-pair review](results/adhesion-comparison.json) and [aligned response CSV](results/adhesion-comparison.csv) for the complete histories and source checks.

The matched runs reach nearly the same maximum indentation under their common imposed path. In this calculation, attraction lowers the peak net compressive reaction, increases the peak reference contact area and raises the maximum surface uplift from 10.10 to 15.83 mm. It also produces a tensile withdrawal interval and later separation. These measured changes apply to this dynamic loading cycle; they do not establish a general monotonic trend with adhesion strength.

### Lateral release and array reactions

| Case / sphere | Reaction y range (N) | Reaction x range (N) | Time of maximum / minimum reaction y (s) | Maximum reaction difference from the undeformed surface (N) |
| --- | ---: | ---: | ---: | ---: |
| Lateral sweep and release / 0 | -13.6451 to 49.2562 | -4.93986 to 4.63554 | 0.408 / 0.476 | 538.916 |
| Three-sphere array / 0 | -14.304 to 57.328 | -1.23741 to 0.308296 | 0.391 / 0.476 | 540.79 |
| Three-sphere array / 1 | -13.894 to 58.6386 | -0.766443 to 0.892147 | 0.476 / 0.578 | 542.897 |
| Three-sphere array / 2 | -13.6146 to 55.6755 | -1.32714 to 1.41063 | 0.578 / 0.68 | 538.459 |

The frozen-surface force uses the same law and prescribed sphere path; it is a diagnostic of deformation feedback. Horizontal resultants come from radial normal forces. These dynamic finite-layer results do not establish frictional dissipation, irreversible adhesion energy, an analytical JKR/MD limit or experimental accuracy.

During the lateral sweep and oblique lift, the contact region migrates and the horizontal resultant changes sign. The interface still supplies only normal forces. In the array, the three withdrawal minima occur successively at 0.476, 0.578 and 0.680 s, with different individual force histories under the staggered paths on the common deformable layer. All tools are clear of their interaction ranges at the final saved state; the layer continues its dynamic response after release.

<!-- ADHESION_RESULTS_END -->

## Basis of the interface law

Maugis introduced a finite-range Dugdale description for the transition between adhesive sphere-contact regimes. Its analytical sphere solution also determines the contact and adhesive zones through elastic compatibility. The present implementation uses the finite-range traction idea with a penalty contact branch; it does not solve those analytical equations. [Maugis, 1992, *Adhesion of spheres: The JKR–DMT transition using a Dugdale model*](https://doi.org/10.1016/0021-9797(92)90285-T).

Bazrafshan, de Rooij, Valefi and Schipper state `Δγ = σ0 h0` and relate the current gap to the deformation and imposed approach in Sections 2–3. Their numerical formulation uses an elastic half-space and a conjugate-gradient contact solver. Those assumptions differ from this finite, hyperelastic, explicitly integrated MPM layer and its reference-area quadrature. [*Numerical method for the adhesive normal contact analysis based on a Dugdale approximation*, Tribology International 112 (2017), 117–128](https://doi.org/10.1016/j.triboint.2017.04.001); [author-hosted full text](https://ris.utwente.nl/ws/portalfiles/portal/17418713/numerical1.pdf).

Section 2 of Johnson's paper describes the Maugis–Dugdale normal-contact construction, including the finite traction range and elastic gap condition. Its later treatment of tangential friction is not implemented here. [Johnson, 1997, *Adhesion and friction between a smooth elastic spherical asperity and a plane surface*, Proceedings of the Royal Society A 453, 163–179](https://bdml.stanford.edu/twiki/pub/Rise/AdhesionModels/JohnsonAdhesionModel.pdf).
