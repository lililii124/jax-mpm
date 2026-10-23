# Woven composites

The mesoscale model separates curved, homogenized yarns from the surrounding matrix. Both phases occupy the same uniform material-point lattice. The matrix fills the entire space outside the yarns; there is no void phase. They share the MPM velocity field, so the interface is perfectly bonded. Fibre-scale strands, yarn slip, debonding, plasticity and damage are outside this model.

The three geometries are 0°/90° plain weave, 0°/+60°/−60° triaxial interlacing, and 0°/90° two-over-two twill. There are no through-thickness binder yarns. The triaxial construction alternates the vertical order at staggered pairwise crossings; it is not three permanently stacked plies. Its matrix-filled specimen is different from an open triaxial fabric.

## Geometry

Each yarn has a planar direction, a smooth height profile and an elliptical section normal to its centreline. Half-cosine spans connect the prescribed over/under crossings with continuous tangents. A normal-plane projection locates each material point in a yarn section and supplies its reference tangent, transverse axis and normal. These three axes form a right-handed orthonormal frame.

All points have the same spacing and reference volume. A point inside two yarns causes construction to fail; the code does not resolve overlaps by giving one yarn priority. A separate surface audit samples the continuous yarn surfaces against the other yarns. It reports clearance, unresolved projections and a section-radius/curvature bound. This sampling is a geometric check, not a proof of continuous clearance.

The current common geometry has the following dimensions:

| Quantity | Value |
| --- | ---: |
| Domain side | 16 mm |
| Specimen size, x × y × z | 10 × 2 × 10 mm |
| In-plane pitch | 2.5 mm, four periods |
| Yarn section width × height | 0.625 × 0.5 mm |
| Vertical clearance at crossings | 0.5 mm |
| Matrix cover per face | 0.25 mm |
| Grid | 96³; cell size 0.166667 mm |
| Material-point spacing | 0.0833333 mm, two points per cell direction |
| Material points | 345,600 |
| Yarn height resolution | Three grid cells, six point spacings |

The three-cell yarn thickness is a starting resolution, not a convergence result. More particles within the same grid do not improve the grid's ability to resolve a thin yarn.

Actual construction at this spacing gives the following geometric checks. The surface audit used 64 samples per crossing span and 128 points around each elliptical section.

| Architecture | Yarn volume fraction | Multiple particle memberships | Sampled surface intersections | Minimum other-yarn ellipse radius squared |
| --- | ---: | ---: | ---: | ---: |
| Plain | 0.109259 | 0 | 0 | 8.919922 |
| Triaxial | 0.186134 | 0 | 0 | 1.597208 |
| Twill 2×2 | 0.104630 | 0 | 0 | 8.919922 |

An ellipse radius squared below one would indicate penetration of another yarn. All normal-plane projections converged in these audits. The largest half-height times curvature bound was 0.592177. These are deliberately sparse, fully infiltrated illustrative geometries, not measured commercial fabrics. Their different yarn volume fractions must be included when comparing structural response; topology alone does not explain a difference.

## Constitutive model

`WovenElastic` uses a compressible Neo-Hookean matrix and a locally orthotropic Saint Venant–Kirchhoff yarn. If `Q0` contains the reference local axes and `F` is the deformation gradient,

```text
E = (Fᵀ F − I) / 2
E_local = Q0ᵀ E Q0
S_local = C : E_local
τ_yarn = F Q0 S_local Q0ᵀ Fᵀ
```

The engineering-strain vector is `(Exx, Eyy, Ezz, 2Eyz, 2Exz, 2Exy)`. Yarn energy is `0.5 eᵀ C e` per reference volume. For the matrix it is `0.5 μ (tr(FᵀF) − 3) − μ ln(J) + 0.5 λ ln(J)²`. Both stresses are derived from these energies and are objective under a superposed rigid rotation.

The yarn law supports finite rotations with small elastic strains. It is not a finite-strain failure model, and large compression can leave its useful range. The examples must report actual elastic strains rather than infer validity from a positive Jacobian alone.

The example stiffnesses are explicitly illustrative: a 2 GPa isotropic matrix and yarn Young's moduli of 40/4/4 GPa. The complete Poisson ratios, shear moduli and phase densities are recorded in each case. These constants are not a calibration of a named fibre/resin system. Density is assigned by phase to particle mass. There is no density or mass scaling for faster computation.

`woven_wave_speed_bound` bounds the initial linearized acoustic speeds using the largest eigenvalue of the yarn's Kelvin stiffness and the matrix stiffness. It does not establish stability at arbitrary finite deformation. Time-step selection and a time-step comparison remain separate numerical checks.

## Interface and checks

`examples.woven_geometry.sample_weave(...)` returns `WeaveSample`: positions, uniform reference volumes, matrix/yarn labels, yarn and family identifiers, reference frames, and compact geometric metadata. `mpm_jax.materials.woven.woven_elastic(...)` builds the phase material. `woven_energy(F, material)` and `woven_kirchhoff_stress(F, material)` accept a batch of deformation gradients.

Geometry tests cover actual phase occupancy, right-handed frames, continuous crossing order, rejection of overlapping yarns, independent surface exclusion and preservation of a uniform lattice. Material tests check the energy derivative, rigid rotation, stress covariance, the infinitesimal orthotropic tangent, phase-dependent orientation and the initial acoustic bound. Passing these local tests is not an experimental validation of a composite specimen.

The loading series uses warp and transverse in-plane tension, in-plane shear, through-thickness compression and out-of-plane bending. Boundary geometry, strain histories and numerical results are recorded by the individual examples. Full deformed geometry is rendered without displacement amplification. Phase/family views describe the actual particle assignments.

## Prescribed loading series

All 15 combinations use a half-cosine displacement over 14.230249 µs, followed by a fixed-displacement hold to 18.973666 µs. These are dynamic loading paths, not quasi-static measurements. The nominal grid96 calculation uses 4,140 steps and 60 output intervals; the initial acoustic bound is approximately 5,401.59 m/s.

| Loading | Constraints and final displacement |
| --- | --- |
| `warp-tension` | Two 1 mm x-end grips move by ±80 µm in x; 8 mm free gauge length and 2% nominal extension. All grip velocity components are prescribed. |
| `weft-tension` | The same arrangement on z ends, moving by ±80 µm in z. In the triaxial model this means global z loading, not loading a 90° yarn family. |
| `in-plane-shear` | Two z-end grips move by ±0.2 mm in x, giving 5% nominal shear over the 8 mm free span. |
| `transverse-compression` | Opposed y-face patches move inward by 25 µm each and allow x/z sliding. The fixed physical patch depth is 0.375 mm per face: the full-thickness strain is −2.5%, and the 1.25 mm free region has −4% nominal strain. |
| `transverse-bending` | Bottom supports at x=4 and 12 mm, each 0.4 mm wide. The left fixes x/y, the right fixes y, and a local left-midwidth patch fixes z. A 1 mm central top patch moves down by 0.25 mm. These bilateral velocity patches are fixture idealizations. |

Moving-grip tracking uses reference points at least 1.5 grid cells inside each face and checks their actual complete support at every output. Thin bending patches may have no such interior particle; their loading is checked through grid reactions and the actual specimen deflection instead of claiming an exact particle displacement.

```powershell
python -m examples.woven_composite --architecture all --loading all --grid 96 --ppc 2 --frames 60 --resume
python -m mpm_jax.visualization outputs/woven/plain/transverse-bending --field family_id --gif
```

The first command writes numerical output without starting ParaView. Each architecture can be selected as `plain`, `triaxial` or `twill-2x2`; each loading can be selected separately using the names in the table. The commands specify the cases and do not imply that all runs have already completed.

Run one case, or the fifteen cases sequentially:

```bash
python -m examples.woven_composite --architecture plain --loading warp-tension
python -m examples.woven_composite --architecture all --loading all
```

The default writes numerical output without starting ParaView. Resume a matching
case with `--resume`. Render a completed case separately:

```bash
python -m mpm_jax.visualization outputs/woven/plain/warp-tension --field family_id --gif
```

Loading lasts 14.23025 µs, followed by a 4.74342 µs displacement hold. The
examples are dynamic tests. For the triaxial architecture, transverse in-plane
tension acts along global z; there is no 90° yarn family. The compression
fixtures constrain only normal velocity in fixed 0.375 mm face layers, leaving
a 1.25 mm free thickness. Bending uses 0.4 mm support patches over an 8 mm span;
the left support fixes x/y, the right support permits axial sliding, and a small
left patch removes lateral rigid motion. These are prescribed grid velocity
regions, not separately solved contact bodies.

## Sources and scope

TexGen represents yarns using centreline interpolation, cross sections and local fibre directions, and separately checks interference. The present normal-plane geometry and constitutive implementation are independently written; TexGen source is not copied. See the pinned [TexGen yarn interface](https://github.com/louisepb/TexGen/blob/99eb6db4ea72679ccbdfea01a3ce2f72fee07b8a/Core/Yarn.h), [2D weave interface](https://github.com/louisepb/TexGen/blob/99eb6db4ea72679ccbdfea01a3ce2f72fee07b8a/Core/TextileWeave2D.h), and [official geometry workshop](https://texgen.sourceforge.io/workshop/TexGenWorkshop_Geometry_modelling.pdf).

The 0°/+60°/−60° family convention and local yarn axes are also used in S. V. Hoa, S. Z. Sheng and P. Ouellette, *Determination of elastic properties of triax composite materials*, Composites Science and Technology 63 (2003), 437–443, [DOI:10.1016/S0266-3538(02)00216-6](https://doi.org/10.1016/S0266-3538(02)00216-6). That paper does not calibrate the matrix-filled geometry or stiffness choices used here. No measured effective modulus, isotropy or failure strength is claimed from these references.
