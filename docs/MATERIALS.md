# Material conventions and scope

Stress is positive in tension; pressure is positive in compression. History
stress is Cauchy stress, and P2G uses Kirchhoff stress J sigma. Densities,
stresses, lengths, times, and temperatures use kg/m³, Pa, m, s, and K.

## Elasticity

See the [elastic model and 3D example map](ELASTIC_MODEL_MAP.md) for runnable
commands, existing verification and remaining spatial validation gaps.

The [MAT002 finite-strain solid](MAT002_SVK.md) adds the separate
`orthotropic-svk` Green-strain branch and its native-kernel check.

- `linear_elastic(E, nu)` uses infinitesimal strain sym(F)−I.
- `orthotropic(E, nu, G, Q)` takes E=(E1,E2,E3), nu=(nu12,nu13,nu23),
  G=(G12,G13,G23). nu_ij is −epsilon_j/epsilon_i in uniaxial stress along i.
  Q columns are material axes in global coordinates. Compliance must be
  positive definite; Voigt order is xx,yy,zz,yz,xz,xy with engineering shear.
- `transverse_isotropic` has longitudinal local axis 3. The examples tilt this
  axis by 30 degrees, rather than rotating only within the isotropic plane.
- `neo_hookean` uses tau=mu(B−I)+lambda log(J)I. `elastic` remains an alias.
- `blatz_ko` implements the supplied MAT007 REF=0 rubber branch, with reference
  Poisson ratio 0.463. Its stress is G[B−J^(1−1/(1−2nu))I]; this is a selected
  branch, not every variant called Blatz–Ko.
- `fluid` uses W=K(J−1)²/2 and sigma=K(J−1)I. Shear history is discarded.

## Incremental models

| Model | Implemented scope | Important limits |
|---|---|---|
| Bilinear J2 | Combined isotropic/kinematic hardening; beta=1 gives isotropic hardening | Rate-independent; no erosion |
| Thermal plasticity | Two-point property tables, thermal expansion, J2 return | Temperature prescribed to the material-point update; spatial example holds initial temperature fixed; no conduction |
| Johnson–Cook | Implicit rate-dependent radial return at old temperature, explicit plastic heating, optional D1–D5 failure | Closure differs from the supplied one-step old-rate implementation; examples leave optional failure disabled |
| Steinberg–Guinan | Pressure/temperature strength, strain hardening cap, linear EOS, work energy | Selected GAMA=F=0 solid branch; no melting/spall; invalid temperatures rejected |
| JH-2 | Intact/fractured strength, damage, delayed bulking, polynomial EOS, tensile cutoff | Supplied MAT110 revision-7 profile; negative FS unsupported |
| HJC | Pressure/rate strength, compaction, lagged unloading modulus, damage | `legacy-v3`, `strain-floor`, and `mat111-compatible` point profiles; cold-start damage seed is off by default |
| RHT | Three surfaces, Lode dependence, hardening, damage, p-alpha compaction, energy coupling | Bounded adaptive integration; unsupported/nonconverged returns are marked invalid; no general interior fallback from the supplied core |
| MAT173-inspired Mohr–Coulomb | Constant-property six-plane principal stress return; nonassociated flow; selected NOVOID=1 hydrostatic tension branch | First-draft Fortran source; JAX adapter is small-strain and selected-branch only; complex Final Check and native history mapping are pending; see [slope response](MAT173_LANDSLIDE.md) |

Incremental kinematics use the right polar stretch of F_new inverse(F_old),
its logarithmic strain, and rotation of tensor history. This is the spatial
integration convention of this package. Material-point replay tests bypass
that kinematic conversion and therefore do not establish equivalence of an
entire MPM solver to the source host solver.

The RHT return solver uses yielding-point batches for large arrays; the
[implementation comparison](RHT_RETURN_MAP.md) records its equivalence checks.

### HJC profiles

`hjc()` and `hjc(profile="legacy-v3")` preserve the supplied v3 conventions,
including the EFMIN activation gate, tensile strength expression and plastic
strain handling at zero strength. The five supplied HJC reference traces apply
to this profile and remain unchanged.

`hjc(profile="strain-floor")` selects the strength and damage expressions in
[Altair Radioss 2025 LAW126, Comments 3–4](https://2025.help.altair.com/2025/hwsolvers/rad/topics/solvers/rad/mat_law126_johnson_holmquist_concrete_starter_r.htm).
With compression-positive pressure, its fracture strain is
`max(D1*((P+T)/fc)^D2, minimum_fracture_strain)` and the damage increment is the
sum of deviatoric and compaction plastic increments divided by that strain.
Thus EFMIN limits the damage increment at low pressure; it does not disable it.
At nonpositive pressure the normalized strength is `A*(1+P/T)*(1-D)`.
The rate multiplier is `1+C*max(log(rate/reference_rate), 0)`. The existing
compressive pressure-hardening term and compressive strength cap remain.
Plastic strain still accumulates when the returned equivalent stress is zero.

These two profiles use the existing compaction/EOS integration, old damage for the
current strength and tensile cutoff, and an explicit damage update afterward.
The new profile does not establish full Radioss equivalence, strain-rate
filtering, nonlocal damage regularization or automatic particle deletion.
At `D=1`, cohesion is lost while compressive confinement can still carry shear
stress. Damage is a local state variable, not a resolved crack surface.
The optional `cold_start_damage=True` retains the supplied conditional seed
in either profile: with `raw=D1*(T/fc)^D2`, initial damage is
`min(1, 1e-4/raw)` only when `raw > minimum_fracture_strain`, otherwise zero.
This is a numerical initialization choice outside the cited evolution
equations, and remains disabled by default.

`hjc(compacted_tension="mat111-linear")` independently selects the supplied
MAT111 compacted-pressure block. With modified compression
`u=(1/J-1-lock_strain)/(1+lock_strain)`, its pressure before the tensile cutoff is
`K1*u+(K2+K3*u)*max(u,0)**2`. Thus fully compacted material unloading below the
locking volume uses a linear tensile branch. The default
`compacted_tension="legacy-polynomial"` retains the original full polynomial,
including its negative-u quadratic/cubic terms. Strength/damage profiles and
the partially compacted unloading branch are independent of this choice.
The reference table for this selected block was generated by compiling the
supplied MAT111 scalar module; it does not certify the whole host adapter.
The published symmetry-constrained beam never reached full compaction and
retains its original source and result.

`hjc(profile="mat111-compatible")` follows the supplied current MAT111
`hjc_point` scalar path, including the uncapped nonpositive-pressure strength,
subreference logarithmic rate factor, EFMIN activation gate, zero-strength
plastic increment, and the zero-trial-stress damage increment. It selects
`mat111-linear` compacted unloading and requires zero cold-start damage.
Eight [compiled-reference points](../tests/data/materials/hjc_current_mat111_points.json)
cover those branches at D1=0.04 and D1=0.004. Four
[sequential reference histories](../tests/data/materials/hjc_current_mat111_histories/manifest.json)
add 248 increments of compaction/unloading/reloading, compressed shear reversal,
active damage, and tension/recompression. The scalar path preserves arithmetic
boundaries because the source's exact-zero trial-stress branch changes the
damage increment; reassociation near hydrostatic loading can select another branch.
These checks cover finite,
fixed-axis material-point updates; host failure/erosion and structural
validation remain separate. The source Final Check does not transfer to JAX.

`python -m examples.material_engineering --case concrete-bending-mat111 --grid 64 --ppc 3 --frames 60 --no-render`
runs this profile on the 12 mm symmetry-constrained beam. Its geometry, supports,
loading and initial state match `concrete-bending-symmetric`; the material profile
and output directory differ. `--stop-after-frame 4` saves a resumable prefix;
repeat with `--resume` to continue. This is a structural comparison case, with
full-loading and energy verification still pending.

The strength profile and compacted unloading option are stored as Boolean
material fields `legacy_v3`, `mat111_compacted_tension`, and
`mat111_compatible` for JAX and serialization. Material signatures include
these choices; resume earlier
checkpoints with their original code version. Completed result records and
their reference data are retained.

## Final Check source audit

`tools/audit_delivery.py` compares the current source SHA-256 to its manifest
and acceptance record. [source_audit.json](source_audit.json) records nine
eligible IDs from `MAT001_296_latest_20261017`:

| Source ID | JAX branch | Scope of source evidence |
|---|---|---|
| MAT001 | Isotropic linear elasticity | Ordinary solid branch |
| MAT002 | Orthotropic elasticity | Supplied engineering constants/orientation branch |
| MAT003 | Mixed-hardening bilinear J2 | Rate off, no erosion |
| MAT004 | Prescribed-temperature plasticity | Fixed-temperature selected branch; dynamic thermal host not validated |
| MAT007 | Blatz–Ko | REF=0, nu=0.463 |
| MAT011 | Steinberg–Guinan | Fixed-axis one-dimensional-strain source checks, GAMA=F=0, linear EOS |
| MAT012 | Isotropic-hardening bilinear J2 | Selected isotropic solid branch |
| MAT015 | Johnson–Cook | Selected source plastic/thermal/damage checks |
| MAT111 | HJC | Selected source D1=0.04 profile |

A source Final Check pass is not inherited by this JAX rewrite. MAT006's
historical result is excluded because its structural Final Check is pending.
MAT110 and MAT272 are not listed as Final Check passes; their JH-2/RHT
implementations have separately documented reference-trace tests.

Demonstration parameters are not calibrated for a particular experimental
material. Damage localization is mesh dependent without a length-scale or
fracture-energy regularization. Failure flags do not remove particle mass.
