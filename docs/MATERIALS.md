# Material conventions and scope

Stress is positive in tension; pressure is positive in compression. History
stress is Cauchy stress, and P2G uses Kirchhoff stress J sigma. Densities,
stresses, lengths, times, and temperatures use kg/m³, Pa, m, s, and K.

## Elasticity

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
| HJC | Pressure/rate strength, compaction, lagged unloading modulus, damage | Supplied v3 tension-fix profile; EFMIN acts as an activation gate; optional cold-start damage seed is off |
| RHT | Three surfaces, Lode dependence, hardening, damage, p-alpha compaction, energy coupling | Bounded adaptive integration; unsupported/nonconverged returns are marked invalid; no general interior fallback from the supplied core |

Incremental kinematics use the right polar stretch of F_new inverse(F_old),
its logarithmic strain, and rotation of tensor history. This is the spatial
integration convention of this package. Material-point replay tests bypass
that kinematic conversion and therefore do not establish equivalence of an
entire MPM solver to the source host solver.

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
