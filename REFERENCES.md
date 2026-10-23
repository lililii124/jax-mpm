# References

- [Taichi MLS-MPM example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py):
  quadratic B-splines, APIC transfers, and stress weighting. Reference blob
  `83b013d039beb274b769d6a271470a79fe240a2e`, Apache-2.0.
- [DiffMPM](https://github.com/geoelements/diffmpm): JAX state organization and
  compiled time integration. Reference solver blob
  `3b1ae01bdcc474d2e77e167c1634370ec05b973d`.
- [JAX](https://github.com/jax-ml/jax): arrays, JIT compilation, automatic
  differentiation, and scan/control-flow primitives.
- [RHT contribution to OpenCourant](https://github.com/OpenCourant/OpenCourant/commit/1a0d691b82bd4f94a3bc2399682551e84e1d5fa4):
  public three-surface implementation and verification work by Yu Zheng.

## Supplied constitutive references

The JAX implementations were developed against the user's material deliveries.
Original host SDK files and reference binaries are not distributed here.

| Model | Selected source |
|---|---|
| JC | `johnson_cook_solid.hpp`, supplied SPHinXsys work, 2026-10-22; the JAX return uses an implicit rate solve |
| JH-2 | `MAT110.f` revision 7 and `jh2_core.h`, supplied 2026-10-21 snapshot |
| HJC | `umat_hjc_user_tested_after_tension_fix.f`, supplied v3 tension-fix profile |
| RHT | `rht_material_mod.F90`, 2026-09-24, AGPL-3.0-or-later; not the older simplified `umat_rht_palpha_verified.f` |
| Selected MAT branches | `MAT001_296_latest_20261017`, audited against current-source Final Check evidence |

The RHT core follows the public three-surface formulation described by
Borrvall and Riedel (2011). Its selected numerical profile includes separate
pre/post-peak return branches, adaptive substeps, and energy iteration.
See [material scopes](docs/MATERIALS.md), [source hashes](docs/source_audit.json),
and the [reference fixture manifest](tests/data/materials/manifest.json).

Reference trace arrays are numerical outputs of the supplied implementations.
The fixtures record source and output SHA-256 hashes. They verify the selected
paths and conventions, not universal equivalence to a commercial material card.

## Example settings and visualization

- [SPHinXsys elastic gate](https://www.sphinxsys.org/html/examples/example2_elasticGate.html)
  and [LS-DYNA dam gate](https://lsdyna.ansys.com/dam5/): water loading of a clamped elastic gate.
- [DualSPHysics test cases](https://github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases):
  dam break and floating-body settings.
- [JAX-FEM hyperelasticity](https://deepmodeling.github.io/jax-fem/learn/hyperelasticity/example.html):
  finite-strain deformation and twist loading.
- [LS-DYNA ball example](https://www.dynaexamples.com/introduction/intro-by-j.-day/misc/ball)
  and [example manual](https://www.dynasupport.com/manuals/additional/ls-dyna-examples-manual):
  gravity, contact and structural dynamics examples. No reference input deck is redistributed.
- [ParaView 6.2 user guide](https://docs.paraview.org/en/v6.2.0/UsersGuide/):
  VTK data, batch rendering and saved visualization states.

Specific adaptations and differences are listed in [EXAMPLES.md](docs/EXAMPLES.md).

## Material engineering examples

- D. R. Lesuer (2000), [*Experimental Investigations of Material Models for
  Ti-6Al-4V Titanium and 2024-T3 Aluminum*, DOT/FAA/AR-00/25](https://ntlrepository.blob.core.windows.net/lib/17000/17400/17488/PB2001101864.pdf).
  Table 6, printed p. 17 (PDF p. 29), reproduces the original AA2024-T351 JC
  strength and failure constants. Equations (1)–(5), printed p. 2 (PDF p. 14),
  define the law and normalization; printed p. 18 (PDF p. 30) gives the original
  torsion-fit rate range of 0.088–123 s⁻¹. Table 8's later 2024-T3 fit is separate
  from the T351 parameter set used here.
  [Official report record](https://rosap.ntl.bts.gov/view/dot/93874).
- Kaiser Aluminum, [*Rod & Bar Alloy 2024*, KA-RBH-2024-8.10](https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1022/Kaiser_Aluminum_2024_Rod_and_Bar.pdf).
  Typical T4/T351 elastic modulus on p. 1; density, heat capacity and melting
  interval on p. 2. The example assumes constant heat capacity, Poisson's ratio
  0.33 and plastic-work heat fraction 0.9.
- [Abaqus documentation: Johnson-Cook dynamic failure](https://abaqus-docs.mit.edu/2017/English/SIMACAEMATRefMap/simamat-c-johnsoncook.htm).
  Documents the sign change of the third failure coefficient when using
  compressive pressure instead of tensile mean stress.

The [material engineering guide](docs/MATERIAL_ENGINEERING.md) identifies which
values are sourced and which settings are assumptions. Its HJC and JH-2 cases
use repository constitutive defaults and demonstration loading histories;
they do not reproduce a calibrated material test. The JC round-bar case reaches
a nominal plateau rate of 1,800 s⁻¹, above the original torsion-fit range.
