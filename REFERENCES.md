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
