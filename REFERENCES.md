# References

- [Taichi MLS-MPM example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py):
  quadratic B-splines, APIC transfers, stress weighting, and grid boundaries.
  Reference blob: `83b013d039beb274b769d6a271470a79fe240a2e`.
  Licensed under Apache-2.0.
- [DiffMPM](https://github.com/geoelements/diffmpm):
  JAX state organization and compiled time integration.
  Reference solver blob: `3b1ae01bdcc474d2e77e167c1634370ec05b973d`.
- [JAX](https://github.com/jax-ml/jax):
  array operations, JIT compilation, automatic differentiation, and `lax.scan`.

The transfers have been rewritten as batched JAX scatter/gather operations.
The solid uses Neo-Hookean elasticity, and deformation is updated after G2P.
The fluid has a volumetric energy model. Neither Taichi's snow plasticity
model nor DiffMPM's USF/USL element implementation is included.
