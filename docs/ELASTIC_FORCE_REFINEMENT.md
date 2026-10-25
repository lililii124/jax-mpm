# Elastic force refinement

The production three-dimensional MLS particle-to-grid force and grid-to-particle acceleration are compared with an independently differentiated smooth deformation. This is a frozen spatial operator study; it advances no dynamic trajectory.

| Material | Grid 32 | Grid 48 | Grid 64 | Observed orders |
|---|---:|---:|---:|---|
| Orthotropic, axes rotated 30° | 14.016% | 6.571% | 3.768% | 1.87, 1.93 |
| Transversely isotropic, longitudinal axis tilted 30° | 14.025% | 6.575% | 3.771% | 1.87, 1.93 |
| Blatz–Ko, reference Poisson ratio 0.463 | 14.308% | 6.713% | 3.844% | 1.87, 1.94 |

The predeclared finest-grid relative acceleration error limit is 5%. All three cases decrease under refinement and meet that limit. Independently reconstructed Kirchhoff stresses, analytical Piola divergences checked by centred differences, full particle support, positive deformation Jacobians, total mass, net force and net torque pass their recorded tolerances.

The reference material cube is `[0.125,0.875]^3 m`. The compatible displacement is `u=A b(X)b(Y)b(Z)(1,0.6,-0.4)`, with `b(s)=sin(pi*(s-0.25)/0.5)^4` inside `(0.25,0.75)` and zero outside. Its gradient supplies `F`; acceleration is `Div_X(P)/rho0`, with `P=J sigma F^-T`. Displacement and stress vanish in an exterior layer, giving traction-free material faces without imposed nodal reactions. The two linear laws use amplitude 20 µm; Blatz–Ko uses 2 mm. Density is 1000 kg/m³.

The grids contain 32, 48 and 64 cells per axis, with two particles per cell direction. The finest case has 884736 particles. Transfers run in batches of 32768 particles through the unchanged production functions. Actual peak USS was below 0.4 GiB.

```sh
python -m tools.check_elastic_force_refinement --output outputs/elastic-force-refinement
```

[Complete result](results/elastic-force-refinement.json) · [Raw run records](../evidence/elastic-force-refinement)

This study does not certify dynamic time integration, large-deformation structural responses, contact or fracture.
