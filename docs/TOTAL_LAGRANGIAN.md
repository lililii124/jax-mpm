# Reference-grid dynamics

`mpm_jax.total_lagrangian` keeps the quadratic basis and material quadrature fixed in reference coordinates. Displacement and velocity remain on nodes, avoiding a particle鈥揼rid velocity projection each step. It uses lumped nodal mass, the first Piola stress and velocity-Verlet integration. The existing material factories supply stresses.

The nodal balance follows the reference formulation described by [de Vaucorbeil, Nguyen and Hutchinson](https://researchmgt.monash.edu/ws/portalfiles/portal/350907422/292716078_oa.pdf). This implementation uses the repository's quadratic basis and lumped inertia; it does not reproduce every element of that paper.

`reference_grid` checks complete support, partition of unity, linear reproduction and mass. `NodalBoundary` prescribes material-node displacement, velocity and acceleration. Body and support impulses, torques and work are recorded. Supply consistent endpoint boundary data; boundary work uses trapezoidal reactions and is not an exact work certificate for arbitrary prescribed motion. The supplied controls cover uniform acceleration, quadratic prescribed motion and a fixed support.

The initial-state helper starts at rest. The momentum ledgers assume zero initial linear and angular momentum. A nonzero initial-velocity extension must preserve its initial momentum baselines. Contact, guide-centroid constraints and fracture separation require separate formulations. Ballistic failure modes are rejected. Sampling particles for output does not change the nodal inertia metric.

## Manufactured dynamics

Both backends use the same three-dimensional reference cube, rotated orthotropic/transversely isotropic materials, Blatz鈥揔o material and independently derived body force. The exact compact displacement is

```
u(X,t) = [1 - cos(2*pi*t/T)] * u0(X),  T = 0.04 s.
```

The independent force derivation and material constants are in `tools/check_elastic_force_refinement.py` and `tools/check_elastic_dynamic_mms.py`. Error norms use the exact full-cycle displacement, velocity and displacement-gradient amplitudes. The target remains 5%. Maxima are sampled at accepted 20-step report times.

The APIC series retains its failed accuracy results: at grid64 the transversely isotropic case has maximum velocity error 5.0223% and deformation-gradient error 5.1404%. Halving its time step increases these to 8.1411% and 7.9439%. Positive Jacobians and momentum checks still pass. These are accuracy failures, not invalid checkpoints.

For the new reference-grid method, the corresponding grid64 errors are 2.5186% in position, 3.6004% in velocity and 3.8602% in deformation gradient. With half the time step they are 2.5399%, 3.6001% and 3.8624%. The final-time nodal differences are also archived. Two runs at one spatial resolution do not establish a temporal convergence rate.

A separate uniform harmonic-acceleration control removes spatial deformation error. Halving time steps from 1 ms to 0.5 ms to 0.25 ms gives observed position and velocity orders within 0.00009 of two. This isolates the time integrator from the material-wave spatial error.

A frozen zero-stress APIC transfer probe measures 0.03865% kinetic-energy loss for one exact quarter-cycle field remap and 0.23463% for one final-checkpoint remap. The affine control is preserved to roundoff. These measurements identify a transfer loss component; multiplying them by the step count would not reconstruct the evolving trajectory's energy budget.

Full case tables, source hashes, checkpoint identities and resource reports are in [the result index](results/elastic-dynamics.json) and [evidence](../evidence/elastic-dynamics/manifest.json). APIC and reference-grid runs have different semidiscrete inertia; neither is presented as an identical restart of the other. The two linear Cauchy-stress branches have only an infinitesimal strain-energy diagnostic. Blatz鈥揔o uses its finite-strain potential. None of these checks certifies contact or post-fracture motion.

## Run

Use a fresh output directory for each command:

```sh
python -m tools.check_total_lagrangian --output outputs/tl-controls
python -m tools.check_total_lagrangian --temporal-only --output outputs/tl-time-control
python -m tools.check_elastic_tl_mms --model transverse-isotropic --grid 64 --output outputs/tl-ti64
python -m tools.check_elastic_tl_mms --model transverse-isotropic --grid 64 --time-refinement 2 --output outputs/tl-ti64-halfdt
python -m tools.check_elastic_dynamic_mms --model transverse-isotropic --grid 64 --output outputs/apic-ti64
```

Other model names are `orthotropic` and `blatz-ko`; grid choices are 32, 48 and 64. Both runners support `--stop-after-steps` and `--resume-from` with a matching contract. A candidate is saved before physical checks; only accepted checkpoints are valid restart inputs. Full-cycle completion and the accuracy target are separate report fields.


## Accepted reference-grid cycles

| Material | Grid | Time-step refinement | Position error | Velocity error | Gradient error |
| --- | ---: | ---: | ---: | ---: | ---: |
| blatz-ko | 64 | 1 | 2.0804% | 3.1508% | 3.2282% |
| orthotropic | 64 | 1 | 2.3347% | 3.7920% | 3.3465% |
| transverse-isotropic | 32 | 1 | 9.4903% | 13.3004% | 14.2293% |
| transverse-isotropic | 48 | 1 | 4.3933% | 6.2003% | 6.6624% |
| transverse-isotropic | 64 | 1 | 2.5186% | 3.6004% | 3.8602% |
| transverse-isotropic | 64 | 2 | 2.5399% | 3.6001% | 3.8624% |
