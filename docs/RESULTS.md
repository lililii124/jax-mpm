# Completed runs

Each row links to the saved configuration and numerical results. Particle trajectories
and ParaView datasets remain in the local output directory. Solver time excludes
compilation, checkpoint I/O and rendering; these CPU runs shared the host with other jobs.

Public-example adaptations are described in [EXAMPLES.md](EXAMPLES.md). Finite states
and a positive determinant establish numerical admissibility, not experimental validation.

| Run | Particles | Steps | Physical time (s) | Minimum J | Maximum damage | Rendering |
|---|---:|---:|---:|---:|---:|---|
| [suite--cantilever--neo-hookean](results/suite--cantilever--neo-hookean/summary.json) | 36,540 | 240 | 0.15 | 0.997722 | 0 | ParaView |
| [suite--compression--neo-hookean](results/suite--compression--neo-hookean/summary.json) | 24,389 | 100 | 0.0636396 | 0.985686 | 0 | ParaView |
| [suite--dam-break--neo-hookean](results/suite--dam-break--neo-hookean/summary.json) | 66,096 | 1,600 | 0.35 | 0.740224 | 0 | ParaView |
| [suite--fsi--neo-hookean](results/suite--fsi--neo-hookean/summary.json) | 85,776 | 1,600 | 0.35 | 0.808814 | 0 | ParaView |
| [suite--head-on--neo-hookean](results/suite--head-on--neo-hookean/summary.json) | 38,686 | 580 | 0.24 | 0.827806 | 0 | ParaView |
| [suite--oblique--neo-hookean](results/suite--oblique--neo-hookean/summary.json) | 38,686 | 580 | 0.24 | 0.756636 | 0 | ParaView |
| [suite--rebound--neo-hookean](results/suite--rebound--neo-hookean/summary.json) | 24,389 | 860 | 0.36 | 0.578802 | 0 | ParaView |
| [suite--shear--neo-hookean](results/suite--shear--neo-hookean/summary.json) | 73,008 | 240 | 0.15 | 0.978174 | 0 | ParaView |
| [suite--sliding--neo-hookean](results/suite--sliding--neo-hookean/summary.json) | 16,704 | 600 | 0.25 | 0.803204 | 0 | ParaView |
| [suite--sloshing--neo-hookean](results/suite--sloshing--neo-hookean/summary.json) | 82,944 | 1,600 | 0.35 | 0.84515 | 0 | ParaView |
| [suite--surface-load--neo-hookean](results/suite--surface-load--neo-hookean/summary.json) | 33,640 | 240 | 0.15 | 0.993425 | 0 | ParaView |
| [suite--torsion--neo-hookean](results/suite--torsion--neo-hookean/summary.json) | 36,540 | 240 | 0.15 | 0.992961 | 0 | ParaView |
| [suite--water-entry--neo-hookean](results/suite--water-entry--neo-hookean/summary.json) | 118,592 | 1,600 | 0.35 | 0.876118 | 0 | ParaView |
| [suite--wave--neo-hookean](results/suite--wave--neo-hookean/summary.json) | 41,472 | 200 | 0.134164 | 0.994805 | 0 | ParaView |
| [elastic_cantilever--cantilever--blatz-ko](results/elastic_cantilever--cantilever--blatz-ko/summary.json) | 36,540 | 520 | 0.15 | 0.999251 | 0 | ParaView |
| [elastic_cantilever--cantilever--linear](results/elastic_cantilever--cantilever--linear/summary.json) | 36,540 | 280 | 0.15 | 0.997705 | 0 | ParaView |
| [elastic_cantilever--cantilever--orthotropic](results/elastic_cantilever--cantilever--orthotropic/summary.json) | 36,540 | 440 | 0.15 | 0.996603 | 0 | ParaView |
| [elastic_cantilever--cantilever--transverse-isotropic](results/elastic_cantilever--cantilever--transverse-isotropic/summary.json) | 36,540 | 460 | 0.15 | 0.99721 | 0 | ParaView |
| [compression--compression--bilinear](results/compression--compression--bilinear/summary.json) | 24,389 | 100 | 5.1063e-05 | 0.985256 | 0 | ParaView |
| [compression--compression--hjc](results/compression--compression--hjc/summary.json) | 24,389 | 100 | 4.57728e-05 | 0.976888 | 0.492128 | ParaView |
| [compression--compression--isotropic-plastic](results/compression--compression--isotropic-plastic/summary.json) | 24,389 | 100 | 5.1063e-05 | 0.985255 | 0 | ParaView |
| [compression--compression--jh2](results/compression--compression--jh2/summary.json) | 24,389 | 100 | 3.78968e-05 | 0.988977 | 1 | ParaView |
| [compression--compression--johnson-cook](results/compression--compression--johnson-cook/summary.json) | 24,389 | 100 | 6.88979e-05 | 0.986423 | 0 | ParaView |
| [compression--compression--rht](results/compression--compression--rht/summary.json) | 24,389 | 100 | 3.06102e-05 | 0.975127 | 0.23272 | ParaView |
| [compression--compression--steinberg-guinan](results/compression--compression--steinberg-guinan/summary.json) | 24,389 | 100 | 7.22002e-05 | 0.987223 | 0 | ParaView |
| [compression--compression--thermal-plastic](results/compression--compression--thermal-plastic/summary.json) | 24,389 | 100 | 3.747e-05 | 0.982193 | 0 | ParaView |
| [benchmarks--elastic-gate--neo-hookean](results/benchmarks--elastic-gate--neo-hookean/summary.json) | 357,912 | 2,800 | 0.35 | 0.640114 | 0 | ParaView |
| [benchmarks--large-bending--neo-hookean](results/benchmarks--large-bending--neo-hookean/summary.json) | 119,944 | 560 | 0.35 | 0.866271 | 0 | Pending |
| [benchmarks--large-twist--neo-hookean](results/benchmarks--large-twist--neo-hookean/summary.json) | 119,944 | 560 | 0.35 | 0.866336 | 0 | Pending |
| [benchmarks--rubber-ball--neo-hookean](results/benchmarks--rubber-ball--neo-hookean/summary.json) | 44,720 | 1,240 | 0.45 | 0.273425 | 0 | Pending |
| [engineering--baffled-tank--neo-hookean](results/engineering--baffled-tank--neo-hookean/summary.json) | 363,352 | 3,200 | 0.7 | 0.737239 | 0 | Pending |
| [engineering--cellular-cushion--neo-hookean](results/engineering--cellular-cushion--neo-hookean/summary.json) | 192,180 | 3,100 | 0.34 | 0.764827 | 0 | ParaView |
| [engineering--perforated-panel--neo-hookean](results/engineering--perforated-panel--neo-hookean/summary.json) | 97,536 | 1,250 | 0.35 | 0.869887 | 0 | ParaView |
| [contact-models--jkr-footprint](results/contact-models--jkr-footprint/summary.json) | 386,019 | 480 | 0.3 | 0.835082 | 0 | ParaView |
| [contact-models--sphere-contact](results/contact-models--sphere-contact/summary.json) | 386,019 | 480 | 0.3 | 0.657805 | 0 | ParaView |
| [production--water-entry--neo-hookean](results/production--water-entry--neo-hookean/summary.json) | 1,327,400 | 2,400 | 0.35 | 0.788163 | 0 | ParaView |
| [refinement--grid32--wave--neo-hookean](results/refinement--grid32--wave--neo-hookean/summary.json) | 12,288 | 140 | 0.134164 | 0.994836 | 0 | Pending |
| [refinement--grid64--wave--linear](results/refinement--grid64--wave--linear/summary.json) | 98,304 | 260 | 0.134164 | 0.994796 | 0 | Pending |
| [refinement--half_dt--wave--neo-hookean](results/refinement--half_dt--wave--neo-hookean/summary.json) | 41,472 | 400 | 0.134164 | 0.994817 | 0 | Pending |

The catalogue contains completed runs only. An active run's `status.json` and checkpoints
are progress records and are not included here. [VALIDATION.md](../VALIDATION.md)
describes the separate constitutive and convergence checks.
