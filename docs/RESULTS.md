# Published runs

Each row links to a selected completed run's configuration and numerical results. Particle trajectories
and ParaView datasets remain in the local output directory. Solver time excludes
compilation, checkpoint I/O and rendering; these CPU runs shared the host with other jobs.

Public-example adaptations are described in [EXAMPLES.md](EXAMPLES.md). Finite states
and a positive determinant establish numerical admissibility, not experimental validation.

| Run | Particles | Steps | Physical time (s) | Minimum J | Maximum damage | Rendering |
|---|---:|---:|---:|---:|---:|---|
| [suite--cantilever--neo-hookean](results/suite--cantilever--neo-hookean/summary.json) | 36,540 | 240 | 0.15 | 0.997722 | 0 | ParaView |
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
| [material-engineering--ceramic-indentation--jh2](results/material-engineering--ceramic-indentation--jh2/summary.json) | 118,580 | 4,080 | 0.0003 | 0.985189 | 1 | ParaView |
| [material-engineering--concrete-bending--hjc](results/material-engineering--concrete-bending--hjc/summary.json) | 121,104 | 2,120 | 0.0003 | 0.976602 | 0.492205 | ParaView |
| [material-engineering--jc-tension--johnson-cook](results/material-engineering--jc-tension--johnson-cook/summary.json) | 146,880 | 10,020 | 0.0003 | 0.992783 | 1 | ParaView |
| [benchmarks--ball-beam--neo-hookean](results/benchmarks--ball-beam--neo-hookean/summary.json) | 88,583 | 1,240 | 0.45 | 0.449918 | 0 | ParaView |
| [benchmarks--elastic-gate--neo-hookean](results/benchmarks--elastic-gate--neo-hookean/summary.json) | 357,912 | 2,800 | 0.35 | 0.640114 | 0 | ParaView |
| [benchmarks--floating-block--neo-hookean](results/benchmarks--floating-block--neo-hookean/summary.json) | 384,373 | 3,720 | 0.5 | 0.945173 | 0 | ParaView |
| [benchmarks--large-bending--neo-hookean](results/benchmarks--large-bending--neo-hookean/summary.json) | 119,944 | 560 | 0.35 | 0.866271 | 0 | ParaView |
| [benchmarks--large-twist--neo-hookean](results/benchmarks--large-twist--neo-hookean/summary.json) | 119,944 | 560 | 0.35 | 0.866336 | 0 | ParaView |
| [benchmarks--rubber-ball--neo-hookean](results/benchmarks--rubber-ball--neo-hookean/summary.json) | 44,720 | 1,240 | 0.45 | 0.273425 | 0 | ParaView |
| [engineering--baffled-tank--neo-hookean](results/engineering--baffled-tank--neo-hookean/summary.json) | 363,352 | 3,200 | 0.7 | 0.737239 | 0 | ParaView |
| [engineering--cellular-cushion--neo-hookean](results/engineering--cellular-cushion--neo-hookean/summary.json) | 192,180 | 3,100 | 0.34 | 0.764827 | 0 | ParaView |
| [engineering--perforated-panel--neo-hookean](results/engineering--perforated-panel--neo-hookean/summary.json) | 97,536 | 1,250 | 0.35 | 0.869887 | 0 | ParaView |
| [engineering--unbaffled-tank--neo-hookean](results/engineering--unbaffled-tank--neo-hookean/summary.json) | 342,144 | 3,200 | 0.7 | 0.88116 | 0 | ParaView |
| [contact-models--jkr-footprint](results/contact-models--jkr-footprint/summary.json) | 386,019 | 480 | 0.3 | 0.835082 | 0 | ParaView |
| [contact-models--sphere-contact](results/contact-models--sphere-contact/summary.json) | 386,019 | 480 | 0.3 | 0.657805 | 0 | ParaView |
| [production--water-entry--neo-hookean](results/production--water-entry--neo-hookean/summary.json) | 1,327,400 | 2,400 | 0.35 | 0.788163 | 0 | ParaView |
| [refinement--grid32--wave--neo-hookean](results/refinement--grid32--wave--neo-hookean/summary.json) | 12,288 | 140 | 0.134164 | 0.994836 | 0 | ParaView |
| [refinement--grid64--wave--linear](results/refinement--grid64--wave--linear/summary.json) | 98,304 | 260 | 0.134164 | 0.994796 | 0 | ParaView |
| [refinement--half_dt--wave--neo-hookean](results/refinement--half_dt--wave--neo-hookean/summary.json) | 41,472 | 400 | 0.134164 | 0.994817 | 0 | ParaView |
| [visible-motion--cantilever--neo-hookean](results/visible-motion--cantilever--neo-hookean/summary.json) | 119,944 | 2,650 | 1.4 | 0.775989 | 0 | ParaView |
| [visible-motion--shear--neo-hookean](results/visible-motion--shear--neo-hookean/summary.json) | 242,208 | 800 | 0.4 | 0.60651 | 0 | ParaView |
| [visible-motion--surface-load--neo-hookean](results/visible-motion--surface-load--neo-hookean/summary.json) | 113,535 | 2,300 | 1.2 | 0.68971 | 0 | ParaView |
| [visible-motion--torsion--neo-hookean](results/visible-motion--torsion--neo-hookean/summary.json) | 119,944 | 2,300 | 1.2 | 0.775237 | 0 | ParaView |
| [visible-motion--wave-grid32--neo-hookean](results/visible-motion--wave-grid32--neo-hookean/summary.json) | 34,560 | 1,250 | 0.969948 | 0.992756 | 0 | ParaView |
| [visible-motion--wave-grid48--neo-hookean](results/visible-motion--wave-grid48--neo-hookean/summary.json) | 112,752 | 1,850 | 0.969948 | 0.989922 | 0 | ParaView |
| [visible-motion--wave-grid64--neo-hookean](results/visible-motion--wave-grid64--neo-hookean/summary.json) | 269,568 | 2,450 | 0.969948 | 0.989943 | 0 | ParaView |
| [visible-motion--wave-half-dt--neo-hookean](results/visible-motion--wave-half-dt--neo-hookean/summary.json) | 112,752 | 3,700 | 0.969948 | 0.9929 | 0 | ParaView |
| [adhesion--adhesive-pull-off](results/adhesion--adhesive-pull-off/summary.json) | 367,744 | 1,200 | 0.85 | 0.784401 | 0 | ParaView |
| [adhesion--lateral-release](results/adhesion--lateral-release/summary.json) | 367,744 | 1,200 | 0.85 | 0.768535 | 0 | ParaView |
| [adhesion--nonadhesive-pull-off](results/adhesion--nonadhesive-pull-off/summary.json) | 367,744 | 1,200 | 0.85 | 0.79432 | 0 | ParaView |
| [adhesion--three-sphere-array](results/adhesion--three-sphere-array/summary.json) | 367,744 | 1,200 | 0.85 | 0.779705 | 0 | ParaView |
| [woven-preview--plain--transverse-bending · coarse preview](results/woven-preview--plain--transverse-bending/summary.json) | 5,400 | 2,100 | 1.89737e-05 | 0.95352 | 0 | ParaView |
| [woven-preview--plain--warp-tension · coarse preview](results/woven-preview--plain--warp-tension/summary.json) | 5,400 | 2,100 | 1.89737e-05 | 0.996464 | 0 | ParaView |
| [woven-preview--plain--weft-tension · coarse preview](results/woven-preview--plain--weft-tension/summary.json) | 5,400 | 2,100 | 1.89737e-05 | 0.996464 | 0 | ParaView |
| [woven-preview--triaxial--in-plane-shear · coarse preview](results/woven-preview--triaxial--in-plane-shear/summary.json) | 5,400 | 2,100 | 1.89737e-05 | 0.963833 | 0 | ParaView |
| [woven-preview--twill-2x2--transverse-compression · coarse preview](results/woven-preview--twill-2x2--transverse-compression/summary.json) | 5,400 | 2,100 | 1.89737e-05 | 0.967532 | 0 | ParaView |
| [woven--plain--transverse-bending](results/woven--plain--transverse-bending/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.912105 | 0 | ParaView |
| [woven--plain--transverse-compression](results/woven--plain--transverse-compression/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.962991 | 0 | ParaView |
| [woven--triaxial--in-plane-shear](results/woven--triaxial--in-plane-shear/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.951967 | 0 | ParaView |
| [woven--plain--in-plane-shear](results/woven--plain--in-plane-shear/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.952719 | 0 | ParaView |
| [woven--plain--warp-tension](results/woven--plain--warp-tension/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.991945 | 0 | ParaView |
| [woven--triaxial--transverse-bending](results/woven--triaxial--transverse-bending/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.911839 | 0 | ParaView |
| [material-engineering--concrete-bending-large--hjc](results/material-engineering--concrete-bending-large--hjc/summary.json) | 121,104 | 10,500 | 0.0015 | 0.965609 | 1 | ParaView |
| [woven--triaxial--transverse-compression](results/woven--triaxial--transverse-compression/summary.json) | 345,600 | 4,140 | 1.89737e-05 | 0.948869 | 0 | ParaView |

The catalogue contains completed runs only. An active run's `status.json` and checkpoints
are progress records and are not included here. [VALIDATION.md](../VALIDATION.md)
describes the separate constitutive and convergence checks.

Rows marked **coarse preview** retain their unsuccessful production-resolution review.
A separate eligibility record permits visualization with explicit resolution and tracking
limitations; it does not convert that review into a production pass.
See [WOVEN_PREVIEWS.md](WOVEN_PREVIEWS.md).

The [transverse shear-wave comparison](results/visible-wave-comparison.json)
compares the new finite-motion runs with one another. The original axial-wave
analytical checks remain separate in [VALIDATION.md](../VALIDATION.md).

The [matched adhesion comparison](results/adhesion-comparison.json) and
[response table](results/adhesion-comparison.csv) use the same sphere path
and dynamic body with zero and positive work of adhesion.
