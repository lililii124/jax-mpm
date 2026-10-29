# AT2 sharp-crack tension

An explicit geometric precrack is loaded in opening tension. The physical
damage field starts at zero everywhere; no crack-face or tip damage is fixed
to one. Damage develops through the irreversible current-energy AT2 solve.

This example uses fixed-reference Galerkin material-point quadrature with
linear grid basis functions and split displacement support at the crack.
It is a quasistatic small-strain formulation, separate from the explicit
APIC solver used by the dynamic examples.

The displayed run has 25 accepted load states. GIF encoding combines identical
images into 23 frames while preserving their duration. The crack traverses
approximately 0.621 m of the 0.625 m ligament; the final work/energy discrepancy
is 1.41%. These are single-grid results. Near-tip asymmetry remains in the
computed damage field; mesh convergence and agreement with an independent FE
solution have not been established.

## Reproduce

```bash
python -m pip install -e ".[fracture]"
python -m examples.at2_sharp_crack_benchmark --n 48 --ny 49 --initial-damage-mode geometry-only --competitor-peak .9 --ell .04 --increments 24 --energy-competitor --adaptive-energy --max-alternations 400 --damage-relaxation 1.6 --displacement .00006 --output outputs/at2-sharp-crack
pvpython --force-offscreen-rendering tools/render_at2_sharp_crack_paraview.py --input outputs/at2-sharp-crack --output outputs/at2-sharp-crack/render --frames 61 --size 400
python tools/gif_review.py outputs/at2-sharp-crack/render/at2-sharp-crack.gif --open
```

The competitor amplitude is an optimization starting guess, not prescribed
physical damage. Every candidate shares the preceding accepted irreversible
lower bound. ParaView renders the actual saved displacement and damage,
including cut-cell interpolation constraints, with no mirror averaging.

[Run summary](results/at2--sharp-crack-zero-damage/summary.json) ·
[Load history](results/at2--sharp-crack-zero-damage/history.csv) ·
[Zero-state receipt](results/at2--sharp-crack-zero-damage/zero-initial-state-receipt.json)
