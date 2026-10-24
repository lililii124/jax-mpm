# JH-2 ceramic failure example

`examples.ceramic_fragmentation` models a 180 × 25 × 180 mm ceramic plate. Four 8 mm edge strips are fixed; there is no backing. A 75 mm diameter rigid sphere makes one prescribed 12 mm impact and withdrawal cycle in 300 µs. The default grid has 96 cells per axis and two particles per cell direction.

The original JH-2 intact/fractured strength and damage parameters are retained. This example additionally selects an illustrative 0.05 equivalent plastic strain failure threshold and ballistic failed particles. Failed mass remains in the result, but ceases to transfer force through the grid or contact the sphere. The calculation therefore represents erosion debris; it does not resolve cohesive ceramic fragments, fragment contact or mesh-independent crack energy. The threshold and loading are not calibrated to a ceramic experiment.

```powershell
python -m examples.ceramic_fragmentation --output outputs/ceramic-fragmentation/jh2
python tools/review_ceramic_fragmentation.py outputs/ceramic-fragmentation/jh2
python -m mpm_jax.visualization outputs/ceramic-fragmentation/jh2 --field damage --sphere-opacity 0.45 --debris-style highlight --gif
```

The review checks saved failure history against the original particle volumes and reports whether failed material has left the initial plate thickness. Only a completed, reviewed numerical run should be rendered for publication.
