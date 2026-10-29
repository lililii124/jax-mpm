# AT2 three-point bending

A 1 m × 0.25 m beam has a 0.05 m bottom-centre geometric notch. Two bottom
rollers constrain vertical motion at x = 0.100775 and 0.899225 m; the left
roller also constrains horizontal motion. A 0.04 m-wide central loading region
prescribes downward displacement while allowing horizontal motion. The other
surfaces, including the notch faces, are traction-free.

The material uses E = 1 MPa, Poisson ratio 0.2, fracture energy
0.001 J/m² and AT2 length scale 0.015 m. The reference grid has 129 × 40 cells.
Damage starts at zero everywhere. The solver minimizes the current mechanical
and fracture energy subject to damage irreversibility, with a spectral tensile
energy split and independent force/KKT checks. See the
[AT2 formulation](AT2_SHARP_CRACK.md).

The animation contains 27 accepted states through a loading displacement of
0.113046875 mm. Reaction drops from 4.5694 to 0.7866 N/m; the final external-work
discrepancy is 0.5081%. The maximum accepted principal strain is 1.999689%.
Larger load attempts violate the 2% small-strain limit, so the original
0.12 mm loading schedule remains unfinished. Mesh convergence has not been
established for this example.

The 400 × 400 ParaView animation shows the whole beam above a close-up of the
notch and crack. Both views use the same saved state, damage range 0–1 and
unscaled displacement.

```bash
python -m examples.at2_three_point_bending --output results/at2-bending
```

[History and numerical checks](results/at2--three-point-bending/summary.json)
and the [rendering receipt](results/at2--three-point-bending/render.json)
record the accepted trajectory and its scope.
