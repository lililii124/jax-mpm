# AT2 crack–hole interaction

A unit square contains a 0.375 m geometric edge crack and a circular hole of
radius 0.10 m centred at (0.65, 0.58) m. The hole is an actual void with a
conforming polygon boundary. Material points, integration cells and unused
degrees of freedom inside it are removed.

The bottom edge has zero vertical displacement, the top edge is pulled upward,
and one horizontal degree of freedom removes rigid translation. The other
surfaces, including the hole and notch, are traction-free. The damage field
has a natural zero-flux condition at the hole; physical initial damage is zero
everywhere and no damage coefficients are prescribed.

The small-strain AT2 model uses E = 1 MPa, Poisson ratio 0.2, fracture energy
0.001 J/m² and length scale 0.04 m. Fixed reference material points integrate
linear triangular background functions. JAX provides the energy derivatives;
the quasistatic solver checks force balance, damage irreversibility and KKT
conditions. Warm and extended initial guesses compete by converged total
energy; neither guess imposes a damage trace or changes the irreversible lower
bound.

The complete 0–0.06 mm loading path contains 25 accepted states. The final
energy/work discrepancy is 0.350%. Two damaged solid components reach the
free hole boundary from opposite sides. No damage values or artificial mesh
connections bridge the void. This is a single-mesh demonstration; mesh and
load-increment convergence have not been established.

```bash
python -m examples.at2_hole_benchmark --output outputs/at2-hole
pvpython --force-offscreen-rendering tools/render_at2_hole_paraview.py \
  --input outputs/at2-hole --output outputs/at2-hole-render
```

The [numerical history](results/at2--offset-hole/history.csv),
[geometry and kernel checks](results/at2--offset-hole/kernel-verification.json)
and [rendering receipt](results/at2--offset-hole/render.json) accompany the
400 × 400 ParaView animation. Displacements are unscaled and damage uses 0–1.
