# AT2 notched-beam bending

`examples.phase_field_bending_free_anchor` uses reference-particle quadrature,
quadratic B-splines, persistent nodal displacement/velocity, and lumped-mass
explicit dynamics. The AT2 damage solve uses a spectral tensile-energy split
and irreversible history. Damage is zero for intact material and one for
fully degraded tensile stiffness.

The beam measures 1.6 × 0.4 × 0.16 m. A central notch spans x=0.72–0.88 m and
extends 0.16 m upward. Two lower shoes at x=0.2 and 1.4 m resist downward
motion; an upper shoe at x=0.8 m imposes indentation. Their tangential
directions are free. These are reference-grid unilateral constraints.

Default parameters are E=1,000 Pa, ν=0.2, ρ=1 kg/m³, Gc=0.005 J/m² and
length scale 0.05 m. The grid spacing is 0.025 m. The punch ramps over 12 ms
to 0.12 m/s, with a prescribed total indentation of 35 mm. This is a numerical
demonstration rather than a calibrated material experiment.

```bash
pip install -e ".[forming]"
python -m examples.phase_field_bending_free_anchor --preflight
python -m examples.phase_field_bending_free_anchor --sparse-gradient \
  --steps 1000 --output outputs/at2-1000
python -m examples.phase_field_bending_free_anchor --sparse-gradient \
  --restart outputs/at2-1000/checkpoint_001000.npz \
  --steps 2000 --output outputs/at2-2000
```

`--steps` is the absolute stop step. Continue in segments of at most 1,000
steps, ending on a multiple of 25 or the final step reported by `--preflight`.
Each segment needs a new output directory. Restart checks bind source code,
runtime, parameters, geometry, operator mode and state. The solver stops when
the projected KKT residual exceeds 1e−7 or the strain component limit exceeds
0.1; completing a segment does not establish a converged fracture solution.

`--sparse-gradient` preassembles the fixed gradient stiffness in bounded
chunks; the history-dependent reaction term remains matrix-free. The default
operator stays matrix-free. The sparse option requires fixed reference basis,
quadrature, Gc and length; it does not provide derivatives through assembly
with respect to those quantities.

The same-state operator comparison agreed within 2.6e−15 relative error. At
step 1001, a four-core warm evaluation decreased from 7.13 to 2.61 s with the
same 197 damage iterations. A subsequent 200-step continuation reached step
1600 with all KKT checks passing, but maximum damage was only 0.159. A
localized crack and full-duration convergence remain unverified.

The [public CLI comparison](../evidence/at2-sparse-gradient/public-cli-comparison.json)
advanced the same step-1600 state to step 1625. It retained identical reference
geometry and mass; maximum nodal damage difference was 2.80e−10 and displacement
difference 8.68e−19 m. Both paths satisfied the original KKT limit. This
comparison measures numerical differences and does not assert bitwise identity.
