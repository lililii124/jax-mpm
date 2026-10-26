# AT2 tension with constrained damage evolution

The three-dimensional tension example uses reference particle quadrature, persistent grid displacement and lumped nodal inertia. A normal displacement is prescribed at the upper edge; the lower edge and three tangential degrees of freedom remove rigid-body motion. The finite precrack is included in the initial fracture energy.

```bash
python -m examples.phase_field_tension_persistent_grid --gc 0.0000125 --speed 0.005 --displacement 0.00045 --steps 600 --damage-driving current-energy --output outputs/at2-current-energy
```

`--damage-driving history` retains the maximum tensile energy as the driving field. `current-energy` minimizes the damage-dependent energy at the current displacement, with the existing bounds `previous_damage <= damage <= 1`. The maximum-history field is still recorded. Material parameters, spectral split, irreversibility, solver tolerance and small-strain checks are identical. The selected driving rule is part of the restart binding.

At fixed displacement the latter solves a constrained convex damage subproblem. This distinction between a variational inequality and history-field enforcement is discussed by [Gerasimov and De Lorenzis](https://arxiv.org/abs/1811.05334). The displacement and damage updates remain staggered; the complete dynamic scheme is not thereby certified energy-conserving.

A frozen-state comparison starting at step 5800 isolates the driving rule over 24 actual steps. The history branch reproduces the original saved per-step audit within `1e-18 J`.

| Driving field | Damage-update potential change | Total energy minus external work |
| --- | ---: | ---: |
| Historical maximum | `+5.87359e-11 J` | `+6.00491e-11 J` |
| Current tensile energy | `-1.26311e-12 J` | `+3.51127e-14 J` |

Both branches pass the original KKT, irreversibility, boundary and strain checks. Every history-driven update in this window increases the current potential; none of the current-energy updates does. These are window increments, not a correction to the earlier history or a full-run convergence result.

The earlier 6000-step history-field result remains separate. Its energy residual is 1.234% of accumulated absolute boundary work, and its damage profile is diffuse over a finite length scale. A complete current-energy loading history and spatial/time refinement remain to be established before restoring the fracture animation to the gallery.

The public source also completed 200 steps from the initial state. All 23 exported numeric fields match the frozen candidate bit for bit; the four nonlinear-window state arrays also match exactly. Source hashes, checkpoint hashes and the actual run receipt are retained in [the evidence directory](../evidence/at2-current-energy/).

The candidate with the frozen baseline dependencies completed a separate 600-step initial-state run: energy residual divided by absolute boundary work is `1.47827e-5`, with no positive damage-update potential increment. This remains an early loading prefix; the source change and its checkpoint are recorded separately.
