# Adhesion with a withdrawal hold

`examples.adhesion_contrast` defines a second matched normal pull-off pair.
The original four cases in `examples.adhesive_suite` retain their parameters,
paths and output directories. Their completed normal pair reached maximum
surface uplifts of 10.10 and 15.83 mm; those separate maxima are recorded in
the [original comparison](results/adhesion-comparison.json).

The new pair uses a softer layer and stronger adhesion, then pauses during
withdrawal. Its numerical and visual acceptance must be established from new
calculations. The design targets below are not measured results.
Earlier coarse trials failed the overlap and final-separation checks; they
are not accepted comparison results. This revision keeps the same comparison
hold and acceptance thresholds.

## Common specimen and motion

Both roles share the original 0.72 × 0.23 × 0.72 m layer, fixed base, 0.16 m
sphere radius and zero gravity. The neo-Hookean material uses Young's modulus
8 kPa, Poisson's ratio 0.25 and density 1000 kg/m³. The interaction range stays
at 25 mm. This revised preset uses a penalty stiffness of 4.8 × 10⁶ Pa/m;
the original suite retains 1.2 × 10⁶ Pa/m. The local penalty time-step limit
is unchanged, so the higher stiffness reduces the allowed time step.

Only work of adhesion differs: zero for `nonadhesive-pull-off`, 72 J/m² for
`adhesive-pull-off`. The latter gives a finite-range attractive traction of
2880 Pa. The [existing interface assumptions](ADHESION_CASES.md#shared-model)
still apply: reference-area surface quadrature, reversible adhesion, normal
forces and a finite dynamic layer. These parameters are illustrative.

The sphere centre stays at x = z = 0.5 m. The following heights refer to its
lowest point relative to the initial sampled material surface. Moving segments
use the same half-cosine interpolation in both runs.

| Time (s) | Sphere motion | Lowest-point height (mm) |
|---|---|---:|
| 0–0.08 | Initial hold | 45 |
| 0.08–0.28 | Indent | 45 to −35 |
| 0.28–0.42 | Hold at indentation | −35 |
| 0.42–0.72 | Partial withdrawal | −35 to 55 |
| 0.72–0.94 | Withdrawal hold | 55 |
| 0.94–1.12 | Final withdrawal | 55 to 295 |
| 1.12–1.30 | Final hold | 295 |

The final segment has a peak prescribed speed of 2.09440 m/s and a maximum
one-sided acceleration magnitude of 36.5541 m/s². At the default resolution,
the highest sphere point is 0.991618 m, within the 1 m domain. These kinematic
checks do not establish that the deforming layer has detached.

At the withdrawal hold, an initially flat surface is outside the interaction
range. A surface point underneath the sphere must rise at least 30 mm to
enter that range. Any remaining adhesive load must therefore result from the
computed surface deformation. No prescribed surface motion or displacement
scale is applied.

## Comparison checks

Within the 0.72–0.94 s hold, require a contiguous set of saved times spanning
at least 0.10 s for which all of the following hold:

- The nonadhesive run has no interacting points and negligible sphere reaction.
- The adhesive run retains adhesive points and a tensile vertical reaction.
- The adhesive maximum surface y displacement exceeds the nonadhesive value
  at the same time by at least 25 mm.

The final state must still satisfy complete separation, and each run must
pass the independent force, moment, overlap, mass and determinant checks.
These thresholds are fixed before the trial runs. A failed comparison remains
a failed comparison; it is not evidence for a clearer physical contrast.

The pair must use the same displacement colour limits, camera, sphere opacity
and physical coordinates. A common contact close-up may crop unrelated parts
of the layer, with the crop recorded in the rendering metadata. Inspect both
movies at the same physical times and display them next to one another.

## Run

Run the two cases sequentially in the project environment:

```powershell
foreach ($case in 'nonadhesive-pull-off', 'adhesive-pull-off') {
    python -m examples.adhesion_contrast --case $case --grid 48 --ppc 3 --frames 50 --no-render --resume
}
```

Outputs go to `outputs/adhesion-visible/<case>`. Default resolution retains
367,744 material points and 51 saved states. Both roles have identical time
steps, limited by the bulk wave speed and `k A0 dt²/m ≤ 0.09`.

A preliminary `--grid 32 --ppc 2` run is only a smoke test: its interaction
range does not span three surface spacings. Use the formal resolution for the
resolved-range checks. Neither constructing the case nor passing its unit
tests establishes that the target contrast has been achieved.
