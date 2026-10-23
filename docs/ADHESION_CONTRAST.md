# Adhesion with a withdrawal hold

`examples.adhesion_contrast` defines a second matched normal pull-off pair.
The original four cases in `examples.adhesive_suite` retain their parameters,
paths and output directories. Their completed normal pair reached maximum
surface uplifts of 10.10 and 15.83 mm; those separate maxima are recorded in
the [original comparison](results/adhesion-comparison.json).

The pair uses a softer layer and stronger adhesion, then pauses during
withdrawal. Profile `v2` keeps the original 55 mm withdrawal hold. Profile
`v3` changes that height to 35 mm; material constants, time intervals, final
withdrawal height and acceptance thresholds are unchanged. The independent
`v4` candidate keeps the `v3` path and doubles the adhesive role's work of
adhesion. Each profile has its own output directory; the defaults remain `v2`.

## Common specimen and motion

Both roles share the original 0.72 × 0.23 × 0.72 m layer, fixed base, 0.16 m
sphere radius and zero gravity. The neo-Hookean material uses Young's modulus
8 kPa, Poisson's ratio 0.25 and density 1000 kg/m³. The interaction range stays
at 25 mm. This revised preset uses a penalty stiffness of 4.8 × 10⁶ Pa/m;
the original suite retains 1.2 × 10⁶ Pa/m. The local penalty time-step limit
is unchanged, so the higher stiffness reduces the allowed time step.

Within each matched pair, only work of adhesion differs between the two roles.
The baseline always uses zero. Profiles `v2` and `v3` use 72 J/m² for the
adhesive role, giving an attractive traction of 2880 Pa. Candidate `v4` uses
144 J/m² and 5760 Pa. Its range, penalty stiffness, bulk material and time-step
controls are unchanged. The [existing interface assumptions](ADHESION_CASES.md#shared-model)
still apply: reference-area surface quadrature, reversible adhesion, normal
forces and a finite dynamic layer. These parameters are illustrative.

The sphere centre stays at x = z = 0.5 m. The following heights refer to its
lowest point relative to the initial sampled material surface. Moving segments
use the same half-cosine interpolation in both runs. This table is profile `v2`;
profiles `v3` and `v4` substitute 35 mm for the three occurrences of 55 mm.

| Time (s) | Sphere motion | Lowest-point height (mm) |
|---|---|---:|
| 0–0.08 | Initial hold | 45 |
| 0.08–0.28 | Indent | 45 to −35 |
| 0.28–0.42 | Hold at indentation | −35 |
| 0.42–0.72 | Partial withdrawal | −35 to 55 |
| 0.72–0.94 | Withdrawal hold | 55 |
| 0.94–1.12 | Final withdrawal | 55 to 295 |
| 1.12–1.30 | Final hold | 295 |

For `v2`, the final segment has a peak prescribed speed of 2.09440 m/s and a maximum
one-sided acceleration magnitude of 36.5541 m/s². At the default resolution,
the highest sphere point is 0.991618 m, within the 1 m domain. These kinematic
checks do not establish that the deforming layer has detached.

At the withdrawal hold, an initially flat surface is outside the interaction
range. A surface point directly underneath the sphere must rise more than
30 mm in `v2`, or 10 mm in `v3`/`v4`, to enter that range. Any remaining adhesive load must therefore result from the
computed surface deformation. No prescribed surface motion or displacement
scale is applied.

## Why the hold height changed

The complete grid48/ppc3 `v2` pair in [run 39965579277](https://github.com/lililii124/jax-mpm/actions/runs/39965579277), commit `4159bd5`, passed both individual numerical reviews but failed the prescribed hold contrast. At all nine saved hold states from 0.728 to 0.936 s, both spheres were force-free with zero interacting points. The maximum simultaneous surface-uplift difference was 9.596 mm, below 25 mm.

Earlier in that same withdrawal, the baseline was already free while the adhesive sphere remained in tension:

| Time (s) | Sphere bottom height (mm) | Adhesive reaction (N) | Simultaneous uplift difference (mm) |
| --- | ---: | ---: | ---: |
| 0.598 | 23.006 | −62.991 | 26.897 |
| 0.624 | 34.112 | −56.562 | 33.970 |
| 0.650 | 43.442 | −47.632 | 41.055 |
| 0.676 | 50.307 | −24.280 | 41.151 |

These qualifying samples span only 0.078 s, not the required 0.10 s, and were recorded during motion rather than a hold. At 54.203 mm the next saved state was already detached. The 35 mm `v3` hold was selected near the earlier state with 460 adhesive surface points and a larger tensile resultant; 43.442 mm had only 364 adhesive points. Stopping at this earlier height is a new loading hypothesis, not evidence that the hold criterion will pass. The unchanged final height of 295 mm increases the last segment's peak speed to 2.26893 m/s. The original `v2` failure remains reproducible with `--profile v2`.

## V3 outcome and V4 candidate

The complete `v3` pair in [run 39983773296](https://github.com/lililii124/jax-mpm/actions/runs/39983773296),
commit `f38bfc8`, passed both individual numerical reviews. The pair comparison
failed only the unchanged requirement for a 25 mm uplift difference over at
least 0.10 s. The baseline was force-free throughout the saved hold states.

| Hold time (s) | Adhesive points | Adhesive reaction (N) | Minimum gap (mm) | Simultaneous uplift difference (mm) |
| --- | ---: | ---: | ---: | ---: |
| 0.728 | 256 | −34.207 | 2.202 | 31.875 |
| 0.754 | 148 | −20.051 | 10.901 | 23.211 |
| 0.780 | 12 | −1.654 | 23.932 | 9.872 |
| 0.806 | 0 | 0 | 42.244 | 0.568 |

Only the first saved hold state met the complete contrast criterion. The
adhesive run's maximum saved penetration was 1.465785 mm against the existing
5.208333 mm limit; its minimum determinant was 0.677599 and recorded mass
drift was zero. No penetration occurred in the saved hold states. These are
results of `v3`, not predictions for stronger adhesion.

`v4` tests whether doubling the effective work of adhesion can retain the
deformed surface at the same 35 mm hold. It changes no waypoint, loading time,
material parameter, interaction range, compressive penalty stiffness or
acceptance threshold. The compressive traction branch is unchanged; the
attractive branch doubles and the potential well becomes deeper. Stronger
adhesion may alter both deformation and dynamic release, so the original
overlap, determinant, force balance and final-separation checks must still
pass. This is an uncalibrated demonstration hypothesis, not a successful
result or an experimental adhesion value. Original `v2` and `v3` records and
profiles are retained.

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
    python -m examples.adhesion_contrast --profile v4 --case $case --grid 48 --ppc 3 --frames 50 --no-render --resume
}
```

Profiles `v3` and `v4` write `outputs/adhesion-visible-v3/<case>` and
`outputs/adhesion-visible-v4/<case>`, respectively. Omitting `--profile`,
or selecting `v2`, retains `outputs/adhesion-visible/<case>` and the 55 mm hold.
An explicit `--output` overrides the profile's directory. Default resolution retains
367,744 material points and 51 saved states. Both roles have identical time
steps, limited by the bulk wave speed and `k A0 dt²/m ≤ 0.09`.

A preliminary `--grid 32 --ppc 2` run is only a smoke test: its interaction
range does not span three surface spacings. Use the formal resolution for the
resolved-range checks. Neither constructing the case nor passing its unit
tests establishes that the target contrast has been achieved.
