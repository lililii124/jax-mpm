# RHT return mapping

Large particle arrays now solve the plastic return only at yielding points, in
batches of 256. The zero-multiplier update still runs over the entire array so
that pressure, compaction and other changes at non-yielding points are retained.
Calls with at most 256 points use the original vector path.

Material equations, tolerances, energy iteration and global adaptive substeps
are unchanged. Padded batch entries cannot overwrite real particle histories.
The original return function remains available for comparisons.

Five focused comparisons cover sparse and dense yielding sets, both return
stages, vector time steps, a partial final batch, and compaction without yield.
All returned fields matched the original JIT path exactly in these fixtures.

```bash
python tools/check_rht_active_return.py --output outputs/rht-return-check.json
```

A separate comparison advances the same RHT impact state from step 105 to 106
in independent processes on four CPU cores:

| Path | Compile | Solve |
| --- | ---: | ---: |
| Original | 7.02 s | 104.55 s |
| Batched | 10.12 s | 3.41 s |

Geometry, validity and contact impulse matched exactly. Six target material
fields had floating-point differences within the existing relative field-scale
tolerance of `1e-12`; the largest stress difference was `1.49e-8 Pa`.

Continuation through step 110 also passed that tolerance against the archived
original calculation, including its change in adaptive substeps. That reference
used iteration-count callbacks, so its timing is not a direct benchmark.
These are implementation comparisons, not a complete perforation validation.
Performance depends on the distribution of yielding points.

[Focused checks](results/rht-return-map/focused-checks.json) ·
[Step 106](results/rht-return-map/step106-comparison.json) ·
[Step 110](results/rht-return-map/step110-comparison.json)
