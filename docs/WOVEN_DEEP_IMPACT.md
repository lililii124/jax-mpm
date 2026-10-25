# Deep woven plate impact

This elastic demonstration uses 819,200 particles in a 160 mm square, 2 mm thick woven plate, four fully clamped 5 mm edge bands and no backing. A 100 mm diameter, 50 g rigid hollow sphere starts at 40 m/s (40 J). The 200 µs protocol saves 61 actual states. Parameters are illustrative; this is not a calibrated impact, fracture or eventual rebound prediction.

Install the project and run a short prefix in a separate output directory:

```sh
pip install -e .
python -m examples.woven_deep_impact --output outputs/deep-probe --probe-steps 20 --persistence controller
```

For the full plan, pause at frame 2 and resume the same plan to frame 60:

```sh
python -m examples.woven_deep_impact --output outputs/deep-impact --stop-after-frame 2 --persistence controller
python -m examples.woven_deep_impact --output outputs/deep-impact --resume --stop-after-frame 60 --persistence controller
```

The probe checks the current checkout's large-model entry and 20 actual steps; it does not revalidate a complete impact. A probe and a full-plan run have different immutable signatures and require separate output directories. Each saved frame writes an atomic checkpoint; preserve the output directory and use `--resume` after a planned pause. Local snapshot mode is also available; controller mode avoids redundant segment copies.

The [published 61-frame result](results/completed--woven-plate-impact/summary.json) remains bound to frozen numerical source `c5eccddd9d0857ecc3e0af6e3bfa356409817986` and continuation source `118c3400e1c9ba69f26a71d11554b62e69b342a3`. Its saved center indentation reaches 6.468 mm; the maximum recorded energy error is 0.877% against the original 10% budget. The [overview GIF](images/home/completed--woven-plate-impact.gif) and [detail GIF](images/woven-plate-impact-detail.gif) use all 61 authentic saved states and exact integrated sphere centers, with no displacement scaling or synthesized history.

For that frozen version, create a separate checkout and use the same full-plan commands above:

```sh
git worktree add ../jax-mpm-deep-frozen 118c3400e1c9ba69f26a71d11554b62e69b342a3
cd ../jax-mpm-deep-frozen
pip install -e .
```

The current library retains the later MAT002 implementation. The new current-checkout prefix and the frozen full result are separate pieces of evidence; the current checkout has not rerun all 14,460 steps of the published result.
