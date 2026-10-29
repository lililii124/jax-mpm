# Experimental fixed-mass transient optimization

`examples.blast_sandwich_reduced` optimizes a linear transient projection of the
[elastic sandwich model](BLAST_SANDWICH_IMPLEMENTATION.md). Its gains are
candidate results. Independent selected-design MPM has exposed model bias;
error correction and complete qualification remain unfinished.

For fixed mass, minimize the worst back-face space-time displacement over the
three prescribed pressure footprints. Constraints are skin von Mises stress
≤ 200 MPa, back-face displacement ≤ 5 mm, and exact mass 1.8468 kg. The filtered
24 × 24 × 3 controls redistribute a mean core fraction of 0.18. A scalar
`lax.custom_root` mass map enforces equality; feasibility rejects relative mass
error above 10⁻¹⁰. The separate `min-mass` action is not this fixed-budget task.

The fixed nodal basis V vanishes on every clamped degree of freedom. Six analytic
families represent bending, independent core shear, compression and membrane
motion, giving 6n² vectors at order n. Mass normalization uses the uniform
reference design. For each control field the solver assembles

`Mr(φ) = Vᵀ diag(mnodal(φ)) V`, `Kr(φ) = Vᵀ Kfull(φ,F=I) V`.

Chunked assembly uses the same quadrature and elastic tangent as full MPM,
without a particle-by-mode-by-mode cache. Reference-face pressure forces are
projected directly. Average-acceleration Newmark advances the transient;
kinetic plus elastic energy is compared with trapezoidal pressure work.
This linear energy identity does not measure omitted modes or nonlinear error.

Every back-face quadrature point, skin quadrature point and saved time enters
response recovery. A stable unnormalized log-sum-exp bounds maxima with at most
0.5% of each prescribed limit as aggregation error. Physical sampled peaks are
saved separately. Fixed-mass SLSQP uses four sequential VJPs: objective plus
three stress rows; displacement reuses the objective derivative. Positive
`--objective-scale` conditions optimizer steps only. Saved objectives and
constraints remain unscaled. The best strict-feasible candidate is retained;
finite iterations and restart do not imply convergence. Restart restores
controls/history and resets SLSQP's internal BFGS matrix.

```bash
pip install -e .[design]
python -m examples.blast_sandwich_reduced forward --order 3 --dt 1e-6 --output outputs/sandwich-rom
python -m examples.blast_sandwich_reduced check --order 3 --dt 1e-6 --output outputs/sandwich-rom-check
python -m examples.blast_sandwich_reduced optimize --stage fixed-mass --order 3 --dt 1e-6 --iterations 8 --objective-scale 1000 --output outputs/sandwich-rom-opt
python -m examples.blast_sandwich_reduced optimize --stage fixed-mass --order 3 --dt 1e-6 --iterations 8 --objective-scale 1000 --resume --output outputs/sandwich-rom-opt
```

These commands rebuild the analytic basis and solve all three loads to 2 ms.
Order 3 is a runnable experimental space, not an accuracy-qualified surrogate.
The measured enriched rank-510 optimizer used four CPUs, a 16 GiB container,
14 GiB USS guard and about 11.55 GB peak USS. Full-field stress recovery is
rematerialized in time chunks. Larger grids/bases need a fresh memory plan.

`--enrichment-snapshots` accepts an NPZ with `displacement` shaped
`(snapshots,nodes,3)`, Boolean `training`, and matching reference `nodes`.
Only marked training snapshots enter residual POD. For the recorded rank-510
study, baseline steps 1,000–24,000 trained 24 added vectors; selected steps
1,000–8,000 supplied eight more for rank 518. The public preparation entry point
uses the exact Gauss-to-node inversion from the face exporter:

```bash
python -m tools.prepare_sandwich_enrichment examples/sandwich_enrichment.json outputs/sandwich-training-510.npz --source initial-eccentric
python -m examples.blast_sandwich_reduced forward --order 9 --dt 1e-6 --enrichment-snapshots outputs/sandwich-training-510.npz --output outputs/sandwich-enriched-initial
python -m tools.prepare_sandwich_enrichment examples/sandwich_enrichment.json outputs/sandwich-training-518.npz
python -m examples.blast_sandwich_reduced forward --order 9 --dt 1e-6 --enrichment-snapshots outputs/sandwich-training-518.npz --design-checkpoint evidence/blast-sandwich-reduced/enriched-v12-best-strict-feasible-controls.npz --output outputs/sandwich-enriched-selected
```

First generate the full-MPM frame directories with the commands in the model
documentation. The JSON recipe names the exact control and physical-source
hashes, baseline/selected training steps and development intervals. The tool
rejects overlap, material changes and incorrect controls, and writes masks and
per-frame/output/training-field hashes. Only `training=True` fields enter POD.
Fresh validation extraction additionally requires a `frozen_model_receipt`
path in the recipe and the predeclared future interval. No future selected
frames are requested by the supplied recipe.

The [preparation check](../evidence/blast-sandwich-reduced/enrichment-preparation-check.json)
recovered all 54 allocated actual frames: the baseline-only pack has 24 training
and 12 development frames; the combined pack has 32 training and 22 development
frames. Maximum Gauss resampling error is 5.21 × 10⁻¹⁸ m; no reserved future
selected frame was read. Five focused recovery/allocation tests pass. Large raw
trajectories are regenerated rather than bundled. The complete public pipeline
has not yet been replayed to re-establish the recorded enriched curve and
basis hashes; POD rotations and numerical runtime can change basis bytes.
These commands close the dataset-generation dependency, not the failed model
accuracy gates or the remaining full pipeline verification.

The canonical selected control field has SHA256
`3bec00a53c79b9e25bddb43be5696e4f428c70d62d28d5e12979d55832d566f3`
over its float64 values. Its [transfer receipt](../evidence/blast-sandwich-reduced/enriched-v12-best-transfer.json)
records the immutable physical source and virgin-state verification binding.
This field is fixed for independent MPM even while further ROM candidates are
explored. It is not a certified optimum.

The [current comparison](../evidence/blast-sandwich-reduced/current-model-comparison.json)
retains the decisive results and unchanged gates. Rank-510 time-step halving
changes all baseline/selected peaks by at most 0.00610%, but a frozen future
baseline interval still differs by 5.04935%, failing the 5% gate. At matched
full-MPM time 0.239636 ms, cumulative initial and selected peaks are 0.846046 mm
and 0.814771 mm: only 3.6966% same-prefix reduction, compared with 21.95% in
the ROM objective. This establishes neither full-horizon gain nor all-case
constraint satisfaction.

Training-only corrections improve the selected early response but still fail
the energy-projection gate: rank 518 omits up to 9.3837% of linear strain energy;
rank 526 with known tangent directions omits 9.1128%. Neither is qualified.
The [data allocation](../evidence/blast-sandwich-reduced/selected-correction-development-policy.json)
marks steps 9,000–18,000 as method-development comparisons, excluded from fitting.
A corrected model must be frozen before fresh selected validation from step
19,000. Further optimization requires model correction and independent complete
MPM responses; small proxy objectives or a stalled round sequence do not prove
global optimality. Spatial refinement and plasticity/crushing/interface physics
remain additional gates.
