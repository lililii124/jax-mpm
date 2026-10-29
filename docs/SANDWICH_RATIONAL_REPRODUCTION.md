# Reproducing the frozen rational sandwich correction

The public tools reproduce the physics-only construction used by the rank130
correction. They do not fit full-MPM displacement snapshots. The selected
material vertex is the round1 control field with SHA256
`3bec00a53c79b9e25bddb43be5696e4f428c70d62d28d5e12979d55832d566f3`.
Its 1728 float64 controls are included in the 34KB JSON fixture. This input is
distinct from later optimizer candidates.

The recipe is pinned in [sandwich_rational.json](../examples/sandwich_rational.json).
It checks the geometry/support/basis source hashes before construction. Python,
NumPy, SciPy and JAX are required; use the repository environment. Reference
construction jobs used four CPU threads, an 8GB container limit and a 6GB USS
guard. No full-MPM rerun is needed to construct the reduced basis.

From the repository root, use fresh output directories:

```console
python -m tools.build_sandwich_rational examples/sandwich_rational.json material
python -m tools.build_sandwich_rational examples/sandwich_rational.json tangent
python -m tools.build_sandwich_rational examples/sandwich_rational.json selected
python -m tools.build_sandwich_rational examples/sandwich_rational.json canonicalize
python -m tools.build_sandwich_rational examples/sandwich_rational.json union
```

`material` evaluates the fixed-mass material map at the uniform and selected
controls. `tangent` assembles the selected full sparse tangent in 2048-point
chunks. `selected` generates the original selected-vertex rational space.
`canonicalize` changes its coordinates to the initial uniform nodal-mass
metric. `union` retains that 78-vector seed and adds directions at the initial
uniform material vertex, producing the recorded rank130 space. It projects
both initial and selected operators into the same final basis.

At each vertex, shifts are 0, 2000, 8000, 32000, 128000 and 512000Hz, in that
order. Each shift uses three inverse powers and, within each power, uniform,
central and eccentric loads. Three mass-inverse load directions follow.
GMRES uses relative residual below 1e-8, restart50, maxiter40, and the original
ILU/coarse-space preconditioner. Two residual projections precede the same
NumPy SVD with a relative 1e-8 cutoff. The port preserves direction ordering,
SVD ordering/signs and the original final Cholesky operation; it does not sort,
sign-normalize or substitute modes. Linear trajectories cover 2ms with the
original pressure sampling and Newmark ordering. The union compares nominal
0.25us and 0.125us time steps.

Each stage records source/input hashes. The construction emits operators,
responses and a review receipt. Large arrays remain under `outputs/` and are
not required in a source publication. The recorded numeric union-basis hash
is `d46b264d211e455ed34724800bd6db9f44e773f2cfd8ba71f3dfd03617eab81b`.
SVD/iterative-solver floating-point behavior can depend on the numerical
runtime. Check the emitted numeric hash; a differing basis must receive its
own frozen receipt and independent validation rather than borrowing an older
receipt. The source port was checked against original numerical AST blocks
and cached forward responses; the entire cold construction was not rerun for
the source port.

Before reading a new validation interval, run:

```console
python -m tools.build_sandwich_rational examples/sandwich_rational.json freeze
```

This writes the selected operators with the union basis to `frozen-basis.npz`
and seals a version2 receipt declaring selected steps31000–42000. It binds
the actual material, controls, tangent, seed, physical sources and complete
frozen basis/operator artifact. Every dependent construction stage verifies
its upstream artifact receipts before computation. Do not replace that
receipt after inspecting validation results. Prepare the selected trajectory
with the explicit extraction specification and frozen receipt; retain its
adjacent JSON provenance receipt:

```console
python -m tools.prepare_sandwich_enrichment examples/sandwich_rational_validation.json outputs/rational-validation/selected.npz
```

The specification is
[sandwich_rational_validation.json](../examples/sandwich_rational_validation.json).
Its training
partition is an extraction-schema requirement and is not used to construct or
change this rational basis. Extract the actual reference history from its
status-bound trajectory directory, stopping at the declared step42000:

```console
python -m tools.assess_sandwich_rational --prepare-history outputs/fullmpm/selected-eccentric --freeze-receipt outputs/rational-union/model-freeze.json --output outputs/rational-validation/history-through-42000.csv
```

The adjacent history receipt binds the prefix hash, source status, pressure
case, controls, physical source signature and frozen model. No rows after
the declared boundary are read. A fresh source directory must already contain
the saved trajectory; this command does not run MPM.
Then adapt input paths in
[sandwich_rational_assessment.json](../examples/sandwich_rational_assessment.json):

```console
python -m tools.assess_sandwich_rational examples/sandwich_rational_assessment.json --output outputs/rational-validation/review.json
```

The assessor checks source pins, material and frozen artifact/numeric hashes,
snapshot output checksum, material/control/load binding, and the history
checksum before using those inputs. Operator overrides are rejected. It also
rejects nonfinite arrays, invalid shapes and time grids, nonpositive norms,
and missing or nonconsecutive history rows. Version1 historical receipts
remain audit evidence and cannot be used as version2 input-bound assessments.
Fresh response and energy intervals must equal the receipt allocation. It
uses unchanged 5% bounds for peak response error, curve relative L2, maximum
curve error divided by the reference peak, and omitted linear strain energy.
The time-sensitivity gate is 1%; linear energy balance must be below 1e-8.
Failure exits nonzero and preserves the report. For previously inspected data,
set `role` to `development` and declare its original windows explicitly.
The original selected windows were9000–18000 and19000–30000. The old baseline
response failure window remains36001–49000. Its separately available energy
snapshots were25000–36000; express that distinction with
`"energy_windows": [[25000, 36000]]`, never describe it as energy validation of
the later response interval. For baseline assessment, make a separate copy
of the construction configuration with `"freeze_vertex": "initial"` and
fresh `frozen_basis`/`freeze_receipt` paths, then run `freeze`. This variant
seals initial operators with the same spatial union; it has no selected fresh
allocation. Use its receipt and the initial material for development checks.
Its history extraction needs the explicit development stop
`--history-end-step 49000`; this option cannot override a selected fresh
allocation.
Selected assessment uses the separately frozen selected operator variant.

For an inexpensive source-port check against saved selected responses:

```console
python -m tools.assess_sandwich_rational --check-frozen outputs/rational-union/frozen-basis.npz outputs/rational-union/selected-fine-response.npz --output outputs/rational-union/public-replay.json
```

This recomputes 96 initial forward steps and reconstructs bottom-face maxima
at 67 saved times across the full horizon. The recorded check matched state
trajectories to relative L2 `1.80e-15`, face maxima exactly, and the saved
full-horizon energy balance to `1.00e-11`. See
[the source-port receipt](../evidence/blast-sandwich-reduced/union130-public-source-port.json)
and [validation proof](../evidence/blast-sandwich-reduced/union130-validation.json).

The frozen selected prefix passed; complete2ms, all-three-load full-MPM
qualification is incomplete. New optimization controls need matched
independent full-MPM confirmation. Neither a successful prefix check nor a
bounded optimizer stall proves a global optimum.
