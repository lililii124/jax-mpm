# 3 × 3 shared-cell blast design

This work in progress optimizes a three-dimensional material field shared by
nine cells. All nine cells are explicitly present in the finite plate model:
their particles, displacement and deformation states are independent. The
central pressure footprint acts on the entire top face, centred on the middle
cell. The perimeter is clamped. The cell pattern is not prescribed and the
initial field can be uniform or seeded random.

To leave the entire thickness free for optimization, use `--skin 0
--skin-layers 0`. In that mode there are no fixed face sheets, and every
voxel belongs to the repeated design cell. Set `--core-young` to the Young
modulus of the chosen solid. For binary replay, the projected pressure acts
on the uppermost solid voxel in each occupied column; it does not model gas
flow or pressure on cavity sides.

The initial configuration uses 100 × 100 × 20 mm cells, 2 mm face sheets,
30% core material fraction and 2.43 kg total material. Each grid cell contains
eight integration particles. The total material mass is fixed by a
differentiable volume projection and checked against the particle mass during
the full-domain solve. The pressure multiplier is a 2 MPa sine-squared pulse
lasting 20 microseconds; the Gaussian footprint is normalized by area and has
an 80 mm standard deviation. These are assumed parameters, not a calibrated
explosive source.

The current search uses compressible Neo-Hookean elasticity, with 70 GPa face
sheets and a core modulus proportional to the square of a continuous material
fraction. Core density scales linearly with this fraction. This soft-material
relaxation is an optimization proxy: low-density regions are not physical
voids. No plasticity, damage, collapse contact or experimental validation is
claimed. Binary reconstructions must be checked for mass, connectivity and
full-domain behavior using an explicit physical model.

The objective is a differentiable 16-norm upper bound on the absolute
back-face displacement sampled across space and time. The model uses a
checkpointed JAX time scan. The following run starts from a reproducible random
field and covers the full pulse and subsequent response over 200 microseconds:

```sh
python -m tools.optimize_multicell_blast --cell 8 8 20 --steps 11765 --dt 1.7e-8 --checkpoint-block 128 --blast-radius .08 --init random --seed 7 --iterations 2 --output outputs/multicell-blast-opt
```

The initial design and each accepted design are stored separately; a resume
checks the physical settings, source hashes and numerical runtime. The
two-iteration budget is not a convergence claim. This case is computationally
expensive: the example above uses 110,592 material points, and the reverse
pass must be run with an explicit memory limit. Higher resolution, full-pulse
gradient accuracy, energy and momentum checks, and independent candidate
verification remain necessary before the resulting topology is treated as an
engineering result.

For the free-thickness initial binary structure, the sampled back-face
displacement was still increasing at 200 microseconds. A preliminary 1 ms
binary replay found its largest sampled value, 0.970 mm, near 357 microseconds.
The 200 µs search is therefore a windowed exploration; a separate 1 ms search
is running. Time-step refinement of the 1 ms replay remains in progress. A
finite-window maximum is not a bound on every later response.

An accepted density field can be converted to equal-volume solid voxels and
voids while preserving the material amount to the nearest voxel per cell:

```sh
python -m tools.reconstruct_multicell_blast outputs/multicell-blast-opt/design-00001.npz outputs/multicell-blast-opt/inputs.json --output outputs/multicell-blast-opt/binary-00001.npz
```

The companion JSON records mass and face-connected component counts. The NPZ
contains the shared and full-domain masks and eight quadrature positions for
each solid voxel. Connectivity is reported rather than repaired; detached
material and disconnected face sheets require a revised design. Reconstruction
alone is not a dynamic simulation or evidence of blast performance.

For subsequent searches, `--filter-passes 2` smooths the design field with
wraparound across the in-plane cell boundaries and a zero normal gradient at
the top and bottom of the core. The fixed-volume projection follows the
filter. This discourages grid-scale variations but does not ensure a connected
or manufacturable topology. The filter operates in voxel units; account for
unequal voxel dimensions when choosing resolution and the number of passes.
Runs without this option retain the unfiltered design. A running calculation
uses its own source snapshot; resume requires the same source hashes and
settings as its checkpoint.

Use `--require-connected-binary` to reject line-search proposals whose
fixed-count binary reconstruction has disconnected components, does not
connect the two thickness boundaries, or leaves an empty projected column.
This preserves the specified material amount without prescribing a cell shape.
The discrete test can restrict or stop the search; connectivity alone does not
establish adequate feature size, manufacturability, or physical performance.

The connected binary geometry can be replayed under the same full-domain
loading and perimeter boundary condition:

```sh
python -m examples.multicell_binary_blast_3d outputs/multicell-blast-opt/binary-00001.npz outputs/multicell-blast-opt/inputs.json --output outputs/multicell-blast-binary
```

The replay stores particle frames and a sampled work, energy, momentum and
minimum-determinant history. It rejects disconnected geometry by default.
This single-field reference-grid calculation has no self-contact, plasticity
or failure; nearby disconnected surfaces can share grid support. A successful
replay alone does not establish physical performance, convergence, or the
accuracy of the continuous-density objective.

Two completed binary replays can be compared with
`python -m tools.compare_multicell_binary_blast BASELINE CANDIDATE --output comparison.json`.
The comparison requires matching load, boundary settings, solver source,
runtime, sampling interval, time horizon and mass. It rejects disconnected
geometries and excessive sampled energy residuals. The reported maximum is
the maximum among recorded frames, so the sampling interval must be refined
when measuring a transient peak.

For ParaView, export a completed replay with
`python -m tools.export_multicell_binary_vtk REPLAY GEOMETRY.npz VTK_OUTPUT`.
The solid voxels form a hexahedral volume; its vertices use the original
reference-grid displacement interpolation and its cell stress is the mean of
the eight material-point values. Render with
`pvpython tools/render_multicell_binary_paraview.py VTK_OUTPUT GIF_OUTPUT --stress-limit-mpa 100 --cutaway`.
Use a common stress limit when comparing designs. The deformation is shown at
physical scale, and the cutaway exposes the interior.
