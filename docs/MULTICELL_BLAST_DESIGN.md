# 3 × 3 shared-cell blast design

This work in progress optimizes a three-dimensional material field shared by
nine cells. All nine cells are explicitly present in the finite plate model:
their particles, displacement and deformation states are independent. The
central pressure footprint acts on the entire top face, centred on the middle
cell. The perimeter is clamped. The cell pattern is not prescribed and the
initial field can be uniform or seeded random.

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
