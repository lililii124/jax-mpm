# Fixed-state consistent-mass projection

This follows a4d2a1a and changes no impact solver, material, load or physical
gate. All three earlier geometric-DELTA raw runs remain unchanged.

The 125 ns seed has 26336 particles and 33575 occupied material/grid nodes.
Each Cartesian velocity component shares a 105344 by 33575 scalar whitened
reconstruction. The two material blocks have 770 and 32805 nodes. Before
equilibration, column norms span 6.14e-20 to 1.48e-4. Blind equilibration
therefore magnifies near-roundoff columns. Thirty-two columns together have
Frobenius norm 2.64e-15, within a standard dimension-scaled backward-error
budget; this observation alone does NOT justify deleting physical modes.

The candidate retains every physical column. For Q=Q0 D^-1, construct a
symmetric LDL factor of Q^T Q + mu I. Use its square-root triangular inverse
as an invertible RIGHT preconditioner for least squares on Q. The shifted
matrix is never the objective and never appears in an energy ledger. Changing
mu does not change the mathematical projector. Known per-body affine grid
modes are explicitly reconstructed, orthonormalized and split off as an exact
known-range component; the remaining solve is orthogonal to them. This also
protects rigid momentum modes from iterative noise.

An important rejected version used a condition limit of 1e12. It failed an
eight-particle fixture with seven exact grid null modes: after entering the
floating-point nullspace it amplified roundoff into a false velocity field,
despite a misleading internal residual. The current guard is eps^(-1/2),
because the square-root factor is obtained from a Gram matrix and squared
singular values below machine precision are unresolved by that factor. This
guard is NOT admission. Every returned result must be tested on Q0, with
projection idempotence, momenta and projection-loss identities independently
checked. No rank truncation or numerical energy compensation is used.

Independent dense SVD comparisons include both a full-rank nonaffine fixture
and the seven-null-mode fixture. The corrected sparse projection differs by
1.13e-13 and 1.20e-14 respectively. The original failed fixture and low-fill
ILU/GMRES failure are retained in local evidence. A bounded PROPACK spectral
exploration was stopped without certifying a rank.

The source-bound fixed-state tool uses exactly original initial.npz SHA256
3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817.
It fixes the iteration bound at 256 and reports every stop flag, original
unscaled and scaled residual, repeated-projection error, momenta, Pythagorean
loss and preconditioner-scale comparison. Stop flags alone neither admit nor
reject a result. Comparisons bound kinetic differences BETWEEN computed
fields; they are not an a priori bound on an unknown exact projection.

The original global energy tolerance remains 1.414599609375e-8 J. No material
or contact threshold changes, no heat generation and no impact advancement
occur. A successful fixed-state check permits further operator development;
it does not establish compatible time-discrete material work or a scalable
contact gauge, and it does not admit a long impact simulation.

Run the two entry points with Python3.12 and SciPy:

```
python tools/prove_metal_affine_projection_reference.py --output new-reference.json
python tools/prove_metal_affine_projection_fixed125.py --state ORIGINAL/initial.npz --output new-fixed.json --shift 1e-8
python tools/prove_metal_affine_projection_fixed125.py --state ORIGINAL/initial.npz --output new-comparison.json --shift 1e-6 --compare new-fixed.npz
```

The fixed-state entry point caps BLAS threads before import. Run processes
sequentially and preserve prior results; output paths must not already exist.
