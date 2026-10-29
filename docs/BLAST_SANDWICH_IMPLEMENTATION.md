# Experimental elastic sandwich model

`examples.blast_sandwich_design` supplies explicit three-dimensional full-MPM
forward histories, restart, derivative checks and particle VTU export.
`examples.blast_sandwich_reduced` supplies a separate transient reduced optimizer.
The reduced model is still undergoing error correction; no optimized sandwich
has completed independent full-horizon, three-load qualification.

The reference panel is 300 × 300 × 24 mm: two 2 mm skins and a 20 mm core,
clamped over a 15 mm perimeter. Eight volume Gauss points per cell use quadratic
B-spline interpolation. Surface quadrature applies pressure on the reference
front face. Uniform, central and eccentric footprints have equal force histories
and different moments: 0.5 MPa nominal sin² pressure for 0.2 ms, observed to 2 ms.

The geometry and skin properties remain fixed: E = 70 GPa, ρ = 2700 kg/m³,
ν = 0.33. The assumed effective core has ρ = 2700φ, E = 0.3 × 70 GPa × φ²,
ν = 0.25, and 0.08 ≤ φ ≤ 0.35. A 15 mm Gaussian control filter and interpolation
precede a differentiable scalar mass projection. The perimeter core stays at
φ = 0.18. At mean fraction 0.18, total mass is 1.8468 kg. Shared displacement
implies perfect bonding. The elastic material is compressible Neo-Hookean;
plasticity, crushing, damage and interface separation are absent.

```bash
pip install -e .[design]
python -m examples.blast_sandwich_design plan --xy-step .01 --z-step .001 --controls 24 24 3
python -m examples.blast_sandwich_design forward --xy-step .01 --z-step .001 --controls 24 24 3 --save-every 1000 --output outputs/sandwich-full
python -m examples.blast_sandwich_design forward --xy-step .01 --z-step .001 --controls 24 24 3 --design-checkpoint evidence/blast-sandwich-reduced/enriched-v12-best-strict-feasible-controls.npz --save-every 1000 --output outputs/sandwich-selected-full
python -m examples.blast_sandwich_design export --output outputs/sandwich-full/eccentric
```

`forward` runs all three footprints from separate virgin states. It transfers
only controls from a supplied design checkpoint. Restart uses `--resume` and
binds settings, source and runtime; `--stop-step` is absolute within the original
horizon. Full checkpoints retain displacement, velocity, force, deformation,
work/impulse ledgers and the accepted CSV prefix. Saved particle fields use
physical deformation scale one. The separate exact face exporter can reconstruct
nodal/face fields from saved Gauss samples.

Run `plan` before allocating a grid. The 10 × 10 × 1 mm grid has 172,800 volume
points and 29,403 nodes. Four-core full forward jobs used an 8 GiB container and
6 GiB USS guard. The 5 × 5 × 1 mm grid has 691,200 points; a measured short prefix
used 4.70 GB USS and extrapolated to about 16.6 hours per footprint. This estimate
does not establish a complete refined-grid result. A production reverse gradient
is not available within that budget: sequential full-MPM JVP optimization is
limited to 32 controls and oversized reverse checkpoint plans are rejected.

The full-MPM CLI's legacy `optimize` action uses mean integrated squared
back-face displacement. The requested fixed-mass worst-peak objective belongs
to the [separate reduced CLI](BLAST_SANDWICH_REDUCED.md). These objectives must
not be substituted for one another.

The retained [numerical checks](../evidence/blast-sandwich-reduced/numerical-validation.json)
include a resolved 8 μs micro-geometry JVP/central-difference comparison
(relative discrepancy 3.81 × 10⁻⁸) and exact short-run restart. Their scope is
implementation verification, not physical optimization or full-horizon accuracy.
The earlier unresolved derivative check was withdrawn. Spatial/time refinement,
selected-design full-horizon confirmation and nonlinear material qualification
remain required. The [design specification](BLAST_SANDWICH_DESIGN.md) defines
those acceptance requirements.
