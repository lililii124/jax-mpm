# Resin filling around formed fibres

`examples.three_strand_resin_rt0` advances a Newtonian fluid around the saved
three-strand forming geometry. The fibre obstacles are voxelized from the
deformed material cells in a completed forming checkpoint. They remain fixed
during filling; curing and two-way fluid–structure coupling are not included.

The solver uses staggered-grid APIC transfers, a mass-metric JAX LSQR pressure
projection, and RT0 characteristic transport through the projected cells.
Particle-owned pressure cells must have supported or prescribed faces. The
solver adds no artificial face mass. Inlet packets and outlet losses enter the
mass and energy accounts explicitly.

| Parameter | Value |
| --- | --- |
| Mold | Cylinder, radius 3.2 mm, axial limits −1 to 25 mm |
| Grid spacing | 0.25 mm |
| Density | 1,100 kg/m³ |
| Dynamic viscosity | 0.05 Pa·s |
| Time step | 40 µs |
| Inlet | Radius 2.5 mm, parabolic axial velocity, peak 0.2 m/s |
| Walls and fibres | Fixed voxel-face velocities |
| Outlet | Zero gauge pressure; explicit removal of exiting parcels |

## Run and resume

Install the geometry dependencies with `pip install -e ".[forming]"`. Use the
checkpoint and plan from a completed `examples.three_strand_forming` run:

```bash
python -m examples.three_strand_resin_rt0 \
  --checkpoint outputs/forming/checkpoint.npz \
  --forming-plan outputs/forming/plan.json \
  --output outputs/fill-0000 --steps 256

python -m examples.three_strand_resin_rt0 \
  --checkpoint outputs/forming/checkpoint.npz \
  --forming-plan outputs/forming/plan.json \
  --resume-from outputs/fill-0000 --output outputs/fill-0256 --steps 256
```

Each output directory must be new and empty. A call advances 1–512 steps.
Resume from a successfully completed segment: its checkpoint and observable
ledger must have the same endpoint. A rejected segment retains diagnostics;
its later ledger must not be paired directly with an earlier periodic checkpoint.
`--source-until` extends the future inlet schedule without changing existing
particles or their masses.

## Verification status

The [bounded continuation evidence](../evidence/resin-dual-refinement/actual-continuation.json)
covers steps 4880–4912 with unchanged physical parameters and acceptance
thresholds. Its maximum pressure residual was 6.39e−8 (limit 1e−7), maximum
stationarity residual 7.24e−11 (limit 1e−10), and maximum cumulative energy
accounting error 7.85e−17 J. No obstacle intrusions or escaped particles occurred.
The pressure velocity uses separate base and correction gradients to avoid
cancellation when their potentials are combined.

A [main-branch integration replay](../evidence/resin-main-integration/checkpoint-comparison.json)
repeated the same 32 physical steps and reproduced every checkpoint field
exactly. The replay completed in 130 s with peak process USS of 1.14 GB.

This remains an experimental filling example. A subsequent segment passed
steps 4913–5011 but rejected step 5012 for a pressure residual of 6.80e−4.
Complete filling, spatial/time convergence, and the smooth-boundary error have
not been established. The saved GIF gallery does not present this partial run
as a completed injected cylinder.
