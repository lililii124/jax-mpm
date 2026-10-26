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

Step 5012 subsequently exhausted its primary pressure iteration budget.
Increasing that budget from 3072 to 5120, with 1024 dual-refinement iterations,
passed a [64-step continuation](../evidence/resin-primary-refinement/actual-continuation.json)
from step 5008 to 5072. All original thresholds remained in force: maximum
primal residual 9.73e−8 and stationarity residual 9.88e−11. Increasing only
the dual budget did not resolve the original failure. The extra primary
iterations also change the previously accepted states at steps 5010–5011;
this is not a bitwise-preserving continuation of the smaller-budget run.

Step 5825 later met the volume constraint but failed stationarity. On the
same saved state, 6144 primary iterations with 1024 dual iterations still
failed; 5120 primary iterations with 2048 dual iterations passed. The
[current budget evidence](../evidence/resin-dual-budget-v6/diagnostic-summary.json)
also covers a complete 64-step continuation through step 5888. Its maximum
pressure and stationarity residuals were 6.84e-8 and 9.26e-11, respectively,
with no obstacle intrusions or escaped particles. The total iteration cap is
7168; the primary cap is the total minus the 2048 dual iterations. Physical
acceptance thresholds are unchanged.

This remains an experimental filling example. Complete filling, spatial/time
convergence, and the smooth-boundary error have not been established. The
gallery does not present this partial run as a completed injected cylinder.
