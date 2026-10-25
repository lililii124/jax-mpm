# MAC face conservative inlet and nominal source clock evidence

This is the accepted replacement for the partial newborn RT0 advection recorded in the older `three-resin-macface-inlet-qualification` directory. That earlier 64-step result is historical and **not a qualification**: its newborns entered a pressure field which did not include them.

## Implemented rule

The RT0 runner creates a parcel only when a full packet volume has crossed its inlet MAC face: nominal birth time `(layer+1)*Vp/Q_face`. Packet volume is `Vp=spacing^3`; transverse coordinates are MAC-face centres; axial position is `left+Vp/(2*dx^2)`. Each face remainder is stored in the checkpoint and observed per step. The source volume balance is `Σ Q_face * source_time = packet_count * Vp + Σ remainder` up to floating-point arithmetic. Births are checked on integer solver ticks `step*dt` with no early-time tolerance. A parcel is injected at the end of its birth tick and first enters P2G, pressure, and RT0 on the next full step. Quantization is bounded by `dt`; the first-order displacement bound is `u_max*dt`. No fractional newborn transport is claimed.

The runner does not overwrite `FlowState.time` to align clocks. It stores the nominal source clock (`step*dt`) and actual accumulated state time independently, records their difference on each row, and validates both on resume.

## Qualification evidence

- `fresh64-*`: fresh empty-fluid qualification from the authenticated forming state. All 64 rows pass pressure coverage, physical domain, source-volume remainder, mass, energy, momentum, wall, speed, intrusion, and escape gates. Final logJ is about `[-9.4e-12, 1.18e-11]`.
- `step624-parent-bridge.json`: nearest complete accepted checkpoint used after the later step640 remainder rejection. It has 5,984 active/emitted particles; all 316 faces had remainder in range and none was ahead of nominal cumulative flux. Its accumulated state clock was 3.47e-16 s behind nominal. The rejected step640 checkpoint was not resumed.
- `to1024-*`: accepted continuation from step624 through1024. All 400 rows pass the same guards, per-face remainder and pressure coverage. At step1024 the logJ range is approximately `[-2.05e-10, 2.39e-10]`; accumulated state time differs from the nominal source clock by less than `1.0e-15 s`.

These runs are bounded inlet-flow qualifications only. They do not certify full fill or production readiness. Full-fill continuation is a separate staged chain and must stop at the original volume/occupancy gates.
