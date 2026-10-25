# Dry three-direction forming station (feature candidate)

The initial material consists of separate, straight circular cords. Family0
is prethreaded warp. Families+60 and-60 start as straight feed lengths outside
the forming zone. All material particles exist from the beginning; no hidden
finished cloth is revealed and no particle is interpolated into a target weave.

Only small reference endcap masks receive machine velocities. The prescribed
cycle is opening/entry alignment, insertion, closing with feed reserve, hold.
Existing cord end guides are raised/lowered to make the next shed. Interior
particles move through the existing APIC update and local Hertz forces.
The periodic geometry helper supplies only directions, circular sampling,
discrete crossing order and an arclength estimate for feed reserve. Its finished
crimp is never used as the initial particle configuration or a body target.

The feed reserve is based on a piecewise geometric lower bound through planned
over/under crossing heights; it is supplied by end motion, not material creation.
This avoids forcing the initially straight reference length into a longer
crimped path by artificial extension. The resulting body shape remains emergent.

Defaults: diameter225um, pitch2.7mm, dx75um, particle spacing60um, ideal elastic
cord E10MPa, nu.25, density1500kg/m3, dormant damage onset20% strain. Material
particles must remain undegraded and logarithmic stretch<=12%; no strength
calibration is claimed. Longitudinal bending at this pitch is much gentler than
the old prebuilt pitch1mm geometry, which cannot simply be bent from straight
with a small-strain material without supplying slack and checking strain.

The old tube grid only surrounds reference particles. A feeding cord moves
farther, so the prototype uses explicit swept AABB packed grids. Contact search
coverage likewise includes the full relative motion and is checked at runtime.
Insertion occurs only during an open shed; the leading material end has passed
the crossings before closure, because the reused Hertz kernel rejects end-cap
contact. Parallel/same-family contacts, footprint overlap, large Hertz radii,
extended connected regions, coverage escape and material failure are rejected.
Hertz potential omits changing-curvature work; all work and momentum records
remain visible, without claiming exact energy conservation from force balance.

## Scale and staged delivery

`--per-family 1` is a three-cord triangular unit with three distinct crossings,
not a single pair contact. It is only the smallest MPM/fixture admission.
The six-cord `--per-family 2` prototype has13032 particles and16 machine stages;
neither is the final user-facing scale.

The intended final process has at least8–12 cords per family, repeated woven
cells and all three colors. Current NumPy plan for8 per family:184224 material
particles. The simple whole-motion dense-grid/contact prototype would require
428291836 grid addresses and up to78315860 contact candidates, so it refuses
to allocate that case under its initial bounds. This is a measured planning
limitation, not a finished large-case implementation.

Before full scale, use stage-local oriented swept corridors for grid support and
per-cord/segment relative-motion coverage, rather than applying the largest feed
skin to every old-old pair. Preserve current material state and contact identity
through any rebuild. Actual support/coverage escape must stop the solve. A
bounded active-grid/map resource pilot must precede the large process. A standard
multi-field MPM normal-contact alternative may be evaluated independently if
Hertz becomes inapplicable; the other forming agent's fixed-three-field PGS
kernel is not silently reused as a six-or24-field law.

## Commands and admission

```bash
python -m examples.triaxial_weaving_process --per-family 1
python -m examples.triaxial_weaving_process --run --per-family 1 --steps 64 --saved-frames 8 --output output
python -m tools.review_triaxial_weaving_process output
```

The actual short job must preserve source hashes, reference masses/IDs,
deformation/contact state, fixture errors, physical times, work and momentum.
It is not a weave animation. Full-stage review must show the measured over/under
order at multiple distinct crossings, not merely the requested guide schedule.
Rendering must use actual saved particles/current cross-section centroids,
three family colors, a full-station view plus a fixed close view of the forming
zone, and visible guide positions. No resin is included. No finished-process GIF
may be published from an opening-only prefix.

An optional `--grid tube` uses existing `build_tube_layout` with static section-centre seeds sampled along machine fixture sweeps. These seeds allocate nodes only and never prescribe material trajectories. Any actual particle stencil outside the allocation invalidates integration. Dense mode retains the original admission baseline. Compression requires comparison of full raw states and does not yet solve the larger station's dense address-map or all-pair coverage cost.

The initial inherited `boundary_work` is the exact grid kinetic-energy change caused by clamping, not physical machine work. The work observer records each fixture impulse dotted with its prescribed velocity and the difference from grid clamping energy as projection loss. For the first observer upgrade after the audited step6000, only that future energy/work window is qualified; the unknown physical work of the earlier prefix is not reconstructed. The solver, material, fixture commands and restored physical arrays remain unchanged. A documented source-specific upgrade is explicit and the prior evidence stays retained.
