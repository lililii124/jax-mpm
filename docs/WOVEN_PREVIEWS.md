# Archived coarse woven runs

The `woven-preview` entries show completed dynamic calculations on a coarse grid.
They use the same geometry, materials and prescribed loading as the
[woven examples](WOVEN_COMPOSITES.md), with actual particle coordinates and no
displacement amplification. ParaView separates yarn families by colour and displays
the matrix translucently. These previews are not grid-converged results.

The original `interpretation.json` remains unchanged with `passed: false`.
`preview-review.json` records the current independent review, including reconstructed
elastic energy, reference section forces and the reference bending moment.
`preview-eligibility.json` binds both reports and their input records by hash.
It permits the insufficient particle count and yarn-height grid resolution for
visualization only; all other physical checks must pass.

The coarse transverse-compression case also lacks particles whose full interpolation
support remains inside each moving platen. Its eligibility record retains
`required_moving_samples_exist: false` and states that exact inner-particle tracking
is unavailable. The prescribed grid boundary and measured surface closure remain
separate checks. A preview does not establish exact particle clamping.

These runs are excluded from the homepage and gallery. Their numerical records
and historical media remain archived; completed, reviewed fine-grid runs are
displayed separately. The original animations contain no captions or axes.

## Archived previews

These five runs retain all 61 saved states. Each animation uses the original ParaView
frames and actual geometry. The [gallery transfer record](results/woven-gallery-relay.json)
binds the compact GIFs to the producer artifacts, source commit and image sequence.

| Weave and loading | Records |
|---|---|
| 0°/90° plain weave · Out-of-plane bending | [Review](results/woven-preview--plain--transverse-bending/preview-review.json) · [Preview eligibility](results/woven-preview--plain--transverse-bending/preview-eligibility.json) |
| 0°/90° plain weave · Warp tension | [Review](results/woven-preview--plain--warp-tension/preview-review.json) · [Preview eligibility](results/woven-preview--plain--warp-tension/preview-eligibility.json) |
| 0°/90° plain weave · Transverse in-plane tension | [Review](results/woven-preview--plain--weft-tension/preview-review.json) · [Preview eligibility](results/woven-preview--plain--weft-tension/preview-eligibility.json) |
| 0°/±60° weave · In-plane shear | [Review](results/woven-preview--triaxial--in-plane-shear/preview-review.json) · [Preview eligibility](results/woven-preview--triaxial--in-plane-shear/preview-eligibility.json) |
| 2×2 twill · Through-thickness compression | [Review](results/woven-preview--twill-2x2--transverse-compression/preview-review.json) · [Preview eligibility](results/woven-preview--twill-2x2--transverse-compression/preview-eligibility.json) |
