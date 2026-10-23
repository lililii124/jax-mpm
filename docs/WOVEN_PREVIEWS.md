# Coarse woven previews

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

Homepage labels identify these runs as **Coarse preview**. A completed, reviewed
formal run can replace its corresponding preview while retaining the coarse
numerical archive and historical media. Rendering does not add captions or axes
to the animation itself.
