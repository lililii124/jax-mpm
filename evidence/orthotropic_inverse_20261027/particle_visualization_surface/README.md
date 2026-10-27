# Continuous surface holdout visualization

This is an antialiased rerender of the already saved float64 MPM particle
trajectories. No JAX calculation or trajectory regeneration was performed.
The original point-cloud GIF remains unchanged in the local-only archive at
`.local/jax-mpm-inverse-surface-local-only-20261027/point_cloud/`; its final
frame is also preserved in `surface_before_after.png` for direct review.

![Smooth outer surface comparison](orthotropic_holdout_particles_surface_400.gif)

`surface_before_after.png` shows the original point-cloud view on the left and
the new continuous surface on the right, both from the same final physical
step and with the same camera and color map.

## Surface construction and image settings

The holdout particle lattice is structured, with 52 × 40 × 36 material points
(74,880 total). For each saved state, the renderer selects the outermost
reference-grid particle layer and connects adjacent material points on each of
the six boundary faces into 10,278 quadrilateral patches. Faces use exact
integer reference-lattice adjacency; no Delaunay reconstruction, particle
smoothing, synthetic points, or remeshed deformation are used. Patch vertices
use their actual MPM positions, and the point scalar is the original per-point
3D displacement magnitude. Area-weighted point normals derived from the
actual deformed quads give smooth lighting.

ParaView 5.11.1 rendered the 26 saved states at 1600 × 1600 with FXAA and 8x
multisampling. Pillow 10.4.0 downsampled each image spatially to 400 × 400 with
LANCZOS and encoded them in the original time order at 12 fps. The GIF contains
26 frames (step 0, 10, ..., 250), covering the full 8 ms holdout protocol. No
temporal interpolation, deformation scaling, scalar bar, labels, or orientation
axes are used. The shared displacement color range is 0–0.0191545 m.

The public package keeps only the 400 × 400 GIF, the compact particle-run and
render receipts, this README, the unlabeled before/after still, and a compact
manifest. The following larger intermediate files are kept locally in
`H:\\test\\mpm\\.local\\jax-mpm-inverse-surface-local-only-20261027`, and are
not part of the public repository:

- the two float64 arrays `truth_particle_positions.npz` and
  `inferred_particle_positions.npz`;
- 52 VTK surface frames (float32 coordinates/scalars/normals);
- 52 point-cloud VTK frames;
- 26 supersampled 1600 × 1600 surface PNG frames;
- 26 point-cloud PNG frames and intermediate render scenes/logs.

The compact receipts retain SHA-256 digests for the source result, holdout
archive, both float64 trajectory files, MPM sources, and final GIF. The local
archive also retains a full per-file SHA-256 inventory.

## Reproduction source and input binding

- Frame source and MPM solver: see `particle_frames_receipt.json`.
- Frozen result SHA-256: `63278383197e515201c573a7ec79a832fb78c6e408d35b96d1c52a911d9dd212`.
- Holdout archive SHA-256: `e7bf3e8d60d8fd27fb0eb0b0ec6e3e70c837ed1b9e969177993e6c64abd7c596`.
- Surface-render script: `tools/render_inverse_holdout_surface_paraview.py`; its
  exact digest and ParaView execution log are recorded in the manifest.
- Sensor trace verification remains bound to the rerun receipt: maximum
  displacement errors are 8.67e-19 m (truth) and 3.47e-17 m (fit).

To regenerate the exact particle trajectories from the published fitted
constants and frozen holdout archive, run in a JAX-enabled environment:

```bash
PYTHONPATH=. python tools/export_inverse_holdout_particle_frames.py \
  --result evidence/orthotropic_inverse_20261027/result.json \
  --holdout evidence/orthotropic_inverse_20261027/holdout_synthetic.npz \
  --output /path/to/local/particle-frames --stride 10
```

Then render the surface GIF with ParaView 5.11.1 and Pillow 10.4.0:

```bash
python tools/render_inverse_holdout_surface_paraview.py \
  --frames /path/to/local/particle-frames \
  --output-dir /path/to/local/surface-render \
  --pvpython /path/to/ParaView/bin/pvpython --fps 12 --supersample 4
```

The frame exporter verifies the rerun against the saved sensor responses at
every exported time. The surface renderer derives connectivity from the known
structured particle counts and reference-lattice adjacency; it does not need
or modify any fitting checkpoint.
