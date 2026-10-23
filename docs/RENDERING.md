# ParaView output

All spatial images and animations use ParaView. The solver exports compressed
VTK PolyData (`.vtp`) and a time collection (`particles.pvd`). Coordinates and
scalars use float32 in VTK; the NPZ trajectories retain the solver precision.
Every particle is exported and rendered.

Install [ParaView](https://www.paraview.org/download/) with its Python executable.
Set `PARAVIEW_PYTHON` if `pvpython` is not on PATH:

```powershell
$env:PARAVIEW_PYTHON = 'C:\Program Files\ParaView 6.2.0\bin\pvpython.exe'
python -m mpm_jax.visualization outputs/suite/oblique/neo-hookean --gif
```

On Linux/macOS, use `export PARAVIEW_PYTHON=/path/to/pvpython`.
The batch renderer has been exercised with ParaView 6.2.0 on Windows.

```bash
python -m mpm_jax.visualization outputs/compression/compression/rht --field damage --gif
python -m mpm_jax.visualization outputs/production/water-entry/neo-hookean --resolution 1600 --final-only
```

The scene uses a fixed orthographic camera, a light background and shaded
particle spheres. Blue/orange distinguish bodies; scalar colours retain one
range across the animation. Axes, grids, legends and text overlays are disabled.
`render.json` records the renderer version, field, camera and particle count.

For small motions, choose a physical scalar that remains visible without
enlarging the geometry:

```bash
python -m mpm_jax.visualization outputs/suite/wave/neo-hookean --field speed --gif
python -m mpm_jax.visualization outputs/suite/surface-load/neo-hookean --field displacement --gif
```

`speed` shows the saved velocity magnitude in m/s. `displacement` is
`norm(x(t) - x(0))` in m for each material particle, using the first saved frame
as reference. The exporter calculates it one frame at a time in chunks of at
most 65,536 particles, without allocating a whole-history displacement array.
It only adds a VTK scalar; source trajectories and rendered coordinates stay
unchanged. Its definition and reference time are stored in `vtk_metadata.json`.
`render.json` records `geometry_scale=1` and one fixed colour range spanning the
complete saved history. By default, speed and displacement start at zero. Waves and cantilever motion use speed; prescribed
shear, torsion and surface loading use displacement magnitude.

## Controlled visual comparisons

The renderer accepts these optional display controls:

| Option | Meaning and checks |
| --- | --- |
| `--color-range LOW HIGH` | A fixed scalar range. Both values must be finite, `LOW < HIGH`, and the interval must contain the complete saved field range. Categorical `body` colouring does not accept this option. |
| `--scene-bounds XMIN XMAX YMIN YMAX ZMIN ZMAX` | Bounds used to calculate the camera. Each pair must be finite and increasing; the box must contain all saved particle coordinates and the prescribed CAD motion envelope. |
| `--sphere-opacity VALUE` | Opacity of the prescribed rigid-sphere CAD only, with `0 < VALUE ≤ 1`; the default is 1. Particle, debris and box displays are unaffected. |
| `--focus-bounds XMIN XMAX YMIN YMAX ZMIN ZMAX` | An explicit detail view. The finite ordered box must overlap the scene but can crop peripheral geometry. Mutually exclusive with `--scene-bounds`; it never scales particle coordinates. |
| `--matrix-opacity VALUE` | Matrix opacity with `--field family_id`, with `0 < VALUE ≤ 1`; the default is 0.008. Yarn particles remain opaque. |

For the matched nonadhesive/adhesive pair, derive one displacement upper limit
from the larger maximum in the two complete `vtk_metadata.json` exports. Use
zero as the lower limit. Form one box enclosing both full-history particle bounds
and both sphere-path envelopes, including each sphere's radius. Use those same
values, field, image resolution and sphere opacity for both renders. Preserve
full precision when passing bounds; an inward-rounded bound is rejected.

The same controls are available as `color_range`, `scene_bounds`, `focus_bounds`,
`sphere_opacity` and `matrix_opacity` keyword arguments to `render_directory` and `render_trajectory`.
They change the display only. For example, the adhesion suite uses translucent
CAD overlays to keep the deforming surface visible:

```bash
python -m mpm_jax.visualization outputs/adhesion/adhesive-pull-off --field displacement --sphere-opacity 0.45 --gif
```

This command alone uses automatic camera and colour limits. A controlled pair
also supplies the common `--color-range` and `--scene-bounds` derived above.
After rendering, compare `color_range`, `camera_bounds`, `camera_position`,
`camera_focal_point`, `camera_parallel_scale` and `resolution` in both
`render.json` files. The sphere `display_opacity` entries must also agree.

The records retain the actual field range and particle/CAD bounds separately
from the chosen camera bounds and colour range, with override sources and
requested values. No coordinate scaling or field clipping is applied. Rigid
sphere transparency does not alter the contact model, particle positions or
particle counts.

For a contact close-up, supply the same `--focus-bounds` to both members of a
controlled pair. `render.json` identifies this as a detail view and retains the
complete data bounds alongside the camera bounds. Compare the same physical
output times and use one colour range for both histories. The full VTK datasets
remain available outside the camera view. When a case records an explicit
`particle_spacing_m`, that spacing sets marker size, including uniformly seeded
rods whose requested particle density has been rounded to a symmetric lattice.

Woven examples save static `phase`, `family_id` and `yarn_id` arrays in
`particle_fields.npz`. The loader checks their shapes, dtypes and SHA-256 hashes
against the summary before exporting them to every VTK frame. With
`--field family_id`, the matrix is translucent and the yarn families use fixed
blue, orange and green colours. Every material particle stays in the scene;
`render.json` records the per-phase counts and matrix opacity. This is a view of
the actual yarn and matrix particles, with no displacement scaling. Prescribed
clamps follow the same half-cosine displacement and hold used by the solver.

| File | Contents |
|---|---|
| `particles.pvd`, `vtk/*.vtp` | Time series and all particle fields |
| `scene.pvsm` | Saved ParaView pipeline and camera |
| `frames/*.png`, `final.png` | ParaView screenshots |
| `simulation.gif` | Screenshot sequence, assembled without changing the frames |
| `simulation.mp4`, `video.json` | H.264 encoding and screenshot/video hashes |
| `vtk_metadata.json`, `render.json` | Export and rendering provenance |

Open `particles.pvd` in ParaView to inspect data. If moving a saved `.pvsm`
scene, select the new data directory when ParaView asks to locate its files.
Simulation and rendering are separate: `--no-render` runs the solver without
ParaView, and the visualization command can be repeated without recomputing.

For a compact video, install `ffmpeg` and encode a complete screenshot sequence:

```bash
python tools/encode_video.py outputs/suite/dam-break/neo-hookean
```

This uses the ParaView frames at 20 fps, without rerunning the simulation or
changing the particle count. The public gallery links these videos from PNG
previews. The homepage embeds compact looping GIFs selected from completed runs,
including the reviewed finite-motion replacements. Each uses the complete ParaView screenshot sequence;
the original six animations also hold their final frame. Full-size GIFs remain
in the output directory.

`docs/images/home/manifest.json` records the run IDs, frame counts and GIF hashes.
Its `render_sha256` checks the corresponding public `render.json` after LF newline
normalization; `source_render_sha256` records the original output file hash.
Regenerate the homepage tables from that manifest with
`python tools/collect_home.py`. Adding runs requires an audited GIF catalogue
passed with `--audit`.

The sphere-contact scene includes a separate smooth CAD sphere driven by the
prescribed boundary motion. Its saved animation track follows the output times.
It is excluded from particle counts and recorded under `auxiliary_geometry`
in `render.json`. The JKR pressure-footprint example has no geometric indenter.

Round-bar grips and concrete-bending fixtures use separate Box geometry.
The bending scene derives two supports and a moving loading block from the saved
velocity-patch definition. Support display height equals the patch width; the
loading block starts at the nominal top face, and all three boxes are clipped
to the specimen depth. These boxes represent bilateral velocity constraints;
they are not separately solved roller or platen contacts. The loading block
follows the prescribed analytic translation. Their dimensions, motion and
interpretation are recorded under `auxiliary_geometry`, and they do not enter
particle counts.
The large-displacement bending block also follows the stopping ramp and remains
at its final prescribed position during the hold.

Gray translucent markers identify failed particles only in bodies configured
with `failure_mode="ballistic"`. Retain-mode bodies keep their normal body or
scalar colours, including particles whose material history has `failed=True`.
Missing configuration defaults to retain for display. The original failed field
remains available in VTK, and `render.json` records active/debris display counts.

For a numerical readback check:

```bash
pvpython tools/check_vtk.py outputs/suite/oblique/neo-hookean
```

This reads the first and last VTK frames with VTK itself and compares particle
counts, coordinates and scalar fields against the saved NPZ data.
