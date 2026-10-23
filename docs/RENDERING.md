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
previews; full GIFs remain in the output directory.

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
