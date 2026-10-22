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
| `vtk_metadata.json`, `render.json` | Export and rendering provenance |

Open `particles.pvd` in ParaView to inspect data. If moving a saved `.pvsm`
scene, select the new data directory when ParaView asks to locate its files.
Simulation and rendering are separate: `--no-render` runs the solver without
ParaView, and the visualization command can be repeated without recomputing.

For a numerical readback check:

```bash
pvpython tools/check_vtk.py outputs/suite/oblique/neo-hookean
```

This reads the first and last VTK frames with VTK itself and compares particle
counts, coordinates and scalar fields against the saved NPZ data.
