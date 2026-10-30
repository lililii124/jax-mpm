# C2 accepted-history collection and ParaView rendering

The formal runner writes one NPZ frame at the initial state and thereafter at
accepted timesteps that cross each 10 ns physical-time bin. These are actual
solver states, not interpolated frames. Each frame stores particle position,
velocity, damage, body, mass, accepted step, and physical time.

## Collect a completed window chain

Pass run output directories in chronological order, beginning with the fresh
virgin run and including every resumed window through the 1 us horizon:

```console
python tools/domain_impact/collect_history.py --output outputs/domain-impact-history path/to/virgin/result path/to/window-01/result path/to/window-02/result
```

The collector checks each run's source map, accepted checkpoint, resume
checkpoint SHA, parent source map, endpoint, and frame order. Equal-time window
boundary frames are removed only when every particle field is identical. By
default it requires the final window to finish at the requested horizon.

## Render with ParaView

Run with ParaView's `pvpython` executable:

```console
pvpython tools/domain_impact/render_paraview.py outputs/domain-impact-history outputs/domain-impact-render
```

The renderer writes a 400 by 400 GIF, its actual VTP frame series, source-time
PVD index, PNG frames, and a hash-bound render audit. It keeps every particle
and colors the cloud by damage. The camera and scale remain fixed across all
frames. It does not interpolate or invent states.

For an intentionally incomplete but real prefix, pass `--allow-prefix` to
both scripts. The collector records the incomplete endpoint; the renderer
marks the GIF `PREFIX — INCOMPLETE`. This output is not a developed cloud.
