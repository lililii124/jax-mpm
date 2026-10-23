# Ceramic spherical indentation

`ceramic-indentation-wide` places a smaller rigid sphere over a wider JH-2 ceramic tile. The original `ceramic-indentation` case and its recorded results remain available.

| Geometry | Original | Wide tile |
| --- | ---: | ---: |
| Tile size, x × y × z | 100 × 25 × 100 mm | 180 × 25 × 180 mm |
| Sphere radius | 75 mm | 37.5 mm |
| Sphere diameter / tile width | 1.5 | 0.416667 |
| Initial sphere centre | (125, 175, 125) mm | (125, 137.5, 125) mm |

Both use a 250 mm cubic domain. The wide tile occupies `(35, 75, 35)` to `(215, 100, 215)` mm. A bonded rigid backing fixes all grid velocity components at `y ≤ 75 mm`; the other exterior faces have zero applied traction. Gravity is omitted.

The sphere starts tangent to the upper face and completes one prescribed indentation–withdrawal cycle:

```text
δ(t) = 4 mm × [1 − cos(2π t / 300 µs)] / 2
```

Maximum indentation occurs at 150 µs. The sphere returns to its initial position at 300 µs, with zero endpoint velocity. This is dynamic indentation, not a quasi-static hardness measurement. The JH-2 parameters are unchanged from the original case.

The `retain` failure mode keeps every particle in the continuum and in grid transfers. Damage changes the constitutive response; `D = 1` does not represent a separated fragment or a resolved crack. Local softening has no fracture-energy regularization.

## Resolution and measurements

The grid96/ppc2 configuration has 386,420 material points. Grid spacing is 2.604167 mm; actual particle spacings are approximately 1.294964/1.25/1.294964 mm. The lattice preserves the exact tile volume. The thickness spans 9.6 grid cells. `sqrt(R δmax)` spans 4.703 cells and is only a contact-resolution estimate; no Hertz validation is claimed for damaged ceramic.

Observation regions remain attached to their initial particle labels:

- The central surface uses the uppermost reference layer within a 4 mm x/z radius of `(125, 125)` mm.
- The local region uses a 30 mm reference radius through the full thickness; its uppermost layer supplies the local surface displacement.
- Damage statistics, mean surface displacement and maximum local downward displacement are recorded separately.

Contact imposes a unilateral normal-velocity constraint on grid nodes. Particle centres can therefore overlap the analytical sphere between nodes. The observer records `max(R − |x − centre|, 0)` over all particles and over the reference top layer, plus the number of centres inside the sphere. These are overlap diagnostics, not contact areas. Reactions are forces on the backing and sphere, evaluated during the next grid step at `boundary_sample_time_s`.

## Run

```powershell
python -m examples.material_engineering --case ceramic-indentation-wide --model jh2 --grid 96 --ppc 2 --frames 40 --no-render --resume
python -m mpm_jax.visualization outputs/material-engineering/ceramic-indentation-wide/jh2 --field damage --sphere-opacity 0.45 --gif
```

The first command writes the complete 300 µs cycle. ParaView renders actual particle coordinates without displacement amplification, axes or annotations. Sphere opacity changes only the auxiliary rigid geometry's display. See [rendering](RENDERING.md), [material scopes](MATERIALS.md) and the [material engineering guide](MATERIAL_ENGINEERING.md).
