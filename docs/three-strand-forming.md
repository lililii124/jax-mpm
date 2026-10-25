# Three-strand forming candidate

Three initially straight, stress-free cylinders are driven by material end
collars. The right collar rotates toward one turn while both ends develop
0.5% extension of the free span. Interior positions follow MPM dynamics;
no helix or intermediate particle positions are prescribed.

The frozen `ab410cf` baseline uses L=24 mm, R=1 mm, initial orbit=1.4 mm,
particle spacing=0.1 mm, grid h=0.2 mm and 227,520 particles. Illustrative
parameters are E=50 kPa, nu=0.25 and rho=1000 kg/m³. The global plan spans
20 ms and 5,164 steps at CFL=0.15. Any damage, invalid material state,
uncovered support or unconverged contact rejects a step.

Separate material grids interact through frictionless nodal normal contact.
The initial runner uses at most 64 projected Gauss–Seidel sweeps and a closing/
KKT tolerance of 1e-8 m/s. The continuation candidate packs all positive shared
nodes, retains the original PGS attempt and uses a bounded active-set fallback
for the same contact QP. Fixture counter-impulses are explicit. This is
grid-scale contact, with no Hertz pressure or sub-grid nonpenetration claim.

Collars use fixed reference-particle masks. The finite-increment prescribed
velocity `(Q-I)x/dt` maps their constrained quadratic support by Q, avoiding
Euler radial expansion. The candidate projects the initial boundary velocity
before force/contact and advances x/F with the beginning/end average velocity.
Its signed work ledger retains transfer losses, contact work and the material
endpoint defect separately. See the [energy review](three-strand-forming-energy-review.md).

## Run from a clean checkout

Use Linux with a bounded CPU/memory allocation. Install the optional runner/
geometry dependencies from the repository root:

```bash
python -m pip install -e '.[forming]'
python -m tools.run_three_strand_forming --output outputs/forming-original --stop-steps 1280 --wall-seconds 3600 --uss-gib 5
python -m tools.run_forming_consistency --parent outputs/forming-original --output outputs/forming-candidate --steps 128
```

The first numerical command creates the original 0→1280 prefix, including
`plan.json`, `checkpoint.npz`, `observables.json` and actual `frame-STEP.npz`
files. If its resource guard stops early, resume that same directory with
`--resume --stop-steps 1280` and unchanged source. A physical gate failure
requires diagnosis; resource continuation does not override it.

The candidate command consumes that exact saved parent and advances
1280→1408 with the original global clock and loading. It writes a full
checkpoint, `progress.json`/`review.json` and saved particle frames. For another
bounded segment, pass `--parent outputs/forming-candidate` and a **new** output
directory; `--steps` is a segment length in 1..512. Candidate continuation needs
an external resource limit, such as a Docker CPU/memory/time guard. Completion
of these commands is subject to the original validity and geometry gates.

## Render saved states

Install official ParaView 6.2.0 separately. Its `pvpython` must be supplied
through `--pvpython`, `PARAVIEW_PYTHON`, or `PATH`. These commands collect
authentic states from both segments, check duplicate-step equality and hashes,
then render with a fixed 400² white-background camera and three colors:

```bash
python -m tools.prepare_three_strand_forming_manifest outputs/forming-original outputs/forming-candidate --output outputs/forming-media-input
python -m tools.render_three_strand_forming_private outputs/forming-media-input outputs/forming-media --pvpython /path/to/ParaView/bin/pvpython
```

The manifest records every source file, SHA, actual step and global time.
Frames are copied without changing arrays; no positions are interpolated.
The renderer produces PNGs, a private GIF and render receipts. A render receipt
is a media check; numerical admission and publication review remain separate.

The available baseline is a partial dynamic forming process, not a certified
full turn, quasistatic result or converged solution. Section-center gaps are
proxies; the F-advected particle-cell intrusion check is a finite-resolution
diagnostic. Stored history and full signed energy records remain necessary.
Resin filling and curing are outside this frozen baseline.
