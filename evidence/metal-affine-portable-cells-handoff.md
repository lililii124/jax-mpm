# Offline independent 125 to 125.002 ns cells

The parent verified dedicated image is
`sha256:639be564c801378d4dd43ea5ea630a3c378c2d2a745177c8a52857656e954180`.
It contains Python 3.12.13, JAX 0.12.1.dev20261022, jaxlib 0.12.0.1 and the
same remaining numerical pins as the successful Actions run 40180997172.
The original dev-client wheel SHA256 is
`8aecd75dba37eb8e76418487445c66efb3877b3724570d3b70237f063090b980`.
No dependency installation or network access occurs at the cell entry point.

Use assigned CPUs 40–43 and an 8 GiB container. The guard requires that CPU
affinity and a finite Docker memory cgroup. It caps aggregate owned USS at
4 GiB, retains at least 2 GiB of cgroup/system headroom, stops on first error,
and caps each cell plus replay at 900 seconds and 2 GiB of new raw data.
The successful first-step Actions run used four CPUs and peaked at 0.922 GB
USS for the complete process group; resource requirements are not extrapolated
from the 64-core host's total RAM.

The capsule entry `run_cell.py --cell both --output /results` runs independent
1 ps × 2 and 0.5 ps × 4 cases sequentially, each starting from the SAME exact
125 ns original input. Each case then replays itself with every final array
compared bitwise. It does not rerun the old 2 ps baseline, continue beyond
125.002 ns, or generate any media. Choose a fresh empty output directory and
keep external console logs outside that directory.

With the extracted capsule and an empty host results directory, the parent
can launch its already-built image as follows:

```
docker run --rm --cpuset-cpus=40-43 --memory=8g --network=none \
  -v "$MPM_CAPSULE:/capsule:ro" -v "$MPM_RESULTS:/results" \
  --entrypoint python \
  sha256:639be564c801378d4dd43ea5ea630a3c378c2d2a745177c8a52857656e954180 \
  /capsule/run_cell.py --cell both --output /results
```

The actual runtime versions, installed dev-client Python file bytes, immutable
source/input capsule, CPU affinity and cgroup limits are checked before any
numerical work. Both original state SHA256s, physical parameters, contact/CFL
and energy gates are unchanged. Output retains every stage, first rejection,
physical channel and replay, with a full output-file SHA256 manifest.

The original successful 2 ps reference is retained without recomputation.
Its physical gates passed; an independent audit verified the real kicks,
drift, full contact coverage, 4864 fixed particles, plastic-only calorimetry,
and global energy/momentum/torque. The first driver's legacy contact diagnostic
maxima/cache records were incomplete. The capsule only corrects recording:
actual NNLS multipliers/pair coordinates are saved, diagnostic maxima are
updated, old cache/iteration counters are explicitly unused, and a rejected
state gets valid=False. Force, material and contact arithmetic are unchanged.
Compare physical arrays and conservation across dt; do not require obsolete
diagnostic counters to match the old baseline.

The 2 ps reference was computed on the four-core Actions host; the finer cells
use the allocated Xeon host. Record this hardware distinction. Contracting
own-method errors are evidence for this tiny window, not a production or
fragmentation admission. The old V4 BE comparison remains revoked; its
correction notice and native-call evidence are preserved separately.
