# Final JC continuation and genuine ParaView delivery

The executable controller is `tools/run_jc_final.py`; its manual private-artifact
workflow is `.github/workflows/jc-final.yml`. No frame-60 result or fracture GIF
exists by virtue of these source files. The active numerical source remains
`08d83a051a9764a5a218fd33e184f3291eb3dd1c`. Do not extend
`run_jc_continuation.py`'s partial-only endpoint list without implementing the
completed-result state machine below.

## Frame 50 to 60 controller

1. Restore the exact independently admitted frame-50 cache with no fallback.
   Run `verify_carry` with its exact key, receipt SHA256 and `from_frame=50`.
   Preserve the entire parent cache, including all ancestor receipts, runtime
   locks, loaded-module reports, and prefix reviews. Verify frozen source and
   controller history using the existing continuation functions.
2. Record `prefix_snapshot(parent/result, 50)` before copying to a separate
   final working directory. Keep the original plan bytes and original source
   record unchanged. Use the recorded package/runtime lock. Admit enough disk
   for both parent and complete final histories (including `trajectory.npz`,
   which duplicates the mmap arrays). The existing one-GB partial carry cache
   budget is not a valid final raw-result storage policy.
3. In a child process select the immutable solver checkout before imports and
   repeat `worker()`'s case-array, material/configuration/signature checks.
   Invoke the same `common.run` with the original 60 frames, 965 substeps,
   `resume=True`, `render_output=False`, `gif=False`, and final endpoint 60.
   The ordinary completed-result branch must execute. Do not rewrite completed
   summary/status into partial schema to reuse `audit_prefix`.
4. Run `tools/review_jc_full_result.py RESULT --parent-cache PARENT
   --parent-key EXACT_KEY --receipt-sha EXACT_SHA` in a fresh process.
   It requires the immutable solver at `.local/jc-frozen-solver` and verifies
   source/controller history, all parent bytes, all 61 raw frames and completed
   mmap histories, State/History descriptors, material invariants, observer
   continuity, independently recomputed final metrics and full 3D positive
   quadratic-grid support at every saved time. It does not run JAX simulation.
   The distinction between `passed` (valid completed protocol) and
   `resolved_saved_history_fracture` (all fracture gates) is intentional.
5. Require separate complete-protocol and fracture gates before describing the
   result as fracture: material failure; final active reference grips on their
   original sides with zero shared positive 3D grid nodes; observed failure
   strictly before observed separation; no separation before failed material.
   Failure and separation first appearing in the same saved frame leave their
   ordering unresolved. Central-layer active count alone is not a sufficient
   gate. Grid decoupling is a numerical-model result, not resolved crack
   surfaces, experimental validation or fracture-energy convergence.
6. Package every final raw file and complete parent cache into bounded complete
   files or reconstructible numbered parts with byte counts and SHA256 for
   each part AND the complete original file. Include exact producer/controller
   identities, final reviewer hash and all parent receipts. Verify a retrieval
   round trip before temporary runner deletion. Keep failed final runs as
   diagnostics and do not issue a successful-result receipt.

## Actual ParaView pipeline

After the final raw review, invoke the frozen solver's
`mpm_jax.visualization.render_trajectory(RESULT, gif=True, field='damage',
resolution=1200, color_range=(0, 1))` in a fresh process with that checkout first
on `PYTHONPATH`. It exports saved particle coordinates and failure labels, runs
real `pvpython --force-offscreen-rendering` through `paraview_scene.py`, saves
PNG frames and `scene.pvsm`, and encodes `simulation.gif` from those PNGs.
Use its automatic complete-scene bounds and geometry scale 1.0. Do not magnify
displacements, hide ballistic debris to create an apparent gap, interpolate a
crack, replace the specimen by a cylinder, or generate synthetic images.

Record ParaView executable/version/hash, export/renderer source hashes, complete
command, raw input hashes, exported VTP/PVD and metadata hashes, all 61 PNG
hashes, state-file hash and GIF hash. Require `render.json` to identify real
ParaView, 61 rendered frames, `geometry_scale=1.0`, `view_scope='complete scene'`,
fixed damage range `[0,1]`, all particles retained and the expected actual
active/debris counts. Independently compare exported points/failure values with
each raw frame (the existing `tools/check_vtk.py` is an initial implementation
reference), and decode the GIF to verify its full frame sequence and dimensions.
Inspect an initial, necking, first-failure, separated and final PNG plus the GIF.
Keep raw audit hashes unchanged before/after rendering. A preview can be rendered
when fracture gates fail, but its manifest and caption must say the actual
diagnostic outcome. The GIF is a rendered particle-model result; ballistic
markers are not resolved fragment surfaces.

## Verification status

The independent completed-result reviewer, dedicated final controller/workflow,
full raw archive transport/round-trip verification, and automated ParaView
provenance gate are implemented. The workflow uploads all multipart raw files,
restores the exact artifact ID and hashes every part and reconstructed original
before issuing a retrieval receipt. It retains those private artifacts for 30
days; permanent publication or longer-term storage remains a separate user
decision. A valid completed protocol whose fracture gates fail retains its
actual rendered diagnostic GIF and fails final success, with the reason recorded.

Static compilation, CLI parsing, YAML parsing, multipart reconstruction and
corrupted-part rejection have passed locally. The final controller, reviewer
and ParaView audit have not yet been exercised together on a real frame-60
dataset, because no admitted frame-50 parent or completed frame-60 result was
available during preparation. Actual CI execution remains necessary; no
completed test, visual inspection or media is claimed here. The workflow
reserves the last 55 minutes of its 360-minute whole-job ceiling for complete
delivery and retrieval and bounds numerical/render subprocesses accordingly.
