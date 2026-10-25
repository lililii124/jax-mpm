# F7 complete RATTLE checkpoint at step 64

This fresh 64-step run replaces the R9 qualification NPZ as the restart base. Its atomic checkpoint contains every `TLState` field, all `BrittleHistory` arrays, source and geometry/material bindings, cumulative work, gate value, and accumulated linear/angular impulses. The checkpoint SHA and complete binding are in `latest.json` and `provenance.json`.

The unchanged R9 geometry, material and 10% energy gate pass. Maximum log strain is below the original 12% cap. Guide position and velocity residuals meet the configured RATTLE tolerances. Global P/L ledgers include both guide impulses: `Gᵀλx/dt` at the position kick and `Gᵀλv` at the velocity kick.

This checkpoint is pre-release. Contact/interlacing and guide withdrawal are not covered; the next bounded continuation stops at step 1024, well before guide release.
