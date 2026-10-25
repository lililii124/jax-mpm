# F7 complete RATTLE qualification receipt at step 64

This fresh 64-step run replaces the earlier nonrestartable 64-step attempt as the restart base. Its atomic checkpoint contains every `TLState` field, all `BrittleHistory` arrays, source and geometry/material bindings, cumulative work, gate value, and accumulated linear/angular impulses. The checkpoint SHA and complete scalar restart summary are in `latest.json`, `qualification.json`, and `provenance.json`. The raw NPZ is retained in the private local/remote audit rather than this Git history.

The original F7 geometry and material are unchanged, and the 10% energy gate passes. Maximum log strain is below the original 12% cap. Guide position and velocity residuals meet the configured RATTLE tolerances. Global P/L ledgers include both guide impulses: `Gᵀλx/dt` at the position kick and `Gᵀλv` at the velocity kick.

The step-64 checkpoint is pre-release. Contact/interlacing and guide withdrawal are not covered; the next bounded continuation stops at step 1024, well before guide release.
