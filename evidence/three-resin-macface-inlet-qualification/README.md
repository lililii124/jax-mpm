# F6 MAC face inlet qualification evidence

This record accompanies a bounded, fresh 64-step resin inlet window using the authenticated forming state. The legacy 8256-step inlet comparison is read-only; no historical particle mass or position was changed.

- `8256-inlet-face-audit.json`: single read-only comparison of legacy 200 µm birth streams against cumulative prescribed 250 µm MAC inlet-face flux through the actual 8256 endpoint. A nearest-face transverse aggregation gives 40.63% L1 discrepancy despite 0.00217% global dose difference; face parcel/MAC ratios span 0.388 to 3.385.
- `fresh64-review.json`, `fresh64-observables.json`, and `fresh64-qualification-summary.json`: actual accepted 0–64 qualification result. All rows valid; 648 births, exact source/active mass closure to 3.4e-21 kg; max divergence 5.04e-7 s^-1; max pressure relative residual 6.84e-9; max stationarity relative 2.90e-14; maximum energy identity 1.85e-23 J.
- `fresh64-plan.json` and `fresh64-source.json`: physical parameters and source capsule hashes.
- `artifact-sha256.json`: forming/fresh checkpoint hashes and private render GIF hash.

Private visual review only: `H:/test/mpm/outputs/three-strand-resin-private-8256/three-strand-resin-private.gif` (17 real snapshots, 8000–8256, 400×400, zero physics replay). It is a late-flow segment, does not certify full filling, and was not published.

The fresh window verifies only short inlet qualification; it is not a full-fill run or production claim.
