# Domain dynamics audit

This records the completed moving-domain compression and release runs. Evidence contains each run's `input.json`, `summary.json`, and `history.csv`, plus the validation report. Large state snapshots and saved frame files are intentionally omitted. The original outputs remain under `outputs/`.

## Compression results

| Run | Grid | dt (s) | Steps | Final numerical loss |
|---|---:|---:|---:|---:|
| `domain-dynamics-grid32` | 32 | 3.125e-6 | 160 | 4.54% |
| `domain-dynamics-grid32-halfdt` | 32 | 1.667e-6 | 300 | 5.87% |
| `domain-dynamics-grid64` | 64 | 1.667e-6 | 300 | 2.20% |
| `domain-dynamics-grid128` | 128 | 8.621e-7 | 580 | 0.85% |

Spatial refinement lowers the reported numerical-loss fraction from 4.54% to 2.20% to 0.85% for these runs. At grid 32, the smaller timestep instead raises it to 5.87%. These results expose a timestep- and resolution-dependent numerical energy defect; they do not establish convergence. The signed changing-moment geometry-energy term is reported in the ledger as a diagnostic defect. It is not physical dissipation and is not assigned to material failure. All four runs report valid status and small energy-balance errors; see their summaries and the validation report for exact values.

## Pre-existing crack release

`domain-crack-release-grid64` starts from a through-crack configuration and reports a final minimum crack opening of 0.5069 mm. It also reports 20.25% numerical loss. This demonstrates release/opening in that configuration only; it does not qualify crack growth or fracture accuracy. The audit does not include cohesive/contact evolution or a validated fracture front. The validation report explicitly records `fracture_validation: false`.

## Reproduction

Run from the repository root with the project environment installed. Each output directory must be empty or absent.

```powershell
python tools/audit_domain_dynamics.py --grid 32 --output outputs/domain-dynamics-grid32
python tools/audit_domain_dynamics.py --grid 32 --dt-scale 0.5 --output outputs/domain-dynamics-grid32-halfdt
python tools/audit_domain_dynamics.py --grid 64 --output outputs/domain-dynamics-grid64
python tools/audit_domain_dynamics.py --grid 128 --output outputs/domain-dynamics-grid128
python tools/audit_domain_dynamics.py --grid 64 --case release --save-frames --output outputs/domain-crack-release-grid64
python tools/check_domain_dynamics.py outputs/domain-dynamics-grid32 outputs/domain-dynamics-grid32-halfdt outputs/domain-dynamics-grid64 outputs/domain-dynamics-grid128 outputs/domain-crack-release-grid64 --output outputs/domain-dynamics-check
```

The check reconstructs final-field energies and checks the supplied histories; it does not replay every timestep. The exact run inputs, summaries, histories, and validation output are preserved in [`evidence/domain-dynamics`](../evidence/domain-dynamics/).

## Source provenance

The compression source corresponds to repository commit `8a409df`. The release driver is preserved in `643c93c`. Exact source SHA-256 values are retained in every copied input file. The driver rounds step counts to twenty output intervals, so `--dt-scale 0.5` gives 300 rather than 320 steps here; the actual steps are reported above.
