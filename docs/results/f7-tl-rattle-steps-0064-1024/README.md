# F7 TL/RATTLE bounded continuation through step 1024

This evidence continues the restartable step-64 checkpoint in `../f7-tl-rattle-bound-cp64r2/` without changing its material, input identity, reference geometry, guide sets, dt, or solver source archive. Each continuation task restores the preceding segment checkpoint and records its parent SHA. `segments.json` binds each checkpoint and task receipt; `step-ledger.csv` retains scalar work, energy, guide, strain, linear-momentum, and angular-momentum diagnostics for all 1024 accepted steps. Raw NPZ checkpoints are retained in the private local/remote task audits; this public evidence records their SHA-256 chain and complete scalar field summaries in `segments.json`.

The exact runtime `source.tar` is nested inside the uploaded `source-capsule.tar` retained with the accepted step-64 qualification. The outer capsule is bound separately by its provenance file. Its SHA-256 is `29cc0dd9e3d63000e4d0328b9f1876bdbc4a16d8cc92ec433082257aa261fca7`; the guide/material/geometry binding SHA-256 is `9d9b48ee7bd9f18251758bad4cfb6756baa0502fddcc346d418bcf8b04d9c98d`. The checkpoint chain ends at SHA-256 `deb95c5579b3beb0bd87eab8b52c73ffcf7983cd2e9eafcf0ec49533716b56bc`.

At step 1024 (`t=6.8692e-5 s`), cumulative absolute machine work is `4.390994652512932e-7 J`; the energy remainder is `-1.4429301028057976e-10 J` (`3.2861e-4` of absolute machine work), below the 0.10 gate. Maximum log strain is `0.00786098`, below 0.12; maximum damage is zero. Across all steps, maximum normalized linear and angular momentum errors are `6.04e-19` and `1.16e-19`; maximum guide position and velocity residuals are `1.36e-20 m` and `4.36e-16 m/s`. All 1024 rows are valid and pass the P/L and strain guards.

The run stopped at step 1024, before any guide release. This is a bounded mechanics qualification only; it does not establish contact, interlacing, guide withdrawal/release, or full-process acceptance. No release/contact computation was run for publication.

## Source closure

The runnable qualification entry point uses the following F7-specific source files in this branch:

- `examples/triaxial_weaving_tl_guided_qualification.py`
- `examples/triaxial_weaving_guided_geometry.py`
- `examples/triaxial_weaving_batch_plan.py`
- `examples/triaxial_weaving_batch_geometry.py`
- `examples/triaxial_weaving_process.py`
- `examples/triaxial_weaving_geometry.py`
- `mpm_jax/fibre_tl_guides.py`
- `mpm_jax/fibre_tlmpm.py`

The entry point also uses the existing `examples/triaxial_fibre_geometry.py`, `mpm_jax/fibre_mpm.py`, and `mpm_jax/materials/isotropic_brittle.py`. The F1 `mpm_jax/fibre_tlmpm.py` baseline at `origin/main` has the same git blob (`a1516cc46735ba738278005cf22bd08c102e876e`) as the 80df08a parent; the no-guide path remains covered by the previously passed one-step smoke artifact in `../f7-tl-guide-diagonal-proof/no-guide-smoke.json`.

## Run the current entry point

```sh
python -m examples.triaxial_weaving_tl_guided_qualification --steps 64 --output outputs/f7-64
python -m examples.triaxial_weaving_tl_guided_qualification --steps 256 --resume-from outputs/f7-64 --output outputs/f7-320
```

The current entry point hashes its Python source closure and runtime versions for restart binding. Its time check uses the forward rounding bound for repeated additions; it preserves the checkpoint's actual time. It rejects a window reaching the first guide release. These bookkeeping changes do not alter the TL material or guide kernels. The published 1024-step evidence remains bound to its archived runtime source, including the recorded IO-only time-check patches; it is not relabelled as a run of the updated entry point.
