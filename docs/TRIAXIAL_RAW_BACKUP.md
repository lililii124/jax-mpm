# Raw triaxial data backup

The normal result artifact is bounded to 192 MiB. It is not a backup of the
complete saved trajectory. The final `f60` cache holds that trajectory, but a
cache is temporary: GitHub may evict it under storage pressure or after inactivity.
Back up a completed case before starting another large case.

`triaxial-cache-export.yml` restores an explicitly named final cache and exports
its original bytes to a **draft** data release. It does not run JAX, advance a
checkpoint, render, save another cache, publish a release, or change any storage
setting. Its source version is recorded separately from the numerical producer.

The backup contains all recorded position, damage, failure, speed and temperature
frames, the latest complete state/contact checkpoint, final state, static particle
fields, reference geometry, runtime metadata, tables and three receipt files.
It does not contain every-frame deformation/stress tensors, earlier checkpoint
bytes, VTK datasets or all rendered PNGs. Keep the original numerical review and
media artifacts alongside it.

## Export sequence

1. Retain the original run/source identity, the `frame-60.json` SHA256 and the
   independent numerical reports. A sealed raw cache alone is not scientific
   acceptance. The exporter labels its output as raw data pending separate review.
2. Create a unique tag `raw-triaxial-<producer-run>-<attempt>-f60` at the exact
   producer commit. Do not move an existing tag. This separate step uses the
   repository owner's existing credentials; the workflow receives no personal token.
3. Dispatch the export workflow with that commit, original run/attempt and receipt
   hash. Its exact restore key and path match the numerical workflow. Cache misses,
   fallback hits, changed source objects, missing/extra files or broken receipt
   chains stop the export.
4. The exporter verifies each raw file and uploads it unchanged. Files over the
   conservative 1.9 GB asset limit are split into ordered 512 MiB parts. The manifest
   binds original paths, sizes, whole-file and part hashes, source identity, receipt
   chain, release asset IDs and exporter identity. Existing assets can only be reused
   if their name, size and GitHub digest match; nothing is overwritten or deleted.
5. Download every release asset into a fresh local directory. Use the manifest SHA
   from the verified export report, then reconstruct and recheck the whole cache:

   ```powershell
   python tools/export_triaxial_cache.py verify-download `
     --producer-commit <commit> --producer-run <run> --producer-attempt 1 `
     --receipt-sha <original-f60-sha256> --manifest-sha <manifest-sha256> `
     --assets .local/raw-assets/<run>-1 --output .local/raw-backups/<run>-1/f60
   ```

   The local repository must contain the producer's Git objects. The command writes
   a backup receipt only after all files, source objects and the receipt chain match.
   Keep this directory outside Git. Release publication remains a separate decision
   after numerical and media review; the exporter never marks it as the latest release.

GitHub permits up to 1,000 assets per release, each smaller than 2 GiB, without a
total release-size or bandwidth limit. This uses release assets rather than the
Actions artifact storage allowance. See [release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)
and [release asset digests](https://docs.github.com/en/rest/releases/assets).

An incomplete upload leaves a draft and any successfully verified assets available
for inspection. A mismatched or partially uploaded asset requires explicit cleanup;
the exporter will not replace it automatically. Creating a tag at an older commit
can require workflow permissions that the job token cannot obtain, so the tag is
prepared separately. See [release creation permissions](https://docs.github.com/en/rest/releases/releases#create-a-release).
