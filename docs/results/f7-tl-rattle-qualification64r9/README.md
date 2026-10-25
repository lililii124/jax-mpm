# F7 TL/RATTLE 64-step qualification

This isolated mechanics qualification uses the exact v4 reference geometry, 600 finite-section guides, fixtures, material, and 75 µm characteristic length. It does not qualify contact, interlacing, guide withdrawal/release, or the full weaving process.

R9 is the accepted record. Its final energy remainder is 0.2424% of absolute machine work, below the unchanged 10% gate. R7 and R8 are retained as failed attempts with explicit source and state hashes in `provenance.json`; the original APIC v4 work-budget rejection and v3 12% strain failure remain unchanged. `qualification.npz` is a final qualification state, not a lossless restart checkpoint.

The report logs per-guide position multipliers, velocity impulses, target and pre/post centroid speeds, ideal prescribed work, actual impulse-midpoint work, and nodal kinetic jumps. It keeps the discrete constraint-work discrepancy visible.
