"""Conservative identity matching on an unchanged coverage-cell list.

This topology foundation is not connected to the Hertz execution path. It does
not select adhesive candidate cells, evolve a contact law, or permit physical
coalescence/splitting. Callers must retain tensile attachments when constructing
current regions and reject the entire step when ``valid`` is false.

Storage is O(N + P**2) for N coverage cells and static region capacity P. Coverage
reordering/rebuilding is unsupported: the same cell index must denote the same
ordered segment pair in both frames. Pair metadata uses canonical fibre IDs.
"""

from typing import NamedTuple

import jax.numpy as jnp


class RegionIdentity(NamedTuple):
    region_ids: object
    previous_slot: object
    next_id: object
    valid: object
    merge: object
    split: object
    lost_active: object
    ambiguous: object
    capacity_exceeded: object
    identity_exhausted: object
    malformed: object


def _counts(values, length, xp):
    # JAX needs an explicit static length, rather than only a minimum length.
    return (xp.bincount(values, length=length) if xp is jnp
            else xp.bincount(values, minlength=length))


def _match(previous_cells, current_cells, previous_ids, previous_active,
           previous_pairs, current_pairs, next_id, current_count, xp):
    """Shared array algebra; NumPy backend permits CPU-only topology checks."""
    old, new, ids, active, old_pairs, pairs, cursor, count = map(xp.asarray, (
        previous_cells, current_cells, previous_ids, previous_active,
        previous_pairs, current_pairs, next_id, current_count))
    p = ids.shape[0]
    if (p < 1 or ids.ndim != 1 or old.ndim != 1 or new.shape != old.shape
            or active.shape != (p,) or old_pairs.shape != (p, 2)
            or pairs.shape != (p, 2) or cursor.shape != () or count.shape != ()):
        raise ValueError("Fixed-capacity region and identical coverage-array shapes required")
    if (any(a.dtype != xp.dtype('int32') for a in (old, new, ids, old_pairs, pairs, cursor, count))
            or active.dtype != xp.dtype('bool')):
        raise ValueError("Region indices/IDs must be int32; attachment flags must be boolean")
    # Static bound also keeps pair encoding and bincount shape representable.
    if p*p >= 2**31-1:
        raise ValueError("Region capacity exceeds int32 overlap encoding")
    slots = xp.arange(p, dtype=xp.int32)
    occupied = ids >= 0
    live = slots < count
    oi, ni = xp.clip(old, 0, p-1), xp.clip(new, 0, p-1)
    old_present, new_present = old >= 0, new >= 0
    old_legal = (old >= -1) & (old < p)
    new_legal = (new >= -1) & (new < p)
    malformed = (xp.any(~old_legal) | xp.any(~new_legal)
        | xp.any(old_present & ~occupied[oi]) | xp.any(new_present & ~live[ni])
        | xp.any(active & ~occupied) | xp.any(ids < -1)
        | xp.any(occupied & (ids >= cursor)) | (cursor < 0) | (count < 0)
        | xp.any(occupied & ((old_pairs[:, 0] < 0) | (old_pairs[:, 0] >= old_pairs[:, 1])))
        | xp.any(live & ((pairs[:, 0] < 0) | (pairs[:, 0] >= pairs[:, 1]))))
    duplicate = (ids[:, None] == ids[None, :]) & occupied[:, None] & occupied[None, :]
    malformed = malformed | xp.any(duplicate & (slots[:, None] < slots[None, :]))
    same_pair = xp.all(old_pairs[oi] == pairs[ni], axis=1)
    both = old_present & new_present & old_legal & new_legal
    incompatible = xp.any(both & ~same_pair)
    encoded = xp.where(both & same_pair, oi*p+ni, p*p)
    overlap = _counts(encoded, p*p+1, xp)[:p*p].reshape(p, p) > 0
    outgoing, incoming = xp.sum(overlap, axis=1), xp.sum(overlap, axis=0)
    merge = xp.any(incoming > 1)
    split = xp.any(outgoing > 1)
    lost = xp.any(active & (outgoing == 0))
    # Empty live regions cannot acquire identity, even when count says they exist.
    current_coverage = _counts(xp.where(new_present & new_legal, ni, p), p+1, xp)[:p]
    previous_coverage = _counts(xp.where(old_present & old_legal, oi, p), p+1, xp)[:p]
    malformed = malformed | xp.any(live & (current_coverage == 0)) | xp.any(occupied & (previous_coverage == 0))
    source = xp.argmax(overlap, axis=0).astype(xp.int32)
    continuing = live & (incoming == 1) & (outgoing[source] == 1)
    births = live & (incoming == 0)
    birth_count = xp.sum(births, dtype=xp.int32)
    exhausted = cursor > (2**31-1-birth_count)
    safe_cursor = xp.where(exhausted, 0, cursor)
    birth_ids = safe_cursor+xp.cumsum(births, dtype=xp.int32)-1
    result_ids = xp.where(continuing, ids[source], xp.where(births, birth_ids, -1))
    capacity = count > p
    ambiguous = incompatible | merge | split
    valid = ~(malformed | ambiguous | lost | capacity | exhausted)
    # Invalid results expose diagnostics, never a partially usable assignment.
    return RegionIdentity(xp.where(valid, result_ids, -1),
        xp.where(valid & continuing, source, -1),
        xp.where(valid, safe_cursor+birth_count, cursor), valid, merge, split,
        lost, ambiguous, capacity, exhausted, malformed)


def match_region_identity(previous_cells, current_cells, previous_ids,
                          previous_active, previous_pairs, current_pairs,
                          next_id, current_count):
    """Match fixed-capacity regions using shared coverage cells, without host I/O.

    Cell owners are region slots, with -1 meaning absent. Previous IDs use -1
    for unused slots; current slots are compact [0,current_count). New IDs are
    monotonic, allocated in current-slot order and never reused. ``previous_slot``
    indexes the entire old law-history tree for continuing regions; births need
    fresh law history. Both pair arrays have shape (P,2), sorted fibre IDs.

    Every occupied old region must retain its old coverage membership, including
    detached regions until retired. An active old region without overlap is
    rejected rather than treated as a birth. All merges/splits are rejected,
    including detached ones; no redistribution of adhesive state is implied.
    Inputs use int32 and a boolean previous_active array. Compatible with jit;
    capacity P and coverage length N are static through their array shapes.
    """
    return _match(previous_cells, current_cells, previous_ids, previous_active,
                  previous_pairs, current_pairs, next_id, current_count, jnp)
