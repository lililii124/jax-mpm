"""Read-only bounded reproduction of the original Hertz extent predicate.

No contact force, history, material, or trajectory state is written here.
Input is the PRE-contact state and metrics returned by that same evaluation.
"""
import numpy as np


def identify(state, data, neighbours, metrics):
    import jax.numpy as jnp
    from mpm_jax.fibre_coupling import section_centres, closest_segments
    from mpm_jax.contact_patches import contact_patches

    centres, _ = section_centres(state.x, jnp.ones_like(state.mass), data)
    near = data.coverage_pairs
    ends = data.segments[near]
    first, second, _, _, distance = closest_segments(
        centres[ends[:, 0, 0]], centres[ends[:, 0, 1]],
        centres[ends[:, 1, 0]], centres[ends[:, 1, 1]])
    gap = distance-jnp.sum(data.fibre_radius[data.segment_fibre[near]], axis=1)
    selection = contact_patches(gap, gap < 0, neighbours, capacity=4096,
                                max_patches=32, max_iterations=256)
    # Transfer only the bounded active-cell gather, never the whole coverage graph.
    cells = np.asarray(selection.cells)
    labels = np.asarray(selection.labels)
    live = cells < len(near)
    safe = jnp.asarray(np.minimum(cells, len(near)-1))
    positions = np.asarray(((first+second)/2)[safe])[live]
    gaps = np.asarray(gap[safe])[live]
    segment_ids = np.asarray(near[safe])[live]
    strand_ids = np.asarray(data.segment_fibre[near[safe]])[live]
    cells, labels = cells[live], labels[live]
    spacing = float(jnp.max(jnp.linalg.norm(
        centres[data.segments[:, 1]]-centres[data.segments[:, 0]], axis=1)))
    radius = float(jnp.max(metrics.contact_radius, initial=0.))
    limit = max(2*spacing, 4*radius)
    patches = []
    for root in np.unique(labels):
        member = labels == root
        low, high = positions[member].min(axis=0), positions[member].max(axis=0)
        extent = float(np.linalg.norm(high-low))
        patches.append(dict(root_slot=int(root), coverage_cell_ids=cells[member].tolist(),
            segment_pairs=segment_ids[member].tolist(), strand_pairs=strand_ids[member].tolist(),
            gaps_m=gaps[member].tolist(), centre_low_m=low.tolist(), centre_high_m=high.tolist(),
            extent_m=extent, limit_m=limit, unsupported_extent=bool(extent > limit)))
    return dict(schema='original-hertz-read-only-patch-identities-v1',
        input_state='pre-contact state of attempted step', topology_valid=bool(selection.valid),
        active_cell_count=int(selection.active_count), patch_count=int(selection.patch_count),
        spacing_m=spacing, maximum_contact_radius_m=radius, limit_m=limit, patches=patches,
        maximum_extent_m=max((p['extent_m'] for p in patches), default=0.))
