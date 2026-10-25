"""Read-only JAX instrumentation of the rejected 8540 contact evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path('/task')
OUT = Path('/work/failed8540-jax-contact.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    jax.config.update('jax_enable_x64', True)
    import sys
    sys.path.insert(0, str(ROOT))
    from examples.triaxial_failure import (restore_contact_model,
        restore_contact_history)
    from examples.triaxial_fibre_geometry import make_geometry
    from mpm_jax.fibre_mpm import FibreState
    from mpm_jax.fibre_coupling import (FibreCouplingHistory, build_contact_data,
        section_centres, closest_segments)
    from mpm_jax.fibre_contact import fibre_contact_parameters
    from mpm_jax.fibre_patch_contact import (initial_patch_history,
        evaluate_hertz_patches, closed_footprint_contains)
    from mpm_jax.contact_patches import patch_neighbours, contact_patches
    from mpm_jax.materials.isotropic_brittle import BrittleHistory as History

    binding = json.loads((ROOT / 'input/binding.json').read_text())
    plan = json.loads((ROOT / 'input/plan.json').read_text())
    kwargs = json.loads((ROOT / 'input/context.json').read_text())['kwargs']
    names = ('dimensions', 'diameter', 'pitch', 'crimp', 'clearance',
             'particle_spacing', 'packing', 'density', 'grip_length', 'max_particles')
    geometry = make_geometry(theta_deg=-kwargs['theta'],
        **{k: kwargs[k] for k in names})
    old = build_contact_data(geometry, search_window=plan['search_window'],
        coverage_skin=plan['coverage_skin_m'])
    data = build_contact_data(geometry, search_window=plan['search_window'],
        coverage_skin=plan['coverage_skin_m'], register_cross_family_coverage=True)
    assert np.array_equal(np.asarray(data.coverage_pairs), np.asarray(old.coverage_pairs))
    assert hashlib.sha256(np.asarray(data.coverage_pairs).tobytes()).hexdigest() == binding['coverage_descriptor']['sha256']
    neighbours = patch_neighbours(np.asarray(data.coverage_pairs), np.asarray(data.segment_fibre))
    assert hashlib.sha256(np.asarray(neighbours).tobytes()).hexdigest() == binding['neighbour_descriptor']['sha256']
    material = plan['material']
    law = fibre_contact_parameters(
        effective_modulus=material['young'] / (2 * (1 - material['poisson'] ** 2)),
        effective_radius=geometry['metadata']['diameter_m'] / 2,
        work_of_adhesion=plan['work_of_adhesion_J_per_m2'], model='hertz')
    cp_path = ROOT / 'output/checkpoint_0008540.npz'
    with np.load(cp_path, allow_pickle=False) as archive:
        payload = {k: archive[k].copy() for k in archive.files}
    state = FibreState(**{
        **{k: jnp.asarray(payload['state_' + k]) for k in FibreState._fields if k != 'history'},
        'history': History(*(jnp.asarray(payload['history_' + k]) for k in History._fields))})
    contact_law = restore_contact_model(law, payload)
    template = initial_patch_history(data, max_patches=plan['contact_regions']['max_patches'])
    contact = FibreCouplingHistory(
        restore_contact_history(template.law, payload, jnp, parameters=contact_law),
        jnp.asarray(payload['contact_radius']), jnp.asarray(payload['contact_segments']),
        jnp.asarray(payload['contact_valid']))
    regions = plan['contact_regions']
    def evaluate(s, c):
        return evaluate_hertz_patches(s, c, data, contact_law, neighbours,
            capacity=regions['capacity'], max_patches=regions['max_patches'],
            max_iterations=regions['max_iterations'])
    force, updated, metrics = jax.jit(evaluate)(state, contact)
    jax.block_until_ready((force, updated, metrics))

    centres, section_mass = section_centres(state.x, state.mass, data)
    near = data.coverage_pairs
    endpoint_sections = data.segments[near]
    p, q = centres[endpoint_sections[:, 0, 0]], centres[endpoint_sections[:, 0, 1]]
    r, s = centres[endpoint_sections[:, 1, 0]], centres[endpoint_sections[:, 1, 1]]
    first, second, wa, wb, distance = closest_segments(p, q, r, s)
    jax.block_until_ready((first, second, wa, wb, distance))
    gap = distance - jnp.sum(data.fibre_radius[data.segment_fibre[near]], axis=1)
    selected = contact_patches(gap, gap < 0, neighbours,
        capacity=regions['capacity'], max_patches=regions['max_patches'],
        max_iterations=regions['max_iterations'])
    jax.block_until_ready(selected)
    selected_representatives = np.asarray(selected.representatives)
    component_labels = np.full(len(near) + 1, regions['capacity'], dtype=np.int32)
    component_labels[np.asarray(selected.cells)] = np.asarray(selected.labels)
    fail_indices = np.flatnonzero(np.asarray(metrics.failed_section_contact))
    active = np.asarray(metrics.active)
    reps = np.asarray(metrics.pair_segments)
    patch_force = np.asarray(metrics.normal_force)
    patch_radius = np.asarray(metrics.contact_radius)
    patch_points = np.asarray(metrics.closest_points)
    patch_normals = np.asarray(metrics.normal)
    centres_np = np.asarray(centres)
    damage = np.zeros(data.section_count, dtype=np.float64)
    np.maximum.at(damage, np.asarray(data.particle_section), np.asarray(state.history.damage))
    failed = damage > 0
    owner_pairs = np.asarray(data.segment_fibre)
    segment_pairs = np.asarray(data.coverage_pairs)
    radius_table = np.asarray(data.fibre_radius)
    sections_table = np.asarray(data.segments)
    records = []
    for idx in fail_indices:
        pair = segment_pairs[idx]
        owners = owner_pairs[pair]
        sections = sections_table[pair]
        weights = np.asarray([[1 - np.asarray(wa)[idx], np.asarray(wa)[idx]],
                              [1 - np.asarray(wb)[idx], np.asarray(wb)[idx]]])
        endpoint_failed = failed[sections]
        loaded_failed = endpoint_failed & (weights > 0)
        matches = np.flatnonzero(active & (selected_representatives == idx))
        same_pair_slots = np.flatnonzero(active & np.all(reps == pair[None, :], axis=1))
        candidate_normal = (np.asarray(second)[idx] - np.asarray(first)[idx]) / max(float(np.asarray(distance)[idx]), 1e-30)
        cand_radii = radius_table[owners]
        candidate_surface = np.stack((np.asarray(first)[idx] + cand_radii[0] * candidate_normal,
                                      np.asarray(second)[idx] - cand_radii[1] * candidate_normal))
        candidate_point = np.mean(candidate_surface, axis=0)
        patch_records = []
        for slot in np.flatnonzero(active):
            patch_pair = reps[slot]
            patch_owner = owner_pairs[patch_pair]
            same_owner = bool(np.array_equal(owners, patch_owner))
            if not same_owner:
                continue
            patch_sections = sections_table[patch_pair]
            patch_radii = radius_table[patch_owner]
            same_pair = bool(np.array_equal(pair, patch_pair))
            patch_surface = np.stack((patch_points[slot, 0] + patch_radii[0] * patch_normals[slot],
                                      patch_points[slot, 1] - patch_radii[1] * patch_normals[slot]))
            starts = centres_np[patch_sections[:, 0]]
            directions = centres_np[patch_sections[:, 1]] - starts
            contained = bool(np.asarray(closed_footprint_contains(
                jnp.asarray(candidate_point), jnp.asarray(patch_surface),
                jnp.asarray(patch_normals[slot]), jnp.asarray(patch_radius[slot]),
                jnp.asarray(starts), jnp.asarray(directions), jnp.asarray(patch_radii))))
            rep_row = int(selected_representatives[slot])
            rw = np.asarray([[1 - np.asarray(wa)[rep_row], np.asarray(wa)[rep_row]],
                             [1 - np.asarray(wb)[rep_row], np.asarray(wb)[rep_row]]])
            damage_on_force_support = failed[patch_sections] & (rw > 0)
            patch_root = int(component_labels[rep_row])
            patch_records.append({'slot': int(slot), 'segments': patch_pair.tolist(),
                'owners': patch_owner.tolist(), 'normal_force_N': float(patch_force[slot]),
                'contact_radius_m': float(patch_radius[slot]), 'same_owner_as_failed_cell': same_owner,
                'same_pair_as_failed_cell': same_pair,
                'representative_coverage_pair_index': rep_row,
                'same_selected_contact_component_as_failed_cell': bool(
                    component_labels[idx] == patch_root and component_labels[idx] != regions['capacity']),
                'candidate_cell_point_inside_this_exact_closed_patch_footprint': contained,
                'live_patch_damaged_positive_weight_endpoint_flags': damage_on_force_support.tolist(),
                'live_patch_has_any_damaged_positive_weight_endpoint': bool(damage_on_force_support.any())})
        records.append({'coverage_pair_index': int(idx), 'coverage_cell': int(idx),
            'segments': pair.tolist(), 'owners': owners.tolist(), 'endpoint_sections': sections.tolist(),
            'closest_weights': [float(np.asarray(wa)[idx]), float(np.asarray(wb)[idx])],
            'failed_endpoint_flags': endpoint_failed.tolist(), 'positive_weight_failed_support': loaded_failed.tolist(),
            'loaded_failed_endpoint_exists': bool(loaded_failed.any()),
            'is_actual_live_patch_representative': bool(len(matches)),
            'representative_slots': matches.tolist(), 'same_segment_pair_live_patch_slots': same_pair_slots.tolist(),
            'actual_live_patches_same_owner_only': patch_records,
            'gap_m': float(np.asarray(distance)[idx] - cand_radii.sum()),
            'search_cell_surface_midpoint_m': candidate_point.tolist()})
    metrics_vec = np.asarray([np.sum(np.asarray(metrics.search_boundary)),
        metrics.uncovered_near_pairs, metrics.same_family_near_pairs,
        metrics.coverage_motion_exceeded, metrics.maximum_nonrigid_displacement,
        metrics.minimum_coverage_gap, np.sum(active), np.min(patch_force, initial=0.0),
        np.max(patch_force, initial=0.0), len(fail_indices), np.sum(np.asarray(metrics.unsupported_end_contact)),
        np.sum(np.asarray(metrics.degenerate_geometry)), np.linalg.norm(np.asarray(metrics.force_residual)),
        np.linalg.norm(np.asarray(metrics.torque_residual)), np.sum(np.asarray(metrics.potential)), metrics.valid], dtype=np.float64)
    record = {'mechanical_steps_executed': 0,
        'input_checkpoint_sha256': sha256(cp_path), 'checkpoint_time_s': float(state.time),
        'checkpoint_aggregate': payload['aggregate'].tolist(), 'checkpoint_last_contact': payload['last_contact'].tolist(),
        'recomputed_contact_metrics': metrics_vec.tolist(),
        'metrics_match_saved_last_contact_bitwise': bool(np.array_equal(metrics_vec, payload['last_contact'])),
        'current_kernel_valid': bool(metrics.valid), 'failed_cell_count': int(len(fail_indices)),
        'source_mpm_jax_fibre_patch_contact_sha256': sha256(ROOT / 'mpm_jax/fibre_patch_contact.py'),
        'source_commit': 'b2a07c2a6d1448c3a82254b2288f8fb03f46ccef',
        'failed_cells': records}
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
