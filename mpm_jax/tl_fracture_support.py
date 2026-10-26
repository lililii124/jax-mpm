"""Conservative, serializable TL nodal-support duplication at cell failures.

This module only changes the incidence map. It never edits particle fields,
history, material strength, or accepted parent checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def hash_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def reference_fingerprint(reference):
    digest = hashlib.sha256()
    # Hash exact geometry and incidence. JAX and NumPy scatter arithmetic can
    # differ by a few ulps in shape values/mass sums, so those are compared
    # below with the original 64-epsilon kinematic tolerance instead of hashed.
    for name in ('positions', 'indices', 'volume', 'particle_mass', 'nodes'):
        digest.update(name.encode())
        digest.update(hash_array(getattr(reference, name)).encode())
    return digest.hexdigest()


def face_connected_components(cell_ijk, body_id, failed, *, spacing,
                               sphere_center=None, sphere_radius=None):
    """Particle-cell components over exact face links, cutting fully_failed cells.

    Sphere voxel face links are retained only where the analytic sphere
    intersects the shared face with positive area. The plate is a full solid
    voxel block. The routine returns global component IDs and an edge receipt.
    """
    cell_ijk = np.asarray(cell_ijk, dtype=np.int64)
    body_id = np.asarray(body_id, dtype=np.int32)
    failed = np.asarray(failed, dtype=bool)
    spacing = np.asarray(spacing, dtype=np.float64)
    if cell_ijk.shape != (len(body_id), 3) or failed.shape != body_id.shape:
        raise ValueError('Cell lattice, body, and failure arrays disagree')
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError('Positive material cell spacings required')
    if len(np.unique(np.c_[body_id, cell_ijk], axis=0)) != len(body_id):
        raise ValueError('Expected exactly one material point per occupied reference voxel')
    if np.any(~np.isin(body_id, (0, 1))):
        raise ValueError('Unexpected body label')
    if body_id.tolist().count(0) and (sphere_center is None or sphere_radius is None):
        raise ValueError('Sphere geometry is required for body-0 face links')
    lookup = {(int(body_id[p]), *map(int, cell_ijk[p])): p
              for p in range(len(body_id))}
    adjacency = [[] for _ in range(len(body_id))]
    all_neighbors = [[] for _ in range(len(body_id))]
    edges = []
    sphere_center = None if sphere_center is None else np.asarray(sphere_center, float)
    for p, (body, i, j, k) in enumerate(np.c_[body_id, cell_ijk]):
        base = (int(i), int(j), int(k))
        for axis in range(3):
            neighbor = list(base)
            neighbor[axis] += 1
            other = lookup.get((int(body), *neighbor))
            if other is None:
                continue
            if body == 0:
                # Sphere voxel indices are centered on the analytic sphere;
                # these coordinates are already relative to its center.
                face = -float(sphere_radius)+neighbor[axis]*spacing[axis]
                normal_distance = face
                tangent2 = 0.
                for tangent in range(3):
                    if tangent == axis:
                        continue
                    lo = -float(sphere_radius)+base[tangent]*spacing[tangent]
                    hi = lo+spacing[tangent]
                    nearest = 0. if lo <= 0. <= hi else min(abs(lo), abs(hi))
                    tangent2 += nearest*nearest
                face_intersection = normal_distance**2+tangent2 < float(sphere_radius)**2
                if not face_intersection:
                    continue
            edges.append((p, other))
            all_neighbors[p].append(other)
            all_neighbors[other].append(p)
            if not failed[p] and not failed[other]:
                adjacency[p].append(other)
                adjacency[other].append(p)
    component = np.full(len(body_id), -1, dtype=np.int64)
    next_id = 0
    components = []
    for seed in range(len(body_id)):
        if failed[seed] or component[seed] >= 0:
            continue
        stack = [seed]
        component[seed] = next_id
        group = []
        while stack:
            p = stack.pop()
            group.append(p)
            for neighbor in adjacency[p]:
                if component[neighbor] < 0:
                    component[neighbor] = next_id
                    stack.append(neighbor)
        components.append(dict(component_id=next_id, body=int(body_id[seed]),
                               particle_count=len(group), particle_ids=group))
        next_id += 1
    for p in np.flatnonzero(failed):
        component[p] = next_id
        components.append(dict(component_id=next_id, body=int(body_id[p]),
                               particle_count=1, particle_ids=[int(p)],
                               fully_failed_singleton=True))
        next_id += 1
    if np.any(component < 0):
        raise FloatingPointError('Component extraction omitted material points')
    return dict(component_id=component, components=components,
        reference_face_edge_count=len(edges),
        severed_failed_incident_edge_count=sum(bool(failed[a] or failed[b]) for a, b in edges),
        failed_ids=np.flatnonzero(failed).tolist(),
        failed_neighbor_ids={str(int(p)): [int(q) for q in adjacency[p]]
                             for p in np.flatnonzero(failed)},
        failed_face_neighbor_ids={str(int(p)): [int(q) for q in all_neighbors[p]]
                             for p in np.flatnonzero(failed)})


def split_reference_support(reference, displacement, velocity, fixed,
                            component_id, body_id=None):
    """Copy each old node per component and remap particle incidence.

    `reference` is the ReferenceGrid-like NamedTuple used by the solver.
    The returned arrays can rebuild a new ReferenceGrid. Empty unused nodes
    are omitted; positive-mass nodes and every particle support are retained.
    """
    indices = np.asarray(reference.indices, dtype=np.int64)
    weights = np.asarray(reference.weights, dtype=np.float64)
    gradients = np.asarray(reference.gradients, dtype=np.float64)
    particle_mass = np.asarray(reference.particle_mass, dtype=np.float64)
    nodal_mass = np.asarray(reference.nodal_mass, dtype=np.float64)
    nodes = np.asarray(reference.nodes, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=bool)
    component_id = np.asarray(component_id, dtype=np.int64)
    body_id = None if body_id is None else np.asarray(body_id, dtype=np.int32)
    if indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError('Reference indices and weights must share [particle,support] shape')
    if (gradients.shape != indices.shape+(3,) or len(component_id) != len(indices)
            or (body_id is not None and body_id.shape != component_id.shape)):
        raise ValueError('Reference gradients or particle component labels have wrong shape')
    if nodes.ndim != 2 or nodes.shape[1] != 3 or displacement.shape != nodes.shape or velocity.shape != nodes.shape:
        raise ValueError('Node geometry/state shapes disagree')
    if fixed.shape != nodes.shape or nodal_mass.shape != (len(nodes),):
        raise ValueError('Node fixed/mass shapes disagree')
    if np.any(indices < 0) or np.any(indices >= len(nodes)) or np.any(component_id < 0):
        raise ValueError('Invalid old node index or negative component label')
    if not np.isfinite(np.r_[weights.ravel(), gradients.ravel(), particle_mass,
                             nodal_mass, nodes.ravel(), displacement.ravel(),
                             velocity.ravel()]).all():
        raise FloatingPointError('Nonfinite reference or state before support duplication')

    keys = sorted({(int(component_id[p]), int(node))
                   for p in range(len(indices)) for node in indices[p]})
    lookup = {key: row for row, key in enumerate(keys)}
    new_indices = np.asarray([[lookup[(int(component_id[p]), int(node))]
                               for node in indices[p]]
                              for p in range(len(indices))], dtype=np.int32)
    old_node = np.asarray([node for _, node in keys], dtype=np.int64)
    new_component = np.asarray([component for component, _ in keys], dtype=np.int64)
    if body_id is None:
        new_body = np.full(len(keys), -1, dtype=np.int32)
    else:
        component_body = {}
        for p, comp in enumerate(component_id):
            value = int(body_id[p])
            if int(comp) in component_body and component_body[int(comp)] != value:
                raise ValueError('One support component spans multiple physical bodies')
            component_body[int(comp)] = value
        new_body = np.asarray([component_body[int(comp)] for comp, _ in keys], dtype=np.int32)
    new_nodes = nodes[old_node].copy()
    new_q = (nodes+displacement)[old_node]
    new_displacement = new_q-new_nodes
    new_velocity = velocity[old_node].copy()
    new_fixed = fixed[old_node].copy()
    new_mass = np.zeros(len(keys), dtype=np.float64)
    np.add.at(new_mass, new_indices.ravel(),
              (particle_mass[:, None]*weights).ravel())
    if np.any(new_mass < 0) or not np.isfinite(new_mass).all():
        raise FloatingPointError('Split support has negative or nonfinite nodal mass')

    old_q = nodes+displacement
    sample_before = np.sum(weights[..., None]*old_q[indices], axis=1)
    sample_after = np.sum(weights[..., None]*new_q[new_indices], axis=1)
    F_before = np.eye(3)+np.einsum('pki,pkj->pij', displacement[indices], gradients)
    F_after = np.eye(3)+np.einsum('pki,pkj->pij',
                                   new_displacement[new_indices], gradients)
    eps = np.finfo(np.float64).eps
    position_scale = max(float(np.max(np.abs(sample_before), initial=0.)), 1.)
    F_scale = max(float(np.max(np.abs(F_before), initial=0.)), 1.)
    if np.max(np.abs(sample_after-sample_before), initial=0.) > 64*eps*position_scale:
        raise FloatingPointError('Support duplication changed a particle position')
    if np.max(np.abs(F_after-F_before), initial=0.) > 64*eps*F_scale:
        raise FloatingPointError('Support duplication changed particle F')

    def ledger(q, v, mass):
        P = np.sum(mass[:, None]*v, axis=0)
        L = np.sum(np.cross(q, mass[:, None]*v), axis=0)
        K = .5*np.sum(mass[:, None]*v*v)
        return P, L, float(K)

    before = ledger(old_q, velocity, nodal_mass)
    after = ledger(new_q, new_velocity, new_mass)
    mass_before, mass_after = float(nodal_mass.sum()), float(new_mass.sum())
    # Use the run's original global conservation admission (1e-10). Grouped
    # scatter summation order changes under duplication, so a machine-epsilon
    # comparison would reject a mathematically identical mass partition.
    tol = 1e-10
    if abs(mass_after-mass_before) > tol*max(mass_before, 1e-300):
        raise FloatingPointError('Support duplication changed total nodal mass')
    if np.linalg.norm(after[0]-before[0]) > tol*max(np.linalg.norm(before[0]), 1e-300):
        recovered_mass = np.bincount(old_node, weights=new_mass, minlength=len(nodes))
        raise FloatingPointError(f'Support duplication changed nodal momentum: before={before[0]} after={after[0]} mass_error={np.max(np.abs(recovered_mass-nodal_mass))} total_mass_error={recovered_mass.sum()-nodal_mass.sum()} components={np.unique(component_id).size}')
    angular_scale = max(mass_before*2.5e-4*5000., 1e-300)
    if np.linalg.norm(after[1]-before[1]) > tol*angular_scale:
        raise FloatingPointError('Support duplication changed nodal angular momentum')
    if abs(after[2]-before[2]) > tol*max(abs(before[2]), 1e-300):
        raise FloatingPointError('Support duplication changed nodal kinetic energy')

    return dict(indices=new_indices, nodes=new_nodes, displacement=new_displacement,
        velocity=new_velocity, nodal_mass=new_mass, fixed=new_fixed,
        old_node_for_new=old_node, component_for_new=new_component, node_body=new_body,
        particle_component=component_id.copy(),
        positions=np.asarray(reference.positions).copy(), weights=weights.copy(),
        gradients=gradients.copy(), volume=np.asarray(reference.volume).copy(),
        particle_mass=particle_mass.copy(),
        old_reference_fingerprint=reference_fingerprint(reference),
        before_mass_kg=mass_before, after_mass_kg=mass_after,
        before_momentum=before[0], after_momentum=after[0],
        before_angular_momentum=before[1], after_angular_momentum=after[1],
        before_kinetic_J=before[2], after_kinetic_J=after[2],
        max_position_change_m=float(np.max(np.abs(sample_after-sample_before), initial=0.)),
        max_F_change=float(np.max(np.abs(F_after-F_before), initial=0.)),
        support_hash=hash_array(new_indices),
        topology_hash=hash_array(component_id))


def save_support_checkpoint(path, parent_checkpoint_sha256,
                            source_manifest_sha256, adapter):
    """Atomically serialize duplicated support plus the complete particle state.

    `state_fields` must contain exactly F,tau,ep,cold,fracture,damage,internal;
    they are copied byte-for-byte and are not recomputed or reset.
    """
    path = Path(path)
    state_fields = adapter.get('state_fields')
    if not isinstance(state_fields, dict) or set(state_fields) != {
            'F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'}:
        raise ValueError('Complete seven-field material state is required')
    arrays = {k: np.asarray(v) for k, v in adapter.items()
              if isinstance(v, np.ndarray)}
    arrays.update({f'material_{k}': np.asarray(v) for k, v in state_fields.items()})
    arrays.update({k: np.asarray(v) for k, v in state_fields.items()})
    metadata = dict(schema='tl-fracture-support-checkpoint-v1',
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        source_manifest_sha256=source_manifest_sha256,
        support_hash=adapter['support_hash'], topology_hash=adapter['topology_hash'],
        old_reference_fingerprint=adapter['old_reference_fingerprint'],
        state_field_hashes={k: hash_array(v) for k, v in state_fields.items()})
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return metadata


def validate_support_checkpoint(path, *, parent_checkpoint_sha256,
                                source_manifest_sha256, reference,
                                component_id):
    """Fail closed if checkpoint ancestry, support, topology, or history differs."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        if metadata.get('schema') != 'tl-fracture-support-checkpoint-v1':
            raise ValueError('Unrecognized support checkpoint schema')
        if metadata.get('parent_checkpoint_sha256') != parent_checkpoint_sha256:
            raise ValueError('Support checkpoint parent hash does not match requested CP')
        if metadata.get('source_manifest_sha256') != source_manifest_sha256:
            raise ValueError('Support checkpoint source manifest hash mismatch')
        if metadata.get('old_reference_fingerprint') != reference_fingerprint(reference):
            raise ValueError('Support checkpoint belongs to a different original reference map')
        if metadata.get('support_hash') != hash_array(saved['indices']):
            raise ValueError('Support incidence hash mismatch')
        if metadata.get('topology_hash') != hash_array(saved['particle_component']):
            raise ValueError('Support topology hash mismatch')
        if not np.array_equal(saved['particle_component'], np.asarray(component_id)):
            raise ValueError('Support checkpoint component map differs from audited CP topology')
        if not np.array_equal(saved['positions'], np.asarray(reference.positions)):
            raise ValueError('Support checkpoint particle reference positions mismatch')
        for name in ('weights', 'gradients'):
            actual, expected = np.asarray(saved[name]), np.asarray(getattr(reference, name))
            scale = max(float(np.max(np.abs(expected), initial=0.)), 1.)
            if actual.shape != expected.shape or np.max(np.abs(actual-expected), initial=0.) > 64*np.finfo(float).eps*scale:
                raise ValueError(f'Support checkpoint reference {name} differs beyond 64eps')
        for name, digest in metadata['state_field_hashes'].items():
            if hash_array(saved[f'material_{name}']) != digest:
                raise ValueError(f'Material history field {name} hash mismatch')
        return metadata
