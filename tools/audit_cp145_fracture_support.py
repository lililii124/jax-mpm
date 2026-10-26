"""Audit and serialize a zero-time conservative support split of accepted CP145.

No mechanical step is taken. The raw parent checkpoint and its source bridge
are verified and remain untouched.
"""
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT/'evidence/material-domain-contact/continue-cp136-midpoint-viscosity-to250'
CP = EVIDENCE/'data/result/accepted.npz'
RUN = EVIDENCE/'data/result/run.json'
FAILED = EVIDENCE/'data/result/first-fully-failed-point.json'
EXPECTED_CP_SHA = '918a8ba8dfe09cadaffccc5ca5e8ddc1151f1a7f8f1aba2abf6968b84434cb47'
PHYSICS_SOURCE_COMMIT = '2c2824d'
DX = .00025
SPACING = np.array([.00025, .000125, .00025])
OFFSETS = np.asarray(tuple(product(range(3), repeat=3)), dtype=np.int64)


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_cp_source():
    expected = hashlib.sha256(CP.read_bytes()).hexdigest()
    if expected != EXPECTED_CP_SHA:
        raise ValueError(f'CP145 SHA mismatch: {expected}')
    run = json.loads(RUN.read_text(encoding='utf-8'))
    receipt = json.loads(FAILED.read_text(encoding='utf-8'))
    if receipt['accepted_checkpoint_sha256'] != expected or int(receipt['step']) != 145:
        raise ValueError('First-failure receipt is not bound to raw accepted CP145')
    # Verify every source path recorded by the producer against the exact
    # source commit, not the subsequently edited worktree contact adapter.
    archive = subprocess.run(['git', 'archive', PHYSICS_SOURCE_COMMIT,
                              'mpm_jax', 'examples', 'tools'], cwd=ROOT,
                             check=True, stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tar:
        archived = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                    for m in tar.getmembers() if m.isfile()}
    archived['run_metal_tl_fresh.py'] = archived['tools/run_metal_tl_fresh.py']
    recorded = run['source_sha256']
    missing = sorted(name for name in recorded if name not in archived)
    changed = sorted(name for name, digest in recorded.items()
                     if archived.get(name) != digest)
    if missing or changed:
        raise ValueError(f'CP source receipt does not match {PHYSICS_SOURCE_COMMIT}: missing={missing[:3]} changed={changed[:3]}')
    manifest_hash = hashlib.sha256(json.dumps(recorded, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return run, receipt, expected, manifest_hash


def geometry_and_reference():
    # This factory is NumPy-only. Rebuild the exact reference stencil without
    # importing JAX, then bind every geometry and mass descriptor to run.json.
    from examples.metal_sphere_impact_fracture import plan_case
    geometry, plan = plan_case(speed=5000., sphere_diameter=.002,
        plate_size=(.02, .02), plate_thickness=.0005, gap=.00075,
        clamp_width=.001, dx=DX, spacing=tuple(SPACING), end_time=1e-6, frames=60)
    if len(geometry['x']) != 26336:
        raise ValueError('Formal C2 particle count changed')
    refs, offsets = [], []
    node_offset = 0
    for body in (0, 1):
        mask = geometry['body_id'] == body
        x = np.asarray(geometry['x'][mask], dtype=np.float64)
        lower = np.floor(x.min(axis=0)/DX)*DX-2*DX
        upper = np.ceil(x.max(axis=0)/DX)*DX+2*DX
        n = max(8, int(np.rint(((upper-lower)/DX).max())))
        q = (x-lower)/DX
        base = np.floor(q-.5).astype(np.int64)
        f = q-base
        w = np.stack((.5*(1.5-f)**2, .75-(f-1.)**2,
                      .5*(f-.5)**2), axis=1)
        nodes_ijk = base[:, None, :]+OFFSETS[None, :, :]
        inside = np.all((nodes_ijk >= 0)&(nodes_ijk <= n), axis=-1)
        axis_w = w[:, OFFSETS, np.arange(3)]
        weights = np.prod(axis_w, axis=-1)
        weights = np.where(inside, weights, 0.)
        dw = np.stack((f-1.5, 2*(1.-f), f-.5), axis=1)/DX
        grads = []
        for d in range(3):
            factors = axis_w.copy()
            factors[:, :, d] = dw[:, OFFSETS[:, d], d]
            grads.append(np.prod(factors, axis=-1))
        gradients = np.where(inside[..., None], np.stack(grads, axis=-1), 0.)
        strides = np.array([(n+1)**2, n+1, 1], dtype=np.int64)
        indices = np.sum(np.clip(nodes_ijk, 0, n)*strides, axis=-1)+node_offset
        grid_id = np.arange((n+1)**3, dtype=np.int64)
        nodes = np.stack(np.unravel_index(grid_id, (n+1, n+1, n+1)), axis=-1)*DX+lower
        nodes_mass = np.zeros(len(nodes))
        pvolume = np.asarray(geometry['volume'][mask], dtype=np.float64)
        pmass = 2700.*pvolume
        np.add.at(nodes_mass, (indices-node_offset).ravel(),
                  (pmass[:, None]*weights).ravel())
        refs.append(dict(x=x, indices=indices, weights=weights, gradients=gradients,
                         volume=pvolume, particle_mass=pmass, nodes=nodes+0.,
                         nodal_mass=nodes_mass, body=np.full(len(x), body, np.int32),
                         lower=lower, n=n))
        offsets.append((node_offset, node_offset+len(nodes)))
        node_offset += len(nodes)

    def cat(name): return np.concatenate([r[name] for r in refs], axis=0)
    indices, weights, gradients = cat('indices'), cat('weights'), cat('gradients')
    positions, volume, particle_mass, body_id = cat('x'), cat('volume'), cat('particle_mass'), cat('body')
    nodes = cat('nodes')
    nodal_mass = np.concatenate([r['nodal_mass'] for r in refs])
    if len(nodes) != 616322 or abs(float(particle_mass.sum())-0.0005513167968750003) > 2e-18:
        raise ValueError('Rebuilt C2 reference grid/mass differs from source-bound run')
    fixed = np.zeros((len(nodes), 3), dtype=bool)
    plate_start = offsets[1][0]
    plate_nodes = nodes[plate_start:]
    fixed[plate_start:] = ((np.abs(plate_nodes[:, 0]) >= .009) |
                           (np.abs(plate_nodes[:, 2]) >= .009))[:, None]
    return SimpleNamespace(positions=positions, indices=indices, weights=weights,
        gradients=gradients, volume=volume, particle_mass=particle_mass,
        nodal_mass=nodal_mass, nodes=nodes), body_id, fixed, refs


def cell_lattice(positions, body_id, geometry, plan):
    sphere_center = np.asarray([0., .00025+.00075+.001, 0.])
    sphere_min = sphere_center-.001
    plate_min = np.asarray([-.01, -.00025, -.01])
    ijk = np.empty((len(positions), 3), dtype=np.int64)
    for p, (x, body) in enumerate(zip(positions, body_id)):
        origin = sphere_min if body == 0 else plate_min
        ijk[p] = np.floor((x-origin)/SPACING+1e-10).astype(np.int64)
    return ijk, sphere_center, .001


def main():
    support_mod = load_local_module('_tl_fracture_support', ROOT/'mpm_jax/tl_fracture_support.py')
    contact_mod = load_local_module('_material_domain_contact', ROOT/'mpm_jax/material_domain_contact.py')
    run, failed_receipt, cp_sha, source_manifest_sha = verify_cp_source()
    reference, body_id, fixed, refs = geometry_and_reference()
    with np.load(CP, allow_pickle=False) as saved:
        cp = {name: saved[name].copy() for name in saved.files}
    if int(cp['step']) != 145 or abs(float(cp['time_s'])-1.741739886706902e-7) > 1e-24:
        raise ValueError('Raw checkpoint is not accepted step145')
    if cp['displacement'].shape != reference.nodes.shape or cp['velocity'].shape != reference.nodes.shape:
        raise ValueError('Checkpoint state does not match rebuilt original node mapping')
    material = {name: cp[name] for name in ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal')}
    failed = material['damage'] >= 1.
    if np.flatnonzero(failed).tolist() != failed_receipt['fully_failed_particle_ids']:
        raise ValueError('Failed point set differs from source-bound CP145 receipt')

    K = 2700.*5350.**2
    shear, rho, cv = 26e9, 2700., 900.
    J = np.linalg.det(material['F'])
    dev = np.sum(material['tau']**2, axis=(1, 2))/(4*shear*rho*np.maximum(1-material['damage'], 1e-12))
    volumetric = K*(1-J)**2/(2*rho)*np.where(J > 1., 1-material['damage'], 1.)
    thermal = material['internal']-material['cold']-material['fracture']-dev-volumetric
    energy_residual = material['internal']-(thermal+material['cold']+material['fracture']+dev+volumetric)
    pressure = K*(1-J)
    temperature = 300.+thermal/cv
    failed_ids = np.flatnonzero(failed)
    mass = reference.particle_mass
    particle_energy = dict(
        total_internal_J=float(np.dot(mass, material['internal'])),
        thermal_residual_J=float(np.dot(mass, thermal)),
        cold_work_J=float(np.dot(mass, material['cold'])),
        fracture_reservoir_J=float(np.dot(mass, material['fracture'])),
        deviatoric_elastic_J=float(np.dot(mass, dev)),
        volumetric_recoverable_J=float(np.dot(mass, volumetric)),
        max_decomposition_error_Jkg=float(np.max(np.abs(energy_residual), initial=0.)))
    failed_energy = []
    for p in failed_ids:
        failed_energy.append(dict(particle_id=int(p), body=int(body_id[p]), J=float(J[p]),
            damage=float(material['damage'][p]), tau_norm_Pa=float(np.linalg.norm(material['tau'][p])),
            dev_elastic_Jkg=float(dev[p]), volume_recoverable_Jkg=float(volumetric[p]),
            thermal_residual_Jkg=float(thermal[p]), cold_Jkg=float(material['cold'][p]),
            fracture_Jkg=float(material['fracture'][p]), internal_Jkg=float(material['internal'][p]),
            pressure_Pa=float(pressure[p]), temperature_K=float(temperature[p]),
            mass_kg=float(mass[p]), internal_J=float(mass[p]*material['internal'][p]),
            volume_recoverable_J=float(mass[p]*volumetric[p]),
            thermal_residual_J=float(mass[p]*thermal[p])))

    ijk, sphere_center, sphere_radius = cell_lattice(reference.positions, body_id, None, None)
    topo = support_mod.face_connected_components(ijk, body_id, failed,
        spacing=SPACING, sphere_center=sphere_center, sphere_radius=sphere_radius)
    component = topo['component_id']
    topology_counts = {}
    for item in topo['components']:
        key = f"body{item['body']}"
        topology_counts.setdefault(key, []).append(item['particle_count'])
    adapter = support_mod.split_reference_support(reference, cp['displacement'],
        cp['velocity'], fixed, component, body_id=body_id)
    adapter['body_id'] = body_id.copy()
    adapter['step'] = cp['step'].copy()
    adapter['time_s'] = cp['time_s'].copy()
    adapter['initial_P'] = cp['initial_P'].copy()
    adapter['initial_L'] = cp['initial_L'].copy()
    adapter['initial_K'] = cp['initial_K'].copy()
    adapter['viscous_loss'] = cp['viscous_loss'].copy()
    adapter['contact_work'] = cp['contact_work'].copy()
    adapter['state_fields'] = material
    # Parent and child complete material arrays must match byte-for-byte.
    for name, field in material.items():
        if not np.array_equal(field, cp[name]):
            raise FloatingPointError(f'Material history changed during zero-time split: {name}')
    U_before = float(np.dot(mass, material['internal']))
    heat_before = float(np.dot(mass, thermal))
    U_after = float(np.dot(adapter['particle_mass'], adapter['state_fields']['internal']))
    heat_after = float(np.dot(adapter['particle_mass'], thermal))
    if U_before != U_after or heat_before != heat_after:
        raise FloatingPointError('Zero-time split changed particle internal or thermal energy')

    support_cp = EVIDENCE/'data/result/cp145-fracture-support-adapter.npz'
    metadata = support_mod.save_support_checkpoint(support_cp, cp_sha,
        source_manifest_sha, adapter)
    support_mod.validate_support_checkpoint(support_cp,
        parent_checkpoint_sha256=cp_sha, source_manifest_sha256=source_manifest_sha,
        reference=reference, component_id=component)
    support_sha = hashlib.sha256(support_cp.read_bytes()).hexdigest()

    xcurrent = reference.positions + np.sum(reference.weights[..., None]
        *cp['displacement'][reference.indices], axis=1)
    candidates = contact_mod.material_domain_candidates(xcurrent, material['F'],
        body_id, SPACING, max_pairs=8192, component_id=component)
    same_body_rows = []
    cross_body_count = 0
    for hit in candidates:
        if body_id[hit.first] != body_id[hit.second]:
            cross_body_count += 1
            continue
        ca = contact_mod.point_velocity_coefficients(reference.weights[hit.first],
            reference.gradients[hit.first], hit.xi_first)
        cb = contact_mod.point_velocity_coefficients(reference.weights[hit.second],
            reference.gradients[hit.second], hit.xi_second)
        va = np.sum(ca[:, None]*cp['velocity'][reference.indices[hit.first]], axis=0)
        vb = np.sum(cb[:, None]*cp['velocity'][reference.indices[hit.second]], axis=0)
        same_body_rows.append(dict(first=hit.first, second=hit.second,
            component_first=int(component[hit.first]), component_second=int(component[hit.second]),
            penetration_m=hit.penetration, closing_m_s=float(np.dot(hit.normal, va-vb)),
            point_m=hit.point.tolist(), normal=hit.normal.tolist()))

    split_reference = SimpleNamespace(positions=adapter['positions'],
        indices=adapter['indices'], weights=adapter['weights'],
        gradients=adapter['gradients'], volume=adapter['volume'],
        particle_mass=adapter['particle_mass'], nodal_mass=adapter['nodal_mass'],
        nodes=adapter['nodes'])
    vpost, contact_impulse, contact_work, contact_loss, contact_report = (
        contact_mod.material_domain_contact_jump(adapter['velocity'], split_reference,
            adapter['particle_mass'], body_id, xcurrent, material['F'], SPACING,
            adapter['fixed'], component_id=component))
    q_split = adapter['nodes']+adapter['displacement']
    p_pre = np.sum(adapter['nodal_mass'][:, None]*adapter['velocity'], axis=0)
    p_post = np.sum(adapter['nodal_mass'][:, None]*vpost, axis=0)
    l_pre = np.sum(np.cross(q_split, adapter['nodal_mass'][:, None]*adapter['velocity']), axis=0)
    l_post = np.sum(np.cross(q_split, adapter['nodal_mass'][:, None]*vpost), axis=0)
    k_pre = .5*np.sum(adapter['nodal_mass'][:, None]*adapter['velocity']**2)
    k_post = .5*np.sum(adapter['nodal_mass'][:, None]*vpost**2)

    receipt = dict(schema='c2-cp145-zero-time-support-split-audit-v1', status='passed',
        parent_checkpoint_sha256=cp_sha, parent_source_commit=PHYSICS_SOURCE_COMMIT,
        parent_source_manifest_sha256=source_manifest_sha,
        support_checkpoint='cp145-fracture-support-adapter.npz',
        support_checkpoint_sha256=support_sha, adapter_metadata=metadata,
        step=int(cp['step']), time_s=float(cp['time_s']),
        original_nodes=int(len(reference.nodes)), split_nodes=int(len(adapter['nodes'])),
        particle_count=int(len(body_id)), failed_ids=failed_ids.tolist(),
        failed_point_energy=failed_energy, material_energy_ledger=particle_energy,
        pressure_convention='positive compression; current failed J<1 keeps EOS compression energy and K(1-J) pressure',
        failed_material_interpretation='all four points have exactly zero deviatoric stress and zero damaged tensile-volume energy; compression-volume storage and positive pressure remain. Existing internal/cold/fracture/elastic/volume decomposition closes without adding another release.',
        support_topology=dict(reference_face_edge_count=topo['reference_face_edge_count'],
            severed_failed_incident_edge_count=topo['severed_failed_incident_edge_count'],
            component_count=len(topo['components']), component_particle_counts=topology_counts,
            failed_face_neighbors=topo['failed_face_neighbor_ids'],
            failed_point_remains_mass_bearing=True),
        zero_time_conservation=dict(
            mass_before_kg=adapter['before_mass_kg'], mass_after_kg=adapter['after_mass_kg'],
            momentum_before_Ns=adapter['before_momentum'].tolist(),
            momentum_after_Ns=adapter['after_momentum'].tolist(),
            angular_before_Nms=adapter['before_angular_momentum'].tolist(),
            angular_after_Nms=adapter['after_angular_momentum'].tolist(),
            K_before_J=adapter['before_kinetic_J'], K_after_J=adapter['after_kinetic_J'],
            U_before_J=U_before, U_after_J=U_after,
            thermal_residual_before_J=heat_before, thermal_residual_after_J=heat_after,
            F_and_all_history_byte_identical=True,
            max_particle_position_change_m=adapter['max_position_change_m'],
            max_particle_F_change=adapter['max_F_change'],
            fixed_component_flags_copied=True),
        same_body_component_SAT=dict(pair_count=len(same_body_rows),
            cross_body_candidate_count=cross_body_count,
            maximum_penetration_m=max((r['penetration_m'] for r in same_body_rows), default=0.),
            max_positive_closing_m_s=max((r['closing_m_s'] for r in same_body_rows), default=0.),
            rows=same_body_rows),
        zero_time_e0_projection=dict(valid=bool(contact_report['valid']),
            candidates=int(contact_report['candidate_pairs']),
            positive_multiplier_pairs=int(contact_report['positive_multiplier_pairs']),
            max_closing_before_m_s=float(contact_report['initial_closing_m_s']),
            physical_route_closing_after_m_s=float(contact_report['physical_route_post_closing_m_s']),
            impulse_norm_Ns=float(contact_report['nodal_impulse_norm_Ns']),
            contact_work_J=float(contact_work), kinetic_loss_J=float(contact_loss),
            K_before_J=float(k_pre), K_after_J=float(k_post),
            P_error_Ns=float(np.linalg.norm(p_post-p_pre)),
            L_error_Nms=float(np.linalg.norm(l_post-l_pre)),
            max_route_power_identity_error_J=float(max(
                contact_report['dot_v_Atlambda_vs_Av_lambda_pre_abs_error'],
                contact_report['dot_v_Atlambda_vs_Av_lambda_post_abs_error']))),
        production_continuation='not started until the first accepted split event, topology receipt, and contact-energy bridge are source-bound')
    out = EVIDENCE/'data/result/cp145-fracture-support-audit.json'
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'step', 'time_s',
        'original_nodes', 'split_nodes', 'failed_ids', 'material_energy_ledger',
        'support_topology', 'zero_time_conservation', 'same_body_component_SAT')},
        indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
