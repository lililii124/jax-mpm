"""Run a fresh-t0 C2 metal TL midpoint window with atomic accepted steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path, **values):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        writer=np.savez if path.name=='accepted.npz' else np.savez_compressed
        writer(stream, **values)
    os.replace(temp, path)



def canonical_ancestry_reference(reference, geometry):
    """Reproduce saved NumPy quadratic basis; keep strict ancestry tolerance."""
    import jax.numpy as jnp
    offsets=np.asarray([(i,j,k) for i in range(3) for j in range(3) for k in range(3)],dtype=np.int32)
    weights=[];gradients=[];h=.00025
    for bid in (0,1):
        xx=np.asarray(geometry['x'][geometry['body_id']==bid])
        lower=np.floor(xx.min(axis=0)/h)*h-2*h
        q=(xx-lower)/h;base=np.floor(q-.5).astype(np.int32);f=q-base
        w=np.stack((.5*(1.5-f)**2,.75-(f-1.)**2,.5*(f-.5)**2),axis=1)
        aw=w[:,offsets,np.arange(3)]
        weights.append(np.prod(aw,axis=-1))
        dw=np.stack((f-1.5,2*(1-f),f-.5),axis=1)/h
        gg=[]
        for d in range(3):
            factors=aw.copy();factors[:,:,d]=dw[:,offsets[:,d],d]
            gg.append(np.prod(factors,axis=-1))
        gradients.append(np.stack(gg,axis=-1))
    return reference._replace(weights=jnp.asarray(np.concatenate(weights)),gradients=jnp.asarray(np.concatenate(gradients)))


def pad_nodal_storage(adapter):
    """Append unused zero-mass nodal storage to stabilize JIT shapes.

    No particle, support incidence, material state or physical mass changes.
    Padding copies an existing clamped node's finite coordinates/state solely
    to preserve exact parent-map ancestry; no particle references these rows.
    """
    real_nodes=len(adapter['nodal_mass'])
    capacity=1<<(max(real_nodes,65536)-1).bit_length()
    if capacity==real_nodes:
        adapter.update(storage_connected_nodes=np.asarray(len(np.unique(adapter['indices'])),dtype=np.int64),
            storage_node_capacity=np.asarray(capacity,dtype=np.int64))
        return adapter
    if np.max(adapter['indices'],initial=-1)>=real_nodes:
        raise ValueError('Physical support exceeds unpadded nodal storage')
    clamped=np.flatnonzero(np.all(adapter['fixed'],axis=1))
    if not len(clamped):raise ValueError('Storage padding requires an existing fixed node')
    copy=int(clamped[0]);extra=capacity-real_nodes
    def ledger():
        mass=adapter['nodal_mass'];v=adapter['velocity'];q=adapter['nodes']+adapter['displacement']
        return float(mass.sum()),np.sum(mass[:,None]*v,axis=0),np.sum(np.cross(q,mass[:,None]*v),axis=0),float(.5*np.sum(mass[:,None]*v*v))
    before=ledger()
    for name in ['nodes','displacement','velocity','fixed','component_for_new','node_body','old_node_for_new','initial_node_velocity']:
        if name in adapter:
            values=np.asarray(adapter[name])
            if len(values)!=real_nodes:raise ValueError('Inconsistent nodal storage field '+name)
            adapter[name]=np.concatenate((values,np.repeat(values[copy:copy+1],extra,axis=0)))
    adapter['nodal_mass']=np.concatenate((adapter['nodal_mass'],np.zeros(extra,dtype=np.float64)))
    after=ledger()
    tol=1e-10
    if (abs(after[0]-before[0])>tol*max(before[0],1e-300)
        or np.linalg.norm(after[1]-before[1])>tol*max(np.linalg.norm(before[1]),1e-300)
        or np.linalg.norm(after[2]-before[2])>tol*max(before[0]*2.5e-4*5000.,1e-300)
        or abs(after[3]-before[3])>tol*max(abs(before[3]),1e-300)):
        raise FloatingPointError('Unused zero-mass storage changed physical ledger')
    adapter.update(storage_connected_nodes=np.asarray(real_nodes,dtype=np.int64),
        storage_node_capacity=np.asarray(capacity,dtype=np.int64),
        storage_padding_unused=np.asarray(True),
        storage_padding_kinetic_roundoff_J=np.asarray(after[3]-before[3]))
    if 'after_kinetic_J' in adapter:
        adapter.update(after_mass_kg=after[0],after_momentum=after[1],after_angular_momentum=after[2],after_kinetic_J=after[3])
    return adapter


def calorimetric_eos(J,internal,cold,fracture,elastic,damage,material):
    """Project mechanical stores to the linear EOS energy convention.

    Actual calorimetry is only accumulated plastic heat. The stored damaged
    volume potential is removed before adding the EOS recoverable volume
    energy; neither this projection nor damage supplies heat or thermal
    pressure. Particle/internal state arrays are never edited here.
    """
    from mpm_jax.fragment_multifield_v6_mpm import damage_volume_energy
    from mpm_jax.materials.linear_metal import eos_state,volume_energy
    heat=internal-cold-fracture-elastic-damage_volume_energy(J,damage,material)
    return eos_state(J,heat+volume_energy(J,material),material)

def make_reference(geometry, dx):
    """Build two disjoint bodywise reference lattices and compact their IDs."""
    import jax
    import jax.numpy as jnp
    from mpm_jax.core import Config
    from mpm_jax.total_lagrangian import ReferenceGrid, reference_grid

    refs = []
    offsets = []
    node_offset = 0
    for body in (0, 1):
        take = geometry['body_id'] == body
        x = np.asarray(geometry['x'][take], dtype=np.float64)
        lower = np.floor(x.min(axis=0)/dx)*dx - 2*dx
        upper = np.ceil(x.max(axis=0)/dx)*dx + 2*dx
        ngrid = np.maximum(8, np.rint((upper-lower)/dx).astype(int))
        domain = float(ngrid.max()*dx)
        # Config is cubic; extend only the storage envelope. Particle and
        # shape locations remain their exact reference coordinates.
        counts = np.ceil((upper-lower)/dx).astype(int)
        n = int(counts.max())
        cfg = Config(n_grid=max(8, n), dt=1e-12, domain_size=max(8, n)*dx,
                     dim=3, boundary_cells=0, material='elastic', failure_mode='retain')
        local = reference_grid(jnp.asarray(x-lower), jnp.asarray(geometry['volume'][take]),
                               2700., cfg)
        indices = np.asarray(local.indices) + node_offset
        nodes = np.asarray(local.nodes) + lower
        refs.append(dict(positions=x, indices=indices, weights=np.asarray(local.weights),
            gradients=np.asarray(local.gradients), volume=np.asarray(local.volume),
            particle_mass=np.asarray(local.particle_mass), nodal_mass=np.asarray(local.nodal_mass),
            nodes=nodes, valid=bool(np.asarray(local.valid)), body=np.full(len(x), body, np.int32),
            lower=lower.tolist(), n_grid=cfg.n_grid))
        offsets.append((node_offset, node_offset+len(nodes)))
        node_offset += len(nodes)
    if not all(r['valid'] for r in refs):
        raise FloatingPointError('Initial reference shape/gradient, mass, or domain admission failed')
    positions = np.concatenate([r['positions'] for r in refs])
    indices = np.concatenate([r['indices'] for r in refs])
    weights = np.concatenate([r['weights'] for r in refs])
    gradients = np.concatenate([r['gradients'] for r in refs])
    volume = np.concatenate([r['volume'] for r in refs])
    particle_mass = np.concatenate([r['particle_mass'] for r in refs])
    nodal_mass = np.concatenate([r['nodal_mass'] for r in refs])
    nodes = np.concatenate([r['nodes'] for r in refs])
    body = np.concatenate([r['body'] for r in refs])
    reference = ReferenceGrid(jnp.asarray(positions), jnp.asarray(indices), jnp.asarray(weights),
        jnp.asarray(gradients), jnp.asarray(volume), jnp.asarray(particle_mass),
        jnp.asarray(nodal_mass), jnp.asarray(nodes), jnp.asarray(True))
    fixed = np.zeros((len(nodes), 3), dtype=bool)
    plate_start = offsets[1][0]
    plate_nodes = nodes[plate_start:]
    fixed[plate_start:] = ((np.abs(plate_nodes[:, 0]) >= .009) |
                           (np.abs(plate_nodes[:, 2]) >= .009))[:, None]
    nodal_velocity = np.zeros((len(nodes), 3), dtype=np.float64)
    sphere_start, sphere_end = offsets[0]
    sphere_mass = np.asarray(nodal_mass[sphere_start:sphere_end])
    nodal_velocity[sphere_start:sphere_end, 1] = np.where(sphere_mass > 0, -5000., 0.)
    return reference, body, fixed, nodal_velocity, dict(
        bodies=[dict(body=b, lower=refs[b]['lower'], n_grid=refs[b]['n_grid'],
                     nodes=offsets[b][1]-offsets[b][0]) for b in (0, 1)],
        particle_mass_kg=float(particle_mass.sum()), nodal_mass_kg=float(nodal_mass.sum()),
        fixed_nodes=int(np.sum(fixed[:, 0])), node_count=len(nodes))


def verify_adjoint_power_identity():
    """Check finite-dimensional A/A^T power identity for identity and full E0."""
    import jax.numpy as jnp
    from mpm_jax.total_lagrangian import ReferenceGrid
    from mpm_jax.metal_tl_midpoint import tl_velocity_map, tl_velocity_map_transpose
    rng = np.random.default_rng(20261026)
    particles, nodes = 7, 11
    indices = rng.integers(0, nodes, size=(particles, 4), dtype=np.int32)
    weights = rng.normal(size=(particles, 4))
    gradients = rng.normal(size=(particles, 4, 3))
    mass = rng.uniform(.2, 2., size=particles)
    reference = ReferenceGrid(jnp.zeros((particles, 3)), jnp.asarray(indices),
        jnp.asarray(weights), jnp.asarray(gradients), jnp.ones((particles,)),
        jnp.asarray(mass), jnp.ones((nodes,)), jnp.zeros((nodes, 3)), jnp.asarray(True))
    velocity = jnp.asarray(rng.normal(size=(nodes, 3)))
    phase_impulse = jnp.asarray(rng.normal(size=(particles, 4, 3)))
    cases = {'identity': jnp.eye(3), 'nonsymmetric': jnp.asarray(
        [[1., .2, -.1], [.3, .8, .05], [-.2, .4, 1.1]])}
    receipt = {}
    for name, E0 in cases.items():
        Av = tl_velocity_map(velocity, reference, jnp.asarray(mass), E0)
        Aty = tl_velocity_map_transpose(phase_impulse, reference,
                                        jnp.asarray(mass), E0, nodes)
        lhs, rhs = float(jnp.vdot(Av, phase_impulse)), float(jnp.vdot(velocity, Aty))
        error = abs(lhs-rhs)
        receipt[name] = {'dot_Av_y': lhs, 'dot_v_Aty': rhs,
                         'absolute_error': error,
                         'relative_error': error/max(abs(lhs), abs(rhs), 1e-30)}
        if error > 2e-12*max(1., abs(lhs), abs(rhs)):
            raise FloatingPointError(f'{name} A/A^T power identity failed: {error:g}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--end-time', type=float, default=160e-9)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-bridge', type=Path)
    parser.add_argument('--resume-parent-checkpoint', type=Path,
                        help='Raw accepted parent for a duplicated-support checkpoint')
    parser.add_argument('--resume-support-audit', type=Path,
                        help='Source-bound zero-time support-split audit receipt')
    parser.add_argument('--audit-only', action='store_true',
                        help='Write read-only physical/material metrics for a source-bound checkpoint')
    args = parser.parse_args()
    if not 150e-9 <= args.end_time <= 1e-6 or not 1 <= args.max_seconds <= 3500:
        parser.error('The bounded contact/fracture-review window is limited to 150 ns to 1 us and 3500 s')
    if bool(args.resume_checkpoint) != bool(args.resume_bridge):
        parser.error('--resume-checkpoint and --resume-bridge must be supplied together')
    if bool(args.resume_parent_checkpoint) != bool(args.resume_support_audit):
        parser.error('--resume-parent-checkpoint and --resume-support-audit must be supplied together')
    if args.resume_parent_checkpoint is not None and args.resume_checkpoint is None:
        parser.error('A support parent requires --resume-checkpoint')
    if args.audit_only and args.resume_checkpoint is None:
        parser.error('--audit-only requires a source-bound accepted checkpoint')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Fresh-t0 TL output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    adjoint_power_check = verify_adjoint_power_identity()
    from examples.metal_sphere_impact_fracture import plan_case
    from mpm_jax.fragment_v6_discrete_work import update as material_update
    from mpm_jax.materials.linear_metal import linear_metal, volume_energy, eos_state
    from mpm_jax.metal_tl_midpoint import (
        initial_state, step, endpoint_e0_jump)
    from mpm_jax.total_lagrangian import ReferenceGrid
    from mpm_jax.material_domain_contact import (
        material_domain_contact_jump, minimum_material_domain_gap,
        _world_corners)

    geometry, plan = plan_case(speed=5000., sphere_diameter=.002, plate_size=(.02, .02),
        plate_thickness=.0005, gap=.00075, clamp_width=.001, dx=.00025,
        spacing=(.00025, .000125, .00025), end_time=1e-6, frames=60)
    if len(geometry['x']) != 26336 or plan['material']['density'] != 2700.:
        raise ValueError('The original C2 geometry or density changed')
    sphere_center = np.asarray((0., .00025+.00075+.001, 0.))
    cell_origins = np.where((np.asarray(geometry['body_id']) == 0)[:, None],
        sphere_center-.001, np.asarray((-.01, -.00025, -.01)))
    material_cell_ijk = np.floor((np.asarray(geometry['x'])-cell_origins)/
        np.asarray((.00025, .000125, .00025))+1e-10).astype(np.int64)
    reference, body_id, fixed, v0, reference_record = make_reference(geometry, .00025)
    original_reference = reference
    support_component_id = None
    node_body_id = np.concatenate((
        np.zeros(reference_record['bodies'][0]['nodes'], dtype=np.int32),
        np.ones(reference_record['bodies'][1]['nodes'], dtype=np.int32)))
    particle_spacing = np.asarray(plan['particle_spacing_m'], dtype=np.float64)
    domain_gap = minimum_material_domain_gap(
        np.asarray(geometry['x']), np.broadcast_to(np.eye(3),
            (len(geometry['x']), 3, 3)), np.asarray(geometry['body_id']), particle_spacing)
    expected_first_contact_time = domain_gap['minimum_sat_gap_m']/5000.
    material = linear_metal(**plan['material'])
    particle_mass = np.asarray(reference.particle_mass)
    eye = jnp.broadcast_to(jnp.eye(3), (len(particle_mass), 3, 3))
    zeros = jnp.zeros((len(particle_mass),))
    material_fields = (eye, jnp.zeros_like(eye), zeros, zeros, zeros, zeros, zeros)
    state = initial_state(reference, material_fields, jnp.asarray(v0))
    p0 = np.asarray(state.initial_momentum)
    l0 = np.asarray(state.initial_angular_momentum)
    k0 = float(state.initial_kinetic)
    if not k0 > 0 or not np.linalg.norm(p0) > 0:
        raise FloatingPointError('Fresh t0 nonzero nodal P/L/K baselines were not bound')
    control_velocity = jnp.asarray(((1., 0., 0.), (-1., 0., 0.)))
    control_mass = jnp.ones((2,))
    control_pairs = jnp.asarray(((0, 1),), dtype=jnp.int32)
    control_normals = jnp.asarray(((1., 0., 0.),))
    control_fixed = jnp.zeros((2, 3), dtype=bool)
    (control_post, _, control_work, control_loss, control_valid,
     control_closure, control_kkt, _) = endpoint_e0_jump(
        control_velocity, control_mass, control_pairs, control_normals,
        control_fixed, tolerance=.005)
    control_record = dict(state='passed' if bool(np.asarray(control_valid)) else 'failed',
        pre_relative_closing_m_s=2., post_velocity_m_s=np.asarray(control_post).tolist(),
        post_closing_m_s=float(control_closure), kkt_m_s=float(control_kkt),
        contact_work_J=float(control_work), kinetic_loss_J=float(control_loss),
        scope='Algebraic independent diagonal-mass e0 control; no physical geometry',
        passed=(bool(np.asarray(control_valid)) and
            np.max(np.abs(np.asarray(control_post))) <= 1e-12 and
            abs(float(control_work)+1.) <= 1e-12 and abs(float(control_loss)-1.) <= 1e-12))
    if not control_record['passed']:
        raise FloatingPointError('Independent newly-active e0 control failed')
    atomic_json(args.output/'e0-startup-control.json', control_record)
    # All hash-bound implementation sources live under /work/source; keeping
    # the runner itself there makes the capsule tree a single immutable root.
    source_root = Path(__file__).resolve().parent
    source_files = ['mpm_jax/metal_tl_midpoint.py', 'mpm_jax/material_domain_contact.py',
                    'mpm_jax/total_lagrangian.py',
                    'mpm_jax/fragment_v6_discrete_work.py', 'mpm_jax/fragment_v6_midpoint_strength.py',
                    'mpm_jax/fragment_multifield_v6_mpm.py', 'mpm_jax/fragment_v6_matrixfree_projection.py',
                    'mpm_jax/materials/linear_metal.py', 'mpm_jax/materials/shock_metal.py',
                    'mpm_jax/fibre_mpm.py', 'mpm_jax/contact.py', 'mpm_jax/core.py',
                    'mpm_jax/fracture_impact_mpm.py',
                    'examples/metal_sphere_impact_fracture.py', 'examples/metal_sphere_geometry.py',
                    'examples/metal_plate_geometry.py', 'examples/woven_geometry.py',
                    'tools/audit_metal_fragments.py', 'tools/verify_material_domain_contact.py',
                    'tools/verify_tl_viscous_power.py',
                    'run_metal_tl_fresh.py']
    # Importing mpm_jax executes package initializers, so bind the complete
    # runtime Python trees as well as the focused physics sources above.
    source_files = sorted(set(source_files + [
        path.relative_to(source_root).as_posix()
        for package in ('mpm_jax', 'examples')
        for path in (source_root/package).rglob('*.py')]))
    hashes = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_files}
    numerical_dt_cap = np.inf
    cap_successes = 0
    resume_record = None
    resumed_steps = 0
    resumed_checked = False
    resumed_metrics = {}
    if args.resume_checkpoint is not None:
        bridge = json.loads(args.resume_bridge.read_text(encoding='utf-8'))
        numerical_dt_cap=float(bridge.get('numerical_dt_cap_s',np.inf))
        checkpoint_hash = hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest()
        if bridge.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('Resume checkpoint differs from the source bridge')
        if bridge.get('continuation_source_sha256') != hashes:
            raise ValueError('Continuation source differs from the source bridge')
        if bridge.get('case_geometry_particles') != len(geometry['x']):
            raise ValueError('Resume bridge geometry differs from the fresh formal case')
        continuation_dir = args.resume_bridge.resolve().parent
        prior_history = continuation_dir/'resume-contact-history.jsonl'
        prior_event = continuation_dir/'resume-first-domain-intersection.json'
        prior_activation = continuation_dir/'resume-e0-activation.json'
        prior_damage = continuation_dir/'resume-damage-history.jsonl'
        prior_failure = continuation_dir/'resume-first-fully-failed-point.json'
        if (not prior_history.is_file() or not prior_event.is_file()
                or not prior_activation.is_file()):
            raise ValueError('Resume capsule omits the source-bound contact/activation receipts')
        if (bridge.get('prior_contact_history_sha256') != hashlib.sha256(prior_history.read_bytes()).hexdigest()
                or bridge.get('prior_first_event_sha256') != hashlib.sha256(prior_event.read_bytes()).hexdigest()
                or bridge.get('prior_e0_activation_sha256') != hashlib.sha256(prior_activation.read_bytes()).hexdigest()):
            raise ValueError('Resume capsule receipts differ from their source bridge hashes')
        if bridge.get('prior_damage_history_sha256') is not None:
            if (not prior_damage.is_file() or bridge['prior_damage_history_sha256'] !=
                    hashlib.sha256(prior_damage.read_bytes()).hexdigest()):
                raise ValueError('Resume damage-history receipt differs from its source bridge')
            (args.output/'damage-history.jsonl').write_bytes(prior_damage.read_bytes())
        if bridge.get('prior_first_failed_sha256') is not None:
            if (not prior_failure.is_file() or bridge['prior_first_failed_sha256'] !=
                    hashlib.sha256(prior_failure.read_bytes()).hexdigest()):
                raise ValueError('Resume first-failure receipt differs from its source bridge')
            atomic_json(args.output/'first-fully-failed-point.json',
                        json.loads(prior_failure.read_text(encoding='utf-8')))
        (args.output/'contact-history.jsonl').write_bytes(prior_history.read_bytes())
        atomic_json(args.output/'first-domain-intersection.json',
                    json.loads(prior_event.read_text(encoding='utf-8')))
        atomic_json(args.output/'e0-activation.json',
                    json.loads(prior_activation.read_text(encoding='utf-8')))
        with np.load(args.resume_checkpoint, allow_pickle=False) as saved:
            required = {'displacement', 'velocity', 'F', 'tau', 'ep', 'cold', 'fracture',
                        'damage', 'internal', 'time_s', 'step', 'initial_P', 'initial_L',
                        'initial_K', 'viscous_loss', 'contact_work'}
            has_split_support = 'particle_component' in saved.files
            has_adapter_metadata = 'metadata_json' in saved.files
            if ((not has_split_support and set(saved.files) != required)
                    or (has_split_support and not required.issubset(saved.files))):
                raise ValueError('Resume checkpoint schema is incomplete or unexpected')
            if has_split_support:
                if bridge.get('support_checkpoint_sha256') != checkpoint_hash:
                    raise ValueError('Duplicated-support state differs from the continuation bridge')
                parent_hash = bridge.get('support_parent_checkpoint_sha256')
                if has_adapter_metadata:
                    if args.resume_parent_checkpoint is None or args.resume_support_audit is None:
                        raise ValueError('A topology event adapter requires its exact parent and split audit')
                    support_mod = __import__('mpm_jax.tl_fracture_support', fromlist=['validate_support_checkpoint'])
                    audit_hash = hashlib.sha256(args.resume_support_audit.read_bytes()).hexdigest()
                    audit_receipt = json.loads(args.resume_support_audit.read_text(encoding='utf-8'))
                    parent_hash = hashlib.sha256(args.resume_parent_checkpoint.read_bytes()).hexdigest()
                    if bridge.get('support_audit_sha256') != audit_hash:
                        raise ValueError('Support split audit differs from the continuation bridge')
                    if bridge.get('support_parent_checkpoint_sha256') != parent_hash:
                        raise ValueError('Support parent checkpoint differs from the continuation bridge')
                    if audit_receipt.get('parent_checkpoint_sha256') != parent_hash:
                        raise ValueError('Support audit is bound to a different parent checkpoint')
                    if audit_receipt.get('support_checkpoint_sha256') != checkpoint_hash:
                        raise ValueError('Support audit is bound to a different duplicated-support state')
                    with np.load(args.resume_parent_checkpoint, allow_pickle=False) as parent:
                        if not required.issubset(parent.files):
                            raise ValueError('Raw support parent is not a complete accepted checkpoint')
                        if 'particle_component' in parent.files:
                            parent_reference = ReferenceGrid(jnp.asarray(parent['positions']),
                                jnp.asarray(parent['indices']), jnp.asarray(parent['weights']),
                                jnp.asarray(parent['gradients']), jnp.asarray(parent['volume']),
                                jnp.asarray(parent['particle_mass']), jnp.asarray(parent['nodal_mass']),
                                jnp.asarray(parent['nodes']), jnp.asarray(True))
                        else:
                            parent_reference = canonical_ancestry_reference(original_reference, geometry)
                        metadata = json.loads(str(saved['metadata_json']))
                        if metadata.get('parent_checkpoint_sha256') != parent_hash:
                            raise ValueError('Serialized support parent SHA does not match raw CP')
                        source_manifest = audit_receipt.get('parent_source_manifest_sha256')
                        if metadata.get('source_manifest_sha256') != source_manifest:
                            raise ValueError('Serialized parent source manifest differs from split audit')
                        support_mod.validate_support_checkpoint(args.resume_checkpoint,
                            parent_checkpoint_sha256=parent_hash,
                            source_manifest_sha256=source_manifest,
                            reference=parent_reference,
                            component_id=np.asarray(saved['particle_component']))
                        old_node = np.asarray(saved['old_node_for_new'], dtype=np.int64)
                        if not np.array_equal(saved['displacement'], (np.asarray(parent_reference.nodes)+parent['displacement'])[old_node]-saved['nodes']):
                            raise ValueError('Duplicated displacements do not copy the parent node map')
                        if not np.array_equal(saved['velocity'], parent['velocity'][old_node]):
                            raise ValueError('Duplicated velocities do not copy the parent node map')
                        for name in ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'):
                            if not np.array_equal(saved[name], parent[name]):
                                raise ValueError(f'Support adapter changed parent material field {name}')
                        for name in ('time_s', 'step', 'initial_P', 'initial_L', 'initial_K',
                                     'viscous_loss', 'contact_work'):
                            if not np.array_equal(saved[name], parent[name]):
                                raise ValueError(f'Support adapter changed parent ledger {name}')
                else:
                    if 'support_map_hash' not in saved.files:
                        raise ValueError('Resumable support checkpoint omits its map hash')
                    expected_map = hashlib.sha256(np.asarray(saved['indices']).tobytes()+
                        np.asarray(saved['particle_component']).tobytes()).hexdigest()
                    if str(saved['support_map_hash']) != expected_map or bridge.get('support_map_hash') != expected_map:
                        raise ValueError('Resumable support map hash differs from the bridge')
                reference = ReferenceGrid(jnp.asarray(saved['positions']),
                    jnp.asarray(saved['indices']), jnp.asarray(saved['weights']),
                    jnp.asarray(saved['gradients']), jnp.asarray(saved['volume']),
                    jnp.asarray(saved['particle_mass']), jnp.asarray(saved['nodal_mass']),
                    jnp.asarray(saved['nodes']), jnp.asarray(True))
                body_id = np.asarray(saved['body_id'], dtype=np.int32)
                node_body_id = np.asarray(saved['node_body'], dtype=np.int32)
                fixed = np.asarray(saved['fixed'], dtype=bool)
                particle_mass = np.asarray(saved['particle_mass'], dtype=np.float64)
                support_component_id = np.asarray(saved['particle_component'], dtype=np.int64)
                if has_adapter_metadata:
                    with np.load(args.resume_parent_checkpoint, allow_pickle=False) as parent:
                        parent_v0 = np.asarray(parent['initial_node_velocity']) if 'initial_node_velocity' in parent.files else v0
                    v0 = parent_v0[old_node]
                else:
                    v0 = np.asarray(saved['initial_node_velocity'], dtype=np.float64)
                reference_record = dict(reference_record,
                    node_count=int(len(reference.nodes)),
                    support_component_count=int(np.unique(support_component_id).size),
                    support_parent_checkpoint_sha256=parent_hash,
                    support_checkpoint_sha256=checkpoint_hash)
            resumed_time = float(saved['time_s'])
            resumed_steps = int(saved['step'])
            fields_resume = tuple(jnp.asarray(saved[k]) for k in
                ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'))
            state = type(state)(jnp.asarray(saved['displacement']), jnp.asarray(saved['velocity']),
                fields_resume, jnp.asarray(resumed_time), jnp.asarray(saved['initial_P']),
                jnp.asarray(saved['initial_L']), jnp.asarray(saved['initial_K']),
                jnp.asarray(saved['contact_work']), jnp.asarray(saved['viscous_loss']))
        if not 0 < resumed_time < args.end_time or resumed_steps < 1:
            raise ValueError('Resume checkpoint is outside the requested continuation window')
        p0 = np.asarray(state.initial_momentum)
        l0 = np.asarray(state.initial_angular_momentum)
        k0 = float(state.initial_kinetic)
        resumed_checked = bool(bridge.get('newly_active_e0_checked', False))
        resumed_metrics = bridge.get('prior_metrics', {})
        resume_record = dict(checkpoint=args.resume_checkpoint.name,
            checkpoint_sha256=checkpoint_hash,
            prior_source_sha256=bridge.get('prior_source_sha256'),
            bridge_reason=bridge.get('bridge_reason'),
            support_checkpoint=bool(support_component_id is not None),
            support_component_count=(None if support_component_id is None else
                                     int(np.unique(support_component_id).size)))
        if resumed_checked:
            prior_e0 = dict(schema='metal-tl-e0-check-receipt-v1',
                status='passed-attested-by-source-bound-checkpoint',
                interpretation='Actual-q1 intersection of advected initial material cuboids; shared-world contact points and independent physical postclosing route.',
                newly_active_e0_checked=True, prior_step=bridge.get('prior_step'),
                prior_time_s=bridge.get('prior_time_s'),
                accepted_checkpoint_sha256=checkpoint_hash,
                prior_run_json_sha256=bridge.get('prior_run_json_sha256'),
                prior_runner_sha256=bridge.get('prior_runner_sha256'),
                prior_capsule_sha256=bridge.get('prior_capsule_sha256'),
                detailed_event_metrics_persisted=False)
            atomic_json(args.output/'e0-prior-activation-attestation.json', prior_e0)
    # Storage is padded only after exact checkpoint/source validation.
    # The original reference remains available for ancestry reconstruction.
    storage_adapter=dict(indices=np.asarray(reference.indices),nodes=np.asarray(reference.nodes),
        displacement=np.asarray(state.displacement),velocity=np.asarray(state.velocity),
        nodal_mass=np.asarray(reference.nodal_mass),fixed=fixed,node_body=node_body_id,
        old_node_for_new=np.arange(len(reference.nodes),dtype=np.int64),initial_node_velocity=v0)
    storage_adapter=pad_nodal_storage(storage_adapter)
    reference=reference._replace(nodes=jnp.asarray(storage_adapter['nodes']),
        nodal_mass=jnp.asarray(storage_adapter['nodal_mass']))
    state=state._replace(displacement=jnp.asarray(storage_adapter['displacement']),
        velocity=jnp.asarray(storage_adapter['velocity']))
    fixed=storage_adapter['fixed'];v0=storage_adapter['initial_node_velocity'];node_body_id=storage_adapter['node_body']
    reference_record.update(node_count=len(reference.nodes),
        storage_connected_nodes=int(storage_adapter.get('storage_connected_nodes',len(reference.nodes))))
    atomic_json(args.output/'startup-storage-padding.json',dict(
        connected_nodes=int(storage_adapter.get('storage_connected_nodes',len(reference.nodes))),
        storage_nodes=len(reference.nodes),physical_particles=len(particle_mass),
        particle_mass_kg=float(particle_mass.sum()),nodal_mass_kg=float(np.asarray(reference.nodal_mass).sum()),
        unused_zero_mass_only=True,kinetic_roundoff_J=float(storage_adapter.get('storage_padding_kinetic_roundoff_J',0.))))

    case = dict(status='running', protocol='C2 fresh-t0 metal TL common-midpoint; endpoint e0; no fracture remap',
        geometry_particles=len(geometry['x']), plan_particles=plan['particles'],
        source_sha256=hashes, reference=reference_record, dt_cfl=.025,
        original_cfl_scale=.25,
        contact_geometry='advected initial material cuboids; 15-axis SAT, clipped shared-world contact points; no background-support candidate search',
        geometry_interpretation='The initial coarse particle quadrature defines a staircase material boundary; this is not an analytic sphere surface.',
        particle_spacing_m=particle_spacing.tolist(),
        initial_material_domain_gap=domain_gap,
        nominal_domain_first_contact_time_s=expected_first_contact_time,
        adjoint_power_check=adjoint_power_check,
        initial_momentum_kg_m_s=p0.tolist(), initial_angular_momentum_kg_m2_s=l0.tolist(),
        initial_kinetic_J=k0, energy_tolerance_J=1.414599609375e-8,
        dt_rule='dt=min(0.025*0.00025/max(longitudinal_speed+particle_speed+1.5*sqrt(3)*0.00025*|L|+viscous_signal), end_time-time); dt is a dynamic scalar input to the compiled midpoint kernel.',
        requested_end_time_s=args.end_time, max_seconds=args.max_seconds)
    if resume_record is not None:
        case['resume'] = resume_record
        case['newly_active_e0_checked'] = resumed_checked
    case['startup_e0_control'] = control_record
    atomic_json(args.output/'run.json', case)
    initial_energy = k0 + float(jnp.sum(jnp.asarray(particle_mass) * material_fields[-1]))
    started = time.monotonic()
    steps = resumed_steps
    dt_last = None
    known_failed_mask = np.asarray(state.material[5]) >= 1.
    support_split_events = []
    newly_active_checked = resumed_checked
    max_momentum_error = float(resumed_metrics.get('max_linear_relative', 0.))
    max_angular_error = float(resumed_metrics.get('max_angular_relative', 0.))
    max_energy_error = float(resumed_metrics.get('max_energy_error_J', 0.))
    angular_scale = max(float(np.sum(particle_mass))*.00025*5000., 1e-30)

    def write_endpoint_audit(status):
        u = np.asarray(state.displacement)
        v = np.asarray(state.velocity)
        Fend, tauend, ep_end, cold_end, fracture_end, damage_end, internal_end = map(np.asarray,
            (state.material[0], state.material[1], state.material[2], state.material[3],
             state.material[4], state.material[5], state.material[6]))
        p_x = np.asarray(reference.positions) + np.sum(
            np.asarray(reference.weights)[..., None]*u[np.asarray(reference.indices)], axis=1)
        half = particle_spacing*.5
        local_corners = _world_corners(np.zeros(3), np.eye(3), half)
        corners = p_x[:, None, :] + np.einsum(
            'pij,kj->pki', Fend, local_corners)
        sphere_mask, plate_mask = body_id == 0, body_id == 1
        local_y = corners[sphere_mask, :, 1]
        flat = int(np.argmin(local_y))
        local_particle, local_corner = np.unravel_index(flat, local_y.shape)
        sphere_ids = np.flatnonzero(sphere_mask)
        nose = corners[sphere_ids[local_particle], local_corner]
        plate_corners = corners[plate_mask].reshape(-1, 3)
        nearby = np.linalg.norm(plate_corners[:, (0, 2)]-nose[[0, 2]], axis=1) <= .0025
        if not np.any(nearby):
            raise FloatingPointError('No deformed plate top support near the sphere nose')
        surface_y = float(np.max(plate_corners[nearby, 1]))
        body_node = {}
        mass_nodes = np.asarray(reference.nodal_mass)
        for b in (0, 1):
            selected_nodes = np.flatnonzero(node_body_id == b)
            mb, vb = mass_nodes[selected_nodes], v[selected_nodes]
            initial_vb = np.asarray(v0)[selected_nodes]
            body_node[str(b)] = dict(
                initial_P_kg_m_s=(mb[:, None]*initial_vb).sum(axis=0).tolist(),
                final_P_kg_m_s=(mb[:, None]*vb).sum(axis=0).tolist(),
                delta_P_kg_m_s=(mb[:, None]*(vb-initial_vb)).sum(axis=0).tolist(),
                initial_K_J=float(.5*np.sum(mb[:, None]*initial_vb**2)),
                final_K_J=float(.5*np.sum(mb[:, None]*vb**2)))
        particle_internal = [float(np.sum(particle_mass[mask]*internal_end[mask]))
                             for mask in (sphere_mask, plate_mask)]
        plate_nodes = np.flatnonzero(node_body_id == 1)
        sym_sigma = .5*(tauend+tauend.swapaxes(-1, -2))/np.maximum(
            np.linalg.det(Fend)[:, None, None], 1e-30)
        dev = sym_sigma-.5*np.trace(sym_sigma, axis1=-2, axis2=-1)[:, None, None]*np.eye(3)
        vm = np.sqrt(np.maximum(1.5*np.sum(dev*dev, axis=(-2, -1)), 0.))
        elastic_end = np.sum(tauend*tauend, axis=(-2, -1))/(
            4*material.shear_modulus*material.density*np.maximum(1-damage_end, 1e-12))
        thermo_end = calorimetric_eos(jnp.asarray(np.linalg.det(Fend)),
            jnp.asarray(internal_end),jnp.asarray(cold_end),jnp.asarray(fracture_end),
            jnp.asarray(elastic_end),jnp.asarray(damage_end),material)
        pressure_end = np.asarray(thermo_end.pressure)
        temperature_end = np.asarray(thermo_end.temperature)
        particle_velocity = np.sum(np.asarray(reference.weights)[..., None]*
                                   v[np.asarray(reference.indices)], axis=1)
        particle_speed = np.linalg.norm(particle_velocity, axis=1)
        node_speed = np.linalg.norm(v, axis=1)
        positive_nodes = mass_nodes > 0
        fastest_node = int(np.argmax(np.where(positive_nodes, node_speed, -np.inf)))
        fastest_particle = int(np.argmax(particle_speed))

        def weighted_quantile(values, weights, q):
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            index = min(int(np.searchsorted(cumulative, q*cumulative[-1])), len(order)-1)
            return float(values[order[index]])

        speed_sampling = dict(
            nodal_max_m_s=float(node_speed[fastest_node]), nodal_max_node=fastest_node,
            nodal_max_node_mass_kg=float(mass_nodes[fastest_node]),
            nodal_max_node_body=int(node_body_id[fastest_node]),
            nodal_mass_weighted_rms_m_s=float(np.sqrt(np.sum(mass_nodes[:, None]*v*v)/np.sum(mass_nodes))),
            particle_sampled_max_m_s=float(particle_speed[fastest_particle]),
            particle_sampled_max_id=fastest_particle,
            particle_sampled_max_mass_kg=float(particle_mass[fastest_particle]),
            particle_sampled_max_body=int(body_id[fastest_particle]),
            particle_mass_weighted_rms_m_s=float(np.sqrt(np.sum(particle_mass*particle_speed**2)/np.sum(particle_mass))),
            particle_mass_weighted_p95_m_s=weighted_quantile(particle_speed, particle_mass, .95),
            particle_mass_weighted_p99_m_s=weighted_quantile(particle_speed, particle_mass, .99))
        material_body = {}
        for b, mask in ((0, sphere_mask), (1, plate_mask)):
            material_body[str(b)] = dict(
                pressure_min_Pa=float(np.min(pressure_end[mask])),
                pressure_max_Pa=float(np.max(pressure_end[mask])),
                temperature_min_K=float(np.min(temperature_end[mask])),
                temperature_max_K=float(np.max(temperature_end[mask])),
                plastic_strain_max=float(np.max(ep_end[mask])),
                damage_max=float(np.max(damage_end[mask])),
                damaged_particle_count=int(np.count_nonzero(damage_end[mask] > 0.)),
                fully_failed_particle_count=int(np.count_nonzero(damage_end[mask] >= 1.)))
        report = dict(status=status, step=steps, time_s=float(state.time),
            sphere_nose_world_m=nose.tolist(), nearby_deformed_plate_surface_y_m=surface_y,
            sphere_nose_to_plate_surface_gap_m=float(nose[1]-surface_y),
            positive_value_means_gap_negative_means_vertical_intrusion=True,
            plate_surface_deflection_near_nose_m=surface_y,
            plate_max_nodal_displacement_m=float(np.max(np.linalg.norm(u[plate_nodes], axis=1))),
            plate_max_particle_von_mises_Pa=float(np.max(vm[plate_mask])),
            plate_min_F_determinant=float(np.min(np.linalg.det(Fend[plate_mask]))),
            sphere_min_F_determinant=float(np.min(np.linalg.det(Fend[sphere_mask]))),
            sphere_internal_energy_J=particle_internal[0], plate_internal_energy_J=particle_internal[1],
            pressure_sign='positive means compression; EOS pressure',
            material_by_body=material_body, speed_sampling=speed_sampling,
            support_topology=('Reference support remains attached; no conservative split/remap.'
                if support_component_id is None else
                f'Cell face-connectivity support split active across {np.unique(support_component_id).size} components; no macroscopic fragment claim.'),
            body_nodal_transfer=body_node,
            geometry='deformed quadrature material cuboid corners; local plate surface within 2.5 mm of sphere nose',
            dt_last_s=dt_last, dt_cfl_limit=.025, original_cfl_scale=.25)
        atomic_json(args.output/'endpoint-physical-audit.json', report)

    def damage_monitor_row(record_step, record_time, record_dt):
        Fm, taum, epm, coldm, fracturem, damagem, internalm = map(np.asarray, state.material)
        detF = np.linalg.det(Fm)
        elastic_m = np.sum(taum*taum, axis=(-2, -1))/(
            4*material.shear_modulus*material.density*np.maximum(1-damagem, 1e-12))
        thermo_m = calorimetric_eos(jnp.asarray(detF),jnp.asarray(internalm),
            jnp.asarray(coldm),jnp.asarray(fracturem),jnp.asarray(elastic_m),jnp.asarray(damagem),material)
        pressure_m = np.asarray(thermo_m.pressure)
        temperature_m = np.asarray(thermo_m.temperature)
        fully_failed = np.flatnonzero(damagem >= 1.)
        failed_details = [dict(particle_id=int(i), body=int(body_id[i]),
            damage=float(damagem[i]), plastic_strain=float(epm[i]),
            detF=float(detF[i]), pressure_Pa=float(pressure_m[i]),
            temperature_K=float(temperature_m[i])) for i in fully_failed[:128]]
        body_summary = {}
        for b in (0, 1):
            mask = body_id == b
            body_summary[str(b)] = dict(
                pressure_min_Pa=float(np.min(pressure_m[mask])),
                pressure_max_Pa=float(np.max(pressure_m[mask])),
                temperature_min_K=float(np.min(temperature_m[mask])),
                temperature_max_K=float(np.max(temperature_m[mask])),
                plastic_strain_max=float(np.max(epm[mask])),
                damage_max=float(np.max(damagem[mask])),
                damaged_particle_count=int(np.count_nonzero(damagem[mask] > 0.)),
                fully_failed_particle_count=int(np.count_nonzero(damagem[mask] >= 1.)))
        return dict(step=int(record_step), time_s=float(record_time),
            dt_s=None if record_dt is None else float(record_dt),
            max_damage=float(damagem.max()), max_plastic_strain=float(epm.max()),
            fully_failed_particle_count=int(len(fully_failed)),
            fully_failed_particle_ids=fully_failed[:128].tolist(),
            fully_failed_point_details=failed_details,
            pressure_min_Pa=float(pressure_m.min()), pressure_max_Pa=float(pressure_m.max()),
            temperature_min_K=float(temperature_m.min()), temperature_max_K=float(temperature_m.max()),
            pressure_sign='positive means compression; EOS pressure', material_by_body=body_summary,
            support_topology='Reference supports remain unchanged; damage is not a conservative split.')

    def append_damage_history(record_step, record_time, record_dt):
        row = damage_monitor_row(record_step, record_time, record_dt)
        with (args.output/'damage-history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            stream.flush()
        return row

    if resumed_steps and not (args.output/'damage-history.jsonl').exists():
        append_damage_history(resumed_steps, float(state.time), None)

    if args.audit_only:
        write_endpoint_audit('accepted-checkpoint-read-only')
        case.update(status='read-only-checkpoint-audit-complete', steps=steps,
                    time_s=float(state.time), newly_active_e0_checked=newly_active_checked,
                    max_linear_relative=max_momentum_error,
                    max_angular_relative=max_angular_error,
                    max_energy_error_J=max_energy_error)
        atomic_json(args.output/'run.json', case)
        atomic_json(args.output/'audit-only-receipt.json', dict(
            schema='c2-tl-read-only-checkpoint-audit-v1',
            checkpoint_sha256=resume_record['checkpoint_sha256'],
            step=steps, time_s=float(state.time), source_sha256=hashes,
            interpretation='No midpoint step was run; all endpoint fields are sampled from the accepted checkpoint.'))
        return

    history_dir = args.output/'history'
    history_dir.mkdir(exist_ok=True)
    def save_render_frame():
        ii=np.asarray(reference.indices);ww=np.asarray(reference.weights)[...,None]
        xx=np.asarray(reference.positions)+np.sum(ww*np.asarray(state.displacement)[ii],axis=1)
        vv=np.sum(ww*np.asarray(state.velocity)[ii],axis=1)
        atomic_npz(history_dir/f'frame-{steps:07d}.npz', x=xx, v=vv,
            damage=np.asarray(state.material[5]), body_id=body_id,
            particle_mass=particle_mass, time_s=np.asarray(state.time), step=np.asarray(steps))
    save_render_frame()
    saved_time_bin=int(float(state.time)/1e-8)

    try:
        while float(state.time) < args.end_time:
            step_wall_started = time.monotonic()
            if (args.output.parent/'STOP_AFTER_CHECKPOINT').exists():
                case.update(status='stopped-after-accepted-checkpoint',steps=steps,time_s=float(state.time))
                atomic_json(args.output/'run.json',case)
                break
            if time.monotonic()-started > args.max_seconds:
                case.update(status='timeout-preserved', steps=steps, elapsed_s=time.monotonic()-started)
                atomic_json(args.output/'run.json', case)
                break
            F, tau, ep, cold, fracture, damage, internal = state.material
            J = jnp.linalg.det(F)
            elastic = jnp.sum(tau*tau, axis=(-2, -1))/(4*material.shear_modulus*material.density*jnp.maximum(1-damage, 1e-12))
            thermo = calorimetric_eos(J,internal,cold,fracture,elastic,damage,material)
            W = jnp.einsum('pki,pkj->pij', state.velocity[reference.indices], reference.gradients)
            L = W @ jnp.linalg.inv(F)
            C = jnp.sqrt(jnp.sum(L*L, axis=(-2, -1)))
            vpart = jnp.linalg.norm(jnp.sum(reference.weights[..., None]*state.velocity[reference.indices], axis=1), axis=-1)
            compression = jnp.maximum(-jnp.trace(L, axis1=-2, axis2=-1), 0.)
            visc_signal = material.viscosity_linear*thermo.sound_speed + 2*material.viscosity_quadratic*.00025*compression
            signal = thermo.longitudinal_speed + vpart + 1.5*np.sqrt(3)*.00025*C + visc_signal
            dt_cfl = .025*.00025/float(jnp.max(signal))  # Qualified original CFL scale 1/4; geometry branch own 1/16 reference.
            dt = min(dt_cfl, numerical_dt_cap, args.end_time-float(state.time))
            if not np.isfinite(dt) or dt <= 0:
                raise FloatingPointError('Original CFL admitted a nonpositive/nonfinite step')

            def contact(vpre, x1, F1):
                if support_component_id is None:
                    return material_domain_contact_jump(vpre, reference, particle_mass,
                        body_id, x1, F1, particle_spacing, fixed)
                return material_domain_contact_jump(vpre, reference, particle_mass,
                    body_id, x1, F1, particle_spacing, fixed,
                    component_id=support_component_id)

            mechanics_wall_started=time.monotonic()
            # Reject unconverged force endpoints without committing state.
            # Substeps retain the same full material/contact/global tolerances.
            for numerical_retry in range(9):
                accepted = step(state, reference, material, dt, jnp.asarray(fixed), contact,
                                cell_length=.00025)
                if bool(np.asarray(accepted.valid)):
                    break
                rr=accepted.contact_report
                if not (rr.get('step_material_valid') and rr.get('step_contact_valid')
                        and rr.get('step_finite') and (float(accepted.force_residual)>1e-10 or not rr.get('step_midpoint_work_valid',True))):
                    break
                with (args.output/'force-substeps.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=steps+1, time_s=float(state.time),
                        rejected_dt_s=dt, residual=float(accepted.force_residual),
                        retry=numerical_retry, state_committed=False))+'\n')
                if numerical_retry < 8:
                    dt *= .5
                    numerical_dt_cap=dt
                    cap_successes=0
            if not bool(np.asarray(accepted.valid)):
                atomic_npz(args.output/'rejected.npz', displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), material=np.asarray(state.material[0]),
                    time_s=np.asarray(state.time), force_residual=np.asarray(accepted.force_residual),
                    contact_work_J=np.asarray(accepted.contact_work), viscous_loss_J=np.asarray(accepted.viscous_loss),
                    kinetic_J=np.asarray(accepted.geometry['kinetic']),
                    internal_energy_J=np.asarray(jnp.sum(jnp.asarray(particle_mass)*state.material[-1])))
                atomic_json(args.output/'rejected.json', dict(
                    step=steps+1, time_s=float(state.time), dt_s=dt,
                    step_valid=bool(np.asarray(accepted.valid)),
                    force_residual=float(np.asarray(accepted.force_residual)),
                    contact=accepted.contact_report, source_sha256=hashes))
                raise FloatingPointError('Midpoint force/material/contact admission rejected')
            mechanics_wall_s=time.monotonic()-mechanics_wall_started
            if bool(np.asarray(accepted.valid)):
                cap_successes += 1
                if cap_successes >= 5 and accepted.contact_report.get('step_picard_iterations',24) <= 16:
                    numerical_dt_cap *= 1.10
                    cap_successes=0
            report = accepted.contact_report
            contact_row = dict(step=steps+1, time_before_s=float(state.time),
                time_after_s=float(state.time)+dt, dt_s=dt,
                candidate_pairs=int(report.get('pairs', 0)),
                resolved_normal_pairs=int(report.get('resolved_normal_pairs', 0)),
                positive_multiplier_pairs=int(report.get('positive_multiplier_pairs', 0)),
                pre_closing_m_s=float(report.get('initial_closing_m_s', 0.)),
                post_closing_m_s=float(report.get('final_closing_m_s', 0.)),
                physical_route_pre_closing_m_s=float(report.get('physical_route_pre_closing_m_s', 0.)),
                physical_route_post_closing_m_s=float(report.get('physical_route_post_closing_m_s', 0.)),
                physical_route_complementarity_m_s=float(report.get('physical_route_complementarity_m_s', 0.)),
                dot_v_Atlambda_vs_Av_lambda_pre_abs_error=float(report.get('dot_v_Atlambda_vs_Av_lambda_pre_abs_error', 0.)),
                dot_v_Atlambda_vs_Av_lambda_pre_roundoff_bound=float(report.get('dot_v_Atlambda_vs_Av_lambda_pre_roundoff_bound', 0.)),
                dot_v_Atlambda_vs_Av_lambda_post_abs_error=float(report.get('dot_v_Atlambda_vs_Av_lambda_post_abs_error', 0.)),
                dot_v_Atlambda_vs_Av_lambda_post_roundoff_bound=float(report.get('dot_v_Atlambda_vs_Av_lambda_post_roundoff_bound', 0.)),
                maximum_material_domain_penetration_m=float(report.get('maximum_penetration_m', 0.)),
                body_contact_impulse_Ns=report.get('body_contact_impulse_Ns', [[0., 0., 0.], [0., 0., 0.]]),
                JphaseA_vs_Jnode_pre_max_abs_m_s=float(report.get('JphaseA_vs_Jnode_pre_max_abs_m_s', 0.)),
                JphaseA_vs_Jnode_post_max_abs_m_s=float(report.get('JphaseA_vs_Jnode_post_max_abs_m_s', 0.)),
                complementarity_m_s=float(report.get('complementarity_m_s', 0.)),
                maximum_multiplier=float(report.get('maximum_multiplier', 0.)),
                nodal_impulse_norm_Ns=float(report.get('nodal_impulse_norm_Ns', 0.)),
                physical_contact_work_J=float(report.get('jump_work_J', 0.)),
                kinetic_loss_J=float(report.get('kinetic_loss_J', 0.)),
                candidate_e0_pass=bool(report.get('newly_active_e0', False)))
            if 'old_mean_readonly_shadow' in report:contact_row['old_mean_readonly_shadow']=report['old_mean_readonly_shadow']
            contact_row['runtime_profile_seconds']=report.get('runtime_profile_seconds',{})
            with (args.output/'contact-history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(contact_row, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            if report.get('pairs', 0) and not (args.output/'first-domain-intersection.json').exists():
                event_time = float(state.time)+dt
                event = dict(step=steps+1, time_s=event_time, dt_s=dt,
                    initial_domain_gap_m=domain_gap['minimum_sat_gap_m'],
                    expected_initial_gap_over_speed_time_s=expected_first_contact_time,
                    activation_time_error_s=event_time-expected_first_contact_time,
                    contact=report, source_sha256=hashes,
                    interpretation='First intersection of actual affine initial material cuboids at q1; coarse particle quadrature produces a staircase domain.')
                atomic_json(args.output/'first-domain-intersection.json', event)
            if (report['pairs'] and report['initial_closing_m_s'] > .005
                    and not newly_active_checked):
                closure = float(report['final_closing_m_s'])
                work = float(report['jump_work_J'])
                loss = float(report['kinetic_loss_J'])
                if (not report['newly_active_e0'] or closure > .005
                        or abs(work+loss) > 1e-10*max(k0, 1e-30)):
                    raise FloatingPointError('New-activation e0/jump-work check failed')
                newly_active_checked = True
                activation = dict(step=steps+1, time_s=float(state.time)+dt,
                    dt_s=dt, contact=report, source_sha256=hashes,
                    state='passed-new-activation-e0')
                atomic_json(args.output/'e0-activation.json', activation)
                case['new_activation_receipt'] = activation
            state = accepted.state
            steps += 1
            dt_last = dt
            P, Lm, K = map(np.asarray, (accepted.geometry['momentum'],
                                       accepted.geometry['angular_momentum'],
                                       accepted.geometry['kinetic']))
            pscale=max(np.linalg.norm(p0), 1e-30)
            max_momentum_error=max(max_momentum_error, float(np.linalg.norm(P-p0)/pscale))
            max_angular_error=max(max_angular_error, float(np.linalg.norm(Lm-l0)/angular_scale))
            total = float(K + jnp.sum(jnp.asarray(particle_mass)*state.material[-1])
                          + state.viscous_loss - state.contact_work)
            max_energy_error=max(max_energy_error, abs(total-initial_energy))
            if (max_momentum_error > 1e-10 or max_angular_error > 1e-10
                    or max_energy_error > 1.414599609375e-8):
                atomic_npz(args.output/'rejected.npz', displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), F=np.asarray(state.material[0]),
                    tau=np.asarray(state.material[1]), ep=np.asarray(state.material[2]),
                    cold=np.asarray(state.material[3]), fracture=np.asarray(state.material[4]),
                    damage=np.asarray(state.material[5]), internal=np.asarray(state.material[6]),
                    time_s=np.asarray(state.time), momentum_relative=np.asarray(max_momentum_error),
                    angular_relative=np.asarray(max_angular_error), energy_error_J=np.asarray(max_energy_error),
                    viscous_loss_J=np.asarray(state.viscous_loss), contact_work_J=np.asarray(state.contact_work),
                    kinetic_J=np.asarray(K), internal_energy_J=np.asarray(jnp.sum(
                        jnp.asarray(particle_mass)*state.material[-1])))
                raise FloatingPointError('Original global energy/linear/angular guard failed')
            monitor = append_damage_history(steps, float(state.time), dt)
            failed_now = np.asarray(state.material[5]) >= 1.
            newly_failed_ids = np.flatnonzero(failed_now & ~known_failed_mask)
            if len(newly_failed_ids):
                # Preserve the exact accepted pre-split state so the new
                # incidence map is always reversible and source-auditable.
                parent_path = args.output/f'pre-split-step{steps}.npz'
                parent_values = dict(displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), F=np.asarray(state.material[0]),
                    tau=np.asarray(state.material[1]), ep=np.asarray(state.material[2]),
                    cold=np.asarray(state.material[3]), fracture=np.asarray(state.material[4]),
                    damage=np.asarray(state.material[5]), internal=np.asarray(state.material[6]),
                    time_s=np.asarray(state.time), step=np.asarray(steps), initial_P=p0,
                    initial_L=l0, initial_K=np.asarray(k0),
                    viscous_loss=np.asarray(state.viscous_loss), contact_work=np.asarray(state.contact_work))
                if support_component_id is not None:
                    parent_values.update(indices=np.asarray(reference.indices),
                        positions=np.asarray(reference.positions), weights=np.asarray(reference.weights),
                        gradients=np.asarray(reference.gradients), volume=np.asarray(reference.volume),
                        particle_mass=particle_mass, nodal_mass=np.asarray(reference.nodal_mass),
                        nodes=np.asarray(reference.nodes), fixed=fixed, body_id=body_id,
                        particle_component=support_component_id, node_body=node_body_id,
                        old_node_for_new=np.arange(len(reference.nodes), dtype=np.int64),
                        initial_node_velocity=v0)
                atomic_npz(parent_path, **parent_values)
                parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
                support_mod = __import__('mpm_jax.tl_fracture_support', fromlist=[
                    'face_connected_components', 'split_reference_support', 'save_support_checkpoint'])
                topology = support_mod.face_connected_components(material_cell_ijk,
                    body_id, failed_now, spacing=particle_spacing,
                    sphere_center=sphere_center, sphere_radius=.001)
                adapter = support_mod.split_reference_support(reference,
                    state.displacement, state.velocity, fixed, topology['component_id'],
                    body_id=body_id)
                adapter.update(body_id=body_id.copy(), step=np.asarray(steps),
                    time_s=np.asarray(state.time), initial_P=p0.copy(), initial_L=l0.copy(),
                    initial_K=np.asarray(k0), viscous_loss=np.asarray(state.viscous_loss),
                    contact_work=np.asarray(state.contact_work),
                    initial_node_velocity=v0[np.asarray(adapter['old_node_for_new'], dtype=np.int64)].copy(),
                    state_fields={name: np.asarray(value) for name, value in zip(
                        ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'), state.material)})
                adapter=pad_nodal_storage(adapter)
                parent_source_manifest = hashlib.sha256(json.dumps(hashes, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()
                support_mod.save_support_checkpoint(args.output/'accepted.npz', parent_sha,
                    parent_source_manifest, adapter)
                reference = ReferenceGrid(jnp.asarray(adapter['positions']),
                    jnp.asarray(adapter['indices']), jnp.asarray(adapter['weights']),
                    jnp.asarray(adapter['gradients']), jnp.asarray(adapter['volume']),
                    jnp.asarray(adapter['particle_mass']), jnp.asarray(adapter['nodal_mass']),
                    jnp.asarray(adapter['nodes']), jnp.asarray(True))
                # Transfer the exact accepted nodal state to the copied support.
                # Particle material and cumulative physical ledgers stay unchanged.
                state = state._replace(displacement=jnp.asarray(adapter['displacement']),
                    velocity=jnp.asarray(adapter['velocity']))
                fixed = np.asarray(adapter['fixed'], dtype=bool)
                support_component_id = topology['component_id']
                node_body_id = np.asarray(adapter['node_body'], dtype=np.int32)
                v0 = v0[np.asarray(adapter['old_node_for_new'], dtype=np.int64)]
                # Rebuild current-domain SAT rows now so the event receipt
                # proves that subsequent same-body contacts use this map.
                from mpm_jax.material_domain_contact import material_domain_candidates
                event_pairs = material_domain_candidates(
                    np.asarray(accepted.geometry['position']),
                    np.asarray(state.material[0]), body_id, particle_spacing,
                    component_id=support_component_id)
                event = dict(schema='c2-tl-component-support-event-v1',
                    status='accepted-zero-time-conservative-support-split',
                    step=steps, time_s=float(state.time),
                    parent_checkpoint=parent_path.name, parent_checkpoint_sha256=parent_sha,
                    source_manifest_sha256=parent_source_manifest,
                    parent_source_manifest_sha256=parent_source_manifest,
                    support_checkpoint='accepted.npz',
                    support_checkpoint_sha256=hashlib.sha256(
                        (args.output/'accepted.npz').read_bytes()).hexdigest(),
                    support_metadata=adapter['support_hash'], topology_hash=adapter['topology_hash'],
                    new_fully_failed_ids=newly_failed_ids.tolist(),
                    fully_failed_count=int(np.count_nonzero(failed_now)),
                    component_count=int(len(topology['components'])),
                    component_particle_counts=[int(g['particle_count']) for g in topology['components']],
                    reference_face_edge_count=int(topology['reference_face_edge_count']),
                    severed_failed_incident_edge_count=int(topology['severed_failed_incident_edge_count']),
                    same_body_and_cross_body_sat_pair_count=len(event_pairs),
                    support_node_count=int(len(adapter['nodes'])),
                    storage_connected_nodes=int(adapter.get('storage_connected_nodes',len(adapter['nodes']))),
                    unused_padding_nodes=int(len(adapter['nodes'])-int(adapter.get('storage_connected_nodes',len(adapter['nodes'])))),
                    max_position_change_m=adapter['max_position_change_m'],
                    max_F_change=adapter['max_F_change'],
                    mass_kg=adapter['after_mass_kg'],
                    momentum_Ns=adapter['after_momentum'].tolist(),
                    angular_momentum_Nms=adapter['after_angular_momentum'].tolist(),
                    kinetic_J=adapter['after_kinetic_J'],
                    material_history_unchanged=True,
                    source_sha256=hashes)
                event_path = args.output/f'support-split-step{steps}.json'
                atomic_json(event_path, event)
                support_split_events.append(event)
                case['support_split_events'] = support_split_events
                case['support_event_receipt'] = event_path.name
                reference_record.update(node_count=int(len(reference.nodes)),
                    support_component_count=int(np.unique(support_component_id).size))
            else:
                checkpoint_values = dict(displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity), F=np.asarray(state.material[0]),
                    tau=np.asarray(state.material[1]), ep=np.asarray(state.material[2]),
                    cold=np.asarray(state.material[3]), fracture=np.asarray(state.material[4]),
                    damage=np.asarray(state.material[5]), internal=np.asarray(state.material[6]),
                    time_s=np.asarray(state.time), step=np.asarray(steps), initial_P=p0,
                    initial_L=l0, initial_K=np.asarray(k0), viscous_loss=np.asarray(state.viscous_loss),
                    contact_work=np.asarray(state.contact_work))
                if support_component_id is not None:
                    checkpoint_values.update(indices=np.asarray(reference.indices),
                        positions=np.asarray(reference.positions), weights=np.asarray(reference.weights),
                        gradients=np.asarray(reference.gradients), volume=np.asarray(reference.volume),
                        particle_mass=particle_mass, nodal_mass=np.asarray(reference.nodal_mass),
                        nodes=np.asarray(reference.nodes), fixed=fixed, body_id=body_id,
                        particle_component=support_component_id, node_body=node_body_id,
                        initial_node_velocity=v0,
                        old_node_for_new=np.arange(len(reference.nodes), dtype=np.int64),
                        support_map_hash=np.asarray(hashlib.sha256(
                            np.asarray(reference.indices).tobytes()+
                            support_component_id.tobytes()).hexdigest()))
                atomic_npz(args.output/'accepted.npz', **checkpoint_values)
            known_failed_mask = failed_now
            case.update(status='running', steps=steps, time_s=float(state.time), dt_last_s=dt_last,
                elapsed_s=time.monotonic()-started, newly_active_e0_checked=newly_active_checked,
                max_linear_relative=max_momentum_error, max_angular_relative=max_angular_error,
                max_energy_error_J=max_energy_error)
            if len(newly_failed_ids):
                failed_ids = np.flatnonzero(np.asarray(state.material[5]) >= 1.)
                first = int(failed_ids[0])
                Fm = np.asarray(state.material[0])
                epm = np.asarray(state.material[2])
                receipt = dict(schema='c2-first-fully-failed-point-stop-v1',
                    status='conservative-support-split-applied-and-continuing', step=steps,
                    time_s=float(state.time), dt_s=dt,
                    accepted_checkpoint_sha256=hashlib.sha256((args.output/'accepted.npz').read_bytes()).hexdigest(),
                    fully_failed_particle_count=int(len(failed_ids)),
                    fully_failed_particle_ids=failed_ids[:128].tolist(),
                    fully_failed_point_details=monitor['fully_failed_point_details'],
                    first_point=dict(particle_id=first, body=int(body_id[first]),
                        position_m=np.asarray(accepted.geometry['position'])[first].tolist(),
                        damage=float(np.asarray(state.material[5])[first]),
                        plastic_strain=float(epm[first]), detF=float(np.linalg.det(Fm[first]))),
                    interpretation='New fully_failed particles retain mass and material state and receive component-specific copied nodal support; this is a cell-scale support split, not a macroscopic fragment claim.',
                    source_sha256=hashes)
                atomic_json(args.output/'first-fully-failed-point.json', receipt)
                case.update(status='running-after-conservative-support-split',
                    first_fully_failed_point=receipt)
            if int(float(state.time)/1e-8) > saved_time_bin:
                save_render_frame()
                saved_time_bin=int(float(state.time)/1e-8)
            atomic_json(args.output/'numerical-controller.json',dict(dt_cap_s=float(numerical_dt_cap) if np.isfinite(numerical_dt_cap) else None,consecutive_accepted=cap_successes,last_dt_s=dt))
            atomic_json(args.output/'run.json', case)
            print(json.dumps(dict(step=steps, time_s=float(state.time), dt_s=dt,
                mechanics_wall_s=mechanics_wall_s, picard_iterations=report.get('step_picard_iterations'), numerical_retries=numerical_retry,
                contact_profile_seconds=report.get('runtime_profile_seconds',{}),
                step_wall_s=time.monotonic()-step_wall_started, elapsed_wall_s=time.monotonic()-started,
                force_residual=float(accepted.force_residual), energy_error_J=max_energy_error,
                momentum_relative=max_momentum_error, angular_relative=max_angular_error),
                separators=(',', ':')), flush=True)
        else:
            case.update(status='completed-short-window', steps=steps, time_s=float(state.time),
                dt_last_s=dt_last, elapsed_s=time.monotonic()-started,
                newly_active_e0_checked=newly_active_checked,
                max_linear_relative=max_momentum_error, max_angular_relative=max_angular_error,
                max_energy_error_J=max_energy_error)
            atomic_json(args.output/'run.json', case)
        write_endpoint_audit(case['status'])
    except BaseException as exc:
        rejected_attempt = dict(step=int(steps), time_s=float(state.time),
            max_linear_relative=float(max_momentum_error),
            max_angular_relative=float(max_angular_error),
            max_energy_error_J=float(max_energy_error), error=repr(exc))
        accepted_path = args.output/'accepted.npz'
        if accepted_path.is_file():
            # A candidate can pass the material/contact solver and still fail
            # the global cumulative guards. Keep endpoint reporting on the last
            # checkpoint that passed every guard, never on that rejected state.
            with np.load(accepted_path, allow_pickle=False) as saved:
                state = state._replace(
                    displacement=jnp.asarray(saved['displacement']),
                    velocity=jnp.asarray(saved['velocity']),
                    material=tuple(jnp.asarray(saved[name]) for name in
                        ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal')),
                    time=jnp.asarray(saved['time_s']),
                    contact_work=jnp.asarray(saved['contact_work']),
                    viscous_loss=jnp.asarray(saved['viscous_loss']))
                steps = int(saved['step'])
            rejected_attempt['accepted_checkpoint_sha256'] = hashlib.sha256(
                accepted_path.read_bytes()).hexdigest()
            rejected_attempt['last_accepted_step'] = steps
            rejected_attempt['last_accepted_time_s'] = float(state.time)
        try:
            write_endpoint_audit('failed-at-last-accepted-state')
        except BaseException as audit_exc:
            case['endpoint_audit_error'] = repr(audit_exc)
        case.update(status='failed-preserved', steps=steps, time_s=float(state.time),
                    elapsed_s=time.monotonic()-started, error=repr(exc),
                    rejected_attempt=rejected_attempt, traceback=traceback.format_exc())
        atomic_json(args.output/'run.json', case)
        raise


if __name__ == '__main__':
    main()
