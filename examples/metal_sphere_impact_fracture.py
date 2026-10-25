"""Independent linear-EOS two-body material demonstration with resumable records."""

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np

from .metal_sphere_geometry import plan_case as geometry_plan


SOURCE_FILES = (
    'examples/metal_sphere_impact_fracture.py', 'examples/metal_sphere_geometry.py',
    'examples/metal_plate_geometry.py', 'examples/woven_geometry.py', 'mpm_jax/fracture_impact_mpm.py',
    'mpm_jax/materials/linear_metal.py', 'mpm_jax/materials/shock_metal.py', 'mpm_jax/fibre_mpm.py',
    'mpm_jax/contact.py', 'mpm_jax/core.py', 'tools/audit_metal_fragments.py',
)
MATERIAL = dict(density=2700., bulk_modulus=2700.*5350.**2, heat_capacity=900.,
    reference_temperature=300., melt_temperature=933.15, shear_modulus=26e9,
    A=100e6, B=300e6, n=.34, C=.01, m=1., reference_rate=1., heat_fraction=.9,
    minimum_J=.4, maximum_J=2., viscosity_linear=1., viscosity_quadratic=2.)
FIELDS = ('x', 'speed', 'temperature', 'damage', 'failed')


def descriptor(value):
    array = np.ascontiguousarray(value)
    return {'shape': list(np.shape(value)), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def source_record():
    root = Path(__file__).resolve().parents[1]
    files = {name: digest(root/name) for name in SOURCE_FILES}
    return {'files': files, 'fingerprint': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def write_table(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def table_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def grid_plan(geometry, definition, *, padding_cells=4, max_nodes=8_000_000):
    """Finite per-body rectangular envelopes; leaving them is a numerical failure."""
    dx = definition['dx_m']
    radius = definition['sphere']['diameter_m']/2
    center = np.asarray(definition['sphere']['center_m'])
    plate = geometry['x'][geometry['body_id'] == 1]
    half = np.asarray(definition['plate']['size_xz_m'])/2
    thickness = definition['plate']['thickness_m']
    travel = definition['speed_m_s']*definition['end_time_s']
    allowance = max(8*dx, radius)
    bounds = np.array([
        [[-2*radius, 2*radius], [center[1]-radius-travel-allowance, center[1]+radius+allowance], [-2*radius, 2*radius]],
        [[-half[0], half[0]], [-thickness/2-travel-allowance, thickness/2+allowance], [-half[1], half[1]]]])
    lower = np.stack([geometry['x'][geometry['body_id'] == i].min(0) for i in (0, 1)])
    upper = np.stack([geometry['x'][geometry['body_id'] == i].max(0) for i in (0, 1)])
    first = np.floor(np.minimum(lower, bounds[..., 0])/dx).astype(np.int64)-padding_cells
    last = np.ceil(np.maximum(upper, bounds[..., 1])/dx).astype(np.int64)+padding_cells
    shapes = last-first+1
    counts = np.prod(shapes, axis=1)
    if counts.sum() > max_nodes:
        raise ValueError('The declared packed grid exceeds its node budget')
    clamp = definition['boundary']['clamp_width_m']
    plate_mask = (np.abs(plate[:, 0]) >= half[0]-clamp) | (np.abs(plate[:, 2]) >= half[1]-clamp)
    if not plate_mask.any():
        raise ValueError('No plate particles occupy the fixed edge strips')
    return {'dx_m': dx, 'origin_m': [0., 0., 0.], 'padding_cells': padding_cells,
        'bounds_m': bounds.tolist(), 'lower': first.tolist(), 'shapes': shapes.tolist(),
        'nodes_by_body': counts.tolist(), 'nodes': int(counts.sum()), 'max_nodes': max_nodes,
        'clamp_definition': 'Plate nodes in the four reference edge strips, extended through the allocated y envelope; all xyz components fixed. The corner union is applied once. No backing plane.',
        'domain_scope': 'Explicit finite allocation, not a certified propagation bound. Invalid support stops the run without enlarging the grid or removing particles.'}


def plan_case(*, cfl=.1, strength=True, **geometry_options):
    if not np.isfinite(cfl) or not 0 < cfl <= .2 or type(strength) is not bool:
        raise ValueError('Require 0<CFL<=.2 and boolean strength selection')
    if geometry_options.get('density', 2700.) != MATERIAL['density']:
        raise ValueError('Geometry reference density must equal the fixed material density')
    geometry, definition = geometry_plan(**geometry_options)
    plan = {'schema': 'metal-sphere-impact-fracture-v1', 'definition': definition,
        'material': MATERIAL.copy(), 'config': {'cfl': float(cfl), 'strength': strength},
        'frames': definition['frames'], 'saved_states': definition['frames']+1,
        'output_times_s': definition['output_times_s'], 'duration_s': definition['end_time_s'],
        'particles': len(geometry['x']), 'geometry_fields': {k: descriptor(v) for k, v in geometry.items()},
        'grid': grid_plan(geometry, definition), 'source': source_record(),
        'geometry_scale': 1., 'particle_spacing_m': definition['particle_spacing_m'],
        'failure_model': 'Hypothesized irreversible plastic-strain damage onset=.10 failure=.30. Deviatoric strength and tensile traction degrade; compressive EOS retained. No calibrated fracture energy or mesh-objectivity.',
        'scope': 'Exploratory local damage MPM; shared-grid recoupling and resolution-dependent failure remain limitations. Actual detached groups require independent position/connectivity audit; no guaranteed cloud or calibrated penetration.',
        'energy_scope': 'Physical energy is translation plus affine plus internal energy. Boundary and signed numerical reservoirs are separate accounting terms, never heat. Only plastic work heats particles; bounded strength softening at Tm does not cap temperature.',
        'body_impulse_scope': 'Contact impulses are inferred from each body momentum change and the actual plate boundary impulse; this is not an independent reconstruction of node contact impulses.'}
    return geometry, plan


def tiny_control_plan():
    """Low-speed transfer and continuation control, distinct from the fixed request."""
    return plan_case(speed=10., sphere_diameter=.004, plate_size=(.012, .012),
        plate_thickness=.002, gap=.003, clamp_width=.002, dx=.001,
        spacing=(.001, .001, .001), end_time=2e-7, frames=4)


def minimal_impact_plan():
    """5 km/s contact control, coarse geometry; not a fragmentation convergence study."""
    return plan_case(speed=5000., sphere_diameter=.002, plate_size=(.006,.006),
        plate_thickness=.0005, gap=.0015, clamp_width=.001, dx=.0005,
        spacing=(.0005,.00025,.0005), end_time=1e-6, frames=10)


def validate_endpoint(value, frames, previous=-1):
    if value is not None and (type(value) is not int or not previous < value <= frames or value < 0):
        raise ValueError('The planned stop must be a later complete output frame of the unchanged protocol')


def check_arrays(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f'Unexpected static fields in {Path(path).name}')
        for key, value in expected.items():
            if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value):
                raise ValueError(f'Static field {key} differs from reference data')


def run_case(geometry, plan, output, *, resume=False, stop_after_frame=None, max_steps_per_frame=20000):
    validate_endpoint(stop_after_frame, plan['frames'])
    if type(max_steps_per_frame) is not int or max_steps_per_frame < 1:
        raise ValueError('max_steps_per_frame must be a positive runtime budget')
    if plan['source'] != source_record() or set(geometry) != set(plan['geometry_fields']):
        raise ValueError('Numerical source or reference fields changed')
    for name, expected in plan['geometry_fields'].items():
        if descriptor(geometry[name]) != expected:
            raise ValueError(f'Reference geometry changed: {name}')
    folder = Path(output)
    if not resume and folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Output must be empty, or explicitly resumed')
    if resume and not (folder/'checkpoint.npz').is_file():
        raise ValueError('Resume requires the original complete saved-frame checkpoint')
    folder.mkdir(parents=True, exist_ok=True)
    return _runtime(geometry, plan, folder, resume, stop_after_frame, max_steps_per_frame)


def _runtime(geometry, plan, folder, resume, stop_after_frame, max_steps_per_frame):
    import jax
    import jax.numpy as jnp
    import jaxlib
    from mpm_jax.fracture_impact_mpm import (ImpactConfig, ImpactState, advance_to, build_grid,
        energy_ledger, make_state, thermodynamics, validate_state)
    from mpm_jax.materials.linear_metal import linear_metal

    jax.config.update('jax_enable_x64', True)
    runtime = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
        jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    material = linear_metal(**plan['material'])
    config = ImpactConfig(**plan['config'])
    gp = plan['grid']
    grid = build_grid(geometry['x'], geometry['body_id'], dx=gp['dx_m'],
        padding_cells=gp['padding_cells'], reference_bounds=gp['bounds_m'], max_nodes=gp['max_nodes'])
    if (grid.layout.node_count != gp['nodes'] or not np.array_equal(np.asarray(grid.layout.shape), gp['shapes'])
            or not np.array_equal(np.asarray(grid.layout.lower), gp['lower'])):
        raise ValueError('Actual grid differs from the immutable NumPy plan')
    # Construct only 1D node axes, rather than a full world-coordinate array.
    shape = np.asarray(grid.layout.shape)[1]
    low = np.asarray(grid.layout.lower)[1]
    xx, zz = (low[0]+np.arange(shape[0]))*gp['dx_m'], (low[2]+np.arange(shape[2]))*gp['dx_m']
    half = np.asarray(plan['definition']['plate']['size_xz_m'])/2
    width = plan['definition']['boundary']['clamp_width_m']
    mask_x, mask_z = np.abs(xx) >= half[0]-width, np.abs(zz) >= half[1]-width
    edge = np.broadcast_to(mask_x[:, None, None] | mask_z[None, None, :], tuple(shape)).ravel()
    clamped = np.zeros((grid.layout.node_count, 3), bool)
    start = int(np.asarray(grid.layout.offsets)[1])
    clamped[start:start+len(edge)] = edge[:, None]
    grid = replace(grid, clamped=jnp.asarray(clamped))
    initial = make_state(geometry['x'], geometry['volume'], geometry['body_id'], material, grid, v=geometry['v'])
    state, count = initial, len(geometry['x'])
    initial_mass = np.asarray(initial.mass)
    initial_P = np.stack([(initial_mass[:, None]*geometry['v'])[geometry['body_id'] == b].sum(0) for b in (0, 1)])
    initial_total_energy = float(energy_ledger(initial, grid).sum())
    static = {'family_id': geometry['body_id'].copy(), 'physical_body_id': geometry['body_id'].copy()}
    shapes = {'x': (plan['saved_states'], count, 3), **{name: (plan['saved_states'], count) for name in FIELDS if name != 'x'}}
    dtypes = {name: np.dtype('uint8' if name == 'failed' else 'float64') for name in FIELDS}
    arrays, rows, diagnostics, times, frames = {}, [], [], [], []
    start_frame = steps = 0
    solver_seconds = 0.
    receipt_names = ('checkpoint.npz', 'final_state_0.npz', 'reference.npz', 'particle_fields.npz', 'trajectory.npz',
                     'observables.csv', 'diagnostics.csv', 'plan.json')

    def endpoint_fields(current):
        thermo = thermodynamics(current, material)
        return {'x': np.asarray(current.x), 'speed': np.linalg.norm(np.asarray(current.v), axis=1),
            'temperature': np.asarray(thermo.temperature), 'damage': np.asarray(current.damage),
            'failed': np.asarray(current.damage >= 1, dtype=np.uint8)}

    def saved_receipt():
        save_json(folder/'checkpoint.json', {'schema': 'metal-impact-saved-frame-v1', 'signature': signature,
            'frame': len(times)-1, 'time': times[-1], 'steps': steps,
            'files': {name: digest(folder/name) for name in receipt_names},
            'scope': 'Whole checkpoint/static/table bytes and the hash of each populated trajectory frame. Unwritten future slots are not accepted history.'})

    def checkpoint():
        save_npz(folder/'checkpoint.npz', **{'state_'+k: np.asarray(v) for k, v in state._asdict().items()},
            frame=np.array(len(times)-1), steps=np.array(steps), signature=np.array(signature),
            runtime=np.array(json.dumps(runtime, sort_keys=True)), solver_seconds=np.array(solver_seconds),
            tables=np.array(json.dumps(dict(rows=rows, diagnostics=diagnostics, times=times, frames=frames), allow_nan=False)))
        save_npz(folder/'final_state_0.npz', **{k: np.asarray(v) for k, v in state._asdict().items()})
        saved_receipt()

    def summary(phase):
        complete = phase == 'completed'
        return dict(case='metal-sphere-impact-fracture', completed=complete, phase=phase,
            segment_complete=phase == 'checkpoint-ready', segment_end_frame=len(times)-1,
            particles=[count], physical_body_particles=[int(np.sum(geometry['body_id'] == b)) for b in (0, 1)],
            frames=plan['frames'], steps=steps, adaptive_steps=True, simulation_time=times[-1],
            target_simulation_time=plan['duration_s'], solver_seconds=solver_seconds,
            timing_scope='All advance_to calls, including compilation; setup, saved-frame reductions and IO excluded.',
            dtype='float64', runtime=runtime, source=plan['source'], definition=plan,
            configs=[dict(dim=3, n_grid=round(plan['definition']['plate']['size_xz_m'][0]/gp['dx_m']),
                domain_size=plan['definition']['plate']['size_xz_m'][0], grid_type='two independent packed rectangular grids',
                gravity=[0.,0.,0.], failure_mode='retain', time_integration='adaptive')],
            materials=[plan['material']], geometry_scale=1., particle_spacing_m=plan['particle_spacing_m'],
            particle_fields={'file':'particle_fields.npz','fields':{k:descriptor(v) for k,v in static.items()}},
            boundary={'clamped_nodes':int(edge.sum()), 'clamp_sha256':descriptor(clamped),
                      'backing':False, 'physical_body':1, 'components':[0,1,2]},
            min_J=min(row['minimum_J'] for row in diagnostics), max_damage=float(np.asarray(state.damage).max()),
            initial=diagnostics[0], final=diagnostics[-1], observables={'initial':rows[0], 'final':rows[-1]},
            production_eligible=False, physical_validation_passed=False,
            scope=plan['scope'], energy_scope=plan['energy_scope'], failure_model=plan['failure_model'])

    def status(phase):
        value = dict(completed=phase == 'completed', segment_complete=phase == 'checkpoint-ready', phase=phase,
            frame=len(times)-1, frames=plan['frames'], steps=steps, time=times[-1],
            signature=signature, state_valid=bool(state.valid), production_eligible=False)
        save_json(folder/'status.json', value)
        save_json(folder/'summary.json', summary(phase))

    def record(frame):
        nonlocal rows, diagnostics
        validate_state(state, material, grid)
        thermo = thermodynamics(state, material)
        fields = endpoint_fields(state)
        for name, value in fields.items():
            arrays[name][frame] = value
            arrays[name].flush()
        frames.append({k:descriptor(v) for k,v in fields.items()})
        instant = float(state.time)
        times.append(instant)
        mass, velocity, C, F = map(np.asarray, (state.mass, state.v, state.C, state.F))
        J = np.linalg.det(F)
        ledger = np.asarray(energy_ledger(state, grid))
        row = dict(time=instant, steps=steps, translation_energy_J=float(ledger[0]),
            affine_energy_J=float(ledger[1]), internal_energy_J=float(ledger[2]),
            boundary_reservoir_J=float(ledger[3]), numerical_reservoir_J=float(ledger[4]),
            physical_energy_J=float(ledger[:3].sum()),
            physical_energy_drift_J=float(ledger[:3].sum()-initial_total_energy),
            accounted_energy_J=float(ledger.sum()), total_energy_J=float(ledger.sum()),
            energy_drift_J=float(ledger.sum()-initial_total_energy),
            boundary_projection_loss_J=float(state.boundary_projection_loss),
            signed_boundary_coupling_work_J=float(state.boundary_coupling_work),
            minimum_entropy_increment_sum_J_kg_K=float(np.min(np.asarray(state.negative_entropy_increments))),
            minimum_entropy_change_J_kg_K=float(np.min(np.asarray(state.minimum_entropy-state.initial_entropy))),
            ever_phase_extrapolated_particles=int(np.asarray(state.ever_phase_extrapolated).sum()),
            ever_tension_extrapolated_particles=int(np.asarray(state.ever_tension_extrapolated).sum()),
            maximum_temperature_history_K=float(np.asarray(state.maximum_temperature).max()),
            minimum_J_history=float(np.asarray(state.minimum_J).min()), maximum_J_history=float(np.asarray(state.maximum_J).max()))
        perbody = []
        for b, label in enumerate(('sphere','plate')):
            select = geometry['body_id'] == b
            P = (mass[select,None]*velocity[select]).sum(0)
            perbody.append(P)
            boundary = np.asarray(state.boundary_impulse) if b == 1 else np.zeros(3)
            inferred = P-initial_P[b]-boundary
            row[f'{label}_translation_energy_J'] = float(.5*np.sum(mass[select]*np.sum(velocity[select]**2,axis=1)))
            row[f'{label}_affine_energy_J'] = float(gp['dx_m']**2/8*np.sum(mass[select]*np.sum(C[select]**2,axis=(1,2))))
            row[f'{label}_internal_energy_J'] = float(np.sum(mass[select]*np.asarray(state.internal_energy)[select]))
            row[f'{label}_contact_projection_loss_J'] = float(state.contact_projection_loss[b])
            row[f'{label}_signed_contact_coupling_work_J'] = float(state.contact_coupling_work[b])
            row[f'{label}_mean_y_m'] = float(np.sum(mass[select]*fields['x'][select,1])/mass[select].sum())
            for a, axis in enumerate('xyz'):
                row[f'{label}_momentum_{axis}_Ns'] = float(P[a])
                row[f'{label}_derived_contact_impulse_{axis}_Ns'] = float(inferred[a])
                row[f'{label}_boundary_impulse_{axis}_Ns'] = float(boundary[a])
        residual = np.sum(perbody,axis=0)-initial_P.sum(0)-np.asarray(state.boundary_impulse)
        diagnostic = dict(time=instant, steps=steps, mass=float(mass.sum()), reference_volume=float(state.volume.sum()),
            current_volume=float(np.sum(np.asarray(state.volume)*J)), minimum_J=float(J.min()), maximum_J=float(J.max()),
            mass_relative_drift=float(mass.sum()/initial_mass.sum()-1), maximum_plastic_strain=float(np.asarray(state.plastic_strain).max()),
            minimum_temperature=float(np.asarray(thermo.temperature).min()), maximum_temperature=float(np.asarray(thermo.temperature).max()),
            momentum_residual_Ns=float(np.linalg.norm(residual)), energy_residual_J=row['energy_drift_J'], physical_energy_residual_J=row['physical_energy_drift_J'],
            max_damage=float(np.asarray(state.damage).max()), failed_mass=float(np.asarray(state.mass)[np.asarray(state.damage)>=1].sum()), state_valid=bool(state.valid))
        rows.append(row); diagnostics.append(diagnostic)
        write_table(folder/'observables.csv', rows); write_table(folder/'diagnostics.csv', diagnostics)
        save_npz(folder/'trajectory.npz', time=np.asarray(times), body=np.zeros(count,dtype=np.int32))
        checkpoint()
        status('running')

    try:
        if resume:
            saved_status = json.loads((folder/'status.json').read_text())
            if saved_status.get('phase') not in ('checkpoint-ready','completed','running') or saved_status.get('signature') != signature:
                raise ValueError('A failed or mismatched run cannot resume')
            receipt = json.loads((folder/'checkpoint.json').read_text())
            if receipt.get('signature') != signature or set(receipt['files']) != set(receipt_names):
                raise ValueError('Incomplete or mismatched saved-frame receipt')
            for name, sha in receipt['files'].items():
                if digest(folder/name) != sha:
                    raise ValueError(f'Saved-frame file bytes changed: {name}')
            check_arrays(folder/'reference.npz', geometry)
            check_arrays(folder/'particle_fields.npz', static)
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
                expected = {'state_'+name for name in initial._fields} | {'frame','steps','signature','runtime','solver_seconds','tables'}
                if set(cp.files) != expected or str(cp['signature']) != signature or json.loads(str(cp['runtime'])) != runtime:
                    raise ValueError('Checkpoint field set, source or numerical runtime differs')
                values = []
                for name, template in initial._asdict().items():
                    value = cp['state_'+name]
                    if value.shape != np.shape(template) or value.dtype != np.asarray(template).dtype or not np.isfinite(value).all():
                        raise ValueError(f'Invalid checkpoint field {name}')
                    values.append(jnp.asarray(value))
                state = ImpactState(*values)
                start_frame, steps = int(cp['frame']), int(cp['steps'])
                solver_seconds = float(cp['solver_seconds'])
                tables = json.loads(str(cp['tables']))
                rows, diagnostics, times, frames = (tables[k] for k in ('rows','diagnostics','times','frames'))
            validate_state(state,material,grid)
            if (not 0 <= start_frame <= plan['frames'] or steps < start_frame
                or any(len(a) != start_frame+1 for a in (rows,diagnostics,times,frames))
                or times != plan['output_times_s'][:start_frame+1] or float(state.time) != times[-1]
                or receipt['frame'] != start_frame or receipt['steps'] != steps or receipt['time'] != times[-1]
                or saved_status['frame'] != start_frame or saved_status['steps'] != steps
                or saved_status.get('time') != times[-1] or saved_status.get('frames') != plan['frames']
                or saved_status.get('state_valid') is not True
                or saved_status.get('completed') is not (saved_status['phase'] == 'completed')
                or saved_status.get('segment_complete') is not (saved_status['phase'] == 'checkpoint-ready')
                or (saved_status['phase'] == 'completed' and start_frame != plan['frames'])
                or any(r['time'] != t or d['time'] != t for r,d,t in zip(rows,diagnostics,times))):
                raise ValueError('Checkpoint, clock, output times and tables differ')
            saved_summary = json.loads((folder/'summary.json').read_text())
            if (saved_summary.get('definition') != plan or saved_summary.get('runtime') != runtime
                    or saved_summary.get('source') != plan['source'] or saved_summary.get('particles') != [count]
                    or saved_summary.get('steps') != steps or saved_summary.get('simulation_time') != times[-1]
                    or saved_summary.get('phase') != saved_status['phase']
                    or saved_summary.get('completed') != saved_status['completed']):
                raise ValueError('Summary differs from the accepted checkpoint and protocol')
            for name, expected_rows in (('observables.csv',rows),('diagnostics.csv',diagnostics)):
                if (folder/name).read_bytes() != table_text(expected_rows).encode('utf-8'):
                    raise ValueError('Saved CSV differs from the checkpoint tables')
            check_arrays(folder/'final_state_0.npz',{k:np.asarray(v) for k,v in state._asdict().items()})
            for name in ('mass','volume','body_id'):
                if not np.array_equal(np.asarray(getattr(state,name)),np.asarray(getattr(initial,name))):
                    raise ValueError(f'Static state field changed: {name}')
            if start_frame < plan['frames']:
                validate_endpoint(stop_after_frame,plan['frames'],start_frame)
            for name,shape in shapes.items():
                arrays[name] = np.load(folder/f'trajectory_{name}.npy',mmap_mode='r',allow_pickle=False)
                if arrays[name].shape != shape or arrays[name].dtype != dtypes[name]:
                    raise ValueError('Trajectory allocation differs from original full plan')
                for f in range(start_frame+1):
                    if descriptor(arrays[name][f]) != frames[f][name] or not np.isfinite(arrays[name][f]).all():
                        raise ValueError(f'Changed or incomplete trajectory frame {name}:{f}')
            for name,value in endpoint_fields(state).items():
                if not np.array_equal(arrays[name][start_frame],value):
                    raise ValueError('Trajectory endpoint differs from actual checkpoint thermodynamics/state')
            if not np.array_equal(arrays['x'][0],geometry['x']):
                raise ValueError('Initial saved positions differ from reference geometry')
            check_arrays(folder/'trajectory.npz',{'time':np.asarray(times),'body':np.zeros(count,dtype=np.int32)})
            for name in arrays:
                arrays[name]._mmap.close()
                arrays[name] = np.load(folder/f'trajectory_{name}.npy',mmap_mode='r+',allow_pickle=False)
        else:
            arrays = {name:np.lib.format.open_memmap(folder/f'trajectory_{name}.npy',mode='w+',dtype=dtypes[name],shape=shape)
                      for name,shape in shapes.items()}
            save_npz(folder/'reference.npz',**geometry)
            save_npz(folder/'particle_fields.npz',**static)
            save_json(folder/'plan.json',plan)
            record(0)
        end_frame = plan['frames'] if stop_after_frame is None else stop_after_frame
        for frame in range(start_frame+1,end_frame+1):
            requested = plan['output_times_s'][frame]
            begun = time.monotonic()
            candidate, iterations = advance_to(state,material,grid,config,jnp.asarray(requested),max_steps=max_steps_per_frame)
            candidate.x.block_until_ready()
            solver_seconds += time.monotonic()-begun
            rejected_reason = None
            if not bool(candidate.valid) or float(candidate.time) != requested:
                rejected_reason = 'Adaptive validity or requested output time was not attained'
            else:
                try:
                    validate_state(candidate,material,grid)
                except FloatingPointError as error:
                    rejected_reason = str(error)
            if rejected_reason is not None:
                save_npz(folder/'failed_state.npz',**{name:np.asarray(value) for name,value in candidate._asdict().items()})
                save_json(folder/'failure.json',dict(completed=False,last_saved_frame=frame-1,
                    requested_frame=frame,requested_time_s=requested,attempted_steps=int(iterations),
                    candidate_time_s=float(candidate.time) if np.isfinite(float(candidate.time)) else None,
                    candidate_valid=bool(candidate.valid),reason=rejected_reason,
                    accepted_checkpoint_sha256=digest(folder/'checkpoint.npz'),
                    scope='The candidate failed within an output interval. The preceding complete accepted checkpoint and original raw failure are retained; no rollback is presented as successful advancement.'))
                status('failed')
                raise FloatingPointError('Adaptive advancement failed; original accepted checkpoint and rejected state retained')
            state = candidate
            steps += int(iterations)
            record(frame)
            print(f'metal-sphere-impact-fracture frame={frame}/{plan["frames"]} adaptive_steps={steps}',flush=True)
        if source_record() != plan['source']:
            raise ValueError('Source changed during numerical execution')
        phase = 'completed' if len(times) == plan['saved_states'] else 'checkpoint-ready'
        status(phase)
        return summary(phase)
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimal-impact',action='store_true',help='Separate coarse 5km/s contact control')
    parser.add_argument('--tiny-control',action='store_true',help='Separate low-speed continuation control; does not change the default protocol')
    parser.add_argument('--output',type=Path,default=Path('outputs/metal-sphere-impact-fracture'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-frame',type=int)
    parser.add_argument('--max-steps-per-frame',type=int,default=20000)
    args = parser.parse_args()
    if args.tiny_control and args.minimal_impact:
        parser.error('Select exactly one control')
    geometry,plan = minimal_impact_plan() if args.minimal_impact else (tiny_control_plan() if args.tiny_control else plan_case())
    if args.dry_run:
        print(json.dumps(plan,indent=2))
    else:
        run_case(geometry,plan,args.output,resume=args.resume,stop_after_frame=args.stop_after_frame,
                 max_steps_per_frame=args.max_steps_per_frame)


if __name__ == '__main__':
    main()
