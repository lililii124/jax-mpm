"""Bending and normal separation of two independent crossed MPM fibres."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import NamedTuple

import numpy as np

from .fibre_contact_common import (
    descriptor, json_value, restore_contact_history, restore_contact_model,
    save_json, save_npz,
)


MODELS = ('hertz', 'jkr', 'md', 'self-consistent')
SOURCE_FILES = (
    'examples/fibre_pull_off.py', 'examples/fibre_contact_common.py',
    'mpm_jax/fibre_mpm.py',
    'mpm_jax/fibre_coupling.py', 'mpm_jax/fibre_contact.py',
    'mpm_jax/local_contact_models.py', 'mpm_jax/maugis_dugdale.py',
    'mpm_jax/cohesive_bem.py', 'mpm_jax/materials/isotropic_brittle.py',
)


def source_record():
    root = Path(__file__).resolve().parents[1]
    files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {'files': files, 'fingerprint': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def make_geometry(*, spacing=.0002, length=.064, radius=.002, initial_gap=.00005,
                  density=1000., grip_length=.005):
    """Uniform Cartesian quadrature clipped to two straight circular cylinders."""
    values = (spacing, length, radius, density, grip_length)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Geometry lengths and density must be finite and positive')
    if not np.isfinite(initial_gap) or initial_gap < 0 or 2*grip_length >= length:
        raise ValueError('Initial gap must be nonnegative and end grips must leave a free span')
    axial_count, radial_count = round(length/spacing), round(2*radius/spacing)
    if (axial_count < 8 or radial_count < 4 or radial_count % 2
            or not np.isclose(axial_count*spacing, length, rtol=0, atol=1e-14)
            or not np.isclose(radial_count*spacing, 2*radius, rtol=0, atol=1e-14)):
        raise ValueError('Length and diameter must be integer multiples of spacing, with an even diameter count >=4')
    axial = (np.arange(axial_count)+.5)*spacing-length/2
    radial = (np.arange(radial_count)+.5)*spacing-radius
    u, v = np.meshgrid(radial, radial, indexing='ij')
    cross = np.column_stack((u.ravel(), v.ravel()))
    cross = cross[np.sum(cross*cross, axis=1) <= radius*radius]
    n = axial_count*len(cross)
    x = np.empty((2*n, 3), dtype=np.float64)
    x[:n] = np.column_stack((np.repeat(axial, len(cross)), np.tile(cross, (axial_count, 1))))
    x[n:] = np.column_stack((np.tile(cross[:, 0], axial_count),
        np.tile(cross[:, 1], axial_count)+2*radius+initial_gap, np.repeat(axial, len(cross))))
    ids = np.repeat(np.arange(2, dtype=np.int32), n)
    volume = np.full(2*n, spacing**3)
    axial_coordinate = np.concatenate((x[:n, 0], x[n:, 2]))
    ends = np.abs(axial_coordinate) >= length/2-grip_length
    masks = ((ids == 0) & ends, (ids == 1) & ends)
    if any(not mask.any() for mask in masks):
        raise ValueError('Each fibre needs occupied end grips')
    fibres = [{'id': i, 'family_id': i, 'longitudinal_sections': axial_count,
        'particle_count': n, 'particle_slice': [i*n, (i+1)*n], 'radius_m': radius}
        for i in range(2)]
    metadata = {'length_m': length, 'radius_m': radius, 'diameter_m': 2*radius,
        'initial_surface_gap_m': initial_gap, 'particle_spacing_m': spacing,
        'cross_section_particles': len(cross), 'fibres': fibres,
        'sampled_cross_section_area_m2': len(cross)*spacing**2,
        'circle_area_m2': math.pi*radius**2,
        'cross_section_area_relative_error': len(cross)*spacing**2/(math.pi*radius**2)-1,
        'grip_length_m': grip_length, 'free_span_m': length-2*grip_length,
        'sampling': 'Uniform cell-centre lattice, clipped by the reference circle; every particle has volume spacing^3.',
        'contact_coordinates': 'Mass-weighted section centroids. Symmetric straight reference sections have zero centroid offset.'}
    return {'x': x, 'volume': volume, 'mass': density*volume, 'fibre_id': ids,
        'family_id': ids.copy(), 'grip_masks': masks, 'metadata': metadata,
        'contact_candidates': [{'fibre_ids': [0, 1],
            'reference_centreline_points_m': [[0., 0., 0.], [0., 2*radius+initial_gap, 0.]]}]}


def fixture_motion(instant, protocol, xp=np):
    """Top-grip displacement/velocity; the lower fibre's end grips remain fixed."""
    t = xp.asarray(instant)
    t1, t2, t3, _ = protocol['times_s']
    down, lift = protocol['indentation_m'], protocol['lift_m']
    one, two = xp.clip(t/t1, 0., 1.), xp.clip((t-t2)/(t3-t2), 0., 1.)
    displacement = -.5*down*(1-xp.cos(xp.pi*one))+.5*(lift+down)*(1-xp.cos(xp.pi*two))
    velocity = (xp.where((t > 0) & (t < t1), -.5*down*xp.pi/t1*xp.sin(xp.pi*one), 0.)
        + xp.where((t > t2) & (t < t3), .5*(lift+down)*xp.pi/(t3-t2)*xp.sin(xp.pi*two), 0.))
    return displacement, velocity


def plan_case(*, model='jkr', dx=.0004, spacing=.0002, frames=60, duration=.03,
              indentation=.0003, lift=.0015, initial_gap=.00005, cfl=.15,
              work_of_adhesion=1.2, interaction_range=.00002, outer_radius=.001,
              rings=32, coverage_skin=.008):
    if model not in MODELS:
        raise ValueError(f'Unknown contact model {model}')
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 2:
        raise ValueError('frames must be an integer >=2')
    if (not np.isfinite((dx, duration, indentation, lift, cfl, work_of_adhesion,
                         interaction_range, outer_radius, coverage_skin)).all()
            or min(dx, duration, indentation, lift, interaction_range, outer_radius, coverage_skin) <= 0
            or not 0 < cfl <= .2 or work_of_adhesion < 0):
        raise ValueError('Require positive finite dimensions/times, 0<CFL<=.2 and W>=0')
    if not isinstance(rings, int) or isinstance(rings, bool) or rings < 8:
        raise ValueError('rings must be an integer >=8')
    material = dict(young=2e6, poisson=.25, density=1000., tensile_strength=3e5,
        fracture_energy=100., characteristic_length=dx)
    if 2*material['young']*material['fracture_energy']/material['tensile_strength']**2 <= dx:
        raise ValueError('The chosen grid cannot resolve this crack-band material')
    geometry = make_geometry(spacing=spacing, initial_gap=initial_gap, density=material['density'])
    radius, length = geometry['metadata']['radius_m'], geometry['metadata']['length_m']
    mu = material['young']/(2*(1+material['poisson']))
    lam = material['young']*material['poisson']/((1+material['poisson'])*(1-2*material['poisson']))
    speed = math.sqrt((lam+2*mu)/material['density'])
    steps = max(frames, math.ceil(duration/(cfl*dx/speed)))
    dt = duration/steps
    times = np.array([10, 12, 25, 30], dtype=float)/30*duration
    excursion = max(.005, lift+indentation+.002)
    bounds = np.array([[[-length/2, length/2], [-radius-excursion, radius+excursion], [-radius-.001, radius+.001]],
        [[-radius-.001, radius+.001], [radius+initial_gap-excursion, 3*radius+initial_gap+excursion], [-length/2, length/2]]])
    padding = 4
    first, last = np.floor(bounds[..., 0]/dx).astype(int)-padding, np.ceil(bounds[..., 1]/dx).astype(int)+padding
    shape = last-first+1
    nodes = int(np.prod(shape, axis=1).sum())
    if nodes > 5_000_000:
        raise ValueError('Packed-grid plan exceeds five million nodes')
    if coverage_skin < 2*(lift+indentation):
        raise ValueError('Coverage skin must at least cover twice the prescribed relative excursion; runtime also checks bending')
    plan = {'series': 'crossed-fibre-pull-off-v1', 'model': model, 'duration_s': duration,
        'frames': frames, 'saved_states': frames+1, 'steps': steps, 'dt': dt,
        'frame_steps': np.rint(np.linspace(0, steps, frames+1)).astype(int).tolist(),
        'geometry': geometry['metadata'], 'geometry_fields': {key: descriptor(geometry[key]) for key in ('x', 'volume', 'mass', 'fibre_id', 'family_id')},
        'grip_masks': [descriptor(mask) for mask in geometry['grip_masks']],
        'contact_candidates': geometry['contact_candidates'],
        'material': material, 'wave_speed_m_per_s': speed, 'cfl': cfl,
        'grid': {'dx_m': dx, 'padding_cells': padding, 'bounds_m': bounds.tolist(), 'shapes': shape.tolist(), 'nodes': nodes, 'max_nodes': 5_000_000},
        'protocol': {'times_s': times.tolist(), 'indentation_m': indentation, 'lift_m': lift,
            'boundary': 'Both ends of lower fibre fixed in xyz; both ends of upper fibre share prescribed y translation and fixed xz.',
            'velocity_discretization': 'Exact displacement increment over each time step divided by dt.'},
        'contact': {'effective_modulus_Pa': material['young']/(2*(1-material['poisson']**2)),
            'reference_effective_radius_m': radius, 'work_of_adhesion_J_per_m2': work_of_adhesion,
            'interaction_range_m': interaction_range, 'outer_radius_m': outer_radius, 'rings': rings,
            'coverage_skin_m': coverage_skin, 'search_window': max(4, math.ceil(.0025/spacing)+2)},
        'gravity_m_per_s2': [0., 0., 0.], 'geometry_scale': 1.,
        'scope': 'Dynamic independent-fibre bending with a circular-equivalent unresolved normal indentation law; no resolved contact-pressure footprint, exact elliptical contact, calibrated adhesion, or quasistatic strength claim.',
        'energy_scope': 'Pair force/torque and total linear impulse are checked. Changing local curvature and adhesion snap are not a complete conservative coupling.',
        'source': source_record()}
    return geometry, plan


class Totals(NamedTuple):
    steps: object
    valid: object
    reason: object
    grip_impulse: object
    grip_work: object
    grid_work: object
    contact_impulse: object
    contact_work: object
    last_grip_impulse: object
    maximum_force: object
    minimum_force: object
    maximum_radius: object
    maximum_nonrigid_displacement: object
    maximum_radius_relative_change: object


def output_directory(output, plan):
    return Path(output)/plan['model']


def _validate_endpoint(endpoint, frames, previous=-1):
    if endpoint is not None and (not isinstance(endpoint, int) or isinstance(endpoint, bool)
            or not 0 <= endpoint <= frames or endpoint <= previous):
        raise ValueError('stop_after_frame must be a later complete frame within the unchanged physical plan')


def _write_table(path, rows):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_static_fields(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError('Static particle fields differ from the reference field set')
        for name, reference in expected.items():
            value = saved[name]
            if value.dtype != reference.dtype or not np.array_equal(value, reference):
                raise ValueError(f'Static particle field {name} differs from reference data')


def run_case(geometry, plan, output, *, resume=False, stop_after_frame=None):
    """Run one immutable protocol; planned stops preserve the complete trajectory prefix."""
    _validate_endpoint(stop_after_frame, plan['frames'])
    if source_record() != plan['source']:
        raise ValueError('Source changed after this plan was built')
    for key, record in plan['geometry_fields'].items():
        if descriptor(geometry[key]) != record:
            raise ValueError(f'Geometry {key} differs from the plan')
    if (geometry['metadata'] != plan['geometry'] or geometry['contact_candidates'] != plan['contact_candidates']
            or [descriptor(mask) for mask in geometry['grip_masks']] != plan['grip_masks']):
        raise ValueError('Reference geometry, contact seeds or grip masks differ from the plan')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    folder = output_directory(output, plan)
    if resume:
        if not (folder/'checkpoint.npz').is_file():
            raise ValueError('Resume requires the complete checkpoint and all trajectory files')
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Output is not empty; use --resume or a new output root')
    folder.mkdir(parents=True, exist_ok=True)
    return _run_runtime(geometry, plan, folder, signature, resume, stop_after_frame)


def _run_runtime(geometry, plan, folder, signature, resume, stop_after_frame):
    import jax
    import jax.numpy as jnp
    import jaxlib
    from mpm_jax.fibre_mpm import (FibreConfig, FibreState, build_layout, make_state,
        reference_grips, step_with_diagnostics, stored_energy, validate_state)
    from mpm_jax.fibre_coupling import (FibreCouplingHistory, FibreCouplingMetrics,
        build_contact_data, contact_data_summary, evaluate_contact, initial_contact_history)
    from mpm_jax.local_contact_models import local_contact_model
    from mpm_jax.materials.isotropic_brittle import BrittleHistory, isotropic_brittle

    jax.config.update('jax_enable_x64', True)
    runtime = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__,
        'numpy': np.__version__, 'python': platform.python_version()}
    material = isotropic_brittle(**plan['material'])
    config = FibreConfig(plan['dt'])
    grid = plan['grid']
    layout = build_layout(geometry['x'], geometry['fibre_id'], dx=grid['dx_m'],
        padding_cells=grid['padding_cells'], reference_bounds=np.asarray(grid['bounds_m']), max_nodes=grid['max_nodes'])
    if layout.node_count != grid['nodes'] or not np.array_equal(np.asarray(layout.shape), grid['shapes']):
        raise ValueError('Packed grid differs from its NumPy plan')
    local = plan['contact']
    capture = local['interaction_range_m'] if plan['model'] in ('md', 'self-consistent') and local['work_of_adhesion_J_per_m2'] > 0 else 0.
    data = build_contact_data(geometry, search_window=local['search_window'],
        coverage_skin=local['coverage_skin_m'], proximity_margin=capture)
    law = local_contact_model(plan['model'], effective_modulus=local['effective_modulus_Pa'],
        effective_radius=local['reference_effective_radius_m'], work_of_adhesion=local['work_of_adhesion_J_per_m2'],
        interaction_range=local['interaction_range_m'], outer_radius=local['outer_radius_m'], rings=local['rings'])
    state = make_state(geometry['x'], geometry['volume'], geometry['fibre_id'], material, layout)
    contact = initial_contact_history(data, law)
    zero3 = jnp.zeros(3, dtype=jnp.float64)

    def prescribed_velocity(t):
        displacement = fixture_motion(t, plan['protocol'], jnp)[0]
        previous = fixture_motion(t-config.dt, plan['protocol'], jnp)[0]
        return zero3.at[1].set((displacement-previous)/config.dt)

    boundary = reference_grips(geometry['grip_masks'], (lambda t: zero3, prescribed_velocity))
    count, saved = len(state.x), plan['saved_states']
    body = jnp.asarray(geometry['fibre_id'])
    static_fields = {'phase': np.ones(count, dtype=np.int8), 'yarn_id': geometry['fibre_id'],
        'family_id': geometry['family_id'],
        'grip_id': geometry['grip_masks'][0].astype(np.int8)+2*geometry['grip_masks'][1].astype(np.int8)}
    shapes = {'x': (saved, count, 3), **{name: (saved, count) for name in ('damage', 'speed', 'temperature', 'failed')}}
    dtypes = {'x': np.float64, 'damage': np.float64, 'speed': np.float64, 'temperature': np.float32, 'failed': np.uint8}
    totals = Totals(jnp.asarray(0, dtype=jnp.int64), jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((2, 3)), jnp.zeros(2), jnp.zeros(2), jnp.zeros((2, 3)), jnp.zeros(2),
        jnp.zeros((2, 3)), jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.))
    rows, diagnostics, times, frame_records = [], [], [], []
    arrays = {}
    start_frame, solver_seconds, compile_seconds = 0, 0., 0.
    force, contact, metrics = jax.jit(evaluate_contact)(state, contact, data, law)
    if not bool(metrics.valid):
        raise ValueError('Initial contact configuration is invalid')

    def checked_array(checkpoint, name, template):
        value, expected = np.asarray(checkpoint[name]), np.asarray(template)
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint field: {name}')
        return jnp.asarray(value)

    try:
        if resume:
            status = json.loads((folder/'status.json').read_text(encoding='utf-8'))
            if status.get('phase') not in ('checkpoint-ready', 'completed', 'running') or status.get('signature') != signature:
                raise ValueError('A failed or mismatched run cannot be resumed')
            _validate_static_fields(folder/'particle_fields.npz', static_fields)
            with np.load(folder/'checkpoint.npz', allow_pickle=False) as cp:
                if str(cp['signature']) != signature or json.loads(str(cp['runtime'])) != runtime:
                    raise ValueError('Resume requires identical physical plan, source and numerical runtime')
                if not bool(cp['complete_saved_frame']):
                    raise ValueError('A diagnostic partial-step checkpoint is not a resumable saved frame')
                values = {key: checked_array(cp, 'state_'+key, value) for key, value in state._asdict().items() if key != 'history'}
                history = BrittleHistory(*(checked_array(cp, 'history_'+key, value) for key, value in state.history._asdict().items()))
                state = FibreState(**values, history=history)
                law = restore_contact_model(law, cp)
                contact = FibreCouplingHistory(restore_contact_history(contact.law, cp, jnp, parameters=law),
                    checked_array(cp, 'contact_radius', contact.radius), checked_array(cp, 'contact_segments', contact.segments),
                    checked_array(cp, 'contact_valid', contact.valid))
                totals = Totals(*(checked_array(cp, 'totals_'+key, value) for key, value in totals._asdict().items()))
                force = checked_array(cp, 'contact_particle_force', force)
                metrics = FibreCouplingMetrics(*(checked_array(cp, 'metrics_'+key, value) for key, value in metrics._asdict().items()))
                start_frame = int(cp['frame'])
                solver_seconds = float(cp['solver_seconds'])
                tables = json.loads(str(cp['tables']))
                rows, diagnostics, times = tables['observables'], tables['diagnostics'], tables['times']
                frame_records = tables['frame_records']
            if not bool(totals.valid) or not bool(contact.valid) or not bool(metrics.valid) or bool(jnp.any(state.history.damage)):
                raise ValueError('Only valid undamaged accepted states can resume this elastic demonstration')
            if (not 0 <= start_frame < saved or len(rows) != start_frame+1 or len(diagnostics) != start_frame+1
                    or len(times) != start_frame+1 or len(frame_records) != start_frame+1
                    or int(totals.steps) != plan['frame_steps'][start_frame]
                    or float(state.time) != times[-1] or status.get('frame') != start_frame):
                raise ValueError('Checkpoint and tables do not identify the same complete saved frame')
            bound = 32*np.finfo(float).eps*max(1, int(totals.steps))*max(config.dt, times[-1])
            expected_times = np.array(plan['frame_steps'][:start_frame+1])*config.dt
            if (np.max(np.abs(np.asarray(times)-expected_times)) > bound
                    or any(row['time'] != instant or diag['time'] != instant for row, diag, instant in zip(rows, diagnostics, times))):
                raise ValueError('Saved clock and protocol disagree')
            for name in ('mass', 'volume', 'fibre_id'):
                if not np.array_equal(np.asarray(getattr(state, name)), geometry[name]):
                    raise ValueError(f'Checkpoint static {name} differs from reference geometry')
            validate_state(state, layout)
            if start_frame < plan['frames']:
                _validate_endpoint(stop_after_frame, plan['frames'], start_frame)
            for name, shape in shapes.items():
                array = np.load(folder/f'trajectory_{name}.npy', mmap_mode='r+', allow_pickle=False)
                arrays[name] = array
                if array.shape != shape or array.dtype != np.dtype(dtypes[name]):
                    raise ValueError(f'Trajectory {name} has an incompatible shape or dtype')
                for frame in range(start_frame+1):
                    if descriptor(array[frame]) != frame_records[frame][name]:
                        raise ValueError(f'Trajectory {name} frame {frame} differs from its saved hash')
                    for first in range(0, count, 16384):
                        if not np.isfinite(array[frame, first:first+16384]).all():
                            raise ValueError(f'Trajectory {name} prefix is incomplete')
            for name, expected in (('x', state.x), ('damage', state.history.damage), ('failed', state.history.failed),
                                    ('speed', np.linalg.norm(np.asarray(state.v), axis=1))):
                if not np.array_equal(arrays[name][start_frame], np.asarray(expected)):
                    raise ValueError(f'Trajectory {name} endpoint differs from checkpoint')
            with np.load(folder/'trajectory.npz', allow_pickle=False) as index:
                if not np.array_equal(index['time'], times) or not np.array_equal(index['body'], np.zeros(count, dtype=np.int32)):
                    raise ValueError('Trajectory index differs from checkpoint')
        else:
            arrays = {name: np.lib.format.open_memmap(folder/f'trajectory_{name}.npy', mode='w+', dtype=dtypes[name], shape=shape)
                      for name, shape in shapes.items()}
            save_npz(folder/'particle_fields.npz', **static_fields)
            save_json(folder/'contact-model.json', contact_data_summary(data))
            save_json(folder/'plan.json', plan)

        @jax.jit
        def advance_block(current, attachment, current_force, current_metrics, progress, steps):
            def body_step(_, carry):
                old, old_contact, applied, old_metrics, total = carry

                def attempt(_):
                    candidate, diag = step_with_diagnostics(old, material, layout, config, applied, boundary=boundary)

                    def next_contact_step(_):
                        next_force, next_contact, next_metrics = evaluate_contact(candidate, old_contact, data, law)
                        accepted = next_metrics.valid
                        speeds = jnp.stack((zero3, prescribed_velocity(candidate.time)))
                        per_body_force = jnp.zeros((2, 3)).at[body].add(applied)
                        per_body_work = jnp.zeros(2).at[body].add(jnp.sum(applied*(candidate.x-old.x), axis=1))
                        updated = Totals(total.steps+1, total.valid, total.reason,
                            total.grip_impulse+diag.group_impulses,
                            total.grip_work+jnp.sum(diag.group_impulses*speeds, axis=1),
                            total.grid_work+diag.group_work,
                            total.contact_impulse+config.dt*per_body_force,
                            total.contact_work+per_body_work, diag.group_impulses,
                            jnp.maximum(total.maximum_force, jnp.max(next_metrics.normal_force)),
                            jnp.minimum(total.minimum_force, jnp.min(next_metrics.normal_force)),
                            jnp.maximum(total.maximum_radius, jnp.max(next_metrics.contact_radius)),
                            jnp.maximum(total.maximum_nonrigid_displacement, next_metrics.maximum_nonrigid_displacement),
                            jnp.maximum(total.maximum_radius_relative_change,
                                jnp.max(jnp.abs(next_metrics.effective_radius/local['reference_effective_radius_m']-1))))
                        return jax.lax.cond(accepted,
                            lambda _: (candidate, next_contact, next_force, next_metrics, updated),
                            lambda _: (old, old_contact, applied, old_metrics, total._replace(valid=jnp.asarray(False), reason=jnp.asarray(1, dtype=jnp.int32))), None)

                    valid = diag.valid & ~jnp.any(candidate.history.damage > 0)
                    return jax.lax.cond(valid, next_contact_step,
                        lambda _: (old, old_contact, applied, old_metrics, total._replace(valid=jnp.asarray(False),
                            reason=jnp.where(diag.valid, jnp.int32(3), jnp.int32(2)))), None)
                return jax.lax.cond(total.valid, attempt, lambda _: carry, None)
            return jax.lax.fori_loop(0, steps, body_step, (current, attachment, current_force, current_metrics, progress))

        def checkpoint(frame, name='checkpoint.npz'):
            values = {'state_'+key: np.asarray(value) for key, value in state._asdict().items() if key != 'history'}
            values.update({'history_'+key: np.asarray(value) for key, value in state.history._asdict().items()})
            values.update({'totals_'+key: np.asarray(value) for key, value in totals._asdict().items()})
            values.update({'metrics_'+key: np.asarray(value) for key, value in metrics._asdict().items()})
            values.update(contact_active=np.asarray(contact.law.active), contact_law_valid=np.asarray(contact.law.valid),
                contact_radius=np.asarray(contact.radius), contact_segments=np.asarray(contact.segments), contact_valid=np.asarray(contact.valid),
                contact_particle_force=np.asarray(force), frame=frame, signature=np.array(signature), solver_seconds=solver_seconds,
                runtime=np.array(json.dumps(runtime, sort_keys=True)),
                tables=np.array(json.dumps(json_value({'observables': rows, 'diagnostics': diagnostics,
                    'times': times, 'frame_records': frame_records}))))
            values.update(last_saved_frame=frame, partial_step_clock_s=float(state.time),
                complete_saved_frame=(name == 'checkpoint.npz' and bool(totals.valid)
                                      and int(totals.steps) == plan['frame_steps'][frame]))
            values.update({'contact_local_'+key: np.asarray(value) for key, value in contact.law._asdict().items() if key not in ('active', 'valid')})
            if hasattr(law, 'base'):
                values.update({'contact_bem_'+key: np.asarray(value) for key, value in law.base._asdict().items()})
            save_npz(folder/name, **values)

        def record(frame, old_steps, old_grip):
            x, v, mass, C, F = (np.asarray(getattr(state, name)) for name in ('x', 'v', 'mass', 'C', 'F'))
            instant, step_number = float(state.time), int(totals.steps)
            J = np.linalg.det(F)
            speed = np.linalg.norm(v, axis=1)
            arrays['x'][frame], arrays['speed'][frame] = x, speed
            arrays['damage'][frame], arrays['failed'][frame], arrays['temperature'][frame] = np.asarray(state.history.damage), np.asarray(state.history.failed), 293.15
            for array in arrays.values():
                array.flush()
            frame_records.append({name: descriptor(array[frame]) for name, array in arrays.items()})
            times.append(instant)
            displacement = fixture_motion(instant, plan['protocol'])[0]
            per_body_momentum = np.vstack([np.sum(mass[body_id == geometry['fibre_id'], None]*v[body_id == geometry['fibre_id']], axis=0) for body_id in range(2)])
            impulses = np.asarray(totals.grip_impulse)
            contact_impulse = np.asarray(totals.contact_impulse)
            interval = (step_number-old_steps)*config.dt
            reaction = (impulses-old_grip)/interval if interval else np.zeros((2, 3))
            ke_translation = .5*float(np.sum(mass*np.sum(v*v, axis=1)))
            ke_affine = grid['dx_m']**2/8*float(np.sum(mass*np.sum(C*C, axis=(1, 2))))
            elastic = float(stored_energy(state, material))
            targets = (np.zeros(3), np.array([0., displacement, 0.]))
            velocities = (np.zeros(3), np.asarray(prescribed_velocity(state.time)) if step_number else np.zeros(3))
            row = {'time': instant, 'accepted_steps': step_number, 'prescribed_upper_grip_displacement_m': float(displacement),
                'contact_force_N': float(metrics.normal_force[0]), 'contact_gap_m': float(metrics.gap[0]),
                'contact_radius_m': float(metrics.contact_radius[0]), 'effective_radius_m': float(metrics.effective_radius[0]),
                'contact_active': bool(metrics.active[0]), 'contact_potential_J': float(metrics.potential[0]),
                'contact_force_balance_N': float(np.linalg.norm(np.asarray(metrics.force_residual))),
                'contact_torque_balance_Nm': float(np.linalg.norm(np.asarray(metrics.torque_residual))),
                'kinetic_translational_energy_J': ke_translation, 'kinetic_affine_energy_J': ke_affine,
                'kinetic_energy_J': ke_translation+ke_affine, 'elastic_energy_J': elastic,
                'kinetic_to_elastic_ratio': (ke_translation+ke_affine)/elastic if elastic > 0 else None,
                'total_clamp_work_J': float(np.sum(np.asarray(totals.grip_work))),
                'grid_kinetic_correction_J': float(np.sum(np.asarray(totals.grid_work))),
                'total_contact_work_J': float(np.sum(np.asarray(totals.contact_work))),
                'maximum_step_contact_force_N': float(totals.maximum_force), 'minimum_step_contact_force_N': float(totals.minimum_force),
                'maximum_step_contact_radius_m': float(totals.maximum_radius),
                'maximum_step_radius_relative_change': float(totals.maximum_radius_relative_change),
                'maximum_step_nonrigid_displacement_m': float(totals.maximum_nonrigid_displacement),
                'grip_position_error_m': max(float(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-target, axis=1))) for mask, target in zip(geometry['grip_masks'], targets)),
                'grip_velocity_error_m_per_s': max(float(np.max(np.linalg.norm(v[mask]-target, axis=1))) for mask, target in zip(geometry['grip_masks'], velocities)),
                'max_damage': float(np.max(np.asarray(state.history.damage))), 'failed_particles': int(np.sum(np.asarray(state.history.failed)))}
            for i, label in enumerate(('lower', 'upper')):
                mask = geometry['fibre_id'] == i
                centre = np.abs(geometry['x'][:, 0 if i == 0 else 2]) <= geometry['metadata']['particle_spacing_m']
                row[label+'_centre_displacement_y_m'] = float(np.mean((x-geometry['x'])[mask & centre, 1]))
                row[label+'_clamp_work_J'] = float(totals.grip_work[i])
                row[label+'_contact_work_J'] = float(totals.contact_work[i])
                for axis, label_axis in enumerate('xyz'):
                    row[f'{label}_contact_force_{label_axis}_N'] = float(np.sum(np.asarray(force)[mask, axis]))
                    row[f'{label}_interval_grip_force_{label_axis}_N'] = float(reaction[i, axis])
                    row[f'{label}_last_step_grip_force_{label_axis}_N'] = float(totals.last_grip_impulse[i, axis]/config.dt)
                    row[f'{label}_grip_impulse_{label_axis}_Ns'] = float(impulses[i, axis])
                    row[f'{label}_contact_impulse_{label_axis}_Ns'] = float(contact_impulse[i, axis])
                    row[f'{label}_momentum_{label_axis}_Ns'] = float(per_body_momentum[i, axis])
            rows.append(row)
            diagnostics.append({'time': instant, 'mass': float(mass.sum()), 'reference_volume': float(np.sum(geometry['volume'])),
                'current_volume': float(np.sum(geometry['volume']*J)), 'min_J': float(J.min()), 'max_J': float(J.max()),
                'mass_relative_drift': float((mass.sum()-geometry['mass'].sum())/geometry['mass'].sum()),
                'body_momentum_balance_residual_Ns': float(np.max(np.linalg.norm(per_body_momentum-impulses-contact_impulse, axis=1))),
                'contact_impulse_balance_Ns': float(np.linalg.norm(contact_impulse.sum(0))),
                'max_damage': row['max_damage'], 'failed_mass': float(mass[np.asarray(state.history.failed)].sum()),
                'state_valid': bool(state.valid), 'coupling_valid': bool(totals.valid), 'failure_reason_code': int(totals.reason)})
            _write_table(folder/'observables.csv', rows)
            _write_table(folder/'diagnostics.csv', diagnostics)
            save_npz(folder/'trajectory.npz', time=np.asarray(times), body=np.zeros(count, dtype=np.int32))
            final = {key: np.asarray(value) for key, value in state._asdict().items() if key != 'history'}
            final.update({'history_'+key: np.asarray(value) for key, value in state.history._asdict().items()})
            save_npz(folder/'final_state_0.npz', **final)
            checkpoint(frame)
            save_json(folder/'status.json', {'completed': False, 'phase': 'running', 'frame': frame, 'frames': plan['frames'],
                'steps': step_number, 'target_steps': plan['steps'], 'signature': signature, 'coupling_valid': bool(totals.valid)})

        def summary(phase):
            complete = phase == 'completed'
            return {'case': f'fibre-pull-off/{plan["model"]}', 'completed': complete, 'production_eligible': False,
                'segment_complete': phase == 'checkpoint-ready', 'phase': phase, 'segment_end_frame': len(rows)-1,
                'last_saved_frame': len(rows)-1, 'last_saved_time_s': times[-1],
                'partial_step_clock_s': float(state.time) if phase == 'failed' else None,
                'particles': [count], 'steps': int(totals.steps), 'target_steps': plan['steps'],
                'simulation_time': float(state.time), 'target_simulation_time': plan['duration_s'],
                'solver_seconds': solver_seconds, 'compile_and_warmup_seconds': compile_seconds,
                'timing_scope': 'Solver time accumulates accepted block calls across segments; compilation time is for the current segment only and excludes setup and checkpoint IO.',
                'dtype': str(state.x.dtype), 'runtime': runtime, 'source': plan['source'], 'definition': plan,
                'configs': [{'dim': 3, 'dt': config.dt, 'domain_size': geometry['metadata']['length_m'],
                    'n_grid': round(geometry['metadata']['length_m']/layout.dx), 'grid_type': 'independent-packed-rectangular-fibre-fields',
                    'failure_mode': 'retain', 'gravity': [0., 0., 0.]}], 'materials': [plan['material']],
                'particle_spacing_m': geometry['metadata']['particle_spacing_m'],
                'particle_fields': {'file': 'particle_fields.npz', 'fields': {
                    name: descriptor(value) for name, value in static_fields.items()}},
                'contact_model': contact_data_summary(data), 'geometry_scale': 1.,
                'min_J': min(item['min_J'] for item in diagnostics), 'max_damage': max(item['max_damage'] for item in rows),
                'scope': plan['scope'], 'energy_scope': plan['energy_scope'],
                'force_sampling': 'Contact force/history correspond to each saved position; grip forces are separately labelled interval averages and last-step impulses/dt.',
                'initial': diagnostics[0], 'final': diagnostics[-1], 'observables': {'initial': rows[0], 'final': rows[-1]}}

        if not resume:
            record(0, 0, np.zeros((2, 3)))
            save_json(folder/'summary.json', summary('running'))
        end_frame = plan['frames'] if stop_after_frame is None else stop_after_frame
        if end_frame > start_frame:
            started = time.monotonic()
            compiled = advance_block.lower(state, contact, force, metrics, totals, jnp.asarray(1, dtype=jnp.int32)).compile()
            compile_seconds = time.monotonic()-started
        for frame in range(start_frame+1, end_frame+1):
            before_steps, before_grip = int(totals.steps), np.asarray(totals.grip_impulse).copy()
            requested = plan['frame_steps'][frame]-before_steps
            started = time.monotonic()
            state, contact, force, metrics, totals = compiled(state, contact, force, metrics, totals, jnp.asarray(requested, dtype=jnp.int32))
            state.x.block_until_ready()
            solver_seconds += time.monotonic()-started
            if not bool(totals.valid):
                checkpoint(frame-1, 'rejected_checkpoint.npz')
                save_json(folder/'summary.json', summary('failed'))
                save_json(folder/'status.json', {'completed': False, 'phase': 'failed', 'frame': frame-1,
                    'steps': int(totals.steps), 'failure_reason_code': int(totals.reason), 'signature': signature})
                raise FloatingPointError(f'Coupled step refused after {int(totals.steps)} accepted steps; reason {int(totals.reason)}')
            record(frame, before_steps, before_grip)
            print(f'{plan["model"]} frame={frame}/{plan["frames"]} step={int(totals.steps)}/{plan["steps"]}', flush=True)
        if source_record() != plan['source']:
            raise ValueError('Source changed during the run')
        phase = 'completed' if int(totals.steps) == plan['steps'] else 'checkpoint-ready'
        result = summary(phase)
        save_json(folder/'summary.json', result)
        save_json(folder/'status.json', {'completed': result['completed'], 'segment_complete': result['segment_complete'],
            'phase': phase, 'frame': len(rows)-1, 'frames': plan['frames'], 'steps': int(totals.steps),
            'target_steps': plan['steps'], 'signature': signature, 'coupling_valid': bool(totals.valid)})
        return result
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='jkr')
    parser.add_argument('--dx', type=float, default=.0004)
    parser.add_argument('--spacing', type=float, default=.0002)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--rings', type=int, default=32)
    parser.add_argument('--output', type=Path, default=Path('outputs/fibre-pull-off'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-frame', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for model in MODELS if args.model == 'all' else (args.model,):
        geometry, plan = plan_case(model=model, dx=args.dx, spacing=args.spacing, frames=args.frames, rings=args.rings)
        if args.dry_run:
            print(json.dumps({'model': model, 'particles': len(geometry['x']), 'steps': plan['steps'],
                'grid_nodes': plan['grid']['nodes'], 'definition': plan}, indent=2))
        else:
            run_case(geometry, plan, args.output, resume=args.resume, stop_after_frame=args.stop_after_frame)


if __name__ == '__main__':
    main()
