"""Explicit extension of separate interlaced fibres with Hertz or JKR contact.

Dry runs use NumPy only. Probe and smoke results are not production envelopes.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .triaxial_fibre_geometry import make_geometry


ANGLES = tuple(range(0, 31, 5))
SOURCE_FILES = (
    "examples/triaxial_failure.py", "examples/triaxial_fibre_geometry.py",
    "mpm_jax/fibre_mpm.py", "mpm_jax/fibre_coupling.py", "mpm_jax/fibre_contact.py",
    "mpm_jax/materials/isotropic_brittle.py",
)


def source_record():
    root = Path(__file__).resolve().parents[1]
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {"files": files, "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def descriptor(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def loading_motion(instant, stroke, load_time, xp=np):
    """Displacement and velocity of both fixtures, with smooth start and stop."""
    t = xp.asarray(instant)
    phase = xp.clip(t/load_time, 0., 1.)
    half_displacement = .25*stroke*(1-xp.cos(xp.pi*phase))
    half_velocity = xp.where((t > 0) & (t < load_time), .25*stroke*xp.pi/load_time*xp.sin(xp.pi*phase), 0.)
    direction = xp.asarray([[-1., 0., 0.], [1., 0., 0.]])
    return half_displacement[..., None, None]*direction, half_velocity[..., None, None]*direction


def packed_grid_plan(geometry, dx, padding_cells, max_nodes):
    bounds = np.asarray([item['reference_tube_bounds_m'] for item in geometry['metadata']['fibres']]).reshape(-1, 3, 2)
    lower_indices = np.floor(bounds[:, :, 0]/dx)-padding_cells
    upper_indices = np.ceil(bounds[:, :, 1]/dx)+padding_cells
    if (not np.isfinite(lower_indices).all() or not np.isfinite(upper_indices).all()
            or np.any(np.abs(lower_indices) >= 2**30) or np.any(np.abs(upper_indices) >= 2**30)):
        raise ValueError("Packed grid coordinates exceed the int32 index range")
    lower = lower_indices.astype(np.int64)
    upper = upper_indices.astype(np.int64)
    shape = upper-lower+1
    if np.any(shape <= 0) or np.any(shape >= 2**30):
        raise ValueError("Packed grid shape exceeds the int32 index range")
    counts = [math.prod(row) for row in shape.tolist()]
    nodes = sum(counts)
    if nodes > max_nodes:
        raise ValueError(f"Packed grids require {nodes} nodes, above max_nodes={max_nodes}")
    return {"origin_m": [0., 0., 0.], "dx_m": dx, "padding_cells": padding_cells,
            "nodes": nodes, "max_nodes": max_nodes, "fibre_lower_indices": lower.tolist(),
            "fibre_shapes": shape.tolist(), "fibre_node_counts": counts,
            "budget_scope": "max_nodes limits dense AABB addressing, including when grid_storage=tube. It is not an active-node or process-memory limit.",
            "seven_float64_grid_scalars_bytes": nodes*7*8,
            "memory_note": "Seven grid scalars are a storage subtotal, not a peak-memory prediction. Particle stencils, constitutive temporaries, contact search, compiled executables and checkpoints require additional memory."}


def plan_case(theta=0, *, law='jkr', smoke=False, probe_steps=0, frames=60, max_strain=.02,
              dx=75e-6, padding_cells=6, max_nodes=35_000_000, coverage_skin=.0005,
              search_window=6, work_of_adhesion=.1, particle_spacing=45e-6, grid_storage='dense',
              load_transits=10., dimensions=None, grip_length=None, diameter=235e-6,
              pitch=.001, crimp=140e-6, clearance=2e-6, young=10e9, poisson=.25,
              density=1500., tensile_strength=100e6, fracture_energy=100., packing=1.,
              max_particles=1_000_000):
    """Return a NumPy geometry and a reproducible plan without importing JAX."""
    if theta not in ANGLES or law not in ('hertz', 'jkr'):
        raise ValueError("theta must be 0..30 degrees in five-degree steps; law must be hertz or jkr")
    if grid_storage not in ('dense', 'tube'):
        raise ValueError("grid_storage must be dense or tube")
    if any(type(value) is not int or value < limit for value, limit in ((frames, 1), (probe_steps, 0), (padding_cells, 2), (search_window, 1), (max_nodes, 1), (max_particles, 1))):
        raise ValueError("Frame, step, padding, window and node budgets must be valid integers")
    if max_nodes >= 2**31:
        raise ValueError("max_nodes must be an int32-compatible dense addressing budget")
    if any(np.ndim(value) != 0 or not np.isfinite(value) or value <= 0 for value in (dx, max_strain, coverage_skin, particle_spacing, load_transits, packing, young, density, tensile_strength, fracture_energy)) or not np.isfinite(work_of_adhesion) or work_of_adhesion < 0:
        raise ValueError("Lengths, strain, loading transits, sampling density and material scales must be positive; adhesion energy must be nonnegative")
    if np.ndim(poisson) != 0 or not np.isfinite(poisson) or not -1 < poisson < .5:
        raise ValueError("poisson must lie strictly between -1 and 0.5")
    dimensions = np.asarray(dimensions if dimensions is not None else ((.006, .004) if smoke else (.024, .016)), dtype=np.float64)
    grip = float(grip_length if grip_length is not None else (.001 if smoke else .003))
    material = dict(young=float(young), poisson=float(poisson), density=float(density),
                    tensile_strength=float(tensile_strength), fracture_energy=float(fracture_energy),
                    characteristic_length=float(dx))
    maximum_h = 2*material['young']*material['fracture_energy']/material['tensile_strength']**2
    if not np.isfinite(maximum_h) or dx >= maximum_h:
        raise ValueError("Grid dx violates the crack-band softening-length limit")
    geometry = make_geometry(theta_deg=-theta, dimensions=dimensions, diameter=diameter, pitch=pitch,
        crimp=crimp, clearance=clearance, particle_spacing=particle_spacing, packing=packing,
        density=material['density'], grip_length=grip, max_particles=max_particles)
    grid = packed_grid_plan(geometry, dx, padding_cells, max_nodes)
    grid['storage'] = grid_storage
    if grid_storage == 'tube':
        grid.update(node_count_scope='Dense AABB upper bound; actual active nodes are constructed and validated at runtime.',
                    dense_to_active_int32_bytes=grid['nodes']*4, planned_active_nodes=None)
    gauge = dimensions[0]-2*grip
    E, nu, rho = material['young'], material['poisson'], material['density']
    wave_speed = math.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    dt = .15*dx/wave_speed
    load_time = load_transits*gauge/wave_speed
    requested_duration = 1.2*load_time
    steps = int(math.ceil(requested_duration/dt-1e-10))
    if probe_steps > steps:
        raise ValueError("The probe cannot extend beyond the complete loading protocol")
    stop_steps = probe_steps or steps
    frame_steps = [0, stop_steps] if probe_steps else np.rint(np.linspace(0, steps, min(frames, steps)+1)).astype(int).tolist()
    masks = (geometry['x'][:, 0] <= grip, geometry['x'][:, 0] >= dimensions[0]-grip)
    if not all(mask.any() for mask in masks) or np.any(masks[0] & masks[1]):
        raise ValueError("Both disjoint reference grips must contain particles")
    scope = 'smoke-probe' if smoke and probe_steps else 'probe' if probe_steps else 'smoke' if smoke else 'formal'
    plan = {"schema": "triaxial-fibre-extension-v1", "scope": scope, "production_eligible": False,
        "planning_environment": {"numpy": np.__version__, "python": platform.python_version()},
        "theta_deg": theta, "theta_definition": "Load direction relative to the reference fibre family. The fixed global X loading axis therefore uses geometry(theta_deg=-theta). No angle-symmetry reduction is assumed.",
        "geometry_theta_deg": -theta, "law": law, "material": material, "geometry": geometry['metadata'],
        "physical_parameters": {"dimensions_m": dimensions.tolist(), "grip_length_m": grip,
            "diameter_m": float(diameter), "pitch_m": float(pitch), "crimp_amplitude_m": float(crimp),
            "clearance_m": float(clearance), "material": dict(material)},
        "sampling_parameters": {"requested_particle_spacing_m": float(particle_spacing),
            "packing": float(packing), "max_particles": max_particles,
            "nominal_sampling_spacing_m": float(particle_spacing/packing**(1/3)),
            "packing_scope": "Sampling-density multiplier only; not a fibre volume fraction, physical packing change or constitutive characteristic length."},
        "fracture_length": {"characteristic_length_m": float(dx), "strict_upper_bound_m": maximum_h,
            "below_upper_bound": bool(dx < maximum_h), "scope": "h=grid dx, independent of particle spacing and sampling packing; h < 2*E*Gf/ft^2."},
        "grid": grid, "particles": len(geometry['x']), "wave_speed_m_per_s": wave_speed,
        "CFL": .15, "dt": dt, "load_time_s": load_time, "requested_duration_s": requested_duration,
        "load_transits": float(load_transits), "gauge_transit_time_s": gauge/wave_speed,
        "hold_fraction_of_loading": .2, "requested_hold_time_s": .2*load_time,
        "requested_hold_transits": .2*float(load_transits),
        "duration_rounding_seconds": steps*dt-requested_duration,
        "duration_s": steps*dt, "steps": steps, "stop_steps": stop_steps, "probe_steps": probe_steps,
        "frame_steps": frame_steps, "saved_states": len(frame_steps), "max_engineering_strain": max_strain,
        "gauge_length_m": gauge, "total_fixture_extension_m": max_strain*gauge,
        "grip_particle_counts": [int(mask.sum()) for mask in masks],
        "grip_x_intervals_m": [[0., grip], [dimensions[0]-grip, dimensions[0]]],
        "coverage_skin_m": coverage_skin, "search_window": search_window,
        "requested_work_of_adhesion_J_per_m2": work_of_adhesion,
        "work_of_adhesion_J_per_m2": work_of_adhesion if law == 'jkr' else 0.,
        "particle_spacing_m": particle_spacing, "sampling_hash": descriptor(geometry['x']),
        "trajectory_estimated_bytes": len(frame_steps)*len(geometry['x'])*(3*8+8+8+4+1),
        "interpretation": f"An explicit dynamic finite coupon with {load_transits:g} gauge-length elastic transit times of loading and a 20% hold. Kinetic/elastic energy ratio is recorded, not assumed negligible. First material failure, dynamic reaction maximum and sampled section-graph connectivity are distinct observations; they do not by themselves define a quasi-static or size-converged failure envelope."}
    plan['source'] = source_record()
    return geometry, plan


def output_directory(output, plan):
    return Path(output)/plan['scope']/f"theta-{plan['theta_deg']:02d}"/plan['law']


def _graph_observation(x, failed, geometry, data, contact_history, dx, grip_masks):
    """Saved-state section graph, explicitly distinct from a full 3D MPM graph."""
    sections = np.asarray(data.particle_section)
    active = ~failed
    occupied = np.bincount(sections, weights=active, minlength=data.section_count) > 0
    parent = np.arange(data.section_count)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for left, right in np.asarray(data.segments):
        if occupied[left] and occupied[right]:
            join(left, right)
    segments = np.asarray(contact_history.segments)
    contact_active = np.asarray(contact_history.law.active)
    for pair in segments[contact_active]:
        endpoints = np.asarray(data.segments)[pair].ravel()
        usable = endpoints[occupied[endpoints]]
        for node in usable[1:]:
            join(usable[0], node)
    roots = [{find(node) for node in np.unique(sections[mask & active])} for mask in grip_masks]
    connected = bool(roots[0] & roots[1])
    cut_count = 0
    for fibre in geometry['metadata']['fibres']:
        values = np.sort(x[active & (geometry['fibre_id'] == fibre['id']), 0])
        if not len(values):
            continue
        q = values/dx
        base = np.floor(q-.5).astype(np.int64)
        f = q-base
        weights = np.column_stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2))
        nodes = base[:, None]+np.arange(3)
        lower = np.where(weights > 0, nodes, np.iinfo(np.int64).max).min(axis=1)
        upper = np.where(weights > 0, nodes, np.iinfo(np.int64).min).max(axis=1)
        cut_count += int(np.count_nonzero(lower[1:] > np.maximum.accumulate(upper)[:-1]))
    return connected, cut_count


def validate_stop_after_frame(stop_after_frame, plan, checkpoint_frame=None):
    if stop_after_frame is None:
        return
    if type(stop_after_frame) is not int or not 0 <= stop_after_frame < plan['saved_states']:
        raise ValueError("stop_after_frame must be a saved frame index in the unchanged full plan")
    if checkpoint_frame is not None and stop_after_frame <= checkpoint_frame:
        raise ValueError("stop_after_frame must be later than the existing checkpoint frame")


def open_trajectory_arrays(folder, shapes, dtypes, resume):
    arrays = {}
    try:
        for name, shape in shapes.items():
            array = np.lib.format.open_memmap(folder/f'trajectory_{name}.npy',
                mode='r+' if resume else 'w+', dtype=dtypes[name], shape=shape)
            arrays[name] = array
            if array.shape != shape or array.dtype != np.dtype(dtypes[name]):
                raise ValueError(f"Saved trajectory {name} does not match the complete planned shape/dtype")
        return arrays
    except BaseException:
        for array in arrays.values():
            array._mmap.close()
        raise


def run_case(geometry, plan, output, *, resume=False, stop_after_frame=None):
    """Allocate the independent fields only after planning and resource admission."""
    folder = output_directory(output, plan)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if folder.exists() and not resume:
        raise ValueError(f"Output already exists: {folder}; use a fresh root or --resume")
    folder.mkdir(parents=True, exist_ok=True)
    if resume:
        previous = json.loads((folder/'plan.json').read_text())
        if previous != plan:
            raise ValueError("Resume requires identical parameters, geometry sampling and source hashes")
        status = json.loads((folder/'status.json').read_text())
        if status.get('completed') is True or status.get('probe_complete') is True:
            return json.loads((folder/'summary.json').read_text())
        if status.get('phase') == 'failed':
            raise ValueError("A failed run is diagnostic evidence, not a planned resumable segment")
        with np.load(folder/'checkpoint.npz', allow_pickle=False) as checkpoint:
            if not bool(checkpoint['aggregate'][4]):
                raise ValueError("A rejected coupled step is not an automatically resumable checkpoint")
            validate_stop_after_frame(stop_after_frame, plan, int(checkpoint['frame']))
    else:
        validate_stop_after_frame(stop_after_frame, plan)
        save_json(folder/'plan.json', plan)
    save_json(folder/'status.json', {"completed": False, "phase": "initializing", "signature": signature})
    started = time.monotonic()
    try:
        if source_record() != plan['source']:
            raise ValueError("Numerical source changed after planning")
        return _run_runtime(geometry, plan, folder, signature, resume, stop_after_frame)
    except BaseException as error:
        save_json(folder/'failure.json', {"completed": False, "production_eligible": False,
            "error_type": type(error).__name__, "error": str(error), "wall_seconds": time.monotonic()-started,
            "signature": signature, "source": plan['source']})
        status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        status.update(completed=False, phase='failed', error_type=type(error).__name__, error=str(error))
        save_json(folder/'status.json', status)
        raise


def _run_runtime(geometry, plan, folder, signature, resume, stop_after_frame=None):
    validate_stop_after_frame(stop_after_frame, plan)
    import jax
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreConfig, FibreState, build_layout, build_tube_layout, make_state, reference_grips, step_with_diagnostics, stored_energy
    from mpm_jax.fibre_coupling import FibreCouplingHistory, build_contact_data, contact_data_summary, evaluate_contact, initial_contact_history
    from mpm_jax.fibre_contact import FibreContactHistory, fibre_contact_parameters
    from mpm_jax.materials.isotropic_brittle import BrittleHistory, isotropic_brittle

    jax.config.update('jax_enable_x64', True)
    config = FibreConfig(plan['dt'])
    material = isotropic_brittle(**plan['material'])
    bounds = np.asarray([item['reference_tube_bounds_m'] for item in geometry['metadata']['fibres']])
    storage = plan['grid'].get('storage', 'dense')
    builder = build_tube_layout if storage == 'tube' else build_layout
    layout = builder(geometry['x'], geometry['fibre_id'], dx=plan['grid']['dx_m'],
        padding_cells=plan['grid']['padding_cells'], reference_bounds=bounds, max_nodes=plan['grid']['max_nodes'])
    dense_nodes = layout.node_count if layout.node_map is None else layout.node_map.shape[0]
    if dense_nodes != plan['grid']['nodes']:
        raise ValueError("Allocated layout differs from the pure NumPy grid plan")
    layout_record = {'storage': storage, 'dx_m': layout.dx, 'fibre_count': layout.fibre_count,
        'dense_aabb_nodes': dense_nodes, 'active_nodes': layout.node_count,
        'padding_cells': plan['grid']['padding_cells'],
        'scope': 'Storage compression only; absent tube support invalidates a step. Material length, physical grid phase and particle discretization are unchanged.'}
    if layout.node_map is not None:
        layout_record.update(node_map=descriptor(np.asarray(layout.node_map)), dense_indices=descriptor(np.asarray(layout.dense_indices)),
            dilation='Three axis-separable maximum filters of width 2*padding+1 on reference quadratic stencil occupancy, intersected with the original dense fibre AABB.')
    if resume and json.loads((folder/'layout.json').read_text()) != layout_record:
        raise ValueError("Resume requires the identical validated grid-storage mapping")
    save_json(folder/'layout.json', layout_record)
    data = build_contact_data(geometry, search_window=plan['search_window'], coverage_skin=plan['coverage_skin_m'])
    contact_law = fibre_contact_parameters(effective_modulus=material.young/(2*(1-material.poisson**2)),
        effective_radius=geometry['metadata']['diameter_m']/2, work_of_adhesion=plan['work_of_adhesion_J_per_m2'], model=plan['law'])
    state = make_state(geometry['x'], geometry['volume'], geometry['fibre_id'], material, layout)
    contact = initial_contact_history(data)
    intervals = plan['grip_x_intervals_m']
    masks = (geometry['x'][:, 0] <= intervals[0][1], geometry['x'][:, 0] >= intervals[1][0])
    motion = lambda t: loading_motion(t, plan['total_fixture_extension_m'], plan['load_time_s'], jnp)
    boundary = reference_grips(masks, (lambda t: motion(t)[1][0], lambda t: motion(t)[1][1]))
    count, saved = len(geometry['x']), plan['saved_states']
    fields = {'phase': np.ones(count, dtype=np.int8), 'yarn_id': geometry['fibre_id'], 'family_id': geometry['family_id']}
    save_npz(folder/'particle_fields.npz', **fields)
    field_record = {"file": "particle_fields.npz", "fields": {name: descriptor(value) for name, value in fields.items()}}
    shapes = {'x': (saved, count, 3), **{name: (saved, count) for name in ('damage', 'speed', 'temperature', 'failed')}}
    dtypes = {'x': np.float64, 'damage': np.float64, 'speed': np.float64, 'temperature': np.float32, 'failed': np.uint8}
    rows, diagnostics, times = [], [], []
    total_impulse, clamp_work, grid_ke_work = np.zeros((2, 3)), np.zeros(2), np.zeros(2)
    # [accepted steps, first failure time, peak axial force, peak force time, valid, reason]
    aggregate = np.array([0., -1., -np.inf, 0., 1., 0.])
    last_contact = np.zeros(16)
    compile_seconds, solver_seconds, start_frame = 0., 0., 0
    initial_momentum = None
    if resume:
        with np.load(folder/'checkpoint.npz', allow_pickle=False) as checkpoint:
            if str(checkpoint['signature']) != signature:
                raise ValueError("Checkpoint signature differs from current source/input")
            if str(checkpoint['jax_version']) != jax.__version__:
                raise ValueError("Resume requires the same JAX version as the saved checkpoint")
            history = BrittleHistory(*(jnp.asarray(checkpoint['history_'+key]) for key in BrittleHistory._fields))
            values = {key: jnp.asarray(checkpoint['state_'+key]) for key in FibreState._fields if key != 'history'}
            state = FibreState(**values, history=history)
            contact = FibreCouplingHistory(FibreContactHistory(jnp.asarray(checkpoint['contact_active']), jnp.asarray(checkpoint['contact_law_valid'])),
                jnp.asarray(checkpoint['contact_radius']), jnp.asarray(checkpoint['contact_segments']), jnp.asarray(checkpoint['contact_valid']))
            total_impulse, clamp_work, grid_ke_work = [checkpoint[key].copy() for key in ('total_impulse', 'clamp_work', 'grid_ke_work')]
            aggregate, last_contact = checkpoint['aggregate'].copy(), checkpoint['last_contact'].copy()
            tables = json.loads(str(checkpoint['tables']))
            rows, diagnostics, times = tables['observables'], tables['diagnostics'], tables['times']
            start_frame = int(checkpoint['frame'])
            solver_seconds = float(checkpoint['solver_seconds'])
            if 'initial_momentum' in checkpoint:
                initial_momentum = checkpoint['initial_momentum'].copy()
        if not bool(aggregate[4]):
            raise ValueError("A rejected coupled step is diagnostic evidence, not an automatically resumable checkpoint")
        if (not 0 <= start_frame < saved or len(rows) != start_frame+1 or len(diagnostics) != start_frame+1
                or len(times) != start_frame+1 or int(aggregate[0]) != plan['frame_steps'][start_frame]
                or float(state.time) != times[-1]):
            raise ValueError("Checkpoint frame, state and saved tables do not identify the same complete frame")
        validate_stop_after_frame(stop_after_frame, plan, start_frame)
    expected_mass = float(np.sum(geometry['mass']))
    if initial_momentum is None:
        initial_momentum = np.sum(np.asarray(state.mass)[:, None]*np.asarray(state.v), axis=0)-np.asarray(state.boundary_impulse)-np.asarray(state.external_impulse)

    def contact_statistics(metrics):
        minimum_gap = jnp.min(metrics.gap, initial=jnp.inf)
        minimum_force = jnp.min(metrics.normal_force, initial=0.)
        maximum_force = jnp.max(metrics.normal_force, initial=0.)
        return jnp.array([jnp.sum(metrics.search_boundary), metrics.uncovered_near_pairs, metrics.same_family_near_pairs,
            metrics.coverage_motion_exceeded, metrics.maximum_nonrigid_displacement, minimum_gap,
            jnp.sum(metrics.active), minimum_force, maximum_force, jnp.sum(metrics.failed_section_contact),
            jnp.sum(metrics.unsupported_end_contact), jnp.sum(metrics.degenerate_geometry),
            jnp.linalg.norm(metrics.force_residual), jnp.linalg.norm(metrics.torque_residual), jnp.sum(metrics.potential), metrics.valid], dtype=jnp.float64)

    @jax.jit
    def advance_block(state, contact, aggregate, impulses, work, ke, last, steps):
        def body(_, carry):
            current, attachment, totals, group_impulse, actual_work, kinetic_correction, last_metrics = carry

            def attempt(_):
                force, next_contact, metrics = evaluate_contact(current, attachment, data, contact_law)
                record = contact_statistics(metrics)

                def mechanical_step(_):
                    candidate, diag = step_with_diagnostics(current, material, layout, config, force, boundary=boundary)
                    speed = motion(candidate.time)[1]
                    tension = .5*(diag.group_impulses[1, 0]-diag.group_impulses[0, 0])/config.dt
                    first = jnp.where((totals[1] < 0) & jnp.any(candidate.history.failed), candidate.time, totals[1])
                    peak = jnp.maximum(totals[2], tension)
                    peak_time = jnp.where(tension > totals[2], candidate.time, totals[3])
                    accepted = totals.at[:4].set(jnp.array([totals[0]+1, first, peak, peak_time]))
                    return jax.lax.cond(diag.valid,
                        lambda _: (candidate, next_contact, accepted, group_impulse+diag.group_impulses,
                                   actual_work+jnp.sum(diag.group_impulses*speed, axis=1), kinetic_correction+diag.group_work, record),
                        lambda _: (current, attachment, totals.at[4:].set(jnp.array([0., 2.])), group_impulse, actual_work, kinetic_correction, record), None)

                return jax.lax.cond(metrics.valid, mechanical_step,
                    lambda _: (current, attachment, totals.at[4:].set(jnp.array([0., 1.])), group_impulse, actual_work, kinetic_correction, record), None)

            return jax.lax.cond(totals[4] > 0, attempt, lambda _: carry, None)
        return jax.lax.fori_loop(0, steps, body, (state, contact, aggregate, impulses, work, ke, last))

    def checkpoint(frame):
        values = {'state_'+key: np.asarray(value) for key, value in state._asdict().items() if key != 'history'}
        values.update({'history_'+key: np.asarray(value) for key, value in state.history._asdict().items()})
        values.update(contact_active=np.asarray(contact.law.active), contact_law_valid=np.asarray(contact.law.valid),
            contact_radius=np.asarray(contact.radius), contact_segments=np.asarray(contact.segments), contact_valid=np.asarray(contact.valid),
            total_impulse=total_impulse, clamp_work=clamp_work, grid_ke_work=grid_ke_work, aggregate=aggregate,
            last_contact=last_contact, initial_momentum=initial_momentum,
            frame=frame, signature=np.array(signature), solver_seconds=solver_seconds,
            jax_version=np.array(jax.__version__),
            tables=np.array(json.dumps(json_value({'observables': rows, 'diagnostics': diagnostics, 'times': times}))))
        save_npz(folder/'checkpoint.npz', **values)
        final = {key: np.asarray(value) for key, value in state._asdict().items() if key != 'history'}
        final.update({'history_'+key: np.asarray(value) for key, value in state.history._asdict().items()})
        save_npz(folder/'final_state_0.npz', **final)

    def record(frame, interval_steps, interval_impulse):
        x, v, mass = np.asarray(state.x), np.asarray(state.v), np.asarray(state.mass)
        damage, failed = np.asarray(state.history.damage), np.asarray(state.history.failed)
        instant = float(state.time)
        arrays['x'][frame], arrays['speed'][frame] = x, np.linalg.norm(v, axis=1)
        arrays['damage'][frame], arrays['failed'][frame], arrays['temperature'][frame] = damage, failed, 293.15
        for array in arrays.values(): array.flush()
        times.append(instant)
        translational = .5*float(np.sum(mass*np.sum(v*v, axis=1)))
        affine = np.asarray(state.C)
        affine_kinetic = layout.dx**2/8*float(np.sum(mass*np.einsum('pij,pij->p', affine, affine)))
        kinetic = translational+affine_kinetic
        elastic = float(stored_energy(state, material))
        J = np.linalg.det(np.asarray(state.F))
        connected, cuts = _graph_observation(x, failed, geometry, data, contact, plan['grid']['dx_m'], masks)
        displacement, velocity = loading_motion(instant, plan['total_fixture_extension_m'], plan['load_time_s'])
        impulse = np.asarray(interval_impulse)
        denominator = interval_steps*config.dt
        reactions = impulse/denominator if denominator else np.zeros((2, 3))
        row = {"time": instant, "accepted_steps": int(aggregate[0]), "prescribed_engineering_strain": float((displacement[1, 0]-displacement[0, 0])/plan['gauge_length_m']),
            "left_applied_force_x_N": float(reactions[0, 0]), "right_applied_force_x_N": float(reactions[1, 0]),
            "mean_grip_tensile_force_N": float(.5*(reactions[1, 0]-reactions[0, 0])),
            "peak_step_dynamic_grip_force_N": float(aggregate[2]) if aggregate[0] else 0., "peak_step_dynamic_grip_force_time_s": float(aggregate[3]),
            "first_material_failure_time_s": float(aggregate[1]) if aggregate[1] >= 0 else None,
            "failed_particles": int(failed.sum()), "max_damage": float(damage.max()),
            "damaged_reference_volume_fraction": float(np.sum(geometry['volume'][damage >= .5])/np.sum(geometry['volume'])),
            "kinetic_energy_J": kinetic, "kinetic_translational_energy_J": translational,
            "kinetic_apic_affine_energy_J": affine_kinetic, "elastic_energy_J": elastic,
            "kinetic_to_elastic_ratio": kinetic/elastic if elastic > 1e-30 else None,
            "actual_clamp_work_J": float(np.sum(clamp_work)), "grid_kinetic_correction_J": float(np.sum(grid_ke_work)),
            "contact_potential_J": float(last_contact[14]), "active_contact_pairs": int(last_contact[6]),
            "minimum_contact_gap_m": float(last_contact[5]), "contact_force_balance_N": float(last_contact[12]),
            "contact_torque_balance_Nm": float(last_contact[13]), "coverage_nonrigid_displacement_m": float(last_contact[4]),
            "sampled_section_graph_connects_grips": connected, "active_fibre_x_support_cuts": cuts,
            "grip_velocity_error_m_per_s": float(max(np.max(np.linalg.norm(v[mask]-target, axis=1)) for mask, target in zip(masks, velocity))),
            "grip_position_error_m": float(max(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-target, axis=1)) for mask, target in zip(masks, displacement)))}
        for group, name in enumerate(('left', 'right')):
            for axis, coordinate in enumerate('xyz'):
                row[f'{name}_applied_force_{coordinate}_N'] = float(reactions[group, axis])
                row[f'{name}_accumulated_impulse_{coordinate}_Ns'] = float(total_impulse[group, axis])
        rows.append(row)
        momentum = np.sum(mass[:, None]*v, axis=0)
        diagnostics.append({"time": instant, "mass": float(mass.sum()), "reference_volume": float(np.sum(geometry['volume'])),
            "current_volume": float(np.sum(geometry['volume']*J)), "min_J": float(J.min()), "max_J": float(J.max()),
            "max_damage": float(damage.max()), "failed_mass": float(mass[failed].sum()),
            "mass_relative_drift": float((mass.sum()-expected_mass)/expected_mass),
            "momentum_balance_residual_Ns": float(np.linalg.norm(momentum-initial_momentum-np.asarray(state.boundary_impulse)-np.asarray(state.external_impulse))),
            "state_valid": bool(state.valid), "coupling_valid": bool(aggregate[4]), "failure_reason_code": int(aggregate[5])})
        for filename, table in (('observables.csv', rows), ('diagnostics.csv', diagnostics)):
            with (folder/filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=table[0], lineterminator='\n')
                writer.writeheader(); writer.writerows(table)
        # The binary failed history compresses well and enables the existing mmap renderer path.
        save_npz(folder/'trajectory.npz', time=np.asarray(times), body=np.zeros(count, dtype=np.int32), failed=arrays['failed'][:frame+1])
        checkpoint(frame)
        save_json(folder/'status.json', {"completed": False, "phase": "running", "frame": frame,
            "frames": saved-1, "steps": int(aggregate[0]), "target_steps": plan['stop_steps'], "time": instant,
            "signature": signature, "coupling_valid": bool(aggregate[4])})

    arrays = open_trajectory_arrays(folder, shapes, dtypes, resume)
    try:
        if resume:
            for name, expected in (('x', state.x), ('damage', state.history.damage), ('failed', state.history.failed)):
                if not np.array_equal(arrays[name][start_frame], np.asarray(expected)):
                    raise ValueError(f"Checkpoint {name} differs from the saved trajectory frame")
        _, initial_contact, initial_metrics = evaluate_contact(state, contact, data, contact_law)
        if not resume:
            last_contact = np.asarray(contact_statistics(initial_metrics))
        metric_fields = ['search_boundary_pairs', 'uncovered_near_pairs', 'same_family_near_pairs', 'coverage_motion_exceeded',
            'maximum_nonrigid_displacement_m', 'minimum_gap_m', 'active_pairs', 'minimum_force_N', 'maximum_force_N', 'failed_section_contacts',
            'unsupported_end_contacts', 'degenerate_pairs', 'force_residual_N', 'torque_residual_Nm', 'potential_J', 'valid']
        save_json(folder/'contact-metrics.json', {"fields": metric_fields, "values": last_contact})
        if not bool(initial_metrics.valid):
            aggregate[4:6] = [0., 1.]
        if not resume:
            contact = initial_contact if bool(initial_metrics.valid) else contact
            record(0, 0, np.zeros((2, 3)))
            save_json(folder/'summary.json', {"case": f"triaxial-failure/theta-{plan['theta_deg']:02d}/{plan['law']}",
                "completed": False, "production_eligible": False, "scope": plan['scope'], "particles": [count], "steps": 0,
                "simulation_time": 0., "solver_seconds": 0., "min_J": 1., "max_damage": 0., "definition": plan,
                "particle_fields": field_record, "source": plan['source'], "layout": layout_record})
        if not bool(initial_metrics.valid):
            raise FloatingPointError("Initial contact coverage/geometry is invalid; inspect status and contact-metrics.json")
        save_json(folder/'contact-model.json', contact_data_summary(data))
        end_frame = saved-1 if stop_after_frame is None else stop_after_frame
        if end_frame > start_frame:
            save_json(folder/'status.json', {"completed": False, "phase": "compiling", "frame": start_frame, "signature": signature})
            started = time.monotonic()
            arguments = (state, contact, jnp.asarray(aggregate), jnp.asarray(total_impulse), jnp.asarray(clamp_work), jnp.asarray(grid_ke_work), jnp.asarray(last_contact), jnp.asarray(1, dtype=jnp.int32))
            compiled = advance_block.lower(*arguments).compile()
            compile_seconds = time.monotonic()-started
        for frame in range(start_frame+1, end_frame+1):
            requested = plan['frame_steps'][frame]-int(aggregate[0])
            before_steps, before_impulse = int(aggregate[0]), total_impulse.copy()
            started = time.monotonic()
            result = compiled(state, contact, jnp.asarray(aggregate), jnp.asarray(total_impulse), jnp.asarray(clamp_work),
                              jnp.asarray(grid_ke_work), jnp.asarray(last_contact), jnp.asarray(requested, dtype=jnp.int32))
            result[0].x.block_until_ready()
            solver_seconds += time.monotonic()-started
            state, contact = result[:2]
            aggregate, total_impulse, clamp_work, grid_ke_work, last_contact = [np.asarray(value) for value in result[2:]]
            record(frame, int(aggregate[0])-before_steps, total_impulse-before_impulse)
            save_json(folder/'contact-metrics.json', {"fields": metric_fields, "values": last_contact})
            print(f"theta={plan['theta_deg']} {plan['law']} frame={frame}/{saved-1} step={int(aggregate[0])}/{plan['stop_steps']} valid={bool(aggregate[4])}", flush=True)
            if not bool(aggregate[4]):
                break
        finished = bool(aggregate[4]) and int(aggregate[0]) == plan['stop_steps']
        segment_complete = (not finished and bool(aggregate[4]) and stop_after_frame is not None
            and len(rows)-1 == stop_after_frame and int(aggregate[0]) == plan['frame_steps'][stop_after_frame])
        if source_record() != plan['source']:
            raise ValueError("Numerical source changed during execution; the run is not eligible for completion")
        completed = finished and not bool(plan['probe_steps'])
        phase = 'completed' if completed else 'probe-complete' if finished else 'checkpoint-ready' if segment_complete else 'failed'
        summary = {"case": f"triaxial-failure/theta-{plan['theta_deg']:02d}/{plan['law']}", "completed": completed,
            "probe_complete": finished and bool(plan['probe_steps']), "production_eligible": False, "scope": plan['scope'],
            "segment_complete": segment_complete, "segment_start_frame": start_frame, "segment_end_frame": len(rows)-1,
            "runtime_stop_after_frame": stop_after_frame, "phase": phase,
            "history_storage": "file-backed arrays", "particles": [count], "steps": int(aggregate[0]), "simulation_time": float(state.time),
            "target_simulation_time": plan['duration_s'], "dtype": str(state.x.dtype), "solver_seconds": solver_seconds,
            "compile_and_warmup_seconds": compile_seconds, "jax_version": jax.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
            "configs": [{"dim": 3, "dt": config.dt, "domain_size": geometry['metadata']['dimensions_m'][0], "n_grid": int(round(geometry['metadata']['dimensions_m'][0]/layout.dx)),
                         "grid_type": "independent-tube-sparse-fibre-fields" if storage == 'tube' else "independent-packed-rectangular-fibre-fields",
                         "grid_storage": storage, "failure_mode": "retain", "gravity": [0., 0., 0.]}],
            "materials": [plan['material']], "definition": {**plan, "particle_spacing_m": plan['particle_spacing_m'], "contact_model": contact_data_summary(data)},
            "particle_fields": field_record, "min_J": min(item['min_J'] for item in diagnostics), "max_damage": max(item['max_damage'] for item in rows),
            "first_material_failure_time_s": float(aggregate[1]) if aggregate[1] >= 0 else None,
            "peak_dynamic_grip_force_N": float(aggregate[2]) if aggregate[0] else None, "peak_dynamic_grip_force_time_s": float(aggregate[3]),
            "first_sampled_section_graph_disconnection_time_s": next((row['time'] for row in rows if not row['sampled_section_graph_connects_grips']), None),
            "first_sampled_active_fibre_x_support_cut_time_s": next((row['time'] for row in rows if row['active_fibre_x_support_cuts']), None),
            "graph_scope": "Section graph: adjacent occupied sections within each fibre plus active contact edges. Its connectivity is an approximation, not an exact 3D MPM support graph. Positive-weight x-node interval cuts are separately reconstructed and are only a sufficient axial-disconnection diagnostic.",
            "peak_scope": "Completed dynamic protocol maximum, not a quasi-static material strength." if completed else "Partial or probe force maximum; not an accepted macroscopic failure load.",
            "actual_clamp_work_J": float(clamp_work.sum()), "grid_kinetic_correction_J": float(grid_ke_work.sum()),
            "work_scope": "Actual clamp work sums each step's group impulse dotted with the prescribed fixture velocity. Core boundary_work is separately reported as a grid kinetic-energy correction. Dynamic contact curvature work and snap dissipation are not fully represented.",
            "kinetic_energy_scope": "Particle translation plus the quadratic-APIC affine term dx^2/8 times sum(mass*Frobenius_norm(C)^2); both components are also saved separately.",
            "source": plan['source'], "layout": layout_record, "initial": diagnostics[0], "final": diagnostics[-1], "observables": {"initial": rows[0], "final": rows[-1]}}
        save_json(folder/'summary.json', summary)
        save_json(folder/'status.json', {"completed": completed, "probe_complete": summary['probe_complete'],
            "segment_complete": segment_complete, "segment_start_frame": start_frame, "segment_end_frame": len(rows)-1,
            "runtime_stop_after_frame": stop_after_frame, "coupling_valid": bool(aggregate[4]),
            "phase": phase,
            "frame": len(rows)-1, "frames": saved-1, "steps": int(aggregate[0]), "target_steps": plan['stop_steps'], "signature": signature})
        if not finished and not segment_complete:
            raise FloatingPointError(f"Coupled advance refused step {int(aggregate[0])+1}; reason code {int(aggregate[5])}; valid partial state/checkpoint retained")
        return summary
    finally:
        for array in arrays.values():
            array.flush(); array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theta', type=int, choices=ANGLES, default=0)
    parser.add_argument('--law', choices=('jkr', 'hertz'), default='jkr')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--probe-steps', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--max-strain', type=float, default=.02)
    parser.add_argument('--load-transits', type=float, default=10., help='Loading time divided by gauge-length elastic transit time; followed by a 20 percent hold')
    parser.add_argument('--dimensions', type=float, nargs=2, metavar=('LENGTH_M', 'WIDTH_M'), help='Coupon dimensions in X and Z; default 24 by 16 mm, or 6 by 4 mm with --smoke')
    parser.add_argument('--grip-length', type=float, help='Each reference grip length in metres; default 3 mm, or 1 mm with --smoke')
    parser.add_argument('--diameter', type=float, default=235e-6, help='Fibre diameter in metres')
    parser.add_argument('--pitch', type=float, default=.001, help='Weave pitch in metres')
    parser.add_argument('--crimp', type=float, default=140e-6, help='Centreline crimp amplitude in metres')
    parser.add_argument('--clearance', type=float, default=2e-6, help='Required reference surface clearance in metres')
    parser.add_argument('--young', type=float, default=10e9, help='Fibre Young modulus in Pa')
    parser.add_argument('--poisson', type=float, default=.25)
    parser.add_argument('--density', type=float, default=1500., help='Fibre mass density in kg/m^3')
    parser.add_argument('--tensile-strength', type=float, default=100e6, help='Fibre tensile strength in Pa')
    parser.add_argument('--fracture-energy', type=float, default=100., help='Fibre fracture energy in J/m^2')
    parser.add_argument('--dx', type=float, default=75e-6)
    parser.add_argument('--coverage-skin', type=float, default=.0005)
    parser.add_argument('--search-window', type=int, default=6)
    parser.add_argument('--max-nodes', type=int, default=35_000_000)
    parser.add_argument('--max-particles', type=int, default=1_000_000, help='Particle sampling budget; never changes the requested spacing automatically')
    parser.add_argument('--padding-cells', type=int, default=6)
    parser.add_argument('--grid-storage', choices=('dense', 'tube'), default='dense')
    parser.add_argument('--work-of-adhesion', type=float, default=.1)
    parser.add_argument('--particle-spacing', type=float, default=45e-6)
    parser.add_argument('--packing', type=float, default=1., help='Sampling-density multiplier only, not physical fibre packing or volume fraction')
    parser.add_argument('--output', type=Path, default=Path('outputs/triaxial-failure'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-frame', type=int,
        help='Pause after this complete saved frame without changing the physical plan; continue with --resume')
    args = parser.parse_args()
    geometry, plan = plan_case(args.theta, law=args.law, smoke=args.smoke, probe_steps=args.probe_steps, frames=args.frames,
        max_strain=args.max_strain, dx=args.dx, coverage_skin=args.coverage_skin, search_window=args.search_window,
        max_nodes=args.max_nodes, work_of_adhesion=args.work_of_adhesion, particle_spacing=args.particle_spacing,
        grid_storage=args.grid_storage, load_transits=args.load_transits, dimensions=args.dimensions,
        grip_length=args.grip_length, diameter=args.diameter, pitch=args.pitch, crimp=args.crimp,
        clearance=args.clearance, young=args.young, poisson=args.poisson, density=args.density,
        tensile_strength=args.tensile_strength, fracture_energy=args.fracture_energy, packing=args.packing,
        max_particles=args.max_particles, padding_cells=args.padding_cells)
    if args.dry_run:
        validate_stop_after_frame(args.stop_after_frame, plan)
        print(json.dumps(json_value(plan), indent=2, allow_nan=False))
    else:
        run_case(geometry, plan, args.output, resume=args.resume, stop_after_frame=args.stop_after_frame)


if __name__ == '__main__':
    main()
