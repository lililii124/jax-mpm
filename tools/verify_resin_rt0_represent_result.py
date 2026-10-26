"""Audit the exact former RT0 cell failure and the accepted 4096 split."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, '/source')
from mpm_jax.resin_apic import MACLayout
from examples.three_strand_resin_geometry import obstacle_from_checkpoint

failure_x = np.asarray([-0.0008161033244065593, -0.0012000000000000005,
                        0.0014376154025911244], dtype=np.float64)
failure_index = np.asarray([3, 10, 21], dtype=np.int32)
forming = Path('/forming')
geometry = obstacle_from_checkpoint(forming / 'checkpoint.npz', forming / 'plan.json')
layout = MACLayout(tuple(geometry['shape']), tuple(geometry['origin']), geometry['dx'])
spec = importlib.util.spec_from_file_location('resin_transport_rt0_fixed',
                                               '/work/overlay/mpm_jax/resin_transport_rt0.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)

position = jnp.asarray(failure_x[None, :])
index = jnp.asarray(failure_index[None, :])
diagnostic_path = Path('/work/original-diagnostic.json')
diagnostic = json.loads(diagnostic_path.read_text())
old_event = next(event for event in diagnostic['events'] if event['kind'] == 'representation')
old_particle = next(item for item in old_event['particles'] if item['id'] == 39204)
assert old_particle['x'] == failure_x.tolist()
assert old_particle['index'] == failure_index.tolist() and old_particle['back'] == [3, 11, 21]
raw_back = np.floor((failure_x - geometry['origin']) / geometry['dx']).astype(np.int32)
assert raw_back.tolist() == [3, 11, 21], (raw_back, geometry['origin'], geometry['dx'])
adjusted, _, valid = fixed.represent(position, index, layout)
jax.block_until_ready(adjusted)
adjusted = np.asarray(adjusted)[0]
assert bool(valid)
back = np.floor((adjusted - geometry['origin']) / geometry['dx']).astype(np.int32)
assert back.tolist() == failure_index.tolist()

q0 = (failure_x - geometry['origin']) / geometry['dx']
q1 = (adjusted - geometry['origin']) / geometry['dx']
boundary = np.where(q0 < failure_index, failure_index, failure_index + 1)
outside = (q0 < failure_index) | (q0 >= failure_index + 1)
allowed = 64 * np.finfo(np.float64).eps * np.maximum(np.abs(boundary), 1.)
assert np.all(~outside | (np.abs(q0 - boundary) <= allowed))
assert np.all(~outside | (np.abs(q1 - boundary) <= allowed))
changed = np.flatnonzero(failure_x != adjusted)
assert changed.tolist() == [1], changed
target_q = boundary + np.where(q0 < failure_index, allowed*.5, -allowed*.5)
seed = geometry['origin'] + target_q * geometry['dx']
value = float(seed[1]); destination = float(adjusted[1]); direction = math.inf if destination > value else -math.inf
steps = 0
while value != destination and steps <= 64:
    value = math.nextafter(value, direction)
    steps += 1
assert value == destination and steps <= 64

parent = Path('/parent')
fixed_output = Path('/fixed')
with np.load(parent / 'checkpoint.npz', allow_pickle=False) as cp:
    assert int(cp['step']) == 4032 and bool(cp['valid'])
    parent_sha = hashlib.sha256((parent / 'checkpoint.npz').read_bytes()).hexdigest()
    parent_particles = len(cp['x'])
with np.load(fixed_output / 'checkpoint.npz', allow_pickle=False) as cp:
    assert int(cp['step']) == 4096 and bool(cp['valid'])
    assert all(np.isfinite(cp[k]).all() for k in cp.files if cp[k].dtype.kind in 'fci')
    state_time = float(cp['time'])
    source_time = float(cp['inlet_source_time_s'])
    output_particles = len(cp['x'])
    assert all(len(cp[k]) == output_particles for k in ['v', 'C', 'mass', 'active'])
    fixed_sha = hashlib.sha256((fixed_output / 'checkpoint.npz').read_bytes()).hexdigest()
assert output_particles >= parent_particles
rows = json.loads((fixed_output / 'observables.json').read_text())
assert [row['step'] for row in rows] == list(range(4033, 4097))
for row in rows:
    assert row['valid'] and row['particle_wall_valid']
    assert row['pressure_coverage_valid'] and row['physical_pressure_domain_valid']
    assert row['transport_extension_valid'] and row['inlet_face_remainder_valid']
    assert row['pressure_relative_residual'] <= 1e-7
    assert row['pressure_stationarity_relative'] <= 1e-10
    assert row['solid_cell_intrusions'] == 0 and row['escaped_particles'] == 0
    assert abs(row['inlet_volume_balance_error_m3']) <= 1e-20
    assert abs(row['mass_identity_kg']) <= 1e-12 * max(row['source_mass_kg'], 1e-30)
    energy_scale = max(row['source_energy_J'] + row['fixture_work_cumulative_J'],
                       row['active_kinetic_J'], 1e-30)
    assert abs(row['global_energy_identity_J']) <= 1e-8 * energy_scale
    assert abs(row['source_clock_error_s']) <= 1e-10
    assert math.isfinite(row['momentum_step_residual_Ns'])
nominal_time = 4096 * 4e-5
assert abs(state_time - rows[-1]['actual_state_time_s']) <= 1e-12
assert abs(state_time - nominal_time) <= 8 * 4096 * math.ulp(nominal_time)
assert abs(source_time - nominal_time) <= 1e-15

result = dict(status='passed', original_rejected_step=4044,
    corrected_checkpoint_step=4096, accepted_steps=len(rows),
    parent_particles=parent_particles, output_particles=output_particles,
    no_particle_state_pruning=True, checkpoint_time_s=state_time,
    nominal_time_s=nominal_time, inlet_source_time_s=source_time,
    final_mass_identity_kg=rows[-1]['mass_identity_kg'],
    final_global_energy_identity_J=rows[-1]['global_energy_identity_J'],
    all_new_rows_valid=True, solid_cell_intrusions_max=0, escaped_particles_max=0,
    original_64eps_boundary_guard_retained=True,
    original_observed_back=old_particle['back'], raw_physical_back=raw_back.tolist(),
    fixed_represent_back=back.tolist(), original_adjusted=old_particle['adjusted'],
    corrected_particle=failure_x.tolist(), corrected_position=adjusted.tolist(),
    original_q=q0.tolist(), corrected_q=q1.tolist(), boundary=boundary.tolist(),
    boundary_distance_corrected=np.abs(q1-boundary).tolist(),
    allowed_boundary_distance=allowed.tolist(),
    target_to_stored_physical_nextafter_steps=steps,
    corrected_physical_displacement_m=float(np.linalg.norm(adjusted-failure_x)),
    input_checkpoint_sha256=parent_sha, output_checkpoint_sha256=fixed_sha,
    forming_checkpoint_sha256=hashlib.sha256((forming/'checkpoint.npz').read_bytes()).hexdigest(),
    fixed_transport_sha256=hashlib.sha256(Path('/work/overlay/mpm_jax/resin_transport_rt0.py').read_bytes()).hexdigest(),
    original_diagnostic_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
    original_source_sha256={name: hashlib.sha256((Path('/source')/name).read_bytes()).hexdigest()
        for name in ['mpm_jax/resin_fill.py','mpm_jax/resin_transport_rt0.py','examples/three_strand_resin_rt0.py']})
(Path('/work')/'review.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
