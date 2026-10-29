"""Elastic sandwich material design under three prescribed reference pressures.

The graded core is an assumed effective material. Plasticity, core crushing,
delamination and nonlinear blast qualification remain outside this example.
"""
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import json
import math
import os
import struct

import numpy as np


@dataclass(frozen=True)
class Settings:
    length: float = .3
    width: float = .3
    skin: float = .002
    core: float = .02
    edge: float = .015
    xy_step: float = .025
    z_step: float = .001
    controls: tuple = (2, 2, 2)
    filter_radius: float = .015
    phi_min: float = .08
    phi_max: float = .35
    target_phi: float = .18
    peak_pressure: float = .5e6
    pulse_duration: float = .0002
    horizon: float = .002
    cfl: float = .15
    displacement_limit: float = .005
    stress_limit: float = 200e6
    block_steps: int = 32
    stage: str = 'fixed-mass'


SOURCE_FILES = ('examples/blast_sandwich_design.py', 'mpm_jax/compact_tl.py',
                'mpm_jax/total_lagrangian.py', 'mpm_jax/materials/elastic.py')
CASE_NAMES = ('uniform', 'central', 'eccentric')
ROW_FIELDS = ('step', 'time_s', 'pressure_Pa', 'back_mean_z_m', 'front_mean_z_m',
              'peak_back_displacement_m', 'peak_skin_stress_Pa', 'min_J',
              'kinetic_J', 'elastic_J', 'pressure_work_J', 'support_work_J',
              'energy_residual_J', 'pressure_impulse_z_Ns', 'support_impulse_z_Ns',
              'momentum_error_Ns', 'angular_momentum_error_Nms', 'maximum_strain', 'valid')


def describe(settings):
    s = settings
    positive = (s.length, s.width, s.skin, s.core, s.xy_step, s.z_step,
                s.filter_radius, s.peak_pressure, s.pulse_duration, s.horizon,
                s.displacement_limit, s.stress_limit)
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError('Finite positive dimensions, filter, loading and limits required')
    if not 0 <= s.edge < min(s.length, s.width)/2 or not 0 < s.cfl <= .2:
        raise ValueError('Invalid clamped edge width or acoustic CFL')
    if not 0 < s.phi_min < s.target_phi < s.phi_max <= 1 or s.pulse_duration >= s.horizon:
        raise ValueError('Require 0<phi_min<target_phi<phi_max<=1 and pulse<horizon')
    if s.stage not in ('fixed-mass', 'min-mass') or s.block_steps < 1:
        raise ValueError('Unknown design stage or invalid block length')
    if len(s.controls) != 3 or any(type(n) is not int or n < 2 for n in s.controls):
        raise ValueError('Three control counts >=2 required')
    nx, ny = math.ceil(s.length/s.xy_step), math.ceil(s.width/s.xy_step)
    skin_cells = max(2, math.ceil(s.skin/s.z_step))
    dz = s.skin/skin_cells
    core_cells = round(s.core/dz)
    if abs(core_cells*dz-s.core) > 1e-12 or core_cells < 2:
        raise ValueError('Core thickness must tile the skin-resolved thickness grid')
    nz = 2*skin_cells+core_cells
    spacing = (s.length/nx, s.width/ny, dz)
    particles = 8*nx*ny*nz
    nodes = (nx+3)*(ny+3)*(nz+3)
    # The fixed step is bounded by the stiffest admissible constituent, not the current design.
    def speed(E, nu, rho):
        return math.sqrt(E*(1-nu)/(rho*(1+nu)*(1-2*nu)))
    wave = max(speed(70e9, .33, 2700), speed(.3*70e9*s.phi_max**2, .25, 2700*s.phi_max))
    dt_cap = s.cfl/(wave*math.sqrt(sum(1/h**2 for h in spacing)))
    steps = math.ceil(s.horizon/dt_cap)
    dt = s.horizon/steps
    t = np.arange(steps+1)*dt
    pulse = np.where(t < s.pulse_duration, np.sin(np.pi*t/s.pulse_duration)**2, 0.)
    impulse = s.peak_pressure*dt*(pulse.sum()-.5*(pulse[0]+pulse[-1]))
    state_bytes = 8*(9*nodes+9*particles+32)
    support_bytes = particles*27*(4+8+3*8)
    reverse_states = math.ceil(steps/s.block_steps)+s.block_steps
    return dict(settings=asdict(s), cells=(nx, ny, nz), skin_cells=skin_cells,
                spacing_m=spacing, particles=particles, nodes=nodes,
                design_variables=math.prod(s.controls)+(s.stage == 'min-mass'),
                steps=steps, dt_s=dt, fixed_wave_speed_bound_m_s=wave,
                nominal_pressure_impulse_Pa_s=s.peak_pressure*s.pulse_duration/2,
                discrete_pressure_impulse_Pa_s=float(impulse),
                resultant_impulse_Ns=float(impulse*s.length*s.width),
                uniform_mass_kg=2700*s.length*s.width*(2*s.skin+s.target_phi*s.core),
                memory_estimate=dict(support_bytes=support_bytes, state_bytes=state_bytes,
                    raw_all_step_state_bytes=state_bytes*steps,
                    reverse_checkpoint_state_bytes=reverse_states*state_bytes,
                    scope='Array lower bounds, excluding compiler, AD residuals and particle gather temporaries.'),
                qualification='elastic_effective_core_only; nonlinear blast qualification incomplete')


def geometry(settings):
    plan = describe(settings)
    s = settings
    nx, ny, nz = plan['cells']
    h = np.asarray(plan['spacing_m'])
    cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'), axis=-1).reshape(-1, 3)
    gauss = .5+np.asarray(list(product((-1., 1.), repeat=3)))/(2*math.sqrt(3))
    points = ((cells[:, None, :]+gauss)*h).reshape(-1, 3)
    volume = np.full(len(points), np.prod(h)/8)
    z = points[:, 2]
    skin = (z < s.skin) | (z > s.skin+s.core)
    perimeter = ((points[:, 0] < s.edge) | (points[:, 0] > s.length-s.edge)
                 | (points[:, 1] < s.edge) | (points[:, 1] > s.width-s.edge))
    design = ~skin & ~perimeter
    if not design.any():
        raise ValueError('Grid leaves no designable core quadrature')
    face_cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij'), axis=-1).reshape(-1, 2)
    face_gauss = .5+np.asarray(list(product((-1., 1.), repeat=2)))/(2*math.sqrt(3))
    xy = ((face_cells[:, None, :]+face_gauss)*h[:2]).reshape(-1, 2)
    face_area = np.full(len(xy), h[0]*h[1]/4)
    footprints = [np.ones(len(xy))]
    for center in ((.5, .5), (.7, .35)):
        scaled = (xy/np.asarray((s.length, s.width))-center)/.2
        footprints.append(np.exp(-.5*np.sum(scaled**2, axis=1)))
    footprints = np.asarray([f/(np.dot(face_area, f)/face_area.sum()) for f in footprints])
    top = np.column_stack((xy, np.full(len(xy), 2*s.skin+s.core)))
    bottom = np.column_stack((xy, np.zeros(len(xy))))
    return dict(plan=plan, points=points, volume=volume, skin=skin, design=design,
                top=top, bottom=bottom, face_area=face_area, footprints=footprints)


def control_interpolation(points, settings):
    s = settings
    position = points/np.asarray((s.length, s.width, s.core))
    position[:, 2] = (points[:, 2]-s.skin)/s.core
    q = np.clip(position, 0, 1)*(np.asarray(s.controls)-1)
    lo = np.minimum(np.floor(q).astype(int), np.asarray(s.controls)-2)
    fraction = q-lo
    indices, weights = [], []
    for offset in product((0, 1), repeat=3):
        ids = lo+offset
        indices.append(np.ravel_multi_index(tuple(ids[:, a] for a in range(3)), s.controls))
        weights.append(np.prod(np.where(np.asarray(offset), fraction, 1-fraction), axis=1))
    return np.asarray(indices, dtype=np.int32).T, np.asarray(weights).T


def source_signature(settings):
    root = Path(__file__).resolve().parents[1]
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(dict(settings=asdict(settings), sources=sources), sort_keys=True).encode()).hexdigest(), sources


def build_model(settings):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax import lax
    from mpm_jax.compact_tl import (rectilinear_support, reference_grid, point_material,
        surface_weights, initial_state, step, elastic_energy, face_stress)
    from mpm_jax.total_lagrangian import momentum

    s = settings
    geo = geometry(s)
    plan = geo['plan']
    support = rectilinear_support(geo['points'], geo['volume'], plan['spacing_m'])
    if len(support.nodes) != plan['nodes']:
        raise ValueError('Compact grid differs from its capacity plan')
    top_ids, top_w = surface_weights(support, geo['top'])
    bottom_ids, bottom_w = surface_weights(support, geo['bottom'])
    pressure_shape = np.zeros((3, len(support.nodes), 3))
    for k in range(3):
        np.add.at(pressure_shape[k, :, 2], top_ids.ravel(),
                  (-geo['face_area'][:, None]*geo['footprints'][k, :, None]*top_w).ravel())
    expected = -s.length*s.width
    if not np.allclose(pressure_shape.sum(axis=1)[:, 2], expected, atol=1e-14, rtol=1e-13):
        raise ValueError('Pressure footprint resultant changed')
    nodes = support.nodes
    fixed_scalar = ((nodes[:, 0] <= s.edge) | (nodes[:, 0] >= s.length-s.edge)
                    | (nodes[:, 1] <= s.edge) | (nodes[:, 1] >= s.width-s.edge))
    fixed = jnp.asarray(np.broadcast_to(fixed_scalar[:, None], nodes.shape))
    indices, weights = map(jnp.asarray, control_interpolation(geo['points'], s))
    design, skin = jnp.asarray(geo['design']), jnp.asarray(geo['skin'])
    volume = jnp.asarray(geo['volume'])
    core_volume = jnp.sum(jnp.where(skin, 0., volume))
    variable_fraction = float(geo['volume'][geo['design']].sum()/geo['volume'][~geo['skin']].sum())
    lower_mean = s.target_phi*(1-variable_fraction)+s.phi_min*variable_fraction
    upper_mean = s.target_phi*(1-variable_fraction)+s.phi_max*variable_fraction
    control_spacing = np.asarray((s.length, s.width, s.core))/(np.asarray(s.controls)-1)
    axes = []
    for pitch in control_spacing:
        radius = max(1, math.ceil(2*s.filter_radius/pitch))
        ax = np.arange(-radius, radius+1)*pitch
        axes.append(np.exp(-.5*(ax/s.filter_radius)**2))
    kernel = jnp.asarray(np.einsum('i,j,k->ijk', *axes))
    kernel = kernel[None, None]/jnp.sum(kernel)
    normalization = lax.conv_general_dilated(jnp.ones((1, 1)+s.controls), kernel, (1, 1, 1), 'SAME',
        dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'))
    load_shape = jnp.asarray(pressure_shape)
    back_ids, back_w = jnp.asarray(bottom_ids), jnp.asarray(bottom_w)
    front_ids, front_w = jnp.asarray(top_ids), jnp.asarray(top_w)
    steps, dt = plan['steps'], plan['dt_s']

    def properties(q):
        field = lax.conv_general_dilated(q[:math.prod(s.controls)].reshape((1, 1)+s.controls),
            kernel, (1, 1, 1), 'SAME', dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'))/normalization
        field = jnp.sum(field.reshape(-1)[indices]*weights, axis=1)
        target = (lower_mean+(upper_mean-lower_mean)*jax.nn.sigmoid(q[-1])
                  if s.stage == 'min-mass' else s.target_phi)
        def fraction(shift):
            return s.phi_min+(s.phi_max-s.phi_min)*jax.nn.sigmoid(field+shift)
        def residual(shift):
            phi = jnp.where(design, fraction(shift), s.target_phi)
            return jnp.sum(jnp.where(skin, 0., volume*phi))/core_volume-target
        def solve(function, initial):
            def body(_, bounds):
                lo, hi = bounds
                mid = .5*(lo+hi)
                return jnp.where(function(mid) > 0, lo, mid), jnp.where(function(mid) > 0, mid, hi)
            lo, hi = lax.fori_loop(0, 64, body, (jnp.asarray(-50.), jnp.asarray(50.)))
            return .5*(lo+hi)
        shift = lax.custom_root(residual, jnp.asarray(0.), solve, lambda g, y: y/g(jnp.asarray(1.)))
        phi = jnp.where(skin, 1., jnp.where(design, fraction(shift), s.target_phi))
        density = 2700*phi
        young = jnp.where(skin, 70e9, .3*70e9*phi**2)
        poisson = jnp.where(skin, .33, .25)
        return phi, density, young, point_material(young, poisson)

    def pressure(n):
        t = n*dt
        return jnp.where(t < s.pulse_duration, s.peak_pressure*jnp.sin(jnp.pi*t/s.pulse_duration)**2, 0.)

    def norm_update(scale, power, values):
        following = lax.stop_gradient(jnp.maximum(scale, jnp.max(jnp.abs(values))))
        power = power*(scale/following)**24+jnp.sum((jnp.abs(values)/following)**24)
        return following, power

    def aggregate_initial():
        return jnp.asarray([0., 1e-12, 1e-24, 1e-12, 1e-24, 0., 0., 1., 0., 1., 0., 0., 0., 0.])

    def advance(state, aggregate, reference, material, case, n):
        following, load_impulse, boundary_impulse = step(state, reference, material, dt,
            pressure(n)*load_shape[case], pressure(n+1)*load_shape[case], fixed)
        back = jnp.sum(following.displacement[back_ids]*back_w[..., None], axis=1)[:, 2]
        front = jnp.sum(following.displacement[front_ids]*front_w[..., None], axis=1)[:, 2]
        vm = jnp.where(skin, face_stress(following.F, material), 0.)
        ds, dp = norm_update(aggregate[1], aggregate[2], back/s.displacement_limit)
        ss, sp = norm_update(aggregate[3], aggregate[4], vm/s.stress_limit)
        peak_d, peak_s = jnp.max(jnp.abs(back)), jnp.max(vm)
        kinetic = .5*jnp.sum(reference.nodal_mass[:, None]*following.velocity**2)
        elastic = elastic_energy(following.F, reference, material)
        error = kinetic+elastic-following.body_work-following.boundary_work
        strain = .5*(following.F+jnp.swapaxes(following.F, -1, -2))-jnp.eye(3)
        maximum_strain = jnp.max(jnp.abs(strain))
        min_J = jnp.min(jnp.linalg.det(following.F))
        updated = jnp.asarray([aggregate[0]+jnp.mean(back**2), ds, dp, ss, sp,
            jnp.maximum(aggregate[5], peak_d), jnp.maximum(aggregate[6], peak_s),
            jnp.minimum(aggregate[7], min_J), jnp.maximum(aggregate[8], jnp.abs(error)),
            jnp.minimum(aggregate[9], following.valid.astype(jnp.float64)),
            jnp.maximum(aggregate[10], maximum_strain), aggregate[11]+load_impulse[2],
            aggregate[12]+boundary_impulse[2], jnp.where(peak_d > aggregate[5], (n+1)*dt, aggregate[13])])
        P, L = momentum(following, reference)
        row = jnp.asarray([n+1, (n+1)*dt, pressure(n+1), jnp.mean(back), jnp.mean(front),
            peak_d, peak_s, min_J, kinetic, elastic, following.body_work, following.boundary_work,
            error, updated[11], updated[12], jnp.linalg.norm(P-following.impulse),
            jnp.linalg.norm(L-following.angular_impulse), maximum_strain, following.valid.astype(jnp.float64)])
        return following, updated, row

    def simulate(q, case):
        _, density, _, material = properties(q)
        reference = reference_grid(support, density)
        state = initial_state(reference, material, pressure(0)*load_shape[case])
        def block(carry, start):
            def one(carry, offset):
                n = start*s.block_steps+offset
                def actual(carry):
                    state, aggregate, _ = advance(*carry, reference, material, case, n)
                    return state, aggregate
                return lax.cond(n < steps, actual, lambda carry: carry, carry), None
            return lax.scan(one, carry, jnp.arange(s.block_steps))[0], None
        (state, a), _ = lax.scan(jax.checkpoint(block, prevent_cse=False),
            (state, aggregate_initial()), jnp.arange(math.ceil(steps/s.block_steps)))
        return jnp.asarray([a[0]/(steps*s.displacement_limit**2), a[1]*a[2]**(1/24),
            a[3]*a[4]**(1/24), a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]])

    baseline_mass = plan['uniform_mass_kg']
    def metrics(q):
        phi, density, _, _ = properties(q)
        mass = jnp.dot(volume, density)
        cases = lax.map(lambda case: jax.checkpoint(lambda q: simulate(q, case), prevent_cse=False)(q), jnp.arange(3), batch_size=1)
        response = jnp.mean(cases[:, 0])
        displacement_bound = jnp.max(cases[:, 1])
        stress_bound = jnp.max(cases[:, 2])
        constraints = jnp.asarray([displacement_bound-1, stress_bound-1, mass/baseline_mass-1])
        penalty = 100*jnp.sum(jax.nn.relu(constraints)**2)
        objective = mass/baseline_mass if s.stage == 'min-mass' else response
        merit = objective+penalty
        return merit, dict(response=response, mass_kg=mass, constraints=constraints,
            displacement_bound=displacement_bound, stress_bound=stress_bound,
            actual_peak_back_m=jnp.max(cases[:, 3]), actual_peak_skin_Pa=jnp.max(cases[:, 4]),
            phi_min=jnp.min(phi), phi_max=jnp.max(phi), cases=cases)

    q0 = np.zeros(math.prod(s.controls)+(s.stage == 'min-mass'))
    if s.stage == 'min-mass':
        ratio = (s.target_phi-lower_mean)/(upper_mean-lower_mean)
        q0[-1] = math.log(ratio/(1-ratio))
    return dict(settings=s, geometry=geo, plan=plan, support=support, properties=properties,
                pressure=pressure, load_shape=load_shape, fixed=fixed, advance=advance,
                aggregate_initial=aggregate_initial, metrics=metrics, q0=q0,
                initial_state=initial_state, reference_grid=reference_grid)


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def numeric(value):
    if isinstance(value, dict):return {k: numeric(v) for k, v in value.items()}
    array = np.asarray(value)
    return array.tolist()


def qualification(record, settings):
    cases = np.asarray(record['cases'])
    valid = bool(np.isfinite(cases).all() and np.all(cases[:, 7] == 1) and np.all(cases[:, 5] > 0))
    feasible = (valid and record['mass_kg'] <= describe(settings)['uniform_mass_kg']*(1+1e-10)
                and record['actual_peak_back_m'] <= settings.displacement_limit
                and record['actual_peak_skin_Pa'] <= settings.stress_limit)
    return valid, bool(feasible)


def forward_case(model, q, case, output, *, resume=False, stop_step=None, save_every=100):
    import jax
    import jax.numpy as jnp
    import jaxlib
    from mpm_jax.total_lagrangian import TLState, sample_nodal, deformation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature, sources = source_signature(model['settings'])
    binding = dict(source_signature=signature, controls=np.asarray(q).tolist(), case=CASE_NAMES[case],
                   jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__)
    binding_text = json.dumps(binding, sort_keys=True)
    if np.asarray(q).shape != model['q0'].shape or not np.isfinite(q).all():
        raise ValueError('Invalid control shape or values')
    phi, density, young, material = model['properties'](jnp.asarray(q))
    reference = model['reference_grid'](model['support'], density)
    state = model['initial_state'](reference, material, model['pressure'](0)*model['load_shape'][case])
    aggregate = model['aggregate_initial']()
    start = 0
    rows = [np.zeros(len(ROW_FIELDS))]
    rows[0][7] = rows[0][-1] = 1.
    path = output/'checkpoint.npz'
    if resume:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['binding']) != binding_text:
                raise ValueError('Resume requires identical geometry, design, source and runtime')
            values = {}
            for k, template in state._asdict().items():
                if k == 'history':continue
                value = saved['state_'+k]
                if value.shape != template.shape or value.dtype != template.dtype or not np.isfinite(value).all():
                    raise ValueError('Invalid saved state field: '+k)
                values[k] = jnp.asarray(value)
            state = TLState(**values, history=None)
            aggregate = jnp.asarray(saved['aggregate'])
            start = int(saved['step'])
            rows = list(saved['rows'])
            if (not 0 <= start <= model['plan']['steps'] or len(rows) != start+1
                    or aggregate.shape != (14,) or not np.isfinite(aggregate).all()
                    or not bool(state.valid) or not np.isfinite(saved['rows']).all()
                    or abs(float(state.time)-start*model['plan']['dt_s']) > 1e-10*model['settings'].horizon):
                raise ValueError('Incomplete or invalid forward checkpoint')
            if not np.allclose(state.F,deformation(state.displacement,reference),rtol=0,atol=1e-13):
                raise ValueError('Saved deformation differs from its nodal displacement')
    elif any(output.iterdir()):
        raise ValueError('Use a fresh output directory or --resume')
    limit = model['plan']['steps'] if stop_step is None else stop_step
    if not start <= limit <= model['plan']['steps'] or save_every < 1:
        raise ValueError('Stop step must lie within the unchanged absolute time plan')
    one = jax.jit(lambda state, aggregate, n: model['advance'](state, aggregate, reference, material, case, n))
    def save(step_number):
        payload = {'state_'+k: np.asarray(v) for k, v in state._asdict().items() if k != 'history'}
        atomic_npz(path, **payload, step=step_number, aggregate=np.asarray(aggregate),
                   rows=np.asarray(rows), binding=np.asarray(binding_text))
        temporary = output/'history.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream);writer.writerow(ROW_FIELDS);writer.writerows(rows)
        temporary.replace(output/'history.csv')
        displacement = np.asarray(sample_nodal(state.displacement, reference))
        atomic_npz(output/f'frame_{step_number:07d}.npz', reference=np.asarray(reference.positions),
                   position=np.asarray(reference.positions)+displacement, displacement=displacement,
                   F=np.asarray(state.F), phi=np.asarray(phi), density=np.asarray(density),
                   young=np.asarray(young), volume=np.asarray(reference.volume), skin=model['geometry']['skin'],
                   time=np.asarray(state.time))
        atomic_json(output/'status.json', dict(step=step_number, planned_steps=model['plan']['steps'],
            completed=step_number == model['plan']['steps'], valid=bool(state.valid),
            time_s=float(state.time), binding=binding, qualification=model['plan']['qualification']))
    if not resume:save(0)
    for n in range(start, limit):
        state, aggregate, row = one(state, aggregate, jnp.asarray(n))
        row = np.asarray(row)
        if not np.isfinite(row).all() or not bool(state.valid):
            raise RuntimeError('Nonfinite state or nonpositive Jacobian; last saved complete checkpoint retained')
        rows.append(row)
        if (n+1)%save_every == 0 or n+1 == limit:save(n+1)
    result = dict(case=CASE_NAMES[case], completed=limit == model['plan']['steps'], steps=limit,
        time_s=float(state.time), mass_kg=float(reference.particle_mass.sum()),
        peak_back_displacement_m=float(aggregate[5]), peak_skin_stress_Pa=float(aggregate[6]),
        minimum_J=float(aggregate[7]), maximum_energy_residual_J=float(aggregate[8]),
        pressure_impulse_z_Ns=float(aggregate[11]), support_impulse_z_Ns=float(aggregate[12]),
        peak_back_time_s=float(aggregate[13]), source_sha256=sources,
        nonlinear_blast_qualified=False, plan=model['plan'])
    atomic_json(output/'summary.json', result)
    return result


def directional_check(model, q, step=1e-4):
    import jax
    import jax.numpy as jnp
    objective = jax.jit(lambda q: model['metrics'](q)[0])
    direction = np.random.default_rng(731).normal(size=len(q))
    direction /= np.linalg.norm(direction)
    value, tangent = jax.jvp(objective, (jnp.asarray(q),), (jnp.asarray(direction),))
    difference = (objective(q+step*direction)-objective(q-step*direction))/(2*step)
    half = (objective(q+.5*step*direction)-objective(q-.5*step*direction))/step
    scale = max(abs(float(tangent)), abs(float(difference)), abs(float(half)), 1e-30)
    error = max(abs(float(tangent-half)), abs(float(difference-half)))/scale
    resolved = scale > 64*np.finfo(float).eps*max(abs(float(value)), 1e-30)/step
    return dict(objective=float(value), directional_jvp=float(tangent), directional_fd=float(difference),
                directional_fd_half_step=float(half), relative_error=error, derivative_resolved=bool(resolved),
                passed=bool(resolved and np.isfinite(error) and error < 2e-3),
                scope='Derivative of this exact configured horizon; a short horizon is an interface check, not optimization evidence.')


def optimize(model, output, *, iterations=1, gradient_mode='jvp', learning_rate=.05, resume=False):
    import jax
    import jax.numpy as jnp
    import jaxlib
    if iterations < 0 or gradient_mode not in ('jvp','reverse') or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid optimization budget, gradient mode or step size')
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    signature, sources = source_signature(model['settings'])
    signature = hashlib.sha256((signature+json.dumps(dict(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__),sort_keys=True)).encode()).hexdigest()
    path = output/'optimizer.npz'
    q = jnp.asarray(model['q0']);first = jnp.zeros_like(q);second = jnp.zeros_like(q)
    start, history = 0, []
    if resume:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['signature']) != signature:raise ValueError('Optimizer model/source changed')
            q, first, second = (jnp.asarray(saved[k]) for k in ('q','first','second'))
            start = int(saved['iteration']);history = json.loads(str(saved['history']))
    elif any(output.iterdir()):raise ValueError('Fresh optimizer directory required')
    objective = jax.jit(lambda q: model['metrics'](q)[0])
    evaluate = jax.jit(model['metrics'])
    if gradient_mode == 'jvp' and len(q) > 32:
        raise ValueError('Sequential JVP optimization is limited to 32 controls; check the target field with one direction before budgeting reverse AD')
    if gradient_mode == 'reverse':
        estimate = model['plan']['memory_estimate']['reverse_checkpoint_state_bytes']
        if estimate > 4*2**30:
            raise ValueError('Reverse checkpoint states alone exceed 4 GiB; this full-horizon gradient needs a new measured memory plan')
        value_gradient = jax.jit(jax.value_and_grad(objective))
    else:
        tangent = jax.jit(lambda q, d: jax.jvp(objective, (q,), (d,)))
        def value_gradient(q):
            gradient = []
            for i in range(len(q)):
                value, derivative = tangent(q, jnp.zeros_like(q).at[i].set(1.))
                gradient.append(derivative)
            return value, jnp.stack(gradient)
    check = directional_check(model, np.asarray(q))
    atomic_json(output/'gradient-check.json', check)
    if not check['passed']:raise RuntimeError('Configured-horizon directional derivative check failed')
    for iteration in range(start, start+iterations):
        value, gradient = value_gradient(q)
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
            raise RuntimeError('Nonfinite objective or gradient')
        first = .9*first+.1*gradient;second = .999*second+.001*gradient**2
        direction = -(first/(1-.9**(iteration+1)))/(jnp.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        _, old = evaluate(q);old_record = numeric(old)
        valid, feasible = qualification(old_record, model['settings'])
        accepted = False
        for k in range(8):
            proposal = jnp.clip(q+learning_rate*(.5**k)*direction, -12., 12.)
            candidate_value, candidate = evaluate(proposal)
            record = numeric(candidate);candidate_valid,candidate_feasible=qualification(record,model['settings'])
            if (candidate_valid and float(candidate_value) < float(value)
                    and (not feasible or candidate_feasible)):
                q=proposal;old_record=record;value=candidate_value;accepted=True;break
        history.append(dict(iteration=iteration+1, accepted=accepted, merit=float(value), **old_record))
        atomic_npz(path, q=np.asarray(q), first=np.asarray(first), second=np.asarray(second),
            iteration=iteration+1, history=np.asarray(json.dumps(history)), signature=np.asarray(signature))
        atomic_json(output/'history.json', history)
    final_value, record = evaluate(q);record = numeric(record)
    valid, feasible = qualification(record, model['settings'])
    result = dict(merit=float(final_value), valid=valid, feasible=feasible, controls=np.asarray(q).tolist(),
        metrics=record, iterations_completed=start+iterations, source_sha256=sources,
        qualification='finite-budget elastic design; nonlinear forward qualification incomplete',
        convergence_claim=False, plan=model['plan'])
    atomic_json(output/'summary.json', result)
    return result


def export_vtk(output):
    """Export only saved physical particle frames, at deformation scale one."""
    output=Path(output)
    frames=sorted(output.glob('frame_*.npz'))
    if not frames:raise ValueError('No saved particle frames to export')
    destination=output/'vtk';destination.mkdir(exist_ok=True)
    def array(name,value,kind,components=1):
        dtype={'Float64':'<f8','Int32':'<i4','UInt8':'u1'}[kind]
        raw=np.asarray(value,dtype=dtype).tobytes()
        binary=base64.b64encode(struct.pack('<I',len(raw))+raw).decode('ascii')
        return f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" format="binary">{binary}</DataArray>'
    records=[]
    for frame in frames:
        with np.load(frame,allow_pickle=False) as data:
            count=len(data['position']);name=frame.stem+'.vtu'
            fields=array('displacement',data['displacement'],'Float64',3)
            for field in ('phi','density','young'):fields+=array(field,data[field],'Float64')
            fields+=array('skin',data['skin'],'UInt8')
            text='<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"><UnstructuredGrid>'
            text+=f'<Piece NumberOfPoints="{count}" NumberOfCells="{count}"><PointData>'+fields+'</PointData><Points>'
            text+=array('position',data['position'],'Float64',3)+'</Points><Cells>'
            text+=array('connectivity',np.arange(count),'Int32')+array('offsets',np.arange(1,count+1),'Int32')
            text+=array('types',np.ones(count),'UInt8')+'</Cells></Piece></UnstructuredGrid></VTKFile>'
            (destination/name).write_text(text,encoding='utf-8')
            records.append((float(data['time']),name))
    datasets=''.join(f'<DataSet timestep="{time:.17g}" group="" part="0" file="{name}"/>' for time,name in records)
    (destination/'particles.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+datasets+'</Collection></VTKFile>',encoding='utf-8')
    return dict(frames=len(records),pvd=str(destination/'particles.pvd'),deformation_scale=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'check', 'forward', 'optimize', 'export'))
    parser.add_argument('--xy-step', type=float, default=.025)
    parser.add_argument('--z-step', type=float, default=.001)
    parser.add_argument('--controls', type=int, nargs=3, default=(2,2,2))
    parser.add_argument('--horizon', type=float, default=.002)
    parser.add_argument('--pulse-duration', type=float, default=.0002)
    parser.add_argument('--cfl', type=float, default=.15)
    parser.add_argument('--block-steps', type=int, default=32)
    parser.add_argument('--target-phi', type=float, default=.18)
    parser.add_argument('--stage', choices=('fixed-mass','min-mass'), default='fixed-mass')
    parser.add_argument('--gradient', choices=('jvp','reverse'), default='jvp')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('outputs/blast-sandwich-design'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--design-checkpoint', type=Path)
    args=parser.parse_args()
    if args.action=='export':
        print(json.dumps(export_vtk(args.output),indent=2));return
    settings=Settings(xy_step=args.xy_step,z_step=args.z_step,controls=tuple(args.controls),horizon=args.horizon,
        pulse_duration=args.pulse_duration,cfl=args.cfl,block_steps=args.block_steps,target_phi=args.target_phi,stage=args.stage)
    if args.action=='plan':
        print(json.dumps(describe(settings),indent=2));return
    model=build_model(settings)
    q=model['q0']
    if args.design_checkpoint:
        with np.load(args.design_checkpoint,allow_pickle=False) as saved:q=saved['q'].copy()
        if q.shape!=model['q0'].shape:raise ValueError('Control lattice differs from the supplied design')
    if args.action=='check':
        result=directional_check(model,q)
        args.output.mkdir(parents=True,exist_ok=True);atomic_json(args.output/'gradient-check.json',result)
        if not result['passed']:raise RuntimeError('Directional derivative check failed')
    elif args.action=='forward':
        result=[forward_case(model,q,k,args.output/CASE_NAMES[k],resume=args.resume,
            stop_step=args.stop_step,save_every=args.save_every) for k in range(3)]
    else:result=optimize(model,args.output,iterations=args.iterations,gradient_mode=args.gradient,resume=args.resume)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
