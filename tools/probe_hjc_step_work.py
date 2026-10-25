"""Bounded diagnostic only: actual MPM boundary work and signed transfer ledger.

The production core/config/material/loading remain unchanged. This is not an
energy-convergence certificate; the stage identity is an accounting identity.
"""
from dataclasses import replace, asdict
from functools import partial
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def instrumented_step_function():
    import jax
    import jax.numpy as jnp
    from mpm_jax import core
    from mpm_jax.materials.history import corotated_increment

    @partial(jax.jit,static_argnames=('config',))
    def measured_step(state,material,config):
        assert config.failure_mode == 'retain'
        support = core.particle_stencil(state,config)
        gm,gp = core.particle_to_grid(state,material,config,support)
        denom = jnp.where(gm>0,gm,1.)[:,None]
        velocity_stress = gp/denom
        momentum_kin = state.mass[:,None,None]*(state.v[:,None,:]+jnp.einsum('pij,pkj->pki',state.C,support.dpos))
        gp_kin = jnp.zeros_like(gp).at[support.indices].add(support.weights[...,None]*momentum_kin)
        velocity_kin = gp_kin/denom
        velocity_gravity = jnp.where((gm>0)[:,None],velocity_stress+config.dt*jnp.asarray(config.gravity),0.)
        kinetic = lambda velocity:.5*jnp.sum(gm[:,None]*velocity**2)
        particle_kinetic = lambda value:.5*jnp.sum(value.mass*jnp.sum(value.v**2,axis=1))
        apic_kinetic = lambda value:particle_kinetic(value)+config.dx**2/8*jnp.sum(value.mass[:,None,None]*value.C**2)
        node = jnp.arange(len(gm))
        strides = (config.n_grid+1)**jnp.arange(config.dim-1,-1,-1)
        coordinates = ((node[:,None]//strides)%(config.n_grid+1))*config.dx
        current = velocity_gravity
        stages = []
        # Domain clipping first, exactly matching grid_boundaries order.
        velocities = [core.grid_boundaries(current,replace(config,boundaries=()),state.time+config.dt)]
        for boundary in config.boundaries:
            velocities.append(boundary.apply(velocities[-1],coordinates,state.time+config.dt))
        for after in velocities:
            impulse = gm[:,None]*(after-current)
            work = jnp.sum(impulse*after)
            delta_kinetic = kinetic(after)-kinetic(current)
            projection = .5*jnp.sum(gm[:,None]*(after-current)**2)
            stages.append(jnp.r_[work,delta_kinetic,projection,jnp.sum(impulse,axis=0)])
            current = after
        # This exact production call drives the state; diagnostic replay cannot
        # alter any velocity used by the material/particle update.
        gv = core.grid_update(gm,gp,config,state.time+config.dt)
        following = core.particle_update(state,material,config,support,gv)
        strain,rotation,_ = corotated_increment(following.F,state.F)
        unrotated_stress = jnp.swapaxes(rotation,-1,-2)@following.history.stress@rotation
        calculated_work = jnp.sum(state.mass*.5*jnp.sum((state.history.stress+unrotated_stress)*strain,axis=(-2,-1))/following.history.density)
        history_work = jnp.sum(state.mass*(following.history.work-state.history.work))
        stages = jnp.stack(stages)
        p2g_change = kinetic(velocity_kin)-apic_kinetic(state)
        stress_change = kinetic(velocity_stress)-kinetic(velocity_kin)
        gravity_change = kinetic(velocity_gravity)-kinetic(velocity_stress)
        g2p_change = apic_kinetic(following)-kinetic(gv)
        energy_change = apic_kinetic(following)-apic_kinetic(state)+history_work
        prescribed_work = jnp.sum(stages[:,0])
        projection = jnp.sum(stages[:,2])
        stage_sum = p2g_change+stress_change+gravity_change+prescribed_work-projection+g2p_change+history_work
        values = jnp.array([state.time,following.time,particle_kinetic(following),apic_kinetic(following),
            history_work,calculated_work,p2g_change,stress_change,gravity_change,g2p_change,
            energy_change,prescribed_work,projection,energy_change-stage_sum,jnp.max(jnp.abs(gv-current)),
            jnp.max(jnp.abs(stages[:,0]-stages[:,1]-stages[:,2]))])
        return following,values,stages
    return measured_step


def numpy_work(previous,following):
    delta = following.F@np.linalg.inv(previous.F)
    eigenvalues,eigenvectors = np.linalg.eigh(delta.swapaxes(-1,-2)@delta)
    strain = (eigenvectors*(.5*np.log(eigenvalues))[:,None,:])@eigenvectors.swapaxes(-1,-2)
    rotation = delta@((eigenvectors*(1/np.sqrt(eigenvalues))[:,None,:])@eigenvectors.swapaxes(-1,-2))
    stress = rotation.swapaxes(-1,-2)@following.history.stress@rotation
    increment = .5*np.sum((previous.history.stress+stress)*strain,axis=(-2,-1))/following.history.density
    measured = following.history.work-previous.history.work
    return dict(max_specific_increment_error_J_per_kg=float(np.max(np.abs(increment-measured))),
        independently_calculated_increment_J=float(np.sum(previous.mass*increment)),
        stored_increment_J=float(np.sum(previous.mass*measured)))


def run(output,parent,time_step_divisor):
    import jax
    from mpm_jax import core
    from mpm_jax.materials.history import History
    from examples.brittle_engineering import setup
    jax.config.update('jax_enable_x64',True)
    output.mkdir(parents=True,exist_ok=False)
    # Build the reference factory from its exact authenticated original Git blob.
    # This validates the default and ensures dt/2 reaches every observer closure.
    import types
    original_source=subprocess.check_output(['git','show','d857cec8d682446a1f4821ebd0fb3b713ef761a2:examples/brittle_engineering.py'],cwd=ROOT)
    reference_module=types.ModuleType('examples._hjc_original_factory_reference')
    reference_module.__package__='examples'
    sys.modules[reference_module.__name__]=reference_module
    exec(compile(original_source,'authenticated-original-brittle-engineering','exec'),reference_module.__dict__)
    original_case=reference_module.setup('concrete-bending-mat111',grid=64,ppc=3,frames=60)
    default_case=setup('concrete-bending-mat111',grid=64,ppc=3,frames=60)
    assert asdict(default_case.configs[0])==asdict(original_case.configs[0])
    assert default_case.description==original_case.description
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(default_case.states),jax.tree.leaves(original_case.states)))
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(default_case.materials),jax.tree.leaves(original_case.materials)))
    case = setup('concrete-bending-mat111',grid=64,ppc=3,frames=60,time_step_divisor=time_step_divisor)
    original_config=asdict(original_case.configs[0])
    refined_config=asdict(case.configs[0])
    assert refined_config.pop('dt')==original_config.pop('dt')/time_step_divisor
    assert refined_config==original_config
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(case.states),jax.tree.leaves(original_case.states)))
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(case.materials),jax.tree.leaves(original_case.materials)))
    for key in original_case.description:
        if key not in ('time_step_s','acoustic_CFL'):
            assert case.description[key]==original_case.description[key],key
    del default_case,original_case,reference_module
    config,material = case.configs[0],case.materials[0]
    assert config.dt == 1.4285714285714285e-7/time_step_divisor and config.boundary_cells == 0
    assert not any(config.gravity) and config.failure_mode == 'retain'
    with np.load(parent/'raw/checkpoint_0_0.npz',allow_pickle=False) as raw:
        fields = {name:jax.numpy.asarray(raw[name]) for name in raw.files}
    checkpoint = json.loads((parent/'raw/checkpoint.json').read_text())
    assert checkpoint['frame'] == 40 and checkpoint['states'] == ['checkpoint_0_0.npz']
    history = History(**{name:fields['history_'+name] for name in History._fields})
    restored = core.State(**{name:history if name=='history' else fields[name] for name in core.State._fields})
    assert abs(float(restored.time)-.001) < 1e-15
    measured_step = instrumented_step_function()
    headers = ['step_start_s','step_end_s','particle_kinetic_J','apic_kinetic_J','history_work_increment_J',
        'formula_work_increment_J','p2g_signed_energy_change_J','stress_signed_grid_energy_change_J',
        'gravity_signed_energy_change_J','g2p_signed_energy_change_J','total_apic_plus_work_change_J',
        'prescribed_boundary_work_J','boundary_projection_loss_J','stage_identity_residual_J',
        'replayed_grid_velocity_error_m_per_s','boundary_work_identity_error_J']
    reports = []
    for label,initial,steps in [('startup',case.states[0],8*time_step_divisor),('active-damaged-f40',restored,64*time_step_divisor)]:
        started = time.monotonic()
        baseline = core.advance(initial,material,config,steps)
        jax.block_until_ready(baseline.x)
        value = initial
        rows,stage_rows = [],[]
        for index in range(steps):
            previous = value
            value,row,stages = measured_step(value,material,config)
            rows.append(np.asarray(row));stage_rows.append(np.asarray(stages))
        final = jax.tree.map(np.asarray,value)
        before = jax.tree.map(np.asarray,previous)
        original = jax.tree.map(np.asarray,baseline)
        errors = [float(np.max(np.abs(a.astype(float)-b.astype(float)))) for a,b in zip(jax.tree.leaves(final),jax.tree.leaves(original))]
        for first,second in zip(jax.tree.leaves(final),jax.tree.leaves(original)):
            assert np.max(np.abs(first.astype(float)-second.astype(float))) <= 1e-12*max(1.,float(np.max(np.abs(second.astype(float)))))
        assert bool(final.valid) and np.all(final.history.valid)
        np.testing.assert_allclose(final.x,original.x,rtol=0,atol=1e-13)
        np.testing.assert_allclose(final.F,original.F,rtol=0,atol=1e-12)
        np.testing.assert_allclose(final.history.stress,original.history.stress,rtol=1e-12,atol=1e-5)
        np.testing.assert_allclose(final.history.work,original.history.work,rtol=1e-12,atol=1e-9)
        matrix,stage_matrix = np.asarray(rows),np.asarray(stage_rows)
        np.savez_compressed(output/(label+'-ledger.npz'),values=matrix,stages=stage_matrix,headers=np.asarray(headers))
        numerical_work = numpy_work(before,final)
        np.savez_compressed(output/(label+'-last-material-increment.npz'),
            old_F=before.F,new_F=final.F,old_stress=before.history.stress,new_stress=final.history.stress,
            old_work=before.history.work,new_work=final.history.work,mass=final.mass,density=final.history.density)
        assert numerical_work['max_specific_increment_error_J_per_kg'] < 1e-7
        assert np.max(np.abs(matrix[:,14])) < 1e-14
        assert np.max(np.abs(matrix[:,15])) < 1e-8
        assert np.max(np.abs(matrix[:,13])) < 1e-8
        assert np.max(np.abs(matrix[:,4]-matrix[:,5])) < 1e-8
        observer = case.observer((previous,),float(previous.time))
        assert abs(observer['boundary_sample_time_s']-float(value.time)) < 1e-15
        actual_loading_reaction=-float(stage_matrix[-1,-1,4])/config.dt
        np.testing.assert_allclose(observer['loading_reaction_y_N'],actual_loading_reaction,rtol=1e-12,atol=1e-5)
        reports.append(dict(label=label,steps=steps,seconds=time.monotonic()-started,start_time_s=float(initial.time),
            end_time_s=float(value.time),max_state_leaf_absolute_difference=max(errors),
            all_state_leaves_bitwise_equal=all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(final),jax.tree.leaves(original))),
            independent_last_step_work=numerical_work,
            cumulative_prescribed_boundary_work_J=float(matrix[:,11].sum()),
            cumulative_projection_loss_J=float(matrix[:,12].sum()),
            cumulative_p2g_change_J=float(matrix[:,6].sum()),cumulative_g2p_change_J=float(matrix[:,9].sum()),
            cumulative_stress_plus_material_change_J=float((matrix[:,7]+matrix[:,4]).sum()),
            max_stage_identity_residual_J=float(np.max(np.abs(matrix[:,13]))),
            min_J=float(np.linalg.det(final.F).min()),max_damage=float(final.history.damage.max()),
            observer_uses_same_config=True,observer_is_next_step_probe=True,
            observer_loading_reaction_N=observer['loading_reaction_y_N'],actual_step_loading_reaction_N=actual_loading_reaction))
    report = dict(passed=True,commit=os.environ.get('GITHUB_SHA'),particles=121104,config_dt_s=config.dt,
        baseline_equations_unchanged=True,cases=reports,steps_per_path=72*time_step_divisor,total_physical_steps_executed=144*time_step_divisor,
        time_step_divisor=time_step_divisor,default_factory_exactly_preserved=True,
        identical_initial_state_geometry_material_and_loading=True,
        refined_f40_seed_scope="Local seeded comparison only; not a globally refined f0-to-f40 history.",
        parent_frame40_checkpoint_sha256=hashlib.sha256((parent/'raw/checkpoint_0_0.npz').read_bytes()).hexdigest(),
        boundary_labels=['domain-clipping',*case.description['boundary_names']],
        work_definition='Sum impulse dot post-constraint velocity at actual t+dt sequential boundary application; work=grid kinetic change+positive projection term.',
        constitutive_definition='Mass times cumulative J/kg trapezoidal old/new unrotated Cauchy stress contracted with incremental right Hencky strain divided by new density; post-update stress rotation undone for independent NumPy check.',
        structural_energy_certified=False,temporal_convergence_certified=False,
        scope='Bounded instrumentation equivalence and work-coverage check; no full rerun or energy/time convergence claim.')
    (output/'step-work-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--time-step-divisor',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    if args.worker:
        run(args.output.resolve(),args.parent.resolve(),args.time_step_divisor)
    else:
        import ctypes
        import psutil
        from tools.run_phase_field_visible_crack import owned_session_uss,terminate_owned_group
        assert sys.platform == 'linux'
        assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) == 0
        output=args.output.resolve()
        output.parent.mkdir(parents=True,exist_ok=True)
        guard=dict(seconds_limit=1800,uss_limit_bytes=int(5.5*1024**3),minimum_available_bytes=2*1024**3,
            peak_uss_bytes=0,guard_reason=None)
        started=time.monotonic()
        process=None
        try:
            with (output.parent/'worker.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',
                    '--output',str(output),'--parent',str(args.parent.resolve()),'--time-step-divisor',str(args.time_step_divisor)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    uss=owned_session_uss(process.pid,psutil)
                    guard['peak_uss_bytes']=max(guard['peak_uss_bytes'],uss)
                    reason=('wall' if time.monotonic()-started>1800 else 'USS' if uss>guard['uss_limit_bytes']
                        else 'available memory' if psutil.virtual_memory().available<guard['minimum_available_bytes'] else None)
                    if reason:
                        guard['guard_reason']=reason
                        break
                    time.sleep(1)
                code=process.poll()
        finally:
            terminate_owned_group(process)
            guard.update(returncode=process.returncode if process else None,wall_seconds=time.monotonic()-started)
            (output.parent/'resource-guard.json').write_text(json.dumps(guard,indent=2)+'\n')
        if code != 0 or guard['guard_reason']:
            raise RuntimeError('Bounded step-work diagnostic stopped; logs retained')
