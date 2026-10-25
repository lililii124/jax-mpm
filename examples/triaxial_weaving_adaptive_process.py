"""Dynamic dry weaving with checkpoint-refreshed fixed-capacity execution support.

All body motion is APIC. Only material endcaps follow actual guide commands.
Grid/hash/search rebuilds change scratch indexing, never mechanical carry.
"""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .triaxial_weaving_process import plan_process,json_value
from .triaxial_weaving_geometry import fixture_position
from .triaxial_weaving_execution import allocate


def run(output,*,per_family=1,steps=1000,window_steps=200,save_stride=1000,resume=False):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreConfig,FibreState,make_state,reference_grips,step_with_diagnostics,validate_state,stored_energy
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle,BrittleHistory
    from mpm_jax.fibre_contact import fibre_contact_parameters,FibreContactHistory
    from mpm_jax.fibre_coupling import FibreCouplingHistory,section_centres,closest_segments
    from mpm_jax.fibre_patch_contact import initial_patch_history,evaluate_hertz_patches
    from mpm_jax.contact_patches import contact_patches
    geometry,plan=plan_process(per_family=per_family)
    if len(geometry['x'])>400000 or window_steps!=200:raise ValueError('An explicit bounded station and200-step refresh required')
    if not 1<=steps<=plan['full_steps']:raise ValueError('Absolute stop must lie in the unchanged protocol')
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),*[root/'examples'/name for name in ['triaxial_weaving_process.py','triaxial_weaving_geometry.py','triaxial_weaving_execution.py','triaxial_fibre_geometry.py']],*sorted((root/'mpm_jax').rglob('*.py'))]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    runtime=dict(jax=jax.__version__,jaxlib=__import__('jaxlib').__version__,numpy=np.__version__,x64=bool(jax.config.jax_enable_x64))
    binding=dict(schema='weaving-v2-moving-hash-v1',physical_plan=json_value(plan),runtime=runtime,source_sha256=hashes,window_steps=window_steps,padding_cells=3,search_skin_m=.0003)
    signature=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();output=Path(output)
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx'])
    capacity=None;start=0;observations=[];accumulator=jnp.zeros(8);flags=jnp.asarray(0,jnp.int32)
    if resume:
        latest=json.loads((output/'latest.json').read_text())
        checkpoint_path=output/latest['checkpoint_file']
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==latest['checkpoint_sha256']
        with np.load(checkpoint_path,allow_pickle=False) as cp:
            assert str(cp['signature'])==signature and json.loads(str(cp['binding']))==binding
            values={name:jnp.asarray(cp['state_'+name]) for name in FibreState._fields if name!='history'}
            values['history']=BrittleHistory(*(jnp.asarray(cp['history_'+name]) for name in BrittleHistory._fields));state=FibreState(**values)
            history=FibreCouplingHistory(FibreContactHistory(jnp.asarray(cp['law_active']),jnp.asarray(cp['law_valid'])),jnp.asarray(cp['contact_radius']),jnp.asarray(cp['contact_segments']),jnp.asarray(cp['contact_valid']))
            accumulator=jnp.asarray(cp['accumulator']);start=int(cp['step']);capacity=int(cp['packed_capacity'])
            assert start<steps and bool(state.valid)
            exact={name:bool(np.array_equal(cp['state_'+name],np.asarray(getattr(state,name)))) for name in FibreState._fields if name!='history'}
            exact.update({'history_'+name:bool(np.array_equal(cp['history_'+name],np.asarray(getattr(state.history,name)))) for name in BrittleHistory._fields})
            exact.update(contact_radius=bool(np.array_equal(cp['contact_radius'],np.asarray(history.radius))),contact_segments=bool(np.array_equal(cp['contact_segments'],np.asarray(history.segments))),law_active=bool(np.array_equal(cp['law_active'],np.asarray(history.law.active))),law_valid=bool(np.array_equal(cp['law_valid'],np.asarray(history.law.valid))),contact_valid=bool(np.array_equal(cp['contact_valid'],np.asarray(history.valid))),accumulator=bool(np.array_equal(cp['accumulator'],np.asarray(accumulator))))
            assert all(exact.values())
        observations=[row for row in json.loads((output/'observations.json').read_text()) if row['step']<=start];assert observations[-1]['step']==start
        (output/'restore-exact.json').write_text(json.dumps(dict(step=start,fields=exact),indent=2))
        provisional=state
    else:
        output.mkdir(parents=True,exist_ok=False);provisional=SimpleNamespace(x=geometry['x'],v=np.zeros_like(geometry['x']),time=0.)
        np.savez_compressed(output/'reference.npz',reference=geometry['x'],mass=geometry['mass'],volume=geometry['volume'],fibre_id=geometry['fibre_id'],family_id=geometry['family_id'],grip_masks=np.asarray(geometry['masks']))
        (output/'geometry.json').write_text(json.dumps(json_value(geometry['metadata']),indent=2))
    (output/'binding.json').write_text(json.dumps(binding,indent=2))
    config=FibreConfig(dt=plan['dt']);times=geometry['metadata']['fixture_times'];positions=geometry['metadata']['fixture_positions']
    def grip_speed(group):return lambda t:(fixture_position(t,times,positions,jnp)[group]-fixture_position(t-config.dt,times,positions,jnp)[group])/config.dt
    from mpm_jax.fibre_grips import sparse_reference_grips
    boundary=sparse_reference_grips(geometry['masks'],tuple(grip_speed(i) for i in range(len(geometry['masks']))))
    law=fibre_contact_parameters(effective_modulus=plan['young_Pa']/(2*(1-plan['poisson']**2)),effective_radius=geometry['metadata']['diameter_m']/2,work_of_adhesion=0.,model='hertz')
    patch_capacity=max(64,1<<int((2*len(geometry['contact_candidates'])-1).bit_length()));active_capacity=16*patch_capacity
    @jax.jit
    def evaluate(s,h,data,graph):
        force,updated,metrics=evaluate_hertz_patches(s,h,data,law,graph,capacity=active_capacity,max_patches=patch_capacity,max_iterations=128)
        centres,_=section_centres(s.x,s.mass,data);near=data.coverage_pairs;ends=data.segments[near]
        first,second,_,_,distance=closest_segments(centres[ends[:,0,0]],centres[ends[:,0,1]],centres[ends[:,1,0]],centres[ends[:,1,1]])
        gap=distance-jnp.sum(data.fibre_radius[data.segment_fibre[near]],axis=1);selection=contact_patches(gap,gap<0,graph,capacity=active_capacity,max_patches=patch_capacity,max_iterations=128)
        cells=selection.cells;live=cells<len(near);points=jnp.concatenate(((first+second)/2,jnp.zeros((1,3))))[cells]
        low=jnp.full((active_capacity+1,3),jnp.inf).at[selection.labels].min(jnp.where(live[:,None],points,jnp.inf));high=jnp.full((active_capacity+1,3),-jnp.inf).at[selection.labels].max(jnp.where(live[:,None],points,-jnp.inf))
        roots=jnp.nonzero(live&(selection.labels==jnp.arange(active_capacity)),size=patch_capacity,fill_value=active_capacity)[0]
        extent=jnp.linalg.norm(jnp.where((roots<active_capacity)[:,None],high[roots]-low[roots],0.),axis=1)
        spacing=jnp.max(jnp.linalg.norm(centres[data.segments[:,1]]-centres[data.segments[:,0]],axis=1))
        local=selection.valid&jnp.all(extent<=jnp.maximum(2*spacing,4*jnp.max(metrics.contact_radius,initial=0.)))
        return force,updated,metrics,jnp.max(extent,initial=0.),local
    @jax.jit
    def advance(s,h,a,n,layout,data,graph):
        def body(_,carry):
            old,attachment,a,previous_flags=carry
            def attempt(_):
                force,new_h,metrics,extent,local=evaluate(old,attachment,data,graph)
                new,diag=step_with_diagnostics(old,material,layout,config,force,boundary=boundary)
                maximum_log=jnp.max(jnp.abs(jnp.log(jnp.linalg.svd(new.F,compute_uv=False))));strain_ok=maximum_log<=.12;damage_ok=jnp.all(new.history.damage<=1e-10)
                bad=(~new.valid).astype(jnp.int32)+2*(~metrics.valid).astype(jnp.int32)+4*(~local).astype(jnp.int32)+8*(~strain_ok).astype(jnp.int32)+16*(~damage_ok).astype(jnp.int32)
                new=new._replace(valid=new.valid&metrics.valid&local&strain_ok&damage_ok)
                targets=jnp.stack([grip_speed(i)(old.time+config.dt) for i in range(len(geometry['masks']))]);group_work=jnp.sum(diag.group_impulses*targets,axis=1);machine_work=jnp.sum(group_work)
                delta=jnp.array([jnp.sum(force*(new.x-old.x)),jnp.abs(diag.boundary_work),machine_work,machine_work-diag.boundary_work,jnp.sum(jnp.abs(group_work)),jnp.linalg.norm(metrics.force_residual),jnp.linalg.norm(metrics.torque_residual),0.])
                updated=a+delta;updated=updated.at[7].set(jnp.maximum(a[7],config.dt*(plan['elastic_wave_speed_m_s']+jnp.max(jnp.linalg.norm(new.v,axis=1)))/plan['dx']))
                return new,new_h,updated,bad
            return jax.lax.cond(old.valid,attempt,lambda _:carry,None)
        return jax.lax.fori_loop(0,n,body,(s,h,a,jnp.asarray(0,jnp.int32)))
    setup=time.perf_counter();layout,data,graph,resource=allocate(provisional,geometry,plan['dx'],config.dt,window_steps,capacity);capacity=layout.node_count
    if not resume:state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout);history=initial_patch_history(data,max_patches=patch_capacity)
    validate_state(state,layout)
    def save(step,advance_seconds,rebuild_seconds):
        begin=time.perf_counter();validate_state(state,layout);_,_,metrics,extent,local=evaluate(state,history,data,graph);jax.block_until_ready(metrics.valid)
        if not bool(metrics.valid&local):raise RuntimeError('Hertz/coverage/connected-region guard failed at saved state')
        target=fixture_position(float(state.time),times,positions);error=max(float(np.linalg.norm(np.asarray(state.x)[mask]-geometry['x'][mask]-(end-initial),axis=1).max()) for mask,end,initial in zip(geometry['masks'],target,positions[0]))
        if error>plan['dx']*.02:raise RuntimeError('Actual fixture displacement mismatch')
        K=float(.5*jnp.sum(state.mass[:,None]*state.v**2));affine=float(plan['dx']**2/8*jnp.sum(state.mass[:,None,None]*state.C**2));U=float(stored_energy(state,material));H=float(jnp.sum(metrics.potential))
        row=dict(step=step,advance_cache_entries=int(advance._cache_size()),time_s=float(state.time),K=K,APIC_affine_K=affine,U=U,Hertz_potential=H,machine_work=float(accumulator[2]),absolute_machine_work=float(accumulator[4]),projection_loss=float(accumulator[3]),grid_projection_work=float(state.boundary_work),energy_minus_machine_work=K+affine+U+H-float(accumulator[2]),maximum_grip_error_m=error,contact_patches=int(metrics.patch_count),maximum_patch_extent_m=float(extent),contact_valid=True,mpm_valid=True,maximum_wave_plus_velocity_cfl=float(accumulator[7]),advance_seconds=advance_seconds,rebuild_seconds=rebuild_seconds,execution=resource)
        payload={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};payload.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
        temporary=output/'checkpoint.partial.npz';np.savez_compressed(temporary,step=step,signature=signature,binding=json.dumps(binding,sort_keys=True),accumulator=np.asarray(accumulator),packed_capacity=capacity,law_active=np.asarray(history.law.active),law_valid=np.asarray(history.law.valid),contact_radius=np.asarray(history.radius),contact_segments=np.asarray(history.segments),contact_valid=np.asarray(history.valid),**payload);checkpoint_file=f'checkpoint_{step:07d}.npz';temporary.replace(output/checkpoint_file)
        frame=dict(x=np.asarray(state.x),v=np.asarray(state.v),F=np.asarray(state.F),C=np.asarray(state.C),damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed),mass=np.asarray(state.mass),fibre_id=geometry['fibre_id'],family_id=geometry['family_id'],time=float(state.time),fixture_targets=target,contact_points=np.asarray(metrics.closest_points),contact_active=np.asarray(metrics.active),contact_pair_segments=np.asarray(metrics.pair_segments),contact_segment_fibre=np.asarray(data.segment_fibre),contact_segments=np.asarray(data.segments),particle_section=np.asarray(data.particle_section))
        temporary=output/'frame_latest.partial.npz';np.savez_compressed(temporary,step=step,**frame);frame_file=f'frame_{step:07d}.npz';temporary.replace(output/frame_file)
        row['save_sync_seconds']=time.perf_counter()-begin;row['checkpoint_sha256']=hashlib.sha256((output/checkpoint_file).read_bytes()).hexdigest();row['checkpoint_file']=checkpoint_file;row['frame_file']=frame_file
        old_latest=json.loads((output/'latest.json').read_text()) if (output/'latest.json').exists() else None
        observations.append(row);temporary=output/'observations.partial.json';temporary.write_text(json.dumps(observations,indent=2));temporary.replace(output/'observations.json')
        temporary=output/'latest.partial.json';temporary.write_text(json.dumps(dict(step=step,checkpoint_file=checkpoint_file,frame_file=frame_file,checkpoint_sha256=row['checkpoint_sha256']),indent=2));temporary.replace(output/'latest.json')
        # Retain segment input, every archive stride and the current complete state.
        if old_latest and old_latest['step'] not in (0,start) and (old_latest['step']-start)%save_stride:
            for name in ['checkpoint_file','frame_file']:(output/old_latest[name]).unlink()
        print(f'step={step} advance_s={advance_seconds:.6f} rebuild_s={rebuild_seconds:.6f} save_s={row["save_sync_seconds"]:.6f} nodes={capacity} occupied={resource["occupied_nodes"]} pairs={resource["real_coverage_pairs"]} patches={int(metrics.patch_count)}',flush=True)
    if not resume:save(0,0.,time.perf_counter()-setup)
    previous=start
    while previous<steps:
        rebuild=time.perf_counter();layout,data,graph,resource=allocate(state,geometry,plan['dx'],config.dt,window_steps,capacity);rebuild_seconds=time.perf_counter()-rebuild;validate_state(state,layout)
        stop=min(previous+window_steps,steps);begin=time.perf_counter();state,history,accumulator,flags=advance(state,history,accumulator,stop-previous,layout,data,graph);jax.block_until_ready((state,history,accumulator,flags));elapsed=time.perf_counter()-begin
        if not bool(state.valid):
            np.savez_compressed(output/'invalid-state-forensic-not-checkpoint.npz',time=np.asarray(state.time),x=np.asarray(state.x),F=np.asarray(state.F),v=np.asarray(state.v),failure_flags=np.asarray(flags));raise RuntimeError(f'Actual guard failure flags={int(flags)} at t={float(state.time)}; valid rolling checkpoint unchanged')
        save(stop,elapsed,rebuild_seconds);previous=stop
    (output/'summary.json').write_text(json.dumps(dict(scope='Actual dynamic dry-weaving prefix; no topology or final-scale claim',steps=steps,full_protocol_completed=steps>=plan['full_steps'],matrix_present=False,signature=signature),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--per-family',type=int,default=1);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',action='store_true');p.add_argument('--save-stride',type=int,default=1000);a=p.parse_args();run(a.output,per_family=a.per_family,steps=a.steps,resume=a.resume,save_stride=a.save_stride)
