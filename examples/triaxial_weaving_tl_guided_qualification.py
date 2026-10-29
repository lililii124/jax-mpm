"""Single 64-step TL/RATTLE qualification for the unchanged F7 guided geometry."""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np


def run(output,*,steps=64,checkpoint_every=64,resume_from=None):
    import jax
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_tube_layout
    from mpm_jax.fibre_tlmpm import make_solver
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle
    from mpm_jax.fibre_tlmpm import TLState
    from mpm_jax.materials.isotropic_brittle import BrittleHistory
    from examples.triaxial_weaving_guided_geometry import plan_guided
    from examples.triaxial_weaving_geometry import fixture_position

    geometry,plan=plan_guided(per_family=8,nominal_cfl=.08);meta=geometry['metadata']
    x=np.asarray(geometry['x']);volume=np.asarray(geometry['volume']);ids=np.asarray(geometry['fibre_id'])
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],
        tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx'])
    # Keep the original six-cell reference margin while compacting empty tube
    # nodes, which are unnecessary for the static TL support and exceed the
    # dense 24-fibre node budget.
    layout=build_tube_layout(x,ids,dx=plan['dx'],padding_cells=6,max_nodes=40_000_000)
    cross=int(meta['cross_section_particles']);guides=np.asarray([g['particle_ids'] for g in meta['placement_guides']],np.int32)
    if guides.shape!=(600,cross):raise RuntimeError(f'Expected actual 600 finite section guides, got {guides.shape}')
    def digest(value,dtype):return hashlib.sha256(np.ascontiguousarray(value,dtype=dtype).tobytes()).hexdigest()
    # These are byte digests from the rejected v4's original reference.npz and
    # geometry.json.  Any geometry, mass, guide, or fixture change fails closed.
    identity=dict(reference_x=digest(x,'<f8'),volume=digest(volume,'<f8'),
        mass=digest(plan['density_kg_m3']*volume,'<f8'),fibre_id=digest(ids,'<i4'),
        grip_masks=digest(np.asarray(geometry['masks']),'|b1'),guide_particle_ids=digest(guides,'<i4'),
        fixture_times=digest(meta['fixture_times'],'<f8'),fixture_positions=digest(meta['fixture_positions'],'<f8'))
    expected=dict(reference_x='cb4f1e40c02c60b2e67758007de4c4f908596ea3e7d20880b2986749ac2eda65',
        volume='0f100b1b11bde952e30812ef9f96bd33137f59fd2e38e3837bb73108a4947951',
        mass='aa96df8de292e7c3b80667ff1930b38303d887cb04a04c9438e69eb02db13467',
        fibre_id='1fa382c20e0304872c4a2a52f12ee9db033fb2f5b48e9f8fbdc2f2c52f959b15',
        grip_masks='96d30a1745ce7d6ced871642f2b223d6f07571ad182af8f47b789e5180ee1027',
        guide_particle_ids='11839e814d646365c618a2128f1cd0b8f90e69040dcde025cba3eb2e66851456',
        fixture_times='bde9c5f177a84fc5873509c75f37f468a6cb1ed32131843c6534326211157bae',
        fixture_positions='8bbeb1b4ac7a3fcd97da6d620e7f9d015c238ad65608a70655e3404ff8f3a649')
    if identity!=expected:raise RuntimeError('F7 input identity mismatch against the original v4 reference geometry')
    expected_material=dict(dx_m=7.5e-05,young_Pa=1.0e7,poisson=.25,density_kg_m3=1500.,
        tensile_strength_Pa=2.0e6,fracture_energy_J_m2=1000.)
    actual_material=dict(dx_m=plan['dx'],young_Pa=plan['young_Pa'],poisson=plan['poisson'],
        density_kg_m3=plan['density_kg_m3'],tensile_strength_Pa=plan['tensile_strength_Pa'],
        fracture_energy_J_m2=plan['fracture_energy_J_m2'])
    if actual_material!=expected_material:raise RuntimeError('F7 material / characteristic length differs from the original v4 plan')
    times=jnp.asarray(meta['fixture_times']);positions=jnp.asarray(meta['fixture_positions']);releases=jnp.asarray([g['release_time_s'] for g in meta['placement_guides']]);fids=jnp.asarray([g['fibre_id'] for g in meta['placement_guides']])
    start_y=positions[0,2*fids,1]
    def guide_position(t):
        tg=jnp.minimum(t,releases);index=jnp.clip(jnp.searchsorted(times,tg,side='right')-1,0,len(times)-2)
        phase=jnp.clip((tg-times[index])/(times[index+1]-times[index]),0.,1.)
        blend=.5*(1-jnp.cos(jnp.pi*phase));a=positions[index,2*fids,1];b=positions[index+1,2*fids,1]
        return a*(1-blend)+b*blend-start_y
    dt=float(plan['dt'])
    def guide_speed(t):return (guide_position(t)-guide_position(t-dt))/dt
    protocol=dict(fixture_times=np.asarray(meta['fixture_times']).tolist(),fixture_positions=np.asarray(meta['fixture_positions']).tolist())
    solver_dt=dt;duration=steps*solver_dt
    initial,step,particles,energy,geometry_check=make_solver(x,volume,layout,material,geometry['masks'],solver_dt,protocol,
        duration=duration,guide_particles=guides,guide_displacement=guide_position,guide_velocity=guide_speed,
        particle_fibre_ids=ids,guide_active=lambda t:t<releases)
    if steps<1 or checkpoint_every<1:raise ValueError('steps and checkpoint interval must be positive')
    source_path=Path('/work/source.tar');source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else 'unbound-local-source'
    source_root=Path(__file__).resolve().parents[1]
    source_files=sorted([*(source_root/'mpm_jax').rglob('*.py'),*(source_root/'examples').rglob('*.py')])
    source_hashes={p.relative_to(source_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    local_source_sha=hashlib.sha256(json.dumps(source_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    binding=dict(schema='f7-tl-rattle-checkpoint-v1',source_archive_sha256=source_sha,
        local_python_source_sha256=local_source_sha,runtime_versions=dict(jax=jax.__version__,numpy=np.__version__),
        input_identity_sha256=identity,material=actual_material,dt_s=solver_dt,
        guide_particle_ids_sha256=identity['guide_particle_ids'],fixture_times_sha256=identity['fixture_times'],
        fixture_positions_sha256=identity['fixture_positions'],guide_target='centroid y follows fixture through min(t,release); no withdrawal before first release',
        guide_release='remove constraint rows at t >= release',
        guide_work='actual midpoint velocity dot each RATTLE impulse; v2',
        reference_grid='six-cell packed tube; fixed quadratic reference stencil',strain_cap_log=.12,work_gate_relative_maximum=.10)
    binding_text=json.dumps(binding,sort_keys=True,separators=(',',':'));binding_sha=hashlib.sha256(binding_text.encode()).hexdigest()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    latest_path=output/'latest.json';observations_path=output/'observations.jsonl';rows=[];absolute_machine_work=0.;start_step=0;parent_cp_sha=None
    if resume_from:
        parent=Path(resume_from);latest=json.loads((parent/'latest.json').read_text(encoding='utf-8'))
        if latest.get('state') not in ('accepted','continuable') or latest.get('binding_sha256')!=binding_sha:
            raise RuntimeError('Resume parent is not an admitted checkpoint with identical physical/source binding')
        cp_path=parent/latest['checkpoint_file'];actual_sha=hashlib.sha256(cp_path.read_bytes()).hexdigest()
        if actual_sha!=latest['checkpoint_sha256']:raise RuntimeError('Resume checkpoint SHA mismatch')
        with np.load(cp_path,allow_pickle=False) as cp:
            if str(cp['binding_sha256'].item())!=binding_sha or str(cp['binding_json'].item())!=binding_text:
                raise RuntimeError('Checkpoint embedded binding mismatch')
            required={'step','binding_sha256','binding_json','source_archive_sha256','input_identity_sha256_json',
                'absolute_machine_work_J','work_gate_relative_maximum','displacement','velocity','force','F','time','valid',
                'work','impulse','angular_impulse','damage_work',*[('history_'+name) for name in BrittleHistory._fields]}
            if not required.issubset(cp.files):raise RuntimeError('Restart checkpoint is missing required TLState/history/ledger fields')
            history=BrittleHistory(*(jnp.asarray(cp['history_'+name]) for name in BrittleHistory._fields))
            state=TLState(jnp.asarray(cp['displacement']),jnp.asarray(cp['velocity']),jnp.asarray(cp['force']),history,
                jnp.asarray(cp['F']),jnp.asarray(cp['time']),jnp.asarray(cp['valid']),jnp.asarray(cp['work']),
                jnp.asarray(cp['impulse']),jnp.asarray(cp['angular_impulse']),jnp.asarray(cp['damage_work']))
            start_step=int(cp['step'].item());absolute_machine_work=float(cp['absolute_machine_work_J'].item())
            eps=np.finfo(np.asarray(cp['time']).dtype).eps
            product=start_step*solver_dt;accumulation=start_step*eps
            if accumulation>=1:raise RuntimeError('Checkpoint step index exceeds the floating-point accumulation bound')
            time_bound=accumulation/(1-accumulation)*abs(product)+2*eps*max(abs(float(state.time)),abs(product),solver_dt)
            if not bool(state.valid) or not np.isfinite(float(state.time)) or abs(float(state.time)-product)>time_bound:
                raise RuntimeError('Checkpoint state/time is invalid or inconsistent with step index')
        parent_cp_sha=actual_sha
        old_obs=parent/'observations.jsonl'
        if old_obs.exists():observations_path.write_bytes(old_obs.read_bytes())
    else:state=initial
    if (start_step+steps)*solver_dt>=float(np.min(np.asarray(releases))):
        raise ValueError('This qualification stops before the first guide release; release and fibre contact require a separate solver')
    with observations_path.open('a',encoding='utf-8') as observations:
        t0=time.perf_counter()
        for local_step in range(1,steps+1):
            state,row=step(state);jax.block_until_ready(row['valid'])
            row_host={k:np.asarray(v).tolist() for k,v in row.items()};rows.append(row_host)
            observations.write(json.dumps(dict(step=start_step+local_step,**row_host),allow_nan=False)+'\n');observations.flush()
            if not bool(row['valid']):
                raise RuntimeError(f'TL/RATTLE/J/log-strain guard rejected actual step {start_step+local_step}')
            absolute_machine_work+=float(row['machine_abs_work_J'])
            step_number=start_step+local_step
            if step_number%checkpoint_every==0 or local_step==steps:
                K,U=energy(state);K=float(K);U=float(U)
                remainder=float(state.work)-K-U-float(state.damage_work)
                work_ratio=abs(remainder)/max(absolute_machine_work,1e-30)
                if work_ratio>.10:
                    raw=output/'work-budget-rejected-state-not-checkpoint.npz'
                    payload=dict(step=np.asarray(step_number),binding_sha256=np.asarray(binding_sha),binding_json=np.asarray(binding_text),
                        source_archive_sha256=np.asarray(source_sha),input_identity_sha256_json=np.asarray(json.dumps(identity,sort_keys=True)),
                        absolute_machine_work_J=np.asarray(absolute_machine_work),work_gate_relative_maximum=np.asarray(.10),work_remainder_J=np.asarray(remainder),
                        displacement=np.asarray(state.displacement),velocity=np.asarray(state.velocity),force=np.asarray(state.force),
                        F=np.asarray(state.F),time=np.asarray(state.time),valid=np.asarray(state.valid),work=np.asarray(state.work),
                        impulse=np.asarray(state.impulse),angular_impulse=np.asarray(state.angular_impulse),damage_work=np.asarray(state.damage_work),
                        **{'history_'+name:np.asarray(getattr(state.history,name)) for name in BrittleHistory._fields})
                    tmp=raw.with_suffix('.partial.npz');np.savez_compressed(tmp,**payload);tmp.replace(raw)
                    (output/'work-budget-rejection.json').write_text(json.dumps(dict(step=step_number,state='rejected',
                        binding_sha256=binding_sha,work_remainder_J=remainder,absolute_machine_work_J=absolute_machine_work,
                        ratio=work_ratio,gate=.10,checkpoint_remains_previous=True),indent=2)+'\n')
                    return
                payload=dict(step=np.asarray(step_number),binding_sha256=np.asarray(binding_sha),binding_json=np.asarray(binding_text),
                    source_archive_sha256=np.asarray(source_sha),input_identity_sha256_json=np.asarray(json.dumps(identity,sort_keys=True)),
                    work_gate_relative_maximum=np.asarray(.10),absolute_machine_work_J=np.asarray(absolute_machine_work),displacement=np.asarray(state.displacement),
                    velocity=np.asarray(state.velocity),force=np.asarray(state.force),F=np.asarray(state.F),time=np.asarray(state.time),
                    valid=np.asarray(state.valid),work=np.asarray(state.work),impulse=np.asarray(state.impulse),
                    angular_impulse=np.asarray(state.angular_impulse),damage_work=np.asarray(state.damage_work),
                    **{'history_'+name:np.asarray(getattr(state.history,name)) for name in BrittleHistory._fields})
                cp_path=output/f'checkpoint_{step_number:07d}.npz';temporary=output/f'checkpoint_{step_number:07d}.partial.npz'
                np.savez_compressed(temporary,**payload);temporary.replace(cp_path)
                cp_sha=hashlib.sha256(cp_path.read_bytes()).hexdigest()
                pointer=dict(state='accepted' if step_number==64 else 'continuable',step=step_number,
                    time_s=float(state.time),checkpoint_file=cp_path.name,checkpoint_sha256=cp_sha,
                    binding=binding,binding_sha256=binding_sha,absolute_machine_work_J=absolute_machine_work,
                    kinetic_J=K,elastic_J=U,work_J=float(state.work),damage_work_J=float(state.damage_work),
                    work_remainder_J=remainder,work_remainder_relative_to_absolute_work=work_ratio,
                    momentum_impulse= np.asarray(state.impulse).tolist(),angular_impulse=np.asarray(state.angular_impulse).tolist(),
                    parent_checkpoint_sha256=parent_cp_sha)
                tmp_pointer=output/'latest.partial.json';tmp_pointer.write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf-8');tmp_pointer.replace(latest_path)
                old_checkpoints=sorted(output.glob('checkpoint_*.npz'))
                for old in old_checkpoints[:-2]:old.unlink()
        runtime=time.perf_counter()-t0
    latest=json.loads(latest_path.read_text(encoding='utf-8'));K=float(latest['kinetic_J']);U=float(latest['elastic_J'])
    report=dict(case='F7 unchanged geometry + TL/BrittleHistory + finite mass-weighted centroid RATTLE',
        start_step=start_step,steps=steps,final_step=start_step+steps,dt_s=solver_dt,duration_s=float(state.time),
        runtime_s=runtime,particles=len(x),guides=len(guides),cpus=sorted(os.sched_getaffinity(0)),
        machine=dict(platform=platform.platform(),jax=jax.__version__),binding=binding,binding_sha256=binding_sha,
        source_archive_sha256=source_sha,geometry=json.loads(json.dumps(meta,default=lambda z:np.asarray(z).tolist())),
        plan=plan,geometry_check={k:np.asarray(v).tolist() for k,v in geometry_check.items()},rows=rows,
        state=latest['state'],valid=bool(state.valid),machine_work_absolute_J=absolute_machine_work,
        work_remainder_J=latest['work_remainder_J'],work_remainder_relative_to_absolute_work=latest['work_remainder_relative_to_absolute_work'],
        work_gate_relative_maximum=.10,work_gate_passed=latest['work_remainder_relative_to_absolute_work']<=.10,
        final_work_J=float(state.work),final_damage_work_J=float(state.damage_work),final_kinetic_J=K,final_elastic_J=U,
        final_max_damage=float(jnp.max(state.history.damage)),final_max_log_strain=float(max((r['max_log_strain'] for r in rows),default=0.)),
        final_linear_impulse=latest['momentum_impulse'],final_angular_impulse=latest['angular_impulse'],
        parent_checkpoint_sha256=parent_cp_sha,checkpoint_file=latest['checkpoint_file'],checkpoint_sha256=latest['checkpoint_sha256'],
        state_schema='Full TLState: displacement, velocity, force, F, all BrittleHistory fields, time, valid, work, linear/angular impulse, damage_work, absolute machine work and immutable binding',
        qualification_scope='Bounded mechanics only; contact/interlacing/guide withdrawal-release/full-process acceptance not claimed')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda z:np.asarray(z).tolist())+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='output');p.add_argument('--steps',type=int,default=64)
    p.add_argument('--checkpoint-every',type=int,default=64);p.add_argument('--resume-from')
    a=p.parse_args();run(a.output,steps=a.steps,checkpoint_every=a.checkpoint_every,resume_from=a.resume_from)
