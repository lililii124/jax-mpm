"""Single 64-step TL/RATTLE qualification for the unchanged F7 guided geometry."""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np


def run(output):
    import jax
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_tube_layout
    from mpm_jax.fibre_tlmpm import make_solver
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle
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
    solver_dt=dt;duration=64*solver_dt
    initial,step,particles,energy,geometry_check=make_solver(x,volume,layout,material,geometry['masks'],solver_dt,protocol,
        duration=duration,guide_particles=guides,guide_displacement=guide_position,guide_velocity=guide_speed,particle_fibre_ids=ids)
    state=initial;rows=[];t0=time.perf_counter()
    for i in range(64):
        state,row=step(state);jax.block_until_ready(row['valid'])
        rows.append({k:np.asarray(v).tolist() for k,v in row.items()})
        if not bool(row['valid']):raise RuntimeError(f'RATTLE/TL guard rejected physical step {i+1}')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    K,U=energy(state);K=float(K);U=float(U);wabs=float(sum(float(r['machine_abs_work_J']) for r in rows))
    remainder=float(state.work)-K-U-float(state.damage_work);work_ratio=abs(remainder)/max(wabs,1e-30)
    work_accepted=work_ratio<=.10;accepted=bool(state.valid)&work_accepted
    array_payload=dict(displacement=np.asarray(state.displacement),velocity=np.asarray(state.velocity),F=np.asarray(state.F),
        kappa=np.asarray(state.history.kappa),damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed))
    state_file='qualification.npz' if accepted else 'rejected-state-not-checkpoint.npz'
    if not accepted:array_payload['resume_forbidden']=np.asarray(True)
    np.savez_compressed(output/state_file,**array_payload)
    report=dict(case='F7 unchanged geometry + TL/BrittleHistory + finite mass-weighted centroid RATTLE',steps=64,
        dt_s=solver_dt,duration_s=float(state.time),runtime_s=time.perf_counter()-t0,particles=len(x),guides=len(guides),
        cpus=sorted(os.sched_getaffinity(0)),machine=dict(platform=platform.platform(),jax=jax.__version__),
        geometry=json.loads(json.dumps(meta,default=lambda z:np.asarray(z).tolist())),plan=plan,
        immutable_input_identity_sha256=identity,material_identity=actual_material,
        geometry_check={k:np.asarray(v).tolist() for k,v in geometry_check.items()},rows=rows,
        valid=bool(state.valid),state='accepted' if accepted else 'work-budget-rejected',
        machine_work_absolute_J=wabs,work_remainder_J=remainder,work_remainder_relative_to_absolute_work=work_ratio,
        work_gate_relative_maximum=.10,work_gate_passed=work_accepted,resume_forbidden=not accepted,
        final_work_J=float(state.work),final_damage_work_J=float(state.damage_work),final_kinetic_J=K,final_elastic_J=U,
        final_max_damage=float(jnp.max(state.history.damage)),guide_projection='position and velocity; mass metric; only centroid y; x/z unconstrained',
        constraint_work_definition=dict(
            position_multiplier='lambda_x has kg*m; correction is M^-1 G^T lambda_x; half-step constraint impulse is G^T lambda_x/dt',
            position_machine_work='sum_g (lambda_x[g]/dt)*v_guide_mid[g], v_guide_mid=(g(t+dt)-g(t))/dt',
            velocity_multiplier='lambda_v has kg*m/s and is the velocity projection impulse',
            velocity_ideal_machine_work='sum_g lambda_v[g]*v_target(t+dt)[g]',
            velocity_actual_jump='sum_g lambda_v[g]*(Gv_before[g]+Gv_after[g])/2; separately compare with nodal delta kinetic energy',
            discrete_constraint_error='ideal prescribed-velocity work minus actual impulse-midpoint work; reported per step, never hidden in heat or tuned away',
            machine_work_absolute='sum over each end group and guide position/velocity channel of absolute channel work'),
        qualification_scope='64-step mechanics/guide integration only; contact/interlacing/release and full-process acceptance not claimed')
    text=json.dumps(report,indent=2,allow_nan=False,default=lambda z:np.asarray(z).tolist())+'\n';(output/'report.json').write_text(text)
    report['state_sha256']=hashlib.sha256((output/state_file).read_bytes()).hexdigest();report['state_file']=state_file
    (output/'latest.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda z:np.asarray(z).tolist())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='output');run(p.parse_args().output)
