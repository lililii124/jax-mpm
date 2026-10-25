"""Actual f40 particle geometry: frozen, force-free APIC transfer isolation.

Only particle v/C change. Positions, F, history, mass and physical time never
advance. Repetition count is not a physical simulation trajectory.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def step_function():
    import jax
    import jax.numpy as jnp
    from mpm_jax import core

    @partial(jax.jit,static_argnames=('config',))
    def transfer(v,C,mass,x,support,gm,config,frozen_time):
        momentum=mass[:,None,None]*v[:,None,:]+jnp.einsum('pij,pkj->pki',mass[:,None,None]*C,support.dpos)
        gp=jnp.zeros((len(gm),3),dtype=v.dtype).at[support.indices].add(support.weights[...,None]*momentum)
        gv=gp/jnp.where(gm>0,gm,1.)[:,None]
        node=jnp.arange(len(gm))
        strides=(config.n_grid+1)**jnp.arange(2,-1,-1)
        coordinates=((node[:,None]//strides)%(config.n_grid+1))*config.dx
        energy=lambda velocity:.5*jnp.sum(gm[:,None]*velocity**2)
        p_energy=lambda velocity,affine:.5*jnp.sum(mass*jnp.sum(velocity**2,axis=1))+config.dx**2/8*jnp.sum(mass[:,None,None]*affine**2)
        angular=lambda velocity,affine:jnp.sum(mass[:,None]*(jnp.cross(x,velocity)+config.dx**2/4*jnp.stack(
            (affine[:,2,1]-affine[:,1,2],affine[:,0,2]-affine[:,2,0],affine[:,1,0]-affine[:,0,1]),axis=1)),axis=0)
        before_energy=p_energy(v,C)
        grid_energy=energy(gv)
        before_momentum=jnp.sum(mass[:,None]*v,axis=0)
        before_angular=angular(v,C)
        grid_angular=jnp.sum(jnp.cross(coordinates,gp),axis=0)
        work,projection,impulse=0.,0.,jnp.zeros(3,dtype=v.dtype)
        stages=[]
        for boundary in config.boundaries:
            following=boundary.apply(gv,coordinates,frozen_time)
            local_impulse=gm[:,None]*(following-gv)
            local_work=jnp.sum(local_impulse*following)
            local_projection=.5*jnp.sum(gm[:,None]*(following-gv)**2)
            stages.append(jnp.r_[local_work,energy(following)-energy(gv),local_projection,jnp.sum(local_impulse,axis=0)])
            work+=local_work
            projection+=local_projection
            impulse+=jnp.sum(local_impulse,axis=0)
            gv=following
        next_v,next_C=core.grid_to_particle(gv,support,config)
        after_energy=p_energy(next_v,next_C)
        p2g_loss=before_energy-grid_energy
        g2p_loss=energy(gv)-after_energy
        row=jnp.r_[before_energy,grid_energy,energy(gv),after_energy,p2g_loss,g2p_loss,work,projection,
            after_energy-before_energy-work+p2g_loss+g2p_loss+projection,
            jnp.max(jnp.abs(jnp.sum(mass[:,None]*next_v,axis=0)-before_momentum-impulse)),
            jnp.max(jnp.abs(grid_angular-before_angular)),
            jnp.max(jnp.abs(angular(next_v,next_C)-jnp.sum(jnp.cross(coordinates,gm[:,None]*gv),axis=0)))]
        return next_v,next_C,row,jnp.stack(stages) if stages else jnp.zeros((0,6),dtype=v.dtype)
    return transfer


def run(output,parent):
    import jax
    import jax.numpy as jnp
    from mpm_jax import core
    from mpm_jax.materials.history import History
    from examples.brittle_engineering import setup
    jax.config.update('jax_enable_x64',True)
    output.mkdir(parents=True,exist_ok=False)
    case=setup('concrete-bending-mat111',grid=64,ppc=3,frames=60)
    config,material=case.configs[0],case.materials[0]
    assert json.loads(json.dumps(case.description))==json.loads((parent/'plan.json').read_text())['definition']
    assert config.dt==1.4285714285714285e-7
    checkpoint=json.loads((parent/'raw/checkpoint.json').read_text())
    assert checkpoint['frame']==40 and checkpoint['states']==['checkpoint_0_0.npz']
    with np.load(parent/'raw/checkpoint_0_0.npz',allow_pickle=False) as raw:
        fields={name:jnp.asarray(raw[name]) for name in raw.files}
    history=History(**{name:fields['history_'+name] for name in History._fields})
    state=core.State(**{name:history if name=='history' else fields[name] for name in core.State._fields})
    assert config.boundary_cells==0 and not any(config.gravity)
    frozen_time=state.time+config.dt
    support=core.particle_stencil(state,config)
    zero_stress=state._replace(history=state.history._replace(stress=jnp.zeros_like(state.history.stress)))
    gm,gp=core.particle_to_grid(zero_stress,material,config,support)
    weights=np.asarray(support.weights)
    dpos=np.asarray(support.dpos)
    zeroth=np.max(np.abs(weights.sum(1)-1))
    first=np.max(np.abs(np.einsum('pk,pki->pi',weights,dpos)))
    second=np.einsum('pk,pki,pkj->pij',weights,dpos,dpos)
    second_error=np.max(np.abs(second-config.dx**2/4*np.eye(3)))
    assert zeroth<1e-14 and first<1e-16 and second_error<1e-18
    grid_mass_error=float(abs(np.asarray(gm).sum()-np.asarray(state.mass).sum()))
    assert grid_mass_error<1e-12
    no_boundaries=replace(config,boundaries=())
    operator=step_function()
    affine=np.array([[.3,.2,0.],[-.1,.4,.1],[0.,-.2,-.1]])
    translation=np.array([.01,-.02,.03])
    v=jnp.asarray(translation+(np.asarray(state.x)-.2)@affine.T)
    C=jnp.broadcast_to(jnp.asarray(affine),state.C.shape)
    av,aC,arow,_=operator(v,C,state.mass,state.x,support,gm,no_boundaries,frozen_time)
    affine_v_error=float(np.max(np.abs(np.asarray(av-v))))
    affine_C_error=float(np.max(np.abs(np.asarray(aC-C))))
    assert affine_v_error<1e-14 and affine_C_error<1e-12
    assert np.max(np.abs(np.asarray(arow)[[4,5,8,9,10,11]]))<1e-12
    reference_v,reference_C=core.grid_to_particle(gp/jnp.where(gm>0,gm,1.)[:,None],support,config)
    checked_v,checked_C,_,_=operator(state.v,state.C,state.mass,state.x,support,gm,no_boundaries,frozen_time)
    core_v_error=float(np.max(np.abs(np.asarray(reference_v-checked_v))))
    core_C_error=float(np.max(np.abs(np.asarray(reference_C-checked_C))))
    assert core_v_error<1e-12 and core_C_error<1e-9
    samples={1,2,4,8,16,32,64,128}
    cases=[]
    headers=['apic_energy_before_J','grid_energy_pre_boundary_J','grid_energy_post_boundary_J','apic_energy_after_J',
        'p2g_loss_J','g2p_loss_J','prescribed_boundary_work_J','boundary_projection_loss_J','accounting_identity_residual_J',
        'linear_momentum_error_Ns','p2g_angular_momentum_error_Nms','g2p_angular_momentum_error_Nms']
    variants=[('no-boundaries',no_boundaries),('stationary-support-symmetry',replace(config,boundaries=config.boundaries[:-1])),
        ('all-original-boundaries-frozen-time',config)]
    for label,cfg in variants:
        v,C=state.v,state.C
        rows,stages=[],[]
        started=time.monotonic()
        records=[]
        for index in range(1,129):
            v,C,row,boundary_rows=operator(v,C,state.mass,state.x,support,gm,cfg,frozen_time)
            rows.append(np.asarray(row));stages.append(np.asarray(boundary_rows))
            current=rows[-1]
            assert np.isfinite(current).all()
            assert current[4]>=-1e-10 and current[5]>=-1e-10 and current[7]>=0
            assert abs(current[8])<1e-10 and max(current[9:])<1e-10
            if index in samples:
                matrix=np.asarray(rows)
                records.append(dict(applications=index,energy_after_J=float(current[3]),
                    single_p2g_loss_J=float(current[4]),single_g2p_loss_J=float(current[5]),
                    cumulative_p2g_loss_J=float(matrix[:,4].sum()),cumulative_g2p_loss_J=float(matrix[:,5].sum()),
                    cumulative_boundary_work_J=float(matrix[:,6].sum()),
                    cumulative_boundary_projection_J=float(matrix[:,7].sum())))
        values=np.asarray(rows)
        np.savez_compressed(output/(label+'-operator.npz'),values=values,stages=np.asarray(stages),headers=np.asarray(headers))
        # Preserve actual end v/C for independent energy and conservation review.
        np.savez_compressed(output/(label+'-end.npz'),v=np.asarray(v),C=np.asarray(C))
        cases.append(dict(label=label,seconds=time.monotonic()-started,applications=128,samples=records,
            max_accounting_residual_J=float(np.max(np.abs(values[:,8]))),
            max_momentum_error_Ns=float(np.max(np.abs(values[:,9]))),
            max_p2g_angular_error_Nms=float(np.max(np.abs(values[:,10]))),
            max_g2p_angular_error_Nms=float(np.max(np.abs(values[:,11]))),
            nonzero_prescribed_boundary_work=label=='all-original-boundaries-frozen-time'))
    result=dict(passed=True,commit=os.environ.get('GITHUB_SHA'),parent_frame=40,
        parent_checkpoint_sha256=hashlib.sha256((parent/'raw/checkpoint_0_0.npz').read_bytes()).hexdigest(),
        particles=121104,frozen_geometry_time_s=float(state.time),frozen_boundary_sample_time_s=float(frozen_time),
        positions_F_history_mass_time_unchanged=True,material_updates=0,physical_steps_executed=0,
        transfer_applications=386,operator_independent_of_dt=True,
        pure_core_zero_stress_comparison=dict(max_v_error=core_v_error,max_C_error=core_C_error),
        support_moments=dict(grid_mass_error_kg=grid_mass_error,max_partition_error=float(zeroth),max_first_moment_m=float(first),
            max_second_moment_error_m2=float(second_error),D_m2=config.dx**2/4,APIC_inverse_D_per_m2=4/config.dx**2),
        affine_reproduction=dict(max_velocity_error_m_per_s=affine_v_error,max_C_error_per_s=affine_C_error,
            initial_apic_energy_J=float(arow[0]),final_apic_energy_J=float(arow[3])),cases=cases,
        structural_energy_certified=False,temporal_convergence_certified=False,
        scope='Actual frozen f40 geometry force-free repeated APIC operator; application count is not physical time. Complete-boundary case includes separately measured frozen prescribed-velocity work.')
    (output/'frozen-transfer-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        run(args.output.resolve(),args.parent.resolve())
    else:
        import ctypes
        import psutil
        from tools.run_phase_field_visible_crack import owned_session_uss,terminate_owned_group
        assert sys.platform=='linux' and ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
        output=args.output.resolve()
        output.parent.mkdir(parents=True,exist_ok=True)
        guard=dict(seconds_limit=1800,uss_limit_bytes=int(5.5*1024**3),minimum_available_bytes=2*1024**3,
            peak_uss_bytes=0,guard_reason=None)
        started=time.monotonic()
        process=None
        try:
            with (output.parent/'worker.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',
                    '--output',str(output),'--parent',str(args.parent.resolve())],cwd=ROOT,
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
        if code!=0 or guard['guard_reason']:
            raise RuntimeError('Frozen transfer diagnostic stopped; logs retained')
