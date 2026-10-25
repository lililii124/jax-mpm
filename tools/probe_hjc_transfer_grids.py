"""Same authenticated f40 particles: background-grid-only frozen comparison."""
from dataclasses import replace,asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def run(output,parent):
    import jax
    import jax.numpy as jnp
    from mpm_jax import core
    from mpm_jax.materials.history import History
    from examples.brittle_engineering import setup
    from tools.probe_hjc_frozen_transfer import step_function
    jax.config.update('jax_enable_x64',True)
    output.mkdir(parents=True,exist_ok=False)
    original=setup('concrete-bending-mat111',grid=64,ppc=3,frames=60)
    reference=original.configs[0]
    definition=json.loads((parent/'plan.json').read_text())['definition']
    assert json.loads(json.dumps(original.description))==definition
    checkpoint=json.loads((parent/'raw/checkpoint.json').read_text())
    assert checkpoint['frame']==40 and checkpoint['states']==['checkpoint_0_0.npz']
    path=parent/'raw/checkpoint_0_0.npz'
    with np.load(path,allow_pickle=False) as raw:
        arrays={name:jnp.asarray(raw[name]) for name in raw.files}
    history=History(**{name:arrays['history_'+name] for name in History._fields})
    state=core.State(**{name:history if name=='history' else arrays[name] for name in core.State._fields})
    assert len(state.x)==121104 and bool(state.valid) and bool(jnp.all(state.history.valid))
    sample_time=state.time+reference.dt  # Same physical sample time for BOTH grids.
    mass,x=np.asarray(state.mass),np.asarray(state.x)
    kinetic=float(.5*np.sum(mass*np.sum(np.asarray(state.v)**2,axis=1)))
    affine_norm=float(np.sum(mass[:,None,None]*np.asarray(state.C)**2))
    operator=step_function()
    reports=[]
    for grid in (64,96):
        config=replace(reference,n_grid=grid,dt=reference.dt*64/grid)
        assert config.boundaries==reference.boundaries and config.domain_size==reference.domain_size
        support=core.particle_stencil(state,config)
        assert bool(support.valid)
        zero=state._replace(history=state.history._replace(stress=jnp.zeros_like(state.history.stress)))
        gm,_=core.particle_to_grid(zero,original.materials[0],config,support)
        grid_mass=np.asarray(gm)
        assert abs(grid_mass.sum()-mass.sum())<1e-12
        weights,dpos=np.asarray(support.weights),np.asarray(support.dpos)
        m0=float(np.max(np.abs(weights.sum(1)-1)))
        m1=float(np.max(np.abs(np.einsum('pk,pki->pi',weights,dpos))))
        m2=float(np.max(np.abs(np.einsum('pk,pki,pkj->pij',weights,dpos,dpos)-config.dx**2/4*np.eye(3))))
        assert m0<1e-14 and m1<1e-16 and m2<1e-18
        coordinate1d=np.arange(grid+1)*config.dx
        node=np.arange((grid+1)**3)
        coordinates=((node[:,None]//((grid+1)**np.arange(2,-1,-1)))%(grid+1))*config.dx
        boundaries=[]
        for name,boundary in zip(definition['boundary_names'],config.boundaries):
            displacement,target=boundary.motion(sample_time) if hasattr(boundary,'motion') else (np.zeros(3),np.asarray(boundary.velocity))
            lower=np.asarray(boundary.lower)+np.asarray(displacement)
            upper=np.asarray(boundary.upper)+np.asarray(displacement)
            mask=np.all((coordinates>=lower)&(coordinates<=upper),axis=1)
            boundaries.append(dict(name=name,nominal_lower_m=list(boundary.lower),nominal_upper_m=list(boundary.upper),
                translated_lower_m=lower.tolist(),translated_upper_m=upper.tolist(),
                components=list(boundary.components),target_velocity_m_per_s=np.asarray(target).tolist(),
                selected_grid_axis_indices={axis:np.flatnonzero((coordinate1d>=lower[i])&(coordinate1d<=upper[i])).tolist()
                    for i,axis in enumerate(('x','y','z'))},selected_grid_nodes=int(mask.sum()),
                occupied_selected_grid_nodes=int(np.count_nonzero(mask&(grid_mass>0))),
                selected_grid_mass_kg=float(grid_mass[mask].sum())))
        del coordinates,node
        none=replace(config,boundaries=())
        B=np.array([[.3,.2,0.],[-.1,.4,.1],[0.,-.2,-.1]])
        av0=jnp.asarray(np.array([.01,-.02,.03])+(x-.2)@B.T)
        aC0=jnp.broadcast_to(jnp.asarray(B),state.C.shape)
        av,aC,arow,_=operator(av0,aC0,state.mass,state.x,support,gm,none,sample_time)
        affine_v_error=float(np.max(np.abs(np.asarray(av-av0))))
        affine_C_error=float(np.max(np.abs(np.asarray(aC-aC0))))
        assert affine_v_error<1e-14 and affine_C_error<1e-12
        assert abs(float(arow[3]-arow[0]))<1e-12
        maximum=128*grid//64
        sample_counts={1,*[base*grid//64 for base in (2,4,8,16,32,64,128)]}
        cases=[]
        for label,cfg in [('no-boundaries',none),('stationary-support-symmetry',replace(config,boundaries=config.boundaries[:-1])),
                          ('all-original-boundaries-frozen-time',config)]:
            started=time.monotonic()
            v,C=state.v,state.C
            rows,stages,samples=[],[],[]
            for index in range(1,maximum+1):
                v,C,row,boundary_rows=operator(v,C,state.mass,state.x,support,gm,cfg,sample_time)
                rows.append(np.asarray(row));stages.append(np.asarray(boundary_rows))
                assert np.isfinite(rows[-1]).all()
                assert min(rows[-1][4:6])>=-1e-10 and rows[-1][7]>=0
                assert abs(rows[-1][8])<1e-10 and max(rows[-1][9:])<1e-10
                if index in sample_counts:
                    matrix=np.asarray(rows)
                    samples.append(dict(applications=index,equivalent_grid64_applications=index*64/grid,
                        nominal_CFL_matched_window_s=index*config.dt,energy_after_J=float(matrix[-1,3]),
                        cumulative_transfer_loss_J=float(matrix[:,4:6].sum()),
                        cumulative_boundary_projection_J=float(matrix[:,7].sum()),
                        cumulative_boundary_work_J=float(matrix[:,6].sum()),
                        single_transfer_loss_J=float(matrix[-1,4:6].sum())))
            values=np.asarray(rows)
            prefix=f'grid{grid}-{label}'
            np.savez_compressed(output/(prefix+'-operator.npz'),values=values,stages=np.asarray(stages))
            np.savez_compressed(output/(prefix+'-end.npz'),v=np.asarray(v),C=np.asarray(C))
            cases.append(dict(label=label,applications=maximum,seconds=time.monotonic()-started,samples=samples,
                initial_apic_energy_J=float(values[0,0]),
                max_identity_error_J=float(np.max(np.abs(values[:,8]))),
                max_momentum_error_Ns=float(np.max(np.abs(values[:,9]))),
                max_p2g_angular_error_Nms=float(np.max(np.abs(values[:,10]))),
                max_g2p_angular_error_Nms=float(np.max(np.abs(values[:,11])))))
        reports.append(dict(grid=grid,dx_m=config.dx,nominal_dt_s=config.dt,particles=len(x),
            full_support=True,mass_kg=float(mass.sum()),grid_mass_kg=float(grid_mass.sum()),
            support_moments=dict(partition_error=m0,first_moment_m=m1,second_moment_error_m2=m2),
            common_particle_kinetic_J=kinetic,grid_metric_affine_kinetic_J=config.dx**2/8*affine_norm,
            affine_reproduction=dict(max_v_error=affine_v_error,max_C_error=affine_C_error),
            actual_boundary_discretization=boundaries,cases=cases))
    result=dict(passed=True,commit=os.environ.get('GITHUB_SHA'),source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        particles=121104,same_particle_positions_F_v_C_history_mass=True,
        physical_boundary_coordinates_identical=True,boundary_selection_changed_with_grid=True,
        physical_time_frozen_s=float(state.time),boundary_time_common_s=float(sample_time),
        material_updates=0,physical_steps_executed=0,
        scope='Frozen transfer only, same actual f40 particles and physical boundary boxes. dt labels set comparison counts by common CFL; no physical trajectory advances. APIC affine kinetic norm changes with D=h²/4 and is reported separately.',
        grids=reports,structural_energy_certified=False,temporal_convergence_certified=False)
    (output/'frozen-grid-comparison-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
