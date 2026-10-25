"""Linux-only authenticated125--135ns shared-CFL candidate diagnostic."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_numpy_scalar(value):
    """Convert NumPy scalar report values without altering any admission gate."""
    if isinstance(value,np.generic): return value.item()
    raise TypeError(f'Unsupported report type: {type(value).__name__}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.name!='posix': raise RuntimeError('Heavy JAX requires Linux cloud')
    if a.output.exists(): raise FileExistsError('Preserve earlier raw')
    a.output.mkdir(parents=True)
    admission=json.loads(a.admission.read_text())
    for name,h in admission['source_sha256'].items():
        if sha(name)!=h: raise ValueError('Source pin differs: '+name)
    for name,h in admission['input_sha256'].items():
        if sha(a.audit/name)!=h: raise ValueError('Audited input differs: '+name)
    import platform,jax,jaxlib
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax import fragment_multifield_v6_mpm as original
    from mpm_jax.fragment_v6_delta_diagnostic import compare_shared_adaptive,candidate_fixed_schedule
    from mpm_jax.fragment_v6_onset_probe import advance_to_traced
    from mpm_jax.materials.linear_metal import linear_metal
    from tools.audit_metal_onset_state import audit_state_pair
    expected=json.loads((a.audit/'result/dt1/equivalence.json').read_text())
    runtime=dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
    if runtime!=expected['runtime']: raise ValueError('Exact producer runtime required')
    with np.load(a.audit/'result/dt1/epoch-initial.npz',allow_pickle=False) as cp:
        fields={k:cp[k].copy() for k in cp.files}
    state=original.ImpactState(*(jnp.asarray(fields[k]) for k in original.ImpactState._fields))
    x=fields['x'];labels=fields['body_id'];reference=fields['reference'];physical=fields['physical_body'];dx=.00025
    if float(state.time)!=125000*1e-12 or not bool(state.valid): raise ValueError('Exact valid125ns required')
    bodies=np.array([np.unique(physical[labels==i]).item() for i in range(int(labels.max())+1)])
    predicted=x+125000*1e-12*fields['v'];lower=np.full((len(bodies),3),np.inf);upper=np.full_like(lower,-np.inf)
    np.minimum.at(lower,labels,np.minimum(x,predicted));np.maximum.at(upper,labels,np.maximum(x,predicted))
    bounds=np.stack((lower-2*dx,upper+2*dx),axis=-1)
    grid=original.build_grid(x,labels,dx=dx,padding_cells=3,reference_bounds=bounds,max_nodes=2_000_000,field_physical_body=bodies,contact_sweeps=(x,predicted))
    spec=json.loads((a.audit/'result/dt1/grid-spec.json').read_text())
    if grid.layout.node_count!=spec['nodes'] or len(grid.pairs)!=spec['pairs'] or hashlib.sha256(np.asarray(grid.pairs).tobytes()).hexdigest()!=spec['pairs_ordered_bytes_sha256']: raise ValueError('Exact original grid required')
    fixed=(physical==1)&((abs(reference[:,0])>=.009)|(abs(reference[:,2])>=.009))
    support=original.stencil(state.x,state.body_id,grid.layout);mask=np.zeros((grid.layout.node_count,3),bool)
    nodes=np.asarray(support.indices)[fixed].ravel();nodes=nodes[(nodes>=0)&(nodes<grid.layout.node_count)];mask[nodes]=True
    grid=replace(grid,clamped=jnp.asarray(mask));material=linear_metal(**admission['material'])
    config=original.ImpactConfig(cfl=.1,contact_iterations=1024,contact_tolerance=.005)
    original.validate_state(state,material,grid)
    with np.load(a.audit/'result/dt1/adaptive-trace.npz',allow_pickle=False) as saved: dt=saved['dt_s'].copy();old_channels=saved['channels_J'].copy()
    if not 1<=len(dt)<=512 or not np.isfinite(dt).all() or not np.all(dt>0) or abs(dt.sum()-10e-9)>1e-18: raise ValueError('Exact bounded10ns trace required')
    # Authenticate through the EXACT producer wrappers. A new compiled scan can
    # change fusion/rounding even when its Python step source is identical.
    target=jnp.asarray(135000*1e-12)
    authenticated,n_original=original.advance_to(state,material,grid,config,target,max_steps=512)
    authenticated.x.block_until_ready()
    authenticated_probe,n_probe,_,_,auth_identity,auth_trace=advance_to_traced(state,material,grid,config,target,max_steps=512)
    authenticated_probe.x.block_until_ready()
    auth_trace=np.asarray(auth_trace)[:int(n_probe)]
    authenticated_dt=auth_trace[:,0];baseline_channels=auth_trace[:,1:]
    saved_dt=dt.copy()
    baseline,changed,count,shared_dt,bd,cd=compare_shared_adaptive(state,material,grid,config,jnp.asarray(135000*1e-12),max_steps=512);changed.x.block_until_ready()
    count=int(count);dt=np.asarray(shared_dt)[:count]
    bd=jax.tree.map(lambda a:a[:count],bd);cd=jax.tree.map(lambda a:a[:count],cd)
    def save(name,current): np.savez(a.output/name,**{k:np.asarray(v) for k,v in current._asdict().items()})
    save('authenticated-dt1-final.npz',authenticated);save('authenticated-dt1-probe-final.npz',authenticated_probe)
    np.savez(a.output/'authenticated-dt1-trace.npz',dt_s=authenticated_dt,channels_J=baseline_channels)
    save('baseline-final.npz',baseline);save('candidate-instrumented-final.npz',changed)
    np.savez(a.output/'step-diagnostics.npz',dt_s=dt,**{'baseline_'+k:np.asarray(v) for k,v in bd._asdict().items()},**{'candidate_'+k:np.asarray(v) for k,v in cd._asdict().items()})
    independent=candidate_fixed_schedule(state,material,grid,config,jnp.asarray(dt));independent.x.block_until_ready();save('candidate-uninstrumented-final.npz',independent)
    bitwise=lambda u,v: all(np.asarray(getattr(u,k)).tobytes()==np.asarray(getattr(v,k)).tobytes() for k in original.ImpactState._fields)
    with np.load(a.audit/'result/dt1/original-final.npz',allow_pickle=False) as cp: saved=original.ImpactState(*(cp[k] for k in original.ImpactState._fields))
    roundoff=admission['energy_roundoff_J']
    parts=np.asarray(cd.p2g_dissipation+cd.g2p_dissipation+cd.contact_compatible_energy+cd.viscosity_work+cd.constitutive_work_defect)
    identity=np.asarray(cd.numerical_energy_increment)-parts
    channel_tolerance=64*np.finfo(float).eps*max(1e-12,float(np.max(abs(old_channels))))
    audit=audit_state_pair(state,changed,reference,fields['edges'],fields['alive'],physical,dx)
    valid=True;validation_errors={}
    for name,current in (('baseline',baseline),('candidate_instrumented',changed),('candidate_uninstrumented',independent)):
        try: original.validate_state(current,material,grid)
        except (ValueError,FloatingPointError) as error:
            valid=False;validation_errors[name]=type(error).__name__+': '+str(error)
    # Temporal-spread envelope is an a-priori diagnostic screen, not validation.
    shapes={};shape_pass=True
    for field in ('x','F','J','C','plastic_strain','damage'):
        original_dt1=np.linalg.det(saved.F) if field=='J' else np.asarray(getattr(saved,field))
        b=np.linalg.det(np.asarray(baseline.F)) if field=='J' else np.asarray(getattr(baseline,field))
        c=np.linalg.det(np.asarray(changed.F)) if field=='J' else np.asarray(getattr(changed,field));envelope=[]
        for divisor in (2,4):
            with np.load(a.audit/f'result/dt{divisor}/original-final.npz',allow_pickle=False) as cp: difference=(np.linalg.det(cp['F']) if field=='J' else cp[field])-original_dt1
            envelope.append((float(np.max(abs(difference))),float(np.linalg.norm(difference))))
        eps=64*np.finfo(float).eps*max(float(np.max(abs(b))),1e-12)
        limit_inf=max(v[0] for v in envelope)+eps;limit_l2=max(v[1] for v in envelope)+eps*np.sqrt(b.size)
        error_inf=float(np.max(abs(c-b)));error_l2=float(np.linalg.norm(c-b));accepted=error_inf<=limit_inf and error_l2<=limit_l2
        shapes[field]=dict(linf=error_inf,l2=error_l2,limit_linf=limit_inf,limit_l2=limit_l2,accepted=accepted);shape_pass &= accepted
    delta=float(changed.numerical_reservoir-state.numerical_reservoir);base_delta=float(baseline.numerical_reservoir-state.numerical_reservoir)
    required_reduction=float(np.asarray(bd.g2p_dissipation).sum())
    checks=dict(all_physical_valid=valid,baseline_saved_bitwise=bitwise(authenticated,saved),candidate_instrumented_bitwise=bitwise(changed,independent),mass_exact=audit['mass_exact'],
        original_producer_probe_bitwise=bitwise(authenticated,authenticated_probe),
        original_producer_schedule=bool(int(n_original)==int(n_probe)==len(saved_dt) and authenticated_dt.shape==saved_dt.shape and authenticated_dt.tobytes()==saved_dt.tobytes()),
        original_producer_identity=float(auth_identity)<=roundoff,
        baseline_step_channels=bool(baseline_channels.shape==old_channels.shape and np.max(abs(baseline_channels-old_channels))<=channel_tolerance),
        every_step_valid=bool(np.all(np.asarray(bd.valid)) and np.all(np.asarray(cd.valid))),
        exact_endpoints=float(changed.time)==float(baseline.time)==135000*1e-12,
        positive_bounded_schedule=bool(1<=len(dt)<=512 and np.isfinite(dt).all() and np.all(dt>0) and abs(dt.sum()-10e-9)<=1e-18),
        common_cfl_gate=bool(np.all(dt<=np.asarray(bd.dt_limit)*(1+1e-12)) and np.all(dt<=np.asarray(cd.dt_limit)*(1+1e-12))),
        energy_identity=bool(np.max(abs(identity))<=roundoff and np.max(abs(np.asarray(cd.energy_residual)))<=roundoff),
        reservoir_sum=abs(delta-float(np.asarray(cd.numerical_energy_increment).sum()))<=roundoff,
        material_reservoir_reduction=delta<=base_delta-required_reduction+roundoff,shape_screen=bool(shape_pass),
        contact_gate=float(changed.maximum_contact_closing_speed)<=.005 and float(changed.maximum_contact_complementarity_residual)<=.005,
        source_unchanged=all(sha(n)==h for n,h in admission['source_sha256'].items()))
    candidate_valid=np.asarray(cd.valid);rejected=np.flatnonzero(~candidate_valid)
    first_rejected=int(rejected[0]) if len(rejected) else None
    execution=dict(attempted_updates=len(dt),accepted_prefix_steps=first_rejected if first_rejected is not None else len(dt),
                   first_rejected_step_1based=first_rejected+1 if first_rejected is not None else None,
                   first_rejected_dt_s=float(dt[first_rejected]) if first_rejected is not None else None,
                   first_rejected_cfl_limit_s=float(np.asarray(cd.dt_limit)[first_rejected]) if first_rejected is not None else None,
                   rows_after_first_rejection='None: shared adaptive loop stops after first rejected update.')
    result=dict(admitted=all(checks.values()),checks=checks,validation_errors=validation_errors,candidate_execution=execution,shape_screen=shapes,physical_audit=audit,candidate_numerical_increment_J=delta,baseline_numerical_increment_J=base_delta,required_reduction_J=required_reduction,maximum_step_transfer_identity_residual_J=float(np.max(abs(identity))),scope='125--135ns shared minimum-CFL diagnostic screen; temporal envelope is not validation. Original dt1 replay authenticates baseline only. Scan halts after first rejection.',production_restart_allowed=False,public_media_allowed=False)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=json_numpy_scalar)+'\n')
    if not result['admitted']: raise RuntimeError('Candidate rejected; retain all raw')


if __name__=='__main__':main()
