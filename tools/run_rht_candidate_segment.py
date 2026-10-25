"""Continue exact full RHT candidate states, with bounded five-step checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

CONTACT_SCOPE=('Finite MLS-MPM support causes early numerical contact: at step140 the saved particle-nose proxy remains4.893mm above the original front surface (1.30 grid cells at h3.75mm), although nodal contact and target damage exist. This is not certified geometric surface contact or an IJIE accuracy validation. The existing baseline is unchanged; grid/geometry contact convergence remains future work.')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(capsule,output,stop_step):
    output.mkdir(parents=True,exist_ok=False)
    seed=capsule/'seed';old=json.loads((seed/'checkpoint.json').read_text())
    start_step=old['step'];assert start_step%5==0 and start_step<stop_step<=280 and stop_step%35==0
    oldproof=json.loads((seed/'source-review.json').read_text())
    for name,item in old['raw_files'].items():assert digest(seed/name)==item['sha256'],name
    runtime=output/'runtime-source';shutil.copytree(capsule/'runtime-source',runtime)
    manifest={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*') if p.is_file()}
    assert manifest==oldproof['runtime_source_manifest']
    versions={n:importlib.metadata.version(n) for n in oldproof['versions']};assert versions==oldproof['versions']
    proof=dict(original_source_commit='0c99377fac0ff77b376958dff2e124e2a4691f21',compact_source_commit='39a58e3e5781d64c8d72a12aa4ac442133517cd9',
        candidate_rht_sha256=digest(runtime/'mpm_jax/materials/rht.py'),versions=versions,
        seed_step=start_step,seed_checkpoint_sha256=digest(seed/'checkpoint.json'),seed_source_review_sha256=digest(seed/'source-review.json'),
        seed_state_raw_files=old['raw_files'],runtime_source_manifest=manifest,
        original_physical_parameters_iterations_tolerances_and_loading_unchanged=True,instrumentation_added=False,
        source_scope='Original0to105, checked compact106 and107to110 bridge, subsequent explicitly versioned exact full-state candidate segments; not the original full run.',
        contact_scope=CONTACT_SCOPE)
    assert proof['candidate_rht_sha256']==old['candidate_rht_sha256']
    (output/'source-review.json').write_text(json.dumps(proof,indent=2)+'\n')
    sys.path.insert(0,str(runtime))
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    from examples.rht_pointed_impact import make_case
    from examples.common import _load_state,_state_arrays,_valid_checkpoint_states,_atomic_npz,_atomic_json
    from mpm_jax import coupled_step
    from mpm_jax.initialization import diagnostics
    from mpm_jax.particle_fields import material_metadata
    case=make_case();configs,materials=case.configs,case.materials;observer=case.observer
    definition=dict(configs=[asdict(c) for c in configs],materials=[material_metadata(m) for m in materials],
        frames=48,substeps=35,label='RHT pointed impact',definition=case.description)
    assert json.loads(json.dumps(definition))==json.loads((seed/'case.json').read_text())
    assert hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()==old['signature']
    (output/'case.json').write_text(json.dumps(definition,indent=2)+'\n')
    initial_momentum=sum(np.sum(np.asarray(s.mass)[:,None]*np.asarray(s.v),axis=0) for s in case.states)
    assert all(all(float(g)==0 for g in c.gravity) for c in configs)
    states=tuple(_load_state(seed/name) for name in old['states'])
    for s,initial in zip(states,case.states):
        assert s.x.shape==initial.x.shape and np.array_equal(s.mass,initial.mass) and np.array_equal(s.volume,initial.volume)
        assert abs(float(s.time)-start_step*configs[0].dt)<1e-15
    assert _valid_checkpoint_states(states);del case
    cumulative=np.asarray(old['contact_impulse']);rows=list(old['rows']);observations=list(old['observations'])
    record=dict(start_step=start_step,target_step=stop_step,accepted_steps=0,compile_seconds=None,steps=[],
        original_full_run_completed=False,target_reached=False,source_scope=proof['source_scope'],contact_scope=CONTACT_SCOPE,
        checkpoint_interval=5,seed_checkpoint_sha256=proof['seed_checkpoint_sha256'],checkpoints=[])
    def progress():_atomic_json(output/'result.json',record)
    def checkpoint(step):
        actual_time=float(states[0].time);energies=[];new_rows=[]
        for body,(s,c,m) in enumerate(zip(states,configs,materials)):
            row=dict(time=actual_time,body=body,**diagnostics(s,c,m));new_rows.append(row)
            mass=np.asarray(s.mass);C=np.asarray(s.C);h=c.domain_size/c.n_grid
            energies.append(dict(body=body,translational_kinetic_J=row['kinetic_energy'],
                quadratic_APIC_affine_kinetic_diagnostic_J=float(h*h/8*np.sum(mass*np.sum(C*C,axis=(1,2)))),
                material_history_energy_J=float(np.sum(mass*np.asarray(s.history.energy))),
                constitutive_work_J=row['constitutive_work'],stored_elastic_J=row['strain_energy']))
        momentum=sum(np.sum(np.asarray(s.mass)[:,None]*np.asarray(s.v),axis=0) for s in states)
        observation=dict(time=actual_time,external_work_trapezoidal=float(old['external_work']),
            **{f'boundary_impulse_{axis}':float((momentum-initial_momentum-np.asarray(old['external_impulse']))[i]) for i,axis in enumerate('xyz')})
        observation.update(observer(states,actual_time))
        if not rows or rows[-1]['time']!=actual_time:rows.extend(new_rows)
        if not observations or observations[-1]['time']!=actual_time:observations.append(observation)
        raw={};names=[]
        for body,s in enumerate(states):
            name=f'state{step}-body{body}.npz';_atomic_npz(output/name,**_state_arrays(s));names.append(name)
            raw[name]=dict(bytes=(output/name).stat().st_size,sha256=digest(output/name))
        record['last_complete_step']=step;record['target_reached']=step==stop_step
        saved=dict(step=step,time_s=actual_time,frame=step//35 if step%35==0 else None,states=names,raw_files=raw,
            bodies=energies,external_work_J=float(old['external_work']),contact_impulse_cumulative_on_body1=cumulative.tolist())
        record['checkpoints'].append(saved)
        cp=dict(signature=old['signature'],step=step,frame=saved['frame'],states=names,raw_files=raw,
            candidate_rht_sha256=proof['candidate_rht_sha256'],time_s=actual_time,contact_impulse=cumulative.tolist(),
            external_impulse=old['external_impulse'],external_work=old['external_work'],rows=rows,observations=observations,
            progress=record,original_full_run_completed=False,source_scope=proof['source_scope'],contact_scope=CONTACT_SCOPE)
        _atomic_json(output/'checkpoint.json',cp);_atomic_json(output/f'checkpoint-step{step}.json',cp)
        _atomic_json(output/'diagnostics.json',rows);_atomic_json(output/'observations.json',observations)
        progress();print(json.dumps({'complete_checkpoint':step,'time_s':actual_time,'energies':energies,'contact_impulse':cumulative.tolist()}),flush=True)
    checkpoint(start_step);start=time.perf_counter();compiled=coupled_step.lower(states,materials,configs).compile()
    record['compile_seconds']=time.perf_counter()-start;progress()
    for step in range(start_step+1,stop_step+1):
        print(json.dumps({'step_begin':step}),flush=True);start=time.perf_counter()
        updated,impulse=compiled(states,materials);jax.block_until_ready((updated,impulse))
        elapsed=time.perf_counter()-start;valid=bool(_valid_checkpoint_states(updated))
        row=dict(step=step,solve_seconds=elapsed,valid=valid,contact_impulse=np.asarray(impulse).tolist(),
            target_min_J=float(np.linalg.det(np.asarray(updated[0].F)).min()),target_max_damage=float(np.asarray(updated[0].history.damage).max()),
            body_valid=[bool(s.valid) for s in updated],all_history_valid=[bool(np.asarray(s.history.valid).all()) for s in updated])
        record['steps'].append(row);record['accepted_steps']+=int(valid);progress();print(json.dumps(row),flush=True)
        if not valid:
            for body,s in enumerate(updated):_atomic_npz(output/f'invalid-step{step}-body{body}.npz',**_state_arrays(s))
            raise RuntimeError('Original validity gate failed; last valid atomic checkpoint preserved')
        cumulative+=np.asarray(impulse);states=updated
        if step%5==0:checkpoint(step)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-step',type=int,required=True)
    a=p.parse_args();run(a.capsule.resolve(),a.output.resolve(),a.stop_step)
