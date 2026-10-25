"""Finish candidate frame4 with five-step atomic full states and original guards."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(capsule,output,resume=False):
    output.mkdir(parents=True,exist_ok=resume)
    parent=capsule/'parent';seed=capsule/'seed110'
    original=json.loads((parent/'raw/checkpoint.json').read_text())
    bridge=json.loads((seed/'result.json').read_text())
    assert bridge['accepted_steps']==4 and bridge['steps'][-1]['step']==110
    for name,item in bridge['steps'][-1]['raw_files'].items():assert digest(seed/name)==item['sha256']
    seedproof=json.loads((seed/'source-review.json').read_text())
    review=json.loads((parent/'extraction-review.json').read_text())
    for name,item in review['selected_members'].items():assert digest(parent/name)==item['sha256'],name
    runtime=output/'runtime-source'
    if not resume:
        for name in review['selected_members']:
            if name.startswith('source/'):
                target=runtime/name[7:];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
        (runtime/'mpm_jax/materials/rht.py').write_bytes((capsule/'compact-rht.py').read_bytes())
    manifest={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert manifest==json.loads((seed/'runtime-source-manifest.json').read_text())
    versions={n:importlib.metadata.version(n) for n in seedproof['versions']}
    assert versions==seedproof['versions']
    proof=dict(original_source_commit='0c99377fac0ff77b376958dff2e124e2a4691f21',compact_source_commit='39a58e3e5781d64c8d72a12aa4ac442133517cd9',
        candidate_rht_sha256=digest(runtime/'mpm_jax/materials/rht.py'),versions=versions,
        parent105_checkpoint_sha256={name:digest(parent/'raw'/name) for name in ['checkpoint.json','checkpoint_1_0.npz','checkpoint_1_1.npz']},
        seed110_raw_files=bridge['steps'][-1]['raw_files'],seed110_source_review_sha256=digest(seed/'source-review.json'),
        seed110_result_sha256=digest(seed/'result.json'),runtime_source_manifest=manifest,
        physical_parameters_and_all_original_validity_tolerances_unchanged=True,instrumentation_added=False,
        source_scope='Original0to105 followed by independently checked compact106 and107to110 bridge, then this compact candidate; not the original complete run.')
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
    assert hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()==original['signature']
    (output/'case.json').write_text(json.dumps(definition,indent=2)+'\n')
    initial_momentum=sum(np.sum(np.asarray(s.mass)[:,None]*np.asarray(s.v),axis=0) for s in case.states)
    assert all(all(float(g)==0 for g in c.gravity) for c in configs)
    seed106=json.loads((capsule/'lineage/compact106-result.json').read_text())
    cumulative=np.asarray(original['contact_impulse'])+np.asarray(seed106['contact_impulse'])
    for row in bridge['steps']:cumulative+=np.asarray(row['contact_impulse'])
    rows=list(original['rows']);observations=list(original['observations'])
    record=dict(start_step=110,target_step=140,accepted_steps=0,compile_seconds=None,steps=[],
        original_full_run_completed=False,candidate_frame4_reached=False,
        source_scope=proof['source_scope'],checkpoint_interval=5,
        parent_solver_seconds=original['solver_seconds'],bridge106_solve_seconds=seed106['solve_seconds'],
        bridge107to110_solve_seconds=sum(r['solve_seconds'] for r in bridge['steps']))
    start_step=110
    if resume:
        checkpoint=json.loads((output/'checkpoint.json').read_text())
        assert checkpoint['signature']==original['signature'] and 110<=checkpoint['step']<140
        assert checkpoint['candidate_rht_sha256']==proof['candidate_rht_sha256']
        for name,item in checkpoint['raw_files'].items():assert digest(output/name)==item['sha256']
        states=tuple(_load_state(output/name) for name in checkpoint['states'])
        start_step=checkpoint['step'];cumulative=np.asarray(checkpoint['contact_impulse'])
        rows=checkpoint['rows'];observations=checkpoint['observations'];record=checkpoint['progress']
    else:states=tuple(_load_state(seed/f'state110-body{i}.npz') for i in range(2))
    for s,initial in zip(states,case.states):
        assert s.x.shape==initial.x.shape and np.array_equal(s.mass,initial.mass) and np.array_equal(s.volume,initial.volume)
        assert abs(float(s.time)-start_step*configs[0].dt)<1e-15
    assert _valid_checkpoint_states(states)
    del case
    def save_progress():
        _atomic_json(output/'result.json',record)
    def checkpoint(step):
        actual_time=float(states[0].time);body_rows=[];energies=[]
        for body,(s,c,m) in enumerate(zip(states,configs,materials)):
            row=dict(time=actual_time,body=body,**diagnostics(s,c,m));body_rows.append(row)
            mass=np.asarray(s.mass);C=np.asarray(s.C);h=c.domain_size/c.n_grid
            energies.append(dict(body=body,translational_kinetic_J=row['kinetic_energy'],
                quadratic_APIC_affine_kinetic_diagnostic_J=float(h*h/8*np.sum(mass*np.sum(C*C,axis=(1,2)))),
                material_history_energy_J=float(np.sum(mass*np.asarray(s.history.energy))),
                constitutive_work_J=row['constitutive_work'],stored_elastic_J=row['strain_energy']))
        rows.extend(body_rows)
        momentum=sum(np.sum(np.asarray(s.mass)[:,None]*np.asarray(s.v),axis=0) for s in states)
        observation=dict(time=actual_time,external_work_trapezoidal=float(original['external_work']),
            **{f'boundary_impulse_{axis}':float((momentum-initial_momentum-np.asarray(original['external_impulse']))[i]) for i,axis in enumerate('xyz')})
        observation.update(observer(states,actual_time));observations.append(observation)
        raw={};names=[]
        for body,s in enumerate(states):
            name=f'state{step}-body{body}.npz';_atomic_npz(output/name,**_state_arrays(s));names.append(name)
            raw[name]=dict(bytes=(output/name).stat().st_size,sha256=digest(output/name))
        record['last_complete_step']=step;record['candidate_frame4_reached']=step==140
        record.setdefault('checkpoint_energy_records',[]).append(dict(step=step,time_s=actual_time,bodies=energies,
            external_work_J=float(original['external_work']),external_work_scope='No applied particle forces or gravity; stationary fixed clamps. Original common external-force work remains zero. No inferred numerical loss is assigned as heat.',
            contact_impulse_cumulative_on_body1=cumulative.tolist()))
        cp=dict(signature=original['signature'],step=step,frame=4 if step==140 else None,states=names,raw_files=raw,
            candidate_rht_sha256=proof['candidate_rht_sha256'],time_s=actual_time,contact_impulse=cumulative.tolist(),
            external_impulse=original['external_impulse'],external_work=original['external_work'],
            rows=rows,observations=observations,progress=record,original_full_run_completed=False)
        _atomic_json(output/'checkpoint.json',cp);_atomic_json(output/'diagnostics.json',rows);_atomic_json(output/'observations.json',observations)
        save_progress();print(json.dumps({'complete_checkpoint':step,'time_s':actual_time,'energies':energies,'contact_impulse':cumulative.tolist()}),flush=True)
    if not resume:checkpoint(110)
    save_progress();start=time.perf_counter();compiled=coupled_step.lower(states,materials,configs).compile()
    record['compile_seconds']=time.perf_counter()-start
    record.setdefault('compile_sessions',[]).append(dict(start_step=start_step,seconds=record['compile_seconds']))
    save_progress()
    for step in range(start_step+1,141):
        print(json.dumps({'step_begin':step}),flush=True);start=time.perf_counter()
        updated,impulse=compiled(states,materials);jax.block_until_ready((updated,impulse))
        elapsed=time.perf_counter()-start;valid=bool(_valid_checkpoint_states(updated))
        row=dict(step=step,solve_seconds=elapsed,valid=valid,contact_impulse=np.asarray(impulse).tolist(),
            target_min_J=float(np.linalg.det(np.asarray(updated[0].F)).min()),target_max_damage=float(np.asarray(updated[0].history.damage).max()),
            body_valid=[bool(s.valid) for s in updated],all_history_valid=[bool(np.asarray(s.history.valid).all()) for s in updated])
        record['steps'].append(row);record['accepted_steps']+=int(valid);save_progress();print(json.dumps(row),flush=True)
        if not valid:
            for body,s in enumerate(updated):_atomic_npz(output/f'invalid-step{step}-body{body}.npz',**_state_arrays(s))
            raise RuntimeError('Original validity gate failed; last valid atomic checkpoint preserved')
        cumulative+=np.asarray(impulse);states=updated
        if step%5==0:checkpoint(step)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.capsule.resolve(),a.output.resolve(),a.resume)
