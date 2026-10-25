"""Four original coupled steps with the compact kernel; preserve every full state."""
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


def run(capsule,output):
    output.mkdir(parents=True,exist_ok=False)
    parent=capsule/'parent';seed=capsule/'seed106'
    baseline=json.loads((seed/'result.json').read_text())
    assert baseline['mode']=='compact' and baseline['accepted_steps']==1 and baseline['valid']
    for name,record in baseline['raw_files'].items():assert digest(seed/name)==record['sha256']
    review=json.loads((parent/'extraction-review.json').read_text())
    for name,item in review['selected_members'].items():assert digest(parent/name)==item['sha256'],name
    runtime=output/'runtime-source'
    for name in review['selected_members']:
        if name.startswith('source/'):
            target=runtime/name[7:];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
    kernel=runtime/'mpm_jax/materials/rht.py'
    kernel.write_bytes((capsule/'compact-rht.py').read_bytes())
    seedproof=json.loads((seed/'source-review.json').read_text())
    assert digest(kernel)==seedproof['candidate_rht_sha256']
    manifest={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*') if p.is_file()}
    assert manifest==json.loads((seed/'runtime-source-manifest.json').read_text())
    (output/'runtime-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    versions={n:importlib.metadata.version(n) for n in seedproof['versions']}
    assert versions==seedproof['versions']
    proof=dict(parent_source_commit=seedproof['parent_source_commit'],candidate_rht_sha256=digest(kernel),
        seed106_raw_sha256=baseline['raw_files'],versions=versions,instrumentation_added=False,
        exact_same_runtime_source_as_independently_checked_compact106=True,
        physical_steps_requested=[107,108,109,110],all_original_iteration_limits_and_tolerances_unchanged=True)
    (output/'source-review.json').write_text(json.dumps(proof,indent=2)+'\n')
    sys.path.insert(0,str(runtime))
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    from examples.rht_pointed_impact import make_case
    from examples.common import _load_state,_state_arrays,_valid_checkpoint_states
    from mpm_jax import coupled_step
    from mpm_jax.particle_fields import material_metadata
    case=make_case();configs,materials=case.configs,case.materials
    definition=dict(configs=[asdict(c) for c in configs],materials=[material_metadata(m) for m in materials],
        frames=48,substeps=35,label='RHT pointed impact',definition=case.description)
    assert json.loads(json.dumps(definition))==json.loads((seed/'case.json').read_text())
    (output/'case.json').write_text(json.dumps(definition,indent=2)+'\n')
    states=tuple(_load_state(seed/f'state106-body{i}.npz') for i in range(2))
    for s,initial in zip(states,case.states):
        assert s.x.shape==initial.x.shape and np.array_equal(s.mass,initial.mass) and np.array_equal(s.volume,initial.volume)
        assert abs(float(s.time)-106*configs[0].dt)<1e-15
    assert _valid_checkpoint_states(states)
    del case
    record=dict(start_step=106,target_step=110,accepted_steps=0,compile_seconds=None,steps=[],
        instrumentation_added=False,production_continuation_complete=False)
    def save():
        (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    save();start=time.perf_counter()
    compiled=coupled_step.lower(states,materials,configs).compile()
    record['compile_seconds']=time.perf_counter()-start;save()
    for step in range(107,111):
        print(json.dumps({'step_begin':step}),flush=True);start=time.perf_counter()
        updated,impulse=compiled(states,materials);jax.block_until_ready((updated,impulse))
        elapsed=time.perf_counter()-start;files={}
        for body,s in enumerate(updated):
            target=output/f'state{step}-body{body}.npz';np.savez_compressed(target,**_state_arrays(s))
            files[target.name]=dict(bytes=target.stat().st_size,sha256=digest(target))
        target=output/f'contact-impulse-step{step}.npy';np.save(target,np.asarray(impulse))
        files[target.name]=dict(bytes=target.stat().st_size,sha256=digest(target))
        valid=bool(_valid_checkpoint_states(updated))
        record['steps'].append(dict(step=step,solve_seconds=elapsed,valid=valid,
            contact_impulse=np.asarray(impulse).tolist(),raw_files=files,
            target_min_J=float(np.linalg.det(np.asarray(updated[0].F)).min()),
            target_max_damage=float(np.asarray(updated[0].history.damage).max()),
            body_valid=[bool(s.valid) for s in updated],
            all_history_valid=[bool(np.asarray(s.history.valid).all()) for s in updated]))
        record['accepted_steps']+=int(valid);save()
        if not valid:raise RuntimeError('Original validity gate failed; full invalid and prior accepted raw retained')
        states=updated


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.capsule.resolve(),a.output.resolve())
