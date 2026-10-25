"""One uninstrumented original impact step from the authenticated step105 state."""
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
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(parent, candidate, mode, output):
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads((parent/'extraction-review.json').read_text())
    for name, item in review['selected_members'].items():
        if digest(parent/name) != item['sha256']:
            raise ValueError('Changed original member: '+name)
    runtime = output/'runtime-source'
    for name in review['selected_members']:
        if name.startswith('source/'):
            target = runtime/name[len('source/'):]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent/name,target)
    original = (runtime/'mpm_jax/materials/rht.py').read_bytes()
    replacement = candidate.read_bytes()
    if mode == 'compact':
        (runtime/'mpm_jax/materials/rht.py').write_bytes(replacement)
    manifest = {str(p.relative_to(runtime)): digest(p) for p in runtime.rglob('*') if p.is_file()}
    (output/'runtime-source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    proof = dict(mode=mode, parent_source_commit='0c99377fac0ff77b376958dff2e124e2a4691f21',
        original_rht_sha256=hashlib.sha256(original).hexdigest(),
        candidate_rht_sha256=hashlib.sha256(replacement).hexdigest(),
        only_allowed_replacement='mpm_jax/materials/rht.py' if mode=='compact' else None,
        parent_checkpoint_sha256={n:digest(parent/'raw'/n) for n in
            ('checkpoint.json','checkpoint_1_0.npz','checkpoint_1_1.npz')},
        exact_original_56_bisections_64_roots_and_tolerances=True,
        instrumentation_added=False, physical_steps_requested=1)
    versions={name:importlib.metadata.version(name) for name in
              ('jax','jaxlib','numpy','scipy','ml_dtypes','opt_einsum','psutil')}
    required=dict(jax='0.12.0.1',jaxlib='0.12.0.1',numpy='2.5.4',scipy='1.18.1',
                  ml_dtypes='0.6.0',opt_einsum='3.4.0',psutil='7.2.2')
    if versions != required:
        raise ValueError('Different original dependency versions: '+str(versions))
    proof['versions']=versions
    (output/'source-review.json').write_text(json.dumps(proof,indent=2)+'\n')
    sys.path.insert(0,str(runtime))
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    from examples.rht_pointed_impact import make_case
    from examples.common import _load_state, _state_arrays, _valid_checkpoint_states
    from mpm_jax import coupled_step
    from mpm_jax.particle_fields import material_metadata
    case=make_case()
    saved=json.loads((parent/'raw/checkpoint.json').read_text())
    assert saved['frame']==3
    configs,materials=case.configs,case.materials
    definition=dict(configs=[asdict(c) for c in configs],materials=[material_metadata(m) for m in materials],
                    frames=48,substeps=35,label='RHT pointed impact',definition=case.description)
    assert hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()==saved['signature']
    (output/'case.json').write_text(json.dumps(definition,indent=2)+'\n')
    states=tuple(_load_state(parent/'raw'/name) for name in saved['states'])
    for current,initial in zip(states,case.states):
        assert current.x.shape==initial.x.shape
        assert np.array_equal(current.mass,initial.mass) and np.array_equal(current.volume,initial.volume)
        assert abs(float(current.time)-105*configs[0].dt)<1e-15
    assert _valid_checkpoint_states(states)
    del case
    record=dict(mode=mode,start_step=105,target_step=106,compile_seconds=None,solve_seconds=None,
                original_initial_state_valid=True,accepted_steps=0,production_continuation_complete=False)
    def save():
        (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    save()
    start=time.perf_counter()
    compiled=coupled_step.lower(states,materials,configs).compile()
    record['compile_seconds']=time.perf_counter()-start
    save()
    start=time.perf_counter()
    updated,impulse=compiled(states,materials)
    jax.block_until_ready((updated,impulse))
    record['solve_seconds']=time.perf_counter()-start
    files={}
    for body,state in enumerate(updated):
        target=output/f'state106-body{body}.npz'
        np.savez_compressed(target,**_state_arrays(state))
        files[target.name]=dict(bytes=target.stat().st_size,sha256=digest(target))
    target=output/'contact-impulse.npy';np.save(target,np.asarray(impulse))
    files[target.name]=dict(bytes=target.stat().st_size,sha256=digest(target))
    valid=bool(_valid_checkpoint_states(updated))
    record.update(valid=valid,accepted_steps=int(valid),contact_impulse=np.asarray(impulse).tolist(),
        time_s=[float(s.time) for s in updated],raw_files=files,
        body_valid=[bool(s.valid) for s in updated],
        all_history_valid=[bool(np.asarray(s.history.valid).all()) for s in updated])
    save()
    if not valid:
        raise RuntimeError('Original validity gate failed; full raw state retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--mode',choices=('full','compact'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    run(a.parent.resolve(),a.candidate.resolve(),a.mode,a.output.resolve())
