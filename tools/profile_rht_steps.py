"""Trace the existing RHT iteration counts from the verified frame-three state."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import types


def instrument(raw):
    text=raw.decode()
    changes=[
        ('    return jax.lax.while_loop(pending, iteration, initial)[3]',
         '    _profile_root = jax.lax.while_loop(pending, iteration, initial)\n'
         '    _rht_profile_trace("root", _profile_root[0], jnp.sum(active), jnp.sum(active & ~_profile_root[4]))\n'
         '    return _profile_root[3]'),
        ('    initial_substeps = jnp.maximum(1, jnp.ceil(largest / 0.001).astype(jnp.int32))',
         '    initial_substeps = jnp.maximum(1, jnp.ceil(largest / 0.001).astype(jnp.int32))\n'
         '    _rht_profile_trace("subdivision_initial", largest, initial_substeps)'),
        ('        result = jax.lax.fori_loop(0, subdivisions, substep, old)',
         '        _rht_profile_trace("subdivision_begin", count, subdivisions)\n'
         '        result = jax.lax.fori_loop(0, subdivisions, substep, old)\n'
         '        _rht_profile_trace("subdivision_end", count, subdivisions, jnp.sum(~result.valid))'),
        ('        result = update_rht_prescribed_energy(old, strain, J, dt, p, energy)',
         '        _rht_profile_trace("energy_begin", count)\n'
         '        result = update_rht_prescribed_energy(old, strain, J, dt, p, energy)'),
        ('        return count + 1, next_energy, result, converged',
         '        _rht_profile_trace("energy_end", count, jnp.sum(~result.valid), jnp.sum(~converged), jnp.max(jnp.abs(next_energy-energy)))\n'
         '        return count + 1, next_energy, result, converged'),
    ]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Unrecognized original RHT function structure')
        text=text.replace(before,after)
    restored=text
    for before,after in reversed(changes):restored=restored.replace(after,before)
    assert restored.encode()==raw
    return ('from _rht_profile_hook import trace as _rht_profile_trace\n'+text).encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=35)
    args=parser.parse_args()
    if not 1<=args.steps<=35:parser.error('Profile at most the next existing frame')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    parent=args.parent.resolve()
    verification=json.loads((parent/'extraction-review.json').read_text())
    for name,record in verification['selected_members'].items():
        if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Changed parent member: '+name)
    source=output/'source'
    shutil.copytree(parent/'source',source)
    kernel=source/'mpm_jax/materials/rht.py';original=kernel.read_bytes()
    modified=instrument(original);kernel.write_bytes(modified)
    proof={'original_kernel_sha256':hashlib.sha256(original).hexdigest(),
           'profiled_kernel_sha256':hashlib.sha256(modified).hexdigest(),
           'removing_explicit_trace_and_temporary_restores_original_bytes':True,
           'parent_checkpoint':json.loads((parent/'raw/checkpoint.json').read_text())['signature'],
           'scope':'Performance diagnostics only; trace callbacks and one-step dispatch may change timing and floating reduction scheduling'}
    (output/'source-review.json').write_text(json.dumps(proof,indent=2)+'\n')
    sys.path.insert(0,str(source))
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    origin=time.perf_counter();current_step=105;events=0;counts={}
    stream=(output/'material-trace.jsonl').open('w',buffering=1)
    def emit(label,*values):
        nonlocal events
        counts[label]=counts.get(label,0)+1
        row={'seconds':time.perf_counter()-origin,'step':current_step,'event':label,
             'values':[np.asarray(value).item() for value in values]}
        events+=1
        if events<=20000:stream.write(json.dumps(row)+'\n')
        if label!='root' or events%100==0:
            temporary=output/'last-material-event.tmp'
            temporary.write_text(json.dumps({**row,'event_counts':counts,'events':events})+'\n')
            temporary.replace(output/'last-material-event.json')
    hook=types.ModuleType('_rht_profile_hook')
    hook.trace=lambda label,*values:jax.debug.callback(lambda *data:emit(label,*data),*values,ordered=True)
    sys.modules[hook.__name__]=hook
    from examples.rht_pointed_impact import make_case
    from examples.common import _load_state,_state_arrays,_valid_checkpoint_states
    from mpm_jax import coupled_step
    from mpm_jax.particle_fields import material_metadata
    case=make_case()
    checkpoint=json.loads((parent/'raw/checkpoint.json').read_text())
    assert checkpoint['frame']==3
    configs,materials=case.configs,case.materials
    signature_data={'configs':[asdict(config) for config in configs],
                    'materials':[material_metadata(material) for material in materials],
                    'frames':48,'substeps':35,'label':'RHT pointed impact','definition':case.description}
    assert hashlib.sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest()==checkpoint['signature']
    (output/'case.json').write_text(json.dumps(signature_data,indent=2)+'\n')
    states=tuple(_load_state(parent/'raw'/name) for name in checkpoint['states'])
    for saved,initial in zip(states,case.states):
        assert saved.x.shape==initial.x.shape
        assert np.array_equal(saved.mass,initial.mass) and np.array_equal(saved.volume,initial.volume)
        assert abs(float(saved.time)-105*configs[0].dt)<1e-15
    assert _valid_checkpoint_states(states)
    del case
    record={'start_step':105,'accepted_steps':0,'compile_seconds':None,'steps':[]}
    target=output/'step-timing.json'
    def save():target.write_text(json.dumps(record,indent=2)+'\n')
    save();emit('compile_begin')
    start=time.perf_counter()
    compiled=coupled_step.lower(states,materials,configs).compile()
    record['compile_seconds']=time.perf_counter()-start
    save();emit('compile_end')
    for offset in range(args.steps):
        current_step=106+offset;emit('step_begin')
        start=time.perf_counter()
        updated,impulse=compiled(states,materials)
        jax.block_until_ready(updated);jax.effects_barrier()
        elapsed=time.perf_counter()-start
        valid=_valid_checkpoint_states(updated)
        row={'step':current_step,'seconds':elapsed,'valid':bool(valid),
             'contact_impulse':np.asarray(impulse).tolist(),
             'target_max_damage':float(np.max(updated[0].history.damage)),
             'target_min_J':float(np.min(np.linalg.det(updated[0].F)))}
        for body,state in enumerate(updated):
            temporary=output/f'last-state-{body}.tmp'
            with temporary.open('wb') as f:np.savez_compressed(f,**_state_arrays(state))
            temporary.replace(output/f'last-state-{body}.npz')
        record['steps'].append(row)
        if valid:record['accepted_steps']+=1
        save();emit('step_end',elapsed)
        if not valid:raise RuntimeError('Original state validity failed; diagnostic state retained')
        states=updated
    stream.close()
    print(json.dumps({'accepted_steps':record['accepted_steps'],'events':events,'event_counts':counts}))


if __name__=='__main__':main()
