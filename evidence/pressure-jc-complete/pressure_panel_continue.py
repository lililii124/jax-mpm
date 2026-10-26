"""Frozen pressure setup and unchanged integrator, with an explicit compile-only bridge."""
import difflib,hashlib,importlib.util,importlib.metadata,json,os,shutil,sys
from pathlib import Path

root=Path('/source');sys.path.insert(0,str(root))
contract=json.loads(Path('/work/contract.json').read_text())
manifest=json.loads((root/'source-capsule.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in manifest['files'].items():assert sha(root/name)==digest,name
assert importlib.metadata.version('jax')==importlib.metadata.version('jaxlib')=='0.12.0.1'
assert importlib.metadata.version('numpy')=='2.5.4'
parent=Path('/parent');output=Path('/work/output')/contract['model'];output.mkdir(parents=True)
cp=json.loads((parent/'checkpoint.json').read_text())
assert cp['frame']==contract['parent_frame'] and cp['signature']==contract['signature']
assert sha(parent/cp['states'][0])==contract['parent_checkpoint_sha256']
for item in parent.rglob('*'):
 if not item.is_file() or item.name.startswith('trajectory') or item.suffix=='.tmp':continue
 relative=item.relative_to(parent);dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 # Only immutable saved history is linked. Mutable JSON/CP are private copies.
 if relative.parts[0]=='frames_data':
  try:os.link(item,dest)
  except OSError:shutil.copy2(item,dest)
 else:shutil.copy2(item,dest)
original=(root/'examples/common.py').read_bytes();text=original.decode()
before='    warm, *_ = batch(states, 0.0, material_arguments)\n    warm[0].x.block_until_ready()\n    compile_seconds = perf_counter() - start\n    del warm\n'
after='    batch = batch.lower(states, 0.0, material_arguments).compile()\n    compile_seconds = perf_counter() - start\n'
assert text.count(before)==1
adapted=text.replace(before,after)
trajectory_before='''    np.savez_compressed(
        output / "trajectory.npz",
        x=positions,
        speed=speeds,
        damage=damages,
        temperature=temperatures,
        time=times,
        body=body,
        contact_impulse=impulses,
        **({"failed": failures} if failures is not None else {}),
    )
'''
assert adapted.count(trajectory_before)==1
trajectory_after='''    _atomic_json(output / "trajectory-storage.json", {
        "authoritative_frames": "frames_data/frame_*.npz",
        "saved_frames": frames + 1,
        "redundant_combined_archive_written": False,
        "reason": "Stream individual saved frames; retain the full checkpoint without mapping the entire derived history into resident memory.",
    })
'''
adapted=adapted.replace(trajectory_before,trajectory_after)
cache_helper='''
def _flush_derived_cache(field):
    """Persist identical derived bytes, then release reconstructible file pages."""
    import mmap
    field.flush()
    field._mmap.madvise(mmap.MADV_DONTNEED)
    descriptor = os.open(str(field.filename), os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)

'''
assert adapted.count('\ndef run(')==1
adapted=adapted.replace('\ndef run(',cache_helper+'\ndef run(')
assert adapted.count('                times.append(float(data["time"]))')==1
adapted=adapted.replace('                times.append(float(data["time"]))','                for cache in (positions, speeds, damages, temperatures, failures):\n                    if cache is not None:\n                        _flush_derived_cache(cache)\n                times.append(float(data["time"]))')
assert adapted.count('            field.flush()')==1
adapted=adapted.replace('            field.flush()','            _flush_derived_cache(field)')
anchor='        if frame % 5 == 0 or frame == frames:\n'
assert adapted.count(anchor)==1
adapted=adapted.replace(anchor,'        if not _valid_checkpoint_states(states):\n            raise FloatingPointError("Original state-validity gate failed at the saved20-step checkpoint")\n'+anchor)
adapter=Path('/work/common_compile_only.py');adapter.write_text(adapted)
bridge=dict(status='explicit-io-compile-and-stop-cadence-bridge',physics_source_commit=manifest['commit'],original_common_sha256=hashlib.sha256(original).hexdigest(),adapter_sha256=sha(adapter),diff=''.join(difflib.unified_diff(text.splitlines(True),adapted.splitlines(True))),parent=contract,scope='Frozen setup and common.run equations; no discarded warm physical steps; existing case.json preserved; derived memmaps flushed byte-identically then Linux file pages released; redundant terminal trajectory.npz omitted while all original frame NPZ and full-state checkpoints remain authoritative; all original resume and validity checks retained, with the same state-validity gate also after each saved20-step checkpoint')
Path('/work/source-bridge.json').write_text(json.dumps(bridge,indent=2)+'\n')
import jax,numpy as np
jax.config.update('jax_enable_x64',True)
from examples.pressure_pulse_panels import setup,UPPER
case=setup(contract['model'],grid=128,ppc=2)
assert json.loads(json.dumps(case.description))==json.loads((output/'case.json').read_text()),'Frozen definition mismatch'
spec=importlib.util.spec_from_file_location('common_compile_only',adapter);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
fields={**{'reference_'+axis:np.asarray(case.states[0].x)[:,d] for d,axis in enumerate('xyz')},'loaded_surface':np.asarray(np.isclose(np.asarray(case.states[0].x)[:,1],UPPER[1]-.5*case.description['particle_spacing_m'][1],rtol=0,atol=1e-12),dtype=np.int32)}
states,summary=common.run(case.states,case.materials,case.configs,frames=case.description['frames'],substeps=20,output=output,label='Pressure pulse panel / '+contract['model'],forces=case.forces,observer=case.observer,metadata=case.description,render_output=False,resume=True,stop_after_frame=contract['target_frame'],point_fields=fields)
assert common._valid_checkpoint_states(states),'Final original state gate failed'
saved=json.loads((output/'checkpoint.json').read_text());assert saved['frame']==contract['target_frame'] and saved['signature']==contract['signature']
receipt=dict(status='passed',model=contract['model'],frame=saved['frame'],steps=saved['frame']*20,time_s=saved['rows'][-1]['time'],completed=bool(summary['completed']),source_commit=manifest['commit'],checkpoint_sha256=sha(output/saved['states'][0]),checkpoint_file=saved['states'][0],original_physical_gates_unchanged=True,parent_checkpoint_sha256=contract['parent_checkpoint_sha256'],last_diagnostics=saved['rows'][-1],last_observation=saved['observations'][-1])
Path('/work/continuation-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
