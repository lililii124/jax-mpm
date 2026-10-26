"""Read-only terminal-state and saved-frame audit of the JH-2 pressure plate."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

task=Path('/original');data=task/'output/jh2';parent=Path('/parent');work=Path('/work')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_text())
case=read(data/'case.json');cp=read(data/'checkpoint.json');status=read(data/'status.json')
contract=read(task/'contract.json');bridge=read(task/'source-bridge.json')
capsule=read(task/'source/source-capsule.json');guard=read(task/'job/status.json')
assert guard['state']=='complete' and guard['returncode']==0 and guard['guard_reason'] is None
assert contract['model']==case['model']=='jh2' and cp['signature']==contract['signature']
assert capsule['commit']==bridge['physics_source_commit']==contract['physics_source_commit']=='3328c738d1a25da63b1244f63e9291c110776e00'
assert cp['frame']==contract['target_frame']==case['frames']==423
assert status['completed'] and status['steps_completed']==case['planned_steps']==8460
assert status['time']==case['duration_s']==.0002 and case['substeps']==20
parent_manifest=read(parent/'checkpoint.json')
assert sha(parent/parent_manifest['states'][0])==contract['parent_checkpoint_sha256']
for name,digest in capsule['files'].items():assert sha(task/'source'/name)==digest,name
assert sha(task/'source/examples/common.py')==bridge['original_common_sha256']
assert sha(task/'common_compile_only.py')==bridge['adapter_sha256']
rows,observations=cp['rows'],cp['observations']
assert len(rows)==len(observations)==424
def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return value is None or isinstance(value,(str,bool))
assert finite(rows) and finite(observations)
dt=case['time_step_s']*20;tolerance=max(case['time_step_s']*1e-5,1e-15)
for i,(row,obs) in enumerate(zip(rows,observations)):
    assert row['body']==0 and abs(row['time']-i*dt)<=tolerance
    assert abs(obs['time']-i*dt)<=tolerance and abs(obs['load_sample_time_s']-i*dt)<=tolerance
    assert row['mass']>0 and row['current_volume']>0 and row['min_J']>0
    assert 0<=row['max_damage']<=1 and 0<=obs['failed_reference_volume_fraction']<=1
    assert obs['all_material_history_valid'] and obs['fully_supported_clamp_max_motion_m']<=1e-12
mass=np.array([row['mass'] for row in rows]);assert np.ptp(mass)<=1e-12*mass[0]
with np.load(data/cp['states'][0],allow_pickle=False) as archive:state={key:archive[key].copy() for key in archive.files}
n=case['particle_count']
for key,shape in {'x':(n,3),'v':(n,3),'F':(n,3,3),'C':(n,3,3),'mass':(n,),'volume':(n,),
                  'time':(),'valid':(),'history_damage':(n,),'history_temperature':(n,),'history_valid':(n,)}.items():
    assert state[key].shape==shape,key
for key,value in state.items():
    if value.dtype.kind in 'fiu':assert np.isfinite(value).all(),key
assert bool(state['valid']) and state['history_valid'].all()
assert np.all(state['mass']>0) and np.all(state['volume']>0)
determinants=np.linalg.det(state['F']);assert np.all(determinants>0)
assert abs(float(state['time'])-.0002)<=tolerance
kinetic=float(.5*np.sum(state['mass'][:,None]*state['v']**2))
assert np.isclose(kinetic,rows[-1]['kinetic_energy'],rtol=1e-12,atol=1e-12)
assert np.isclose(state['mass'].sum(),mass[-1],rtol=1e-12,atol=0.)
assert np.isclose(determinants.min(),rows[-1]['min_J'],rtol=1e-12,atol=1e-14)
assert np.isclose(determinants.max(),rows[-1]['max_J'],rtol=1e-12,atol=1e-14)
hashes=[];previous_damage=None;reference=None;global_displacement=0.
shapes={'x':(n,3),'speed':(n,),'damage':(n,),'temperature':(n,),'time':(),'contact_impulse':(3,)}
for i in range(424):
    path=data/'frames_data'/f'frame_{i:05d}.npz'
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)==set(shapes)
        frame={key:archive[key] for key in archive.files}
    for key,shape in shapes.items():assert frame[key].shape==shape and np.isfinite(frame[key]).all(),(i,key)
    assert abs(float(frame['time'])-i*dt)<=tolerance
    assert np.all((frame['damage']>=0)&(frame['damage']<=1)) and np.all(frame['temperature']>0)
    if previous_damage is not None:assert np.min(frame['damage']-previous_damage)>=-1e-12
    previous_damage=frame['damage'].copy()
    if reference is None:reference=frame['x'].copy()
    peak=float(np.linalg.norm(frame['x']-reference,axis=1).max())
    global_displacement=max(global_displacement,peak)
    assert np.isclose(peak,observations[i]['maximum_displacement_m'],rtol=1e-10,atol=1e-14)
    if i==423:
        for key,state_key in [('x','x'),('damage','history_damage'),('temperature','history_temperature')]:
            assert np.array_equal(frame[key],state[state_key]),key
    hashes.append(dict(index=i,file=path.name,sha256=sha(path),bytes=path.stat().st_size))
    if i%100==0:print(json.dumps({'audited_frame':i}),flush=True)
receipt=dict(review_status='passed',completed=True,model='jh2',source_commit=capsule['commit'],
             process_exit_code=guard['returncode'],frame=423,steps=8460,time_s=float(state['time']),particle_count=n,
             checkpoint_sha256=sha(data/cp['states'][0]),parent_checkpoint_sha256=contract['parent_checkpoint_sha256'],
             all_raw_frames_finite=True,damage_irreversible_in_saved_frames=True,terminal_frame_matches_full_state=True,
             mass_range_kg=[float(mass.min()),float(mass.max())],positive_terminal_deformation_determinants=True,
             min_J_over_course=min(row['min_J'] for row in rows),max_J_over_course=max(row['max_J'] for row in rows),
             max_damage_over_course=max(row['max_damage'] for row in rows),max_temperature_K=max(row['max_temperature'] for row in rows),
             fully_supported_clamp_max_motion_m=max(row['fully_supported_clamp_max_motion_m'] for row in observations),
             global_displacement_peak_m=global_displacement,terminal_translational_kinetic_J=kinetic,
             final_observation=observations[-1],frame_sha256=hashes,original_files_unchanged=True,
             scope='Full terminal state and all424 raw saved frames checked. Course Jacobians are recorded producer diagnostics; independent Jacobians recomputed at terminal. No experimental, geometric-fracture, or global APIC energy-convergence qualification.')
(work/'independent-terminal-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='frame_sha256'}),flush=True)
