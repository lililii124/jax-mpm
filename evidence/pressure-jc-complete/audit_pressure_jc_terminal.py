import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np

data=Path('/input'); work=Path('/work'); parent=Path('/parent')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
case=json.loads((data/'case.json').read_text())
cp=json.loads((data/'checkpoint.json').read_text())
status=json.loads((data/'status.json').read_text())
contract=json.loads((work/'contract.json').read_text())
bridge=json.loads((work/'source-bridge.json').read_text())
capsule=json.loads((Path('/source')/'source-capsule.json').read_text())
assert contract['model']=='jc' and contract['target_frame']==334 and contract['signature']==cp['signature']
assert contract['physics_source_commit']==capsule['commit']==bridge['physics_source_commit']=='3328c738d1a25da63b1244f63e9291c110776e00'
assert cp['frame']==334 and status['frame']==334 and status['steps_completed']==6680
assert status['time']==case['duration_s']==0.0002 and case['planned_steps']==6680 and case['frames']==334 and case['substeps']==20
assert sha(parent/'checkpoint_0_0.npz')==contract['parent_checkpoint_sha256']
assert sha(Path('/source')/'examples/common.py')==bridge['original_common_sha256']
rows,obs=cp['rows'],cp['observations']
assert len(rows)==len(obs)==335
dt=case['time_step_s']*case['substeps']; tol=max(case['time_step_s']*1e-5,1e-15)
def finite_json(value):
    if isinstance(value,dict): return all(finite_json(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_json(v) for v in value)
    if isinstance(value,(int,float)): return math.isfinite(float(value))
    return value is None or isinstance(value,(str,bool))
assert finite_json(rows) and finite_json(obs)
mass=[float(r['mass']) for r in rows]
for i,(r,o) in enumerate(zip(rows,obs)):
    expected=i*dt
    assert r['body']==0 and abs(float(r['time'])-expected)<=tol
    assert abs(float(o['time'])-expected)<=tol and abs(float(o['load_sample_time_s'])-expected)<=tol
    assert float(r['mass'])>0 and float(r['current_volume'])>0 and float(r['min_J'])>0
    assert 0<=float(r['max_damage'])<=1
    assert o['all_material_history_valid'] is True and float(o['failed_reference_volume_fraction'])==0
    assert float(o['fully_supported_clamp_max_motion_m'])<=1e-12
assert max(mass)-min(mass)<=1e-12*mass[0]
state_path=data/cp['states'][0]
with np.load(state_path,allow_pickle=False) as z:
    state={k:z[k] for k in z.files}
n=int(case['particle_count']); dim=3
for key,shape in {'x':(n,dim),'v':(n,dim),'F':(n,dim,dim),'C':(n,dim,dim),'mass':(n,),
                  'volume':(n,),'time':(),'valid':(),'history_damage':(n,),
                  'history_temperature':(n,),'history_valid':(n,)}.items():
    assert key in state and state[key].shape==shape,(key,state.get(key,None).shape if key in state else None,shape)
for key,value in state.items():
    if value.dtype.kind in 'fiu': assert np.isfinite(value).all(),key
assert bool(np.asarray(state['valid']).all()) and bool(np.asarray(state['history_valid']).all())
assert abs(float(state['time'])-float(status['time']))<=tol and np.all(state['mass']>0) and np.all(state['volume']>0)
assert np.all(np.linalg.det(state['F'])>0)
assert np.all((state['history_damage']>=0)&(state['history_damage']<=1))
assert np.all(state['history_temperature']>0)
frames=[]; hashes=[]; frame_fields={'x':(n,3),'speed':(n,),'damage':(n,),
 'temperature':(n,),'time':(),'contact_impulse':(3,)}
for i in range(335):
    p=data/'frames_data'/f'frame_{i:05d}.npz'; assert p.is_file()
    hashes.append({'index':i,'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)==set(frame_fields),(i,z.files)
        vals={k:z[k] for k in z.files}
    for key,shape in frame_fields.items():
        assert vals[key].shape==shape and np.isfinite(vals[key]).all(),(i,key,vals[key].shape)
    assert abs(float(vals['time'])-i*dt)<=tol
    assert np.all((vals['damage']>=0)&(vals['damage']<=1)) and np.all(vals['temperature']>0)
    if i==334:
        assert np.array_equal(vals['x'],state['x'])
        assert np.array_equal(vals['damage'],state['history_damage'])
        assert np.array_equal(vals['temperature'],state['history_temperature'])
    frames.append(float(vals['time']))
receipt={'schema':'jc-saved-terminal-checkpoint-independent-review-v1',
 'review_status':'passed','completed':True,'process_exit_code':1,
 'process_failure':'Resource guard terminated common.run postprocessing at final trajectory compression after the final frame/checkpoint/status were saved; USS 6575616000 > limit 6442450944 bytes. No solver step was interrupted.',
 'physical_solver_protocol_complete_at_saved_frame':True,'source_commit':capsule['commit'],
 'frame':334,'steps':6680,'time_s':float(state['time']),'particle_count':n,
 'checkpoint_file':cp['states'][0],'checkpoint_sha256':sha(state_path),
 'parent_checkpoint_sha256_verified':contract['parent_checkpoint_sha256'],
 'checkpoint_signature':cp['signature'],'diagnostic_rows':len(rows),'observation_rows':len(obs),
 'frame_count':len(hashes),'all_frames_and_times_valid':True,'all_particle_state_arrays_finite':True,
 'state_valid_all':True,'history_valid_all':True,'positive_deformation_determinants_all':True,
 'positive_mass_and_volume_all':True,'material_damage_in_[0,1]':True,'material_temperature_positive':True,
 'mass_range_kg':[min(mass),max(mass)],'min_J_over_course':min(float(r['min_J']) for r in rows),
 'max_J_over_course':max(float(r['max_J']) for r in rows),
 'max_damage_over_course':max(float(r['max_damage']) for r in rows),
 'max_temperature_over_course_K':max(float(o['maximum_temperature_K']) for o in obs),
 'max_fixed_clamp_motion_m':max(float(o['fully_supported_clamp_max_motion_m']) for o in obs),
 'final_observation':obs[-1], 'frame_sha256':hashes,
 'source_completed_flags_unchanged':True,'final_trajectory_archive_rebuilt':False,
 'scope':'Read-only audit of saved real terminal CP and its 335 frame files. No physics was run and source summary/status completed fields were not changed.'}
(work/'independent-terminal-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='frame_sha256'}),flush=True)
