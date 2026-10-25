"""Resume exact original woven-impact state; immutable CPs and rolling derived cache."""
import argparse
import csv
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();contract = json.loads(a.contract.read_text())
    root = Path(__file__).resolve().parents[1]
    capsule = json.loads((root/'source-capsule.json').read_text())
    for name, digest in capsule['files'].items():assert sha(root/name)==digest, name
    for name, pin in contract['versions'].items():assert importlib.metadata.version(name)==pin, name
    if shutil.disk_usage(a.output.parent).free < 10*1024**3:raise MemoryError('Insufficient disk reserve')
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64', True)
    from examples import woven_large_plate as base
    from examples import woven_deep_impact as deep
    from tools.review_woven_large import _numpy_elastic, saved_energy_gate
    parent = a.parent/'output/plain/deep-sphere-impact'
    assert sha(parent/'checkpoint.npz')==contract['parent_checkpoint_sha256']
    if contract['parent_frame']==34:
        # A seed mounted at /parent is itself the immutable original raw tree.
        original = json.loads((a.parent/'source.json').read_text())
        assert original['commit']=='c5eccddd9d0857ecc3e0af6e3bfa356409817986'
        assert all(original['files'][name]['sha256']==sha(root/name) for name in base.SOURCE_FILES)
    folder = a.output/'plain/deep-sphere-impact';folder.mkdir(parents=True)
    copied = [];linked = []
    for source in parent.iterdir():
        if not source.is_file() or source.suffix=='.tmp' or source.name=='final_state_0.npz' or source.name.startswith('trajectory_'):continue
        target = folder/source.name
        shutil.copy2(source, target);os.chmod(target,0o600);copied.append(source.name)
    for directory in ('atomic-frames', 'frames_data'):
        target = a.output.parent/directory;target.mkdir()
        prior = a.parent/directory
        if prior.exists():
            for source in prior.iterdir():
                if source.is_file():os.link(source,target/source.name)
    sample, plan = base.load_saved_case(folder)
    assert plan['deep_impact']['initial_energy_J']==40 and plan['frames']==60 and plan['particles']==819200
    assert 34<=contract['parent_frame']<contract['target_frame']<=60 and contract['target_frame']-contract['parent_frame']<=3
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as data:
        base.validate_checkpoint_control(data,plan)
        assert int(data['frame'])==contract['parent_frame']
        tables=json.loads(str(data['tables']))
        current_frame=dict(x=data['state_x'],speed=np.linalg.norm(data['state_v'],axis=1).astype(np.float32),
                           damage=np.zeros(len(sample.positions),np.float32),temperature=np.full(len(sample.positions),293.15,np.float32))
        current_time=float(data['state_time'])
        initial_material_energy,*_= _numpy_elastic(data['state_F'],sample.volume,sample.phase,sample.orientation,plan['material_parameters'])
        assert np.isclose(initial_material_energy,tables['rows'][-1]['stored_elastic_energy_J'],rtol=1e-8,atol=1e-8)
    assert saved_energy_gate(tables['rows'],float(sample.volume.sum()),40e9)['passed']
    for row in tables['rows']:
        accounted=row['kinetic_energy_J']+row['stored_elastic_energy_J']+row['ball_kinetic_energy_J']+row['contact_kinetic_loss_J']-row['grid_projection_work_J']-row['particle_external_work_J']
        assert abs(accounted-40)/40<=plan['deep_impact']['energy_relative_tolerance']
    for key,value in current_frame.items():assert base.array_descriptor(value)==tables['trajectory_frames'][contract['parent_frame']][key]
    initial_frame=a.output.parent/'frames_data'/f"frame_{contract['parent_frame']:05d}.npz"
    if not initial_frame.exists():base.save_npz(initial_frame,**current_frame,time=np.asarray(current_time))
    # Never allocate fictitious slots for unavailable historical coordinates.
    # Original physics sees the exact carry; only saved-frame persistence changes.
    class AvailableFrames:
        def __init__(self,name,shape,dtype):self.name=name;self.shape=shape;self.dtype=np.dtype(dtype);self._mmap=self;self.latest=None;self.value=None
        def __getitem__(self,index):
            if not isinstance(index,(int,np.integer)):raise ValueError('Unavailable-history bulk access is forbidden')
            if index==self.latest:return self.value
            path=a.output.parent/'frames_data'/f'frame_{index:05d}.npz'
            if not path.exists():raise FileNotFoundError('Historical coordinates unavailable: frame '+str(index))
            with np.load(path,allow_pickle=False) as data:return data[self.name]
        def __setitem__(self,index,value):
            if index<=contract['parent_frame']:raise ValueError('Cannot overwrite an admitted parent frame')
            self.latest=index;self.value=np.broadcast_to(np.asarray(value,dtype=self.dtype),self.shape[1:]).copy()
        def flush(self):pass
        def close(self):self.value=None
    original_text=(root/'examples/woven_large_plate.py').read_text()
    before='array = np.lib.format.open_memmap(path, mode="r+" if start_frame >= 0 else "w+", dtype=dtype, shape=shape)'
    after='array = _available_frame_storage(name, shape, dtype)'
    assert original_text.count(before)==1
    adapted=original_text.replace(before,after)
    exec(compile(adapted,str(root/'examples/woven_large_plate.py'),'exec'),base.__dict__)
    base._available_frame_storage=AvailableFrames
    def validate_available_prefix(arrays,reference,current_x,current_v,saved,frame,output_steps,dt):
        rows,diagnostics,times=saved['rows'],saved['diagnostics'],saved['times'];descriptors=saved['trajectory_frames']
        assert all(len(t)==frame+1 for t in (rows,diagnostics,times,descriptors))
        assert np.allclose(times,np.asarray(output_steps[:frame+1])*dt,rtol=5e-12,atol=0)
        assert all(float(row['time'])==instant and int(row['accepted_steps'])==step for row,instant,step in zip(rows,times,output_steps))
        assert all(float(row['time'])==instant for row,instant in zip(diagnostics,times))
        for path in sorted((a.output.parent/'frames_data').glob('frame_*.npz')):
            index=int(path.stem.split('_')[1])
            if index>frame:continue
            with np.load(path,allow_pickle=False) as data:
                assert all(base.array_descriptor(data[name])==descriptors[index][name] for name in arrays)
        assert np.array_equal(arrays['x'][frame],current_x)
        assert np.array_equal(arrays['speed'][frame],np.linalg.norm(current_v,axis=1).astype(np.float32))
        assert np.all(arrays['damage'][frame]==0) and np.all(arrays['temperature'][frame]==np.float32(293.15))
    base.validate_saved_prefix=validate_available_prefix
    # Recover only metadata from the exact atomic CP after a resource stop.
    # Keep any incomplete working metadata as evidence, never alter parent.
    for filename, key in [('observables.csv','rows'),('diagnostics.csv','diagnostics')]:
        path=folder/filename
        if path.exists():shutil.copy2(path,path.with_suffix('.parent.csv'))
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=tables[key][0],lineterminator='\n');writer.writeheader();writer.writerows(tables[key])
    bridge=dict(physics_commit='c5eccddd9d0857ecc3e0af6e3bfa356409817986',capsule_commit=capsule['commit'],parent=contract,
        numerical_files={name:sha(root/name) for name in base.SOURCE_FILES},copied_files=copied,rolling_cache_hardlinks=linked,
        persistence_only=True,original_parent_raw_read_only=True,original_source_and_physical_plan_unchanged=True,
        storage_adapter_diff=''.join(difflib.unified_diff(original_text.splitlines(True),adapted.splitlines(True))),
        original_full_historical_coordinate_check_applicable=False,historical_coordinate_frames_unavailable=list(range(34)),
        available_frames_checked_against_original_descriptors=True,
        scope='Original base.run_case dynamics; replace full trajectory memmap with actual available-frame NPZ storage, snapshot hook and exact atomic-CP metadata recovery. Full historical coordinate gate is explicitly not claimed; original tables/source/carry/layout/control gates remain. No zero-filled historical frame or missing physical frame invented.')
    (a.output.parent/'source-bridge.json').write_text(json.dumps(bridge,indent=2)+'\n')
    original_save = base.save_npz
    def save_with_immutable_frame(path, **values):
        if Path(path).name=='trajectory.npz':
            values.update(available_frame_indices=np.asarray(sorted(int(f.stem.split('_')[1]) for f in (a.output.parent/'frames_data').glob('frame_*.npz'))),historical_coordinates_complete=np.asarray(False))
        original_save(path, **values)
        if Path(path).name!='checkpoint.npz':return
        frame=int(values['frame']);target=a.output.parent/'atomic-frames'/f'checkpoint_{frame:05d}.npz'
        assert not target.exists();os.link(path,target)
        frame_values=dict(x=values['state_x'],speed=np.linalg.norm(values['state_v'],axis=1).astype(np.float32),
                          damage=np.zeros(len(sample.positions),np.float32),temperature=np.full(len(sample.positions),293.15,np.float32))
        descriptors=json.loads(str(values['tables']))['trajectory_frames'][frame]
        assert all(base.array_descriptor(v)==descriptors[k] for k,v in frame_values.items())
        original_save(a.output.parent/'frames_data'/f'frame_{frame:05d}.npz',**frame_values,time=values['state_time'])
        base.save_json(a.output.parent/'last-atomic-checkpoint.json',dict(frame=frame,steps=int(values['aggregate'][0]),checkpoint_sha256=sha(target),coupling_valid=bool(values['aggregate'][1])))
    base.save_npz=save_with_immutable_frame
    summary=base.run_case(sample,plan,a.output,resume=True,stop_after_frame=contract['target_frame'])
    summary.update(history_storage='Actual f34+ frame NPZ and complete atomic CP; earlier coordinate frames unavailable',
        historical_coordinates_complete=False,available_saved_frame_indices=sorted(int(f.stem.split('_')[1]) for f in (a.output.parent/'frames_data').glob('frame_*.npz')))
    base.save_json(folder/'summary.json',summary)
    review=deep.review(folder);base.save_json(folder/'deep-review.json',review)
    assert all(review[key] for key in ('stable','clamp_passed','material_energy_passed','energy_passed'))
    assert summary['segment_end_frame']==contract['target_frame'] and (summary['segment_complete'] or summary['completed'])
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as data, np.load(parent/'checkpoint.npz',allow_pickle=False) as parent_data:
        base.validate_checkpoint_control(data,plan);tables=json.loads(str(data['tables']))
        assert tables['trajectory_frames'][:contract['parent_frame']+1]==json.loads(str(parent_data['tables']))['trajectory_frames']
        assert saved_energy_gate(tables['rows'],float(sample.volume.sum()),40e9)['passed']
    result=dict(status='passed',frame=contract['target_frame'],steps=summary['steps'],time_s=summary['simulation_time'],
        completed=summary['completed'],checkpoint_sha256=sha(folder/'checkpoint.npz'),parent_checkpoint_sha256=contract['parent_checkpoint_sha256'],
        original_source_and_physical_plan_unchanged=True,parent_trajectory_descriptors_preserved=True,
        available_saved_frame_hashes_verified=True,historical_coordinates_independently_verified=False,
        review=review,user_depth_goal_m=.005,user_depth_goal_passed=review['peak_center_indentation_m']>=.005,
        publication_admitted=False,scope='Actual original40J free impact continued only; five-millimetre user goal assessed separately from original two-millimetre exploratory criterion')
    base.save_json(a.output.parent/'segment-review.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
