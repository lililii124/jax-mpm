"""Resume exact original woven-impact state; immutable CPs and rolling derived cache."""
import argparse
import csv
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
        if not source.is_file() or source.suffix=='.tmp' or source.name=='final_state_0.npz':continue
        target = folder/source.name
        if source.name.startswith('trajectory_') and source.suffix=='.npy' and contract['parent_frame']>34:
            # Only the new rolling cache is shared. Immutable original seed is
            # copied once. Admitted frame bytes are guarded by original hashes.
            os.link(source, target);linked.append(source.name)
        else:
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
        scope='Original base.run_case; only output snapshot hook and exact atomic-CP metadata recovery; saved prefix descriptors checked by original resume gate. No missing physical frame invented.')
    (a.output.parent/'source-bridge.json').write_text(json.dumps(bridge,indent=2)+'\n')
    original_save = base.save_npz
    def save_with_immutable_frame(path, **values):
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
    review=deep.review(folder);base.save_json(folder/'deep-review.json',review)
    assert all(review[key] for key in ('stable','clamp_passed','material_energy_passed','energy_passed'))
    assert summary['segment_end_frame']==contract['target_frame'] and (summary['segment_complete'] or summary['completed'])
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as data, np.load(parent/'checkpoint.npz',allow_pickle=False) as parent_data:
        base.validate_checkpoint_control(data,plan);tables=json.loads(str(data['tables']))
        assert tables['trajectory_frames'][:contract['parent_frame']+1]==json.loads(str(parent_data['tables']))['trajectory_frames']
    result=dict(status='passed',frame=contract['target_frame'],steps=summary['steps'],time_s=summary['simulation_time'],
        completed=summary['completed'],checkpoint_sha256=sha(folder/'checkpoint.npz'),parent_checkpoint_sha256=contract['parent_checkpoint_sha256'],
        original_source_and_physical_plan_unchanged=True,original_saved_prefix_frame_hashes_unchanged=True,
        review=review,user_depth_goal_m=.005,user_depth_goal_passed=review['peak_center_indentation_m']>=.005,
        publication_admitted=False,scope='Actual original40J free impact continued only; five-millimetre user goal assessed separately from original two-millimetre exploratory criterion')
    base.save_json(a.output.parent/'segment-review.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
