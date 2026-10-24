import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("radial_cloud",ROOT/"tools/radial_sinusoidal_cloud.py")
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_whole_raw_multipart_roundtrip_and_tamper(tmp_path,monkeypatch):
    base=tmp_path/'raw'; base.mkdir()
    (base/'receipt.json').write_text('{"status":"failedraw"}')
    (base/'result').mkdir()
    (base/'result'/'failed-state.bin').write_bytes(bytes(range(256))*100)
    parts=tmp_path/'parts'
    monkeypatch.setattr(c,'BASE',base);monkeypatch.setattr(c,'PARTS',parts)
    c.package()
    original=c.inventory(base)
    manifest=c.retrieve(parts)
    assert manifest['files']==original
    assert manifest['part_limit_bytes']==128*1024**2
    destination=tmp_path/'restored'
    monkeypatch.setattr(c,'BASE',destination)
    c.retrieve(parts,extract=True)
    assert c.inventory(destination)==original
    part=next(parts.glob('part-*.tarpart'))
    with part.open('r+b') as stream: stream.write(b'BAD')
    with pytest.raises(RuntimeError,match='Corrupt/missing'):
        c.retrieve(parts)


def test_media_retrieval_exact_files(tmp_path,monkeypatch):
    media=tmp_path/'media';media.mkdir()
    c.save(media/'provenance.json',{'raw_run_id':'123'})
    (media/'simulation.gif').write_bytes(b'test-gif')
    c.save(media/'delivery.json',c.inventory(media))
    monkeypatch.setattr(c,'ROOT',tmp_path)
    c.verify_media(media)
    assert c.read(tmp_path/'.local/radial-sinusoidal-media-retrieval.json')['status']=='all-delivery-files-verified'
    (media/'unexpected.txt').write_text('extra')
    with pytest.raises(ValueError,match='Retrieved media'):
        c.verify_media(media)


def test_source_binding_includes_actual_registered_workflows():
    source=c.sources()
    assert all(name in source for name in c.WORKFLOWS)
    c.check_sources(source)
    stale=dict(source);stale['mpm_jax/adhesion.py']='0'*64
    with pytest.raises(ValueError,match='source differs'):
        c.check_sources(stale)


def test_completed_plan_requires_51_saved_states(tmp_path):
    c.save(tmp_path/'summary.json',{'completed':True,'particles':[367744],
      'steps':3700,'simulation_time':1.3,'definition':{'radial_sinusoid':{
      'amplitude_m':.008,'wavelength_m':.08}}})
    (tmp_path/'frames_data').mkdir()
    for i in range(50): (tmp_path/'frames_data'/f'frame_{i:05d}.npz').touch()
    with pytest.raises(ValueError,match='exactly 51'):
        c.validate_complete(tmp_path)


def test_actual_workflow_bounds_and_no_media_solver_call():
    import ast
    tree=ast.parse((ROOT/'tools/radial_sinusoidal_cloud.py').read_text())
    media=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='media')
    calls={node.func.id for node in ast.walk(media) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    assert not calls.intersection({'worker','numerical','run','step','advance'})
    full=(ROOT/c.WORKFLOWS[0]).read_text()
    assert full.count('id: part')==64
    assert 'pattern: radial-raw-${{ github.run_id }}-${{ github.run_attempt }}-*' in full
    assert 'restore-keys' not in full
    final=(ROOT/c.WORKFLOWS[1]).read_text()
    assert 'artifact-ids: ${{ steps.media.outputs.artifact-id }}' in final


def test_gif_preserves_identical_saved_hold_states(tmp_path):
    from PIL import Image
    (tmp_path/'frames').mkdir()
    for i in range(3):
        Image.new('RGB',(32,32),(80,90,100)).save(tmp_path/'frames'/f'frame_{i:05d}.png')
    c.encode_saved_frames(tmp_path,3)
    with Image.open(tmp_path/'simulation.gif') as gif:
        assert gif.n_frames==3
        for i in range(3):
            gif.seek(i)
            assert gif.info['duration']==60


def test_wall_guard_kills_worker_and_preserves_receipt(tmp_path,monkeypatch):
    import shutil
    monkeypatch.setattr(c.shutil,'disk_usage',lambda folder:shutil._ntuple_diskusage(100*c.GIB,0,100*c.GIB))
    with pytest.raises(RuntimeError,match='wall-time-limit'):
        c.guard([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'worker.log',0,tmp_path)
    receipt=c.read(tmp_path/'worker.guard.json')
    assert receipt['stop_reason']=='wall-time-limit'
    assert receipt['returncode'] is not None


@pytest.fixture
def fake_numerical(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'BASE',tmp_path/'base')
    monkeypatch.setattr(c,'RAW',tmp_path/'base'/'result')
    monkeypatch.setattr(c,'registered',lambda name:None)
    monkeypatch.setattr(c,'sources',lambda:{'source.py':'abc'})
    monkeypatch.setattr(c,'check_sources',lambda expected:None)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda folder:c.shutil._ntuple_diskusage(100*c.GIB,0,100*c.GIB))
    pins={line.split('==')[0]:line.strip().split('==')[1] for line in c.RUNTIME.read_text().splitlines() if line.strip()}
    monkeypatch.setattr(c.importlib.metadata,'version',lambda name:pins[name])
    monkeypatch.setattr(c.importlib.metadata,'distributions',lambda:[])
    commands=[]
    def guarded(command,*args):
        commands.append(command)
        c.RAW.mkdir(parents=True,exist_ok=True)
        (c.RAW/'checkpoint.json').write_text('{"frame":10}')
        if command[-1]=='audit-prefix':
            c.save(c.BASE/'prefix-audit.json',{'status':'accepted-prefix','frame':10,'saved_states':11})
        return {'returncode':0}
    monkeypatch.setattr(c,'guard',guarded)
    return commands


def test_true_f10_is_accepted_prefix_and_package_keeps_status(fake_numerical,tmp_path,monkeypatch):
    c.numerical(stop_after_frame=10)
    assert [cmd[-1] for cmd in fake_numerical]==['10','audit-prefix']
    receipt=c.read(c.BASE/'receipt.json')
    assert receipt['status']=='accepted-prefix' and receipt['requested_stop_frame']==10
    monkeypatch.setattr(c,'PARTS',tmp_path/'parts')
    c.package()
    assert c.read(c.BASE/'receipt.json')['status']=='accepted-prefix'


def test_failed_worker_is_failedraw_not_accepted_prefix(fake_numerical,monkeypatch):
    def fail(*args): raise RuntimeError('intentional numerical failure')
    monkeypatch.setattr(c,'guard',fail)
    with pytest.raises(RuntimeError,match='intentional'):
        c.numerical(stop_after_frame=10)
    assert c.read(c.BASE/'receipt.json')['status']=='failedraw'


def configure_registered_env(monkeypatch):
    commit='a'*40
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    for name in ('SOURCE_SHA','GITHUB_SHA','GITHUB_WORKFLOW_SHA'):monkeypatch.setenv(name,commit)
    monkeypatch.setenv('GITHUB_REF',c.FEATURE_REF)
    monkeypatch.setenv('GITHUB_REPOSITORY',c.REPOSITORY)
    monkeypatch.setattr(c,'git_output',lambda *args:(commit+'\n').encode())


def test_wrong_actual_registered_workflow_ref_rejected(monkeypatch):
    configure_registered_env(monkeypatch)
    monkeypatch.setenv('GITHUB_WORKFLOW_REF',f'{c.REPOSITORY}/.github/workflows/wrong.yml@{c.FEATURE_REF}')
    with pytest.raises(ValueError,match='registered workflow ref'):
        c.registered(c.WORKFLOWS[0])


def test_actual_registered_override_bytes_must_match_template(tmp_path,monkeypatch):
    configure_registered_env(monkeypatch)
    monkeypatch.setenv('GITHUB_WORKFLOW_REF',f'{c.REPOSITORY}/{c.REGISTERED}@{c.FEATURE_REF}')
    monkeypatch.setattr(c,'ROOT',tmp_path)
    template=tmp_path/c.WORKFLOWS[0];template.parent.mkdir(parents=True)
    template.write_text('name: feature radial sinusoidal full numerical run\non: workflow_dispatch\n')
    registered=tmp_path/c.REGISTERED
    registered.write_text('name: wrong committed registered bytes\n')
    monkeypatch.setattr(c,'sources',lambda:{c.WORKFLOWS[0]:'x',c.REGISTERED:'y'})
    monkeypatch.setattr(c,'git_output',lambda *args:('a'*40+'\n').encode() if args[0]=='rev-parse' else (tmp_path/args[1].removeprefix('HEAD:')).read_bytes())
    with pytest.raises(ValueError,match='registered workflow bytes'):
        c.registered(c.WORKFLOWS[0])
    registered.write_bytes(c.private_workflow_bytes())
    c.registered(c.WORKFLOWS[0])


def test_exact_accepted_prefix_continues_only_f20_to_f50(fake_numerical,tmp_path,monkeypatch):
    import shutil
    c.numerical(stop_after_frame=10)
    parent=tmp_path/'parent'
    shutil.copytree(c.BASE,parent)
    parent_hash=c.digest(parent/'receipt.json')
    monkeypatch.setattr(c,'BASE',tmp_path/'continuation')
    monkeypatch.setattr(c,'RAW',tmp_path/'continuation'/'result')
    monkeypatch.setattr(c,'validate_complete',lambda folder:{})
    fake_numerical.clear()
    c.numerical(stop_after_frame=50,continue_from=parent,parent_receipt_sha256=parent_hash)
    assert [cmd[-1] for cmd in fake_numerical]==['20','30','40','50']
    receipt=c.read(c.BASE/'receipt.json')
    assert receipt['status']=='complete'
    assert receipt['continuation_parent_sha256']==parent_hash
    assert c.digest(c.BASE/'parent-receipt.json')==parent_hash


def test_failedraw_cannot_be_continued(fake_numerical,tmp_path):
    parent=tmp_path/'failed-parent';parent.mkdir()
    c.save(parent/'receipt.json',{'status':'failedraw'})
    with pytest.raises(ValueError,match='Only accepted-prefix'):
        c.numerical(stop_after_frame=50,continue_from=parent,
                    parent_receipt_sha256=c.digest(parent/'receipt.json'))
