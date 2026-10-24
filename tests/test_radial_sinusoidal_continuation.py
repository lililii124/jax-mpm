import importlib.util
from pathlib import Path
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('radial_dual',ROOT/'tools/radial_sinusoidal_continuation.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_frozen_parent_and_independent_proof_identity():
    assert c.PRODUCER_COMMIT=='de2bfea446e66e6245e33329a663996c5d5b9e84'
    assert c.PARENT_RECEIPT=='d0af82b261b0433c024b80472cb5da67ee5d0113f11d2876f0510ee8f631c591'
    assert c.PROOF_SHA=='74840b9e9bf12f479787bdfb359bd34098354402fdcf3c13deccee698043836f'
    proof=ROOT/'.local/radial-sinusoidal-feature/40092005172-1/12559346521/independent-admission.json'
    if proof.exists():assert c.digest(proof)==c.PROOF_SHA


@pytest.fixture
def parent(tmp_path,monkeypatch):
    base=tmp_path/'parent';raw=base/'result';raw.mkdir(parents=True)
    (raw/'checkpoint.json').write_bytes(b'{"frame":10}')
    source={'physics.py':'sourcehash'};runtime={'jax':'pinned'}
    receipt={'status':'accepted-prefix','commit':c.PRODUCER_COMMIT,'run_id':c.PARENT_RUN,
        'cache_key':c.PARENT_CACHE,'source_files':source,'runtime':runtime,'raw_files':c.inventory(raw)}
    c.save(base/'receipt.json',receipt)
    monkeypatch.setattr(c,'PARENT',base)
    monkeypatch.setattr(c,'PARENT_RECEIPT',c.digest(base/'receipt.json'))
    proof={'source_files':source,'runtime':runtime,'all_raw_files':c.inventory(raw)}
    return proof


def test_exact_parent_admission_and_raw_tamper(parent):
    c.verify_parent(parent)
    (c.PARENT/'result'/'checkpoint.json').write_text('tampered')
    with pytest.raises(ValueError,match='raw differs'):c.verify_parent(parent)


def test_f20_to_f50_calls_original_worker_and_preserves_parent(parent,tmp_path,monkeypatch):
    producer=tmp_path/'producer';work=producer/'.local/radial-sinusoidal-full'
    monkeypatch.setattr(c,'PRODUCER',producer)
    monkeypatch.setattr(c,'WORK',work);monkeypatch.setattr(c,'RAW',work/'result')
    files={'new-controller.py':'newhash'}
    monkeypatch.setattr(c,'bind_controller',lambda media=False:('b'*40,files))
    monkeypatch.setattr(c,'own_files',lambda:files)
    monkeypatch.setattr(c,'proof_and_producer',lambda:parent)
    proof=tmp_path/'proof.json';proof.write_text('proof')
    monkeypatch.setattr(c,'PROOF',proof)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda folder:c.shutil._ntuple_diskusage(100*c.LIMIT,0,100*c.LIMIT))
    original=c.inventory(c.PARENT)
    calls=[]
    def guard(command,*args):
        calls.append(command)
        if 'worker' in command:(c.RAW/'checkpoint.json').write_text('{"frame":'+command[-1]+'}')
        return {'returncode':0}
    monkeypatch.setattr(c,'producer_helpers',lambda:types.SimpleNamespace(guard=guard))
    c.continue_run()
    assert [cmd[-1] for cmd in calls]==['audit-prefix','20','30','40','50','audit-full']
    assert all(str(producer/'tools/radial_sinusoidal_cloud.py') in cmd for cmd in calls[:-1])
    receipt=c.read(work/'continuation-receipt.json')
    assert receipt['status']=='complete' and receipt['producer_commit']==c.PRODUCER_COMMIT
    assert receipt['controller_commit']=='b'*40 and receipt['parent_raw_unchanged'] is True
    assert c.inventory(c.PARENT)==original
    assert c.digest(work/'receipt.json')==c.PARENT_RECEIPT


def test_continuation_failure_is_preserved_separately(parent,tmp_path,monkeypatch):
    work=tmp_path/'work';monkeypatch.setattr(c,'WORK',work);monkeypatch.setattr(c,'RAW',work/'result')
    monkeypatch.setattr(c,'bind_controller',lambda media=False:('b'*40,{}))
    def reject():raise ValueError('proof mismatch')
    monkeypatch.setattr(c,'proof_and_producer',reject)
    with pytest.raises(ValueError,match='proof mismatch'):c.continue_run()
    assert c.read(work/'continuation-receipt.json')['status']=='failedraw'
    assert c.digest(c.PARENT/'receipt.json')==c.PARENT_RECEIPT


def test_wrong_new_registered_bytes_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    for template,first in ((c.CONTINUATION,'name: feature radial sinusoidal continuation'),
                           (c.MEDIA_WORKFLOW,'name: feature radial sinusoidal continuation media')):
        path=tmp_path/template;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(first+'\non: workflow_dispatch\n')
    controller=tmp_path/c.CONTROLLER;controller.parent.mkdir(parents=True);controller.write_text('controller')
    (tmp_path/c.REGISTERED).write_text('wrong registered bytes')
    commit='b'*40
    monkeypatch.setattr(c,'head',lambda root:commit)
    monkeypatch.setattr(c,'git',lambda root,*args:(tmp_path/args[1].removeprefix('HEAD:')).read_bytes())
    for key,value in {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':c.REPOSITORY,
        'GITHUB_REF':c.REF,'GITHUB_SHA':commit,'GITHUB_WORKFLOW_SHA':commit,
        'GITHUB_WORKFLOW_REF':f'{c.REPOSITORY}/{c.REGISTERED}@{c.REF}'}.items():monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='registered new workflow bytes'):c.bind_controller()
    (tmp_path/c.REGISTERED).write_bytes(c.private_workflow_bytes())
    assert c.bind_controller()[0]==commit


def test_media_never_executes_numerical_worker_or_mutates_producer_gates():
    import ast
    source=(ROOT/c.CONTROLLER).read_text();tree=ast.parse(source)
    media=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='media')
    text=ast.get_source_segment(source,media)
    assert "'worker'" not in text and 'continue_run(' not in text
    assert 'old.media(' not in source and 'old.numerical(' not in source
    assert 'os.environ[' not in source.split('def media():')[0] or 'GITHUB_WORKFLOW_REF' in source
    assert 'producer_commit' in text and 'numerical_controller_commit' in text and 'media_controller_commit' in text


def test_immutable_producer_bytes_and_exact_runtime_required(tmp_path,monkeypatch):
    producer=tmp_path/'producer';producer.mkdir()
    source=producer/'physics.py';source.write_bytes(b'original immutable physics')
    proof={'admitted_prefix':True,'raw_unchanged':True,'solver_executed':False,
        'producer_commit':c.PRODUCER_COMMIT,'producer_run':c.PARENT_RUN,
        'producer_receipt_sha256':c.PARENT_RECEIPT,'cache_key':c.PARENT_CACHE,
        'prefix_audit':{'saved_states':11},'source_files':{'physics.py':c.digest(source)},
        'runtime':{'jax':'pinned'}}
    proofpath=tmp_path/'proof.json';c.save(proofpath,proof)
    monkeypatch.setattr(c,'PROOF',proofpath);monkeypatch.setattr(c,'PROOF_SHA',c.digest(proofpath))
    monkeypatch.setattr(c,'PRODUCER',producer)
    monkeypatch.setattr(c,'head',lambda root:c.PRODUCER_COMMIT)
    monkeypatch.setattr(c,'git',lambda root,*args:source.read_bytes())
    monkeypatch.setattr(c.importlib.metadata,'version',lambda name:'pinned')
    assert c.proof_and_producer()==proof
    source.write_bytes(b'changed physical law')
    with pytest.raises(ValueError,match='Immutable producer source differs'):c.proof_and_producer()
    source.write_bytes(b'original immutable physics')
    monkeypatch.setattr(c.importlib.metadata,'version',lambda name:'different')
    with pytest.raises(ValueError,match='runtime differs'):c.proof_and_producer()


def test_retained_original_prefix_frames_are_hash_checked(tmp_path):
    raw=tmp_path/'raw';(raw/'frames_data').mkdir(parents=True)
    frame=raw/'frames_data/frame_00000.npz';frame.write_bytes(b'original prefix')
    proof={'all_raw_files':c.inventory(raw)}
    c.verify_prefix_frames(raw,proof)
    frame.write_bytes(b'changed prefix')
    with pytest.raises(ValueError,match='prefix frame changed'):c.verify_prefix_frames(raw,proof)
