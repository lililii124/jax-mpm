import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from tools.assess_sandwich_rational import integrate,check_frozen,assess,prepare_history
from tools.build_sandwich_rational import check_recipe,digest,source_receipt,material_fields_digest


def test_newmark_first_step_and_load_scaling():
    M=np.array([[2.]])
    K=np.array([[3e7]])
    L=np.array([[1.,2.,-3.]])
    result=integrate(M,K,L,np.array([[1.]]),dt=2e-5)
    h=result['time'][1]
    pressure=.5e6*np.sin(np.pi*h/.0002)**2
    expected=.25*h*h*pressure*L/(M+.25*h*h*K)
    np.testing.assert_allclose(result['displacement'][1],expected,rtol=1e-14)
    np.testing.assert_allclose(result['displacement'][:,:,1],2*result['displacement'][:,:,0],rtol=1e-12)
    assert result['energy_relative_error']<1e-8


def test_frozen_replay_detects_field_mismatch(tmp_path):
    M=np.array([[2.]])
    K=np.array([[3e7]])
    L=np.array([[1.,2.,-3.]])
    back=np.array([[1.]])
    basis=tmp_path/'basis.npz';response=tmp_path/'response.npz'
    np.savez_compressed(basis,basis=np.ones((1,3,1)),mass=M,stiffness=K,loads=L,back=back)
    result=integrate(M,K,L,back,dt=2e-5)
    arrays={key:value for key,value in result.items() if isinstance(value,np.ndarray)}
    np.savez_compressed(response,**arrays)
    assert check_frozen(basis,response)['passed']
    arrays['curves']=arrays['curves']*1.01
    np.savez_compressed(response,**arrays)
    with pytest.raises(ValueError,match='replay mismatch'):check_frozen(basis,response)


def test_gate_and_recipe_changes_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='Frozen recipe mismatch'):check_recipe({})
    path=tmp_path/'assessment.json'
    path.write_text(json.dumps(dict(response_gate=.06)))
    with pytest.raises(ValueError,match='Frozen gate mismatch'):assess(path)


def test_nonfinite_frozen_trajectory_rejected(tmp_path):
    basis=tmp_path/'basis.npz';response=tmp_path/'response.npz'
    np.savez_compressed(basis,basis=np.ones((1,3,1)),mass=np.eye(1),stiffness=np.eye(1),loads=np.ones((1,3)),back=np.ones((1,1)))
    np.savez_compressed(response,time=np.linspace(0,.002,101),displacement=np.full((101,1,3),np.nan),curves=np.full((101,3),np.nan),energy=np.full((100,3,3),np.nan))
    with pytest.raises(ValueError,match='Nonfinite'):check_frozen(basis,response)


def bound_assessment(tmp_path):
    material=tmp_path/'material.npz';snapshots=tmp_path/'snapshots.npz';history=tmp_path/'history.csv'
    np.savez_compressed(material,phi=np.ones(1),young=np.ones(1),density=np.ones(1),volume=np.ones(1),skin=np.ones(1,dtype=bool))
    np.savez_compressed(snapshots,displacement=np.zeros((1,1,3)),nodes=np.zeros((1,3)),steps=np.array([31000]))
    history.write_text('step,time_s,peak_back_displacement_m\n31000,.001,1\n')
    fields_sha=material_fields_digest(material);controls='a'*64;signature='b'*64
    freeze=dict(schema_version=2,source_sha256=source_receipt({}),operator_vertex='selected',inputs=dict(selected_material=dict(sha256=digest(material))),
        material_fields_sha256=dict(selected=fields_sha),construction_configuration=dict(selected_controls_sha256=controls),full_mpm_source_signature=signature)
    (tmp_path/'freeze.json').write_text(json.dumps(freeze))
    provenance=dict(output_sha256=digest(snapshots),sources=[dict(controls_sha256=controls,material_sha256=fields_sha,case='eccentric')],specification=dict(sources=[dict(source_signature=signature)]))
    snapshots.with_suffix('.json').write_text(json.dumps(provenance))
    receipt=dict(reference_history_sha256=digest(history),controls_sha256=controls,case='eccentric',source_signature=signature)
    (tmp_path/'history.json').write_text(json.dumps(receipt))
    config=dict(role='development',freeze_receipt='freeze.json',material='material.npz',snapshots='snapshots.npz',history='history.csv',history_receipt='history.json',case_index=2,
        response_gate=.05,omitted_energy_gate=.05,time_gate=.01,linear_energy_gate=1e-8)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    return path,config


@pytest.mark.parametrize('change,match',[('operators','Operator overrides'),('material','material artifact'),('snapshots','Snapshot artifact'),('history','History artifact')])
def test_frozen_input_tampering_rejected_before_computation(tmp_path,change,match):
    path,config=bound_assessment(tmp_path)
    if change=='operators':
        config['operators']='foreign.npz';path.write_text(json.dumps(config))
    else:
        target=tmp_path/config[change]
        with target.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match=match):assess(path)


def test_physical_source_pins_cannot_be_omitted():
    config=json.loads((Path(__file__).resolve().parents[1]/'examples/sandwich_rational.json').read_text())
    del config['expected_source_sha256']
    with pytest.raises(ValueError,match='Complete physical source pins'):check_recipe(config)


def test_initial_history_stop_is_explicit_and_fresh_stop_cannot_change(tmp_path):
    directory=tmp_path/'trajectory';directory.mkdir()
    q=np.zeros(2,dtype=np.float64);qhash=hashlib.sha256(q.tobytes()).hexdigest()
    status=dict(valid=True,binding=dict(controls=q.tolist(),case='eccentric',source_signature='source'))
    (directory/'status.json').write_text(json.dumps(status))
    (directory/'history.csv').write_text('step,time_s,peak_back_displacement_m\n0,0,0\n1,.1,1\n2,.2,2\n3,.3,3\n')
    freeze=dict(schema_version=2,source_sha256=source_receipt({}),operator_vertex='initial',construction_configuration=dict(initial_controls_sha256=qhash),full_mpm_source_signature='source',fresh_validation=None)
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='history-end-step'):prepare_history(directory,path,tmp_path/'missing.csv')
    assert prepare_history(directory,path,tmp_path/'initial.csv',3)['reference_history_maximum_step']==3
    freeze['fresh_validation']=dict(history_interval_steps=[1,3]);path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='Cannot override'):prepare_history(directory,path,tmp_path/'changed.csv',4)
