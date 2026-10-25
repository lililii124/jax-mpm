import json
import numpy as np
import pytest
from tools.compare_metal_v6_dt_convergence import CHANNELS,compare


def make_cells(root):
    for divisor,value in ((1,4.),(2,3.),(4,2.5)):
        path=root/f'dt{divisor}';path.mkdir()
        signed={k:value for k in CHANNELS};absolute=signed.copy()
        proof=dict(admitted=True,dt_divisor=divisor,cfl=.1/divisor,steps_probe=divisor,
                   signed_J=signed,absolute_J=absolute,fields={str(i):dict(bitwise_equal=True) for i in range(35)},
                   energy_identity_tolerance_J=1e-8,parent_checkpoint_sha256='same',actual_start_ns=125,actual_stop_ns=135)
        (path/'equivalence.json').write_text(json.dumps(proof))
        np.savez(path/'adaptive-trace.npz',dt_s=np.full(divisor,10e-9/divisor),channels_J=np.full((divisor,9),value/divisor),channel_names=np.array(CHANNELS))
        np.savez(path/'probe-final.npz',**{k:np.zeros((2,3)) for k in ('x','v','C','F','damage')},mass=np.ones(2))


def test_reports_differences_without_admitting_repair(tmp_path):
    make_cells(tmp_path);result=compare(tmp_path)
    assert result['channels']['p2g']['difference_ratio']==.5
    assert result['channels']['p2g']['absolute_difference_contracts']
    assert not result['repair_admitted'] and not result['validated_fragments']


def test_rejects_incomplete_nine_channel_trace(tmp_path):
    make_cells(tmp_path)
    path=tmp_path/'dt2/adaptive-trace.npz'
    np.savez(path,dt_s=np.array([5e-9,5e-9]),channels_J=np.zeros((2,8)),channel_names=np.array(CHANNELS))
    with pytest.raises(ValueError,match='channel trace'):compare(tmp_path)


def test_rejects_non_bitwise_original_instrumented_pair(tmp_path):
    make_cells(tmp_path);path=tmp_path/'dt4/equivalence.json';proof=json.loads(path.read_text())
    proof['fields']['0']['bitwise_equal']=False;path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='equivalence'):compare(tmp_path)
