import ast
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def functions():
    tree = ast.parse((Path(__file__).resolve().parents[1]/'examples/common.py').read_text(encoding='utf-8'))
    selected = [node for node in tree.body if getattr(node, 'name', '') in ('run', '_valid_checkpoint_states', '_validate_resume_prefix')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<actual-common-continuation>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('stop', [0, -1, 5, True, 2.5, '2'])
def test_bad_runtime_endpoint_is_rejected_before_allocating_or_writing(functions, tmp_path, stop):
    folder = tmp_path/'not-created'
    with pytest.raises(ValueError, match='stop_after_frame'):
        functions['run']((), (), (), frames=4, substeps=1, output=folder, label='test', stop_after_frame=stop)
    assert not folder.exists()


@pytest.fixture
def state():
    History = namedtuple('History', 'stress damage temperature failed valid')
    State = namedtuple('State', 'x v F C mass volume history time valid')
    return State(np.zeros((2, 3)), np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0),
                 np.zeros((2, 3, 3)), np.ones(2), np.ones(2),
                 History(np.zeros((2, 3, 3)), np.zeros(2), np.full(2, 293.), np.zeros(2, dtype=bool),
                         np.ones(2, dtype=bool)), np.array(0.), np.array(True))


def test_valid_saved_state_is_admitted(functions, state):
    assert functions['_valid_checkpoint_states']((state,))


@pytest.mark.parametrize('defect', ['state_flag', 'history_flag', 'negative_J', 'negative_mass', 'nonfinite_v', 'bad_C_shape'])
def test_invalid_checkpoint_cannot_be_marked_ready(functions, state, defect):
    if defect == 'state_flag':
        state = state._replace(valid=np.array(False))
    elif defect == 'history_flag':
        state.history.valid[0] = False
    elif defect == 'negative_J':
        state.F[0, 0, 0] = -1
    elif defect == 'negative_mass':
        state.mass[0] = -1
    elif defect == 'nonfinite_v':
        state.v[0, 0] = np.nan
    else:
        state = state._replace(C=np.zeros((1, 3, 3)))
    assert not functions['_valid_checkpoint_states']((state,))


@pytest.fixture
def prefix(state, tmp_path):
    state = state._replace(time=np.array(.1))
    saved = {'frame': 1, 'rows': [{'time': 0., 'body': 0}, {'time': .1, 'body': 0}],
             'observations': [{'time': 0.}, {'time': .1}]}
    (tmp_path/'frames_data').mkdir()
    for index in range(2):
        np.savez_compressed(tmp_path/'frames_data'/f'frame_{index:05d}.npz',
                            x=state.x, speed=np.zeros(2), damage=state.history.damage,
                            temperature=state.history.temperature, failed=state.history.failed,
                            time=np.array(index*.1), contact_impulse=np.zeros(3))
    return tmp_path, saved, state


def test_complete_consistent_prefix_is_admitted_before_reallocation(functions, prefix):
    folder, saved, state = prefix
    functions['_validate_resume_prefix'](folder, saved, (state,), 2, 1, .1, True)


@pytest.mark.parametrize('defect', ['wrong_frame', 'short_table', 'observation_time', 'state_time', 'missing_frame', 'changed_position'])
def test_inconsistent_prefix_is_rejected(functions, prefix, defect):
    folder, saved, state = prefix
    if defect == 'wrong_frame':
        saved['frame'] = 3
    elif defect == 'short_table':
        saved['rows'].pop()
    elif defect == 'observation_time':
        saved['observations'][-1]['time'] = .2
    elif defect == 'state_time':
        state = state._replace(time=np.array(.2))
    elif defect == 'missing_frame':
        (folder/'frames_data/frame_00000.npz').unlink()
    else:
        state.x[0, 0] = .2
    with pytest.raises((ValueError, FileNotFoundError)):
        functions['_validate_resume_prefix'](folder, saved, (state,), 2, 1, .1, True)
