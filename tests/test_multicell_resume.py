import json

from tools.optimize_multicell_binary_loop import preserve_incomplete_replay, preserve_partial_directory


def test_incomplete_replay_is_preserved_before_restart(tmp_path):
    replay = tmp_path/'replay-000'
    replay.mkdir()
    metadata = dict(geometry_sha256='geometry', support_mode='split-disconnected',
                    settings={'steps': 4, 'dt': .1}, source_sha256={'solver': 'digest'},
                    frame_interval=2, steps=4, diagnostic_only=False, connected_faces=True,
                    detached_voxels=0)
    row = dict(step=0, time_s=0., valid=True, min_J=1., back_displacement_max_m=0.,
               kinetic_J=0., elastic_J=0., pressure_work_J=0., boundary_work_J=0.,
               energy_residual_J=0., momentum_residual=0.)
    (replay/'inputs.json').write_text(json.dumps(metadata))
    (replay/'history.json').write_text(json.dumps([row]))
    preserve_incomplete_replay(replay, 'geometry', metadata)
    assert not replay.exists()
    assert json.loads((tmp_path/'replay-000.interrupted-001/history.json').read_text()) == [row]


def test_partial_proposals_are_preserved_before_regeneration(tmp_path):
    proposals = tmp_path/'proposals'
    proposals.mkdir()
    (proposals/'candidate.npz').write_bytes(b'partial')
    preserve_partial_directory(proposals, ('manifest.json',))
    assert not proposals.exists()
    assert (tmp_path/'proposals.interrupted-001/candidate.npz').read_bytes() == b'partial'
