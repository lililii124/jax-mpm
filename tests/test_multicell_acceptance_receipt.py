from pathlib import Path

from tools.optimize_multicell_binary_loop import acceptance_receipt


def test_receipt_changes_when_replay_or_validator_changes(tmp_path):
    root = tmp_path/'root'
    for name in ('tools/optimize_multicell_binary_loop.py', 'tools/select_multicell_binary_step.py',
                 'tools/compare_multicell_binary_blast.py', 'tools/reconstruct_multicell_blast.py',
                 'tools/multicell_feature_filter.py'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    def file(name):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name); return path
    def replay(name):
        folder = tmp_path/name; folder.mkdir()
        for filename in ('inputs.json', 'history.json'): (folder/filename).write_text(name+filename)
        return folder
    inputs, binary, design, geometry = (file(name) for name in ('inputs', 'binary', 'design', 'geometry'))
    baseline, candidate = replay('baseline'), replay('candidate')
    proposals = tmp_path/'proposals'; proposals.mkdir()
    for filename in ('manifest.json', 'candidate.npz', 'geometry.npz'): (proposals/filename).write_text(filename)
    selection = tmp_path/'selection'; selection.mkdir()
    for filename in ('selection.json', 'selected-design.npz'): (selection/filename).write_text(filename)
    proposed = [{'design': 'candidate.npz', 'geometry': 'geometry.npz'}]
    def receipt():
        return acceptance_receipt(root, inputs, binary, design, geometry, baseline, geometry, baseline,
                                  proposals, proposed, [candidate], selection)
    initial = receipt()
    (candidate/'history.json').write_text('changed')
    assert receipt() != initial
    (candidate/'history.json').write_text('candidatehistory.json')
    (root/'tools/select_multicell_binary_step.py').write_text('new validator')
    assert receipt() != initial
