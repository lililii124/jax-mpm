"""Audit the immutable driver's entire read-set before starting its runtime."""
import ast
import importlib.util
from pathlib import Path
import subprocess

import pytest


def controller_and_source():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('zero_resume_contract_test', root / 'tools/run_triaxial_zero_recovery.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    frozen = m.FROZEN / 'examples/triaxial_failure.py'
    source = frozen.read_text() if frozen.exists() else subprocess.check_output(
        ['git', '-C', str(root), 'show', m.ORIGINAL_COMMIT + ':examples/triaxial_failure.py']).decode()
    return m, source


def test_all_original_resume_reads_classified():
    m, source = controller_and_source()
    assert set(m.resume_file_contract(source)) == {'plan.json', 'status.json', 'summary.json',
        'layout.json', 'contact-model.json', 'checkpoint.npz',
        *(f'trajectory_{field}.npy' for field in m.FIELDS)}


def test_every_absent_dependency_rejected_before_checkpoint_or_runtime(tmp_path):
    m, source = controller_and_source()
    required = m.resume_file_contract(source)
    for name in required:
        (tmp_path / name).write_bytes(b'not-opened-during-missing-file-check')
    for name in required:
        path = tmp_path / name
        path.unlink()
        with pytest.raises(ValueError, match='dependency absent'):
            m.audit_resume_directory(tmp_path, {}, {}, source=source)
        path.write_bytes(b'not-opened-during-missing-file-check')


def test_unclassified_new_dependency_blocks_admission():
    m, source = controller_and_source()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_case')
    function.body.insert(0, ast.parse("(folder / 'another-parent-file.json').read_text()").body[0])
    with pytest.raises(ValueError, match='read-set changed'):
        m.resume_file_contract(ast.unparse(tree))
