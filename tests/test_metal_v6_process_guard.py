import ast
from pathlib import Path
from tools import metal_v6_process_guard as guard


def test_extracted_guard_is_identical_to_proven_controller():
    root=Path(__file__).resolve().parents[1]
    def functions(path):
        return {node.name:ast.dump(node) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    old=functions(root/'tools/run_metal_v6_probe125.py')
    new=functions(Path(guard.__file__))
    for name in ('owned_session_uss','terminate_owned_group'):
        assert new[name]==old[name]


def test_guard_has_only_stdlib_top_level_imports():
    tree=ast.parse(Path(guard.__file__).read_text())
    imports=[name.name for node in tree.body if isinstance(node,ast.Import) for name in node.names]
    assert set(imports)=={'os','signal','time'}
    assert not any(isinstance(node,ast.ImportFrom) for node in tree.body)
