"""Original fibre renderer with the exact shared homepage low camera direction."""
import ast
from pathlib import Path

base = Path(__file__).resolve().parents[1]/'mpm_jax/paraview_scene.py'
tree = ast.parse(base.read_text(encoding='utf-8'))
changed = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'elevation'
                                          for target in node.targets):
        node.value = ast.Constant(value=.2)
        changed += 1
if changed != 1:
    raise RuntimeError('The original camera policy changed; explicit review required')
ast.fix_missing_locations(tree)
exec(compile(tree, str(base), 'exec'), {'__name__':'__main__', '__file__':str(base)})
