"""Original fibre renderer with the exact shared homepage low camera direction."""
import ast
from pathlib import Path

base = Path(__file__).resolve().parents[1]/'mpm_jax/paraview_scene.py'
tree = ast.parse(base.read_text(encoding='utf-8'))
changed = 0
pure_partition_checks = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'elevation'
                                          for target in node.targets):
        node.value = ast.Constant(value=.2)
        changed += 1
    # This standalone case contains fibres only (phase1), so the legacy woven
    # matrix selection is correctly empty. Preserve both family selections and
    # exact total particle coverage; merely allow the known empty matrix slot.
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'any'
            and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)):
        generator = node.args[0]
        if (len(generator.generators) == 1 and isinstance(generator.generators[0].iter, ast.Name)
                and generator.generators[0].iter.id == 'counts'
                and isinstance(generator.elt, ast.Compare)
                and isinstance(generator.elt.left, ast.Name) and generator.elt.left.id == 'count'
                and len(generator.elt.comparators) == 1
                and isinstance(generator.elt.comparators[0], ast.Constant)
                and generator.elt.comparators[0].value == 0):
            generator.generators[0].iter = ast.Subscript(value=ast.Name(id='counts',ctx=ast.Load()),
                slice=ast.Slice(lower=ast.Constant(value=1)),ctx=ast.Load())
            pure_partition_checks += 1
if changed != 1:
    raise RuntimeError('The original camera policy changed; explicit review required')
if pure_partition_checks != 1:
    raise RuntimeError('Expected exactly one pure-fibre partition check adaptation')
ast.fix_missing_locations(tree)
exec(compile(tree, str(base), 'exec'), {'__name__':'__main__', '__file__':str(base)})
