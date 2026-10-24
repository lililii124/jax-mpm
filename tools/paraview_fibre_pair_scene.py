"""Actual ParaView side-elevation view; common scene/geometry/colors unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]/'mpm_jax/paraview_scene.py'
raw=BASE.read_bytes();tree=ast.parse(raw.decode('utf-8'));changed=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='elevation' for t in node.targets):
        node.value=ast.Constant(value=.2);changed+=1
if changed!=1:raise ValueError('The common ParaView camera policy changed; review explicit elevation override')
ast.fix_missing_locations(tree)
namespace={'__name__':'__main__','__file__':str(BASE)}
exec(compile(tree,str(BASE),'exec'),namespace)
output=Path(sys.argv[1]);record=json.loads((output/'render.json').read_text())
record.update(camera_elevation_policy=.2,camera_direction_policy=[1.4,.2,1.8],camera_policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_scene_source_sha256=hashlib.sha256(raw).hexdigest())
(output/'render.json').write_text(json.dumps(record,indent=2)+'\n')
