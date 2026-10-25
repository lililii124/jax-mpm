"""Import smoke check executed inside the exact deployed runtime-source tree."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def snapshot(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py') if '__pycache__' not in p.parts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-source',type=Path,required=True)
    parser.add_argument('--source-admission',type=Path,required=True)
    parser.add_argument('--frozen-admission',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--numerical-imports',action='store_true')
    args=parser.parse_args();root=args.runtime_source.resolve()
    if Path(__file__).resolve()!=root/'tools/smoke_metal_v6_dt_runtime.py':raise ValueError('Smoke must execute from exact deployed tree')
    if args.output.exists():raise FileExistsError('Preserve prior smoke evidence')
    sys.dont_write_bytecode=True
    before=snapshot(root)
    expected=json.loads(args.source_admission.read_text())
    frozen=json.loads(args.frozen_admission.read_text())
    for name,digest in {**frozen['numerical_source'],**expected}.items():
        if before.get(name)!=digest:raise ValueError('Deployed source differs: '+name)
    if before.get('mpm_jax/fragment_v6_energy_probe.py')!=frozen['probe_sha256']:raise ValueError('Frozen energy probe differs')
    modules=['tools.run_metal_v6_dt_convergence','tools.metal_v6_process_guard',
             'tools.compare_metal_v6_dt_convergence','tools.audit_metal_onset_state',
             'examples.metal_fragment_v6_dt_convergence']
    if args.numerical_imports:
        modules+=['psutil','mpm_jax.fragment_v6_onset_probe','mpm_jax.materials.linear_metal',
                  'examples.metal_fragment_multifield_compact']
    for name in modules:importlib.import_module(name)
    project_origins={}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] not in ('tools','examples','mpm_jax'):continue
        origin=getattr(module,'__file__',None)
        if origin:
            resolved=Path(origin).resolve()
            if not resolved.is_relative_to(root):raise ValueError('Project import escaped deployed tree: '+name)
            project_origins[name]=resolved.relative_to(root).as_posix()
        else:
            for location in getattr(module,'__path__',()):
                if not Path(location).resolve().is_relative_to(root):raise ValueError('Namespace escaped deployed tree: '+name)
    if any('recover_metal' in name or 'run_metal_v6_probe125' in name for name in project_origins):raise ValueError('Unexpected recovery-controller dependency')
    after=snapshot(root)
    if before!=after:raise ValueError('Runtime-source changed during import smoke')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,numerical_imports=args.numerical_imports,
                          project_module_origins=project_origins,all_python_source_sha256=before,
                          source_unchanged=True,mechanical_steps_executed=0),indent=2)+'\n')


if __name__=='__main__':main()
