"""Compare surrogate rankings with two completed physical binary replays."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('initial_design', type=Path)
    parser.add_argument('candidate_design', type=Path)
    parser.add_argument('initial_geometry', type=Path)
    parser.add_argument('candidate_geometry', type=Path)
    parser.add_argument('initial_replay', type=Path)
    parser.add_argument('candidate_replay', type=Path)
    parser.add_argument('--density-floor', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    binding = inputs['binding']; settings = binding['settings']
    if hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != inputs['signature']:
        raise ValueError('Optimizer input binding differs')
    root = Path(__file__).resolve().parents[1]
    for name in ('mpm_jax/shared_cell_design.py', 'mpm_jax/periodic_filter.py',
                 'tools/reconstruct_multicell_blast.py'):
        if digest(root / name) != binding['source_sha256'][name]: raise ValueError('Projection source differs: ' + name)
    import jax
    import jaxlib
    if (jax.__version__, jaxlib.__version__, np.__version__) != (binding['jax'], binding['jaxlib'], binding['numpy']):
        raise ValueError('Runtime differs from the optimizer binding')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from examples.multicell_blast_3d import build
    from mpm_jax.periodic_filter import filter_cell_logits
    from mpm_jax.shared_cell_design import project_fixed_volume
    from tools.compare_multicell_binary_blast import compare, read_run, verify_physical_model_compatibility
    from tools.reconstruct_multicell_blast import mask_digest, reconstruct

    initial_meta, _, _ = read_run(args.initial_replay, .02)
    candidate_meta, _, _ = read_run(args.candidate_replay, .02)
    verify_physical_model_compatibility(settings, initial_meta['settings'])
    if initial_meta['geometry_sha256'] != digest(args.initial_geometry) or candidate_meta['geometry_sha256'] != digest(args.candidate_geometry):
        raise ValueError('Physical replays and geometries differ')
    designs = []
    for design, geometry, meta in [(args.initial_design, args.initial_geometry, initial_meta),
                                   (args.candidate_design, args.candidate_geometry, candidate_meta)]:
        with np.load(design, allow_pickle=False) as data:
            q = np.asarray(data['logits'], dtype=np.float64).reshape(-1)
            rho = np.asarray(data['rho_cell'], dtype=np.float64)
        nx, ny, nz = settings['cell']
        calculated = np.asarray(project_fixed_volume(filter_cell_logits(jnp.asarray(q).reshape(nz, ny, nx),
                                                         settings['filter_passes']), settings['volume_fraction'], beta=settings['beta']))
        if not np.allclose(rho, calculated, rtol=0, atol=1e-12): raise ValueError('Design logits and density differ')
        report = reconstruct(rho, settings)[-1]
        with np.load(geometry, allow_pickle=False) as data:
            mask = np.asarray(data['solid_mask'], dtype=bool)
        if (report['solid_mask_sha256'] != meta['solid_mask_sha256'] or mask_digest(mask) != meta['solid_mask_sha256'] or
                not np.any(mask, axis=0).all() or report['components'] != 1):
            raise ValueError('Physical design is disconnected or lacks its pressure footprint')
        designs.append(q)
    physical = compare(args.initial_replay, args.candidate_replay)
    keys = ('cell', 'cell_size', 'skin', 'skin_layers', 'volume_fraction', 'beta', 'filter_passes',
            'skin_young', 'core_young', 'solid_density', 'pressure', 'pulse', 'blast_radius', 'dt',
            'cfl_limit', 'steps', 'checkpoint_block', 'displacement_scale')
    results = {}
    for placement in ('top', 'exposed'):
        model = {key: settings[key] for key in keys}
        if 'fixed_edge_width' in settings: model['fixed_edge_width'] = settings['fixed_edge_width']
        model.update(pressure_placement=placement, density_floor=args.density_floor)
        objective, plan = build(argparse.Namespace(**model))
        evaluate = jax.jit(objective)
        values = []
        for name, q in zip(('initial', 'candidate'), designs):
            print(json.dumps({'stage': 'evaluating', 'pressure_placement': placement, 'design': name}), flush=True)
            started = time.monotonic(); value = float(evaluate(jnp.asarray(q)))
            if not np.isfinite(value) or value <= 0: raise FloatingPointError('Proxy objective is invalid')
            values.append(value)
            print(json.dumps({'stage': 'evaluated', 'seconds': time.monotonic()-started, 'value': value}), flush=True)
        reduction = (values[0]-values[1])/values[0]
        results[placement] = dict(initial_objective=values[0], candidate_objective=values[1],
                                  relative_reduction=reduction,
                                  same_improvement_sign=(reduction > 0) == (physical['sampled_peak_reduction_fraction'] > 0),
                                  plan=plan)
    sources = ('examples/multicell_blast_3d.py', 'mpm_jax/exposed_pressure.py', 'mpm_jax/periodic_filter.py',
               'mpm_jax/shared_cell_design.py', 'mpm_jax/compact_tl.py', 'mpm_jax/total_lagrangian.py',
               'mpm_jax/materials/elastic.py', 'tools/compare_multicell_binary_blast.py',
               'tools/compare_multicell_proxy.py')
    result = dict(physical=physical, proxy=results, density_floor=args.density_floor,
                  source_sha256={name: digest(root/name) for name in sources},
                  inputs_sha256=digest(args.inputs), initial_design_sha256=digest(args.initial_design),
                  candidate_design_sha256=digest(args.candidate_design),
                  qualification='Ranking at one pair of designs only; proxy objectives are not binary peak displacement and do not include cavity-side pressure.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8'); temp.replace(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
