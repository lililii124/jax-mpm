"""Propose distinct, connected fixed-mass designs along a stored JAX gradient."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('gradient', type=Path)
    parser.add_argument('--steps', type=float, nargs='+', required=True)
    parser.add_argument('--logit-bound', type=float, default=12.)
    parser.add_argument('--fallback-directions', type=int, default=0)
    parser.add_argument('--fallback-seed', type=int, default=370)
    parser.add_argument('--max-single-voxel-fraction', type=float,
                        help='Maximum fraction of solid voxels in an axis-aligned one-voxel run')
    parser.add_argument('--min-bottom-coverage', type=float)
    parser.add_argument('--min-common-bottom-coverage', type=float)
    parser.add_argument('--constraint-reference-geometry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not np.isfinite(args.steps).all() or min(args.steps) <= 0 or
            not np.isfinite(args.logit_bound) or args.logit_bound <= 0 or args.fallback_directions < 0):
        raise ValueError('Step sizes and logit bound must be finite and positive')
    if args.max_single_voxel_fraction is not None and not 0 <= args.max_single_voxel_fraction <= 1:
        raise ValueError('Single-voxel fraction limit must lie in [0, 1]')
    for value in (args.min_bottom_coverage, args.min_common_bottom_coverage):
        if value is not None and not 0 <= value <= 1:
            raise ValueError('Bottom coverage limit must lie in [0, 1]')
    if (args.min_common_bottom_coverage is None) != (args.constraint_reference_geometry is None):
        raise ValueError('Common coverage requires its reference geometry')
    meta = json.loads(args.inputs.read_text(encoding='utf-8'))
    binding = meta['binding']
    if hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != meta['signature']:
        raise ValueError('Inputs signature differs')
    root = Path(__file__).resolve().parents[1]
    for name in ('mpm_jax/shared_cell_design.py', 'mpm_jax/periodic_filter.py',
                 'tools/reconstruct_multicell_blast.py'):
        if sha256(root / name) != binding['source_sha256'][name]:
            raise ValueError(f'Frozen projection source differs: {name}')
    import jax
    import jaxlib
    if (jax.__version__, jaxlib.__version__, np.__version__) != (
            binding['jax'], binding['jaxlib'], binding['numpy']):
        raise ValueError('Runtime versions differ from the gradient binding')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from mpm_jax.periodic_filter import filter_cell_logits
    from mpm_jax.shared_cell_design import project_fixed_volume
    from tools.reconstruct_multicell_blast import reconstruct
    from tools.multicell_feature_filter import single_voxel_run_fraction

    with np.load(args.gradient, allow_pickle=False) as data:
        if str(data['signature'].item()) != meta['signature'] or str(data['inputs_sha256'].item()) != sha256(args.inputs):
            raise ValueError('Gradient does not belong to these optimizer inputs')
        q = np.asarray(data['logits'], dtype=np.float64)
        gradient = np.asarray(data['gradient'], dtype=np.float64)
        saved_rho = np.asarray(data['rho_cell'], dtype=np.float64)
        selected_mask = str(data['solid_mask_sha256'].item())
    settings = binding['settings']
    nx, ny, nz = settings['cell']
    if (q.shape != (nx * ny * nz,) or gradient.shape != q.shape or
            not np.isfinite(q).all() or not np.isfinite(gradient).all()):
        raise ValueError('Invalid stored logits or gradient')
    norm = float(np.linalg.norm(gradient))
    if not norm > 0 or not np.isfinite(norm):
        raise ValueError('Gradient norm must be finite and positive')

    def density(logits):
        filtered = filter_cell_logits(jnp.asarray(logits).reshape(nz, ny, nx), settings['filter_passes'])
        return np.asarray(project_fixed_volume(filtered, settings['volume_fraction'], beta=settings['beta']))

    if not np.allclose(density(q), saved_rho, rtol=0, atol=1e-12):
        raise ValueError('Gradient density does not match its logits')
    base = reconstruct(saved_rho, settings)[-1]
    if (base['solid_mask_sha256'] != selected_mask or base['components'] != 1 or
            not base['faces_connected'] or base['populated_column_fraction'] != 1.):
        raise ValueError('Selected baseline must be connected and cover the projected pressure footprint')
    reference_bottom = None
    if args.constraint_reference_geometry is not None:
        with np.load(args.constraint_reference_geometry, allow_pickle=False) as data:
            reference_bottom = np.asarray(data['solid_mask'][0], dtype=bool).copy()
    args.output.mkdir(parents=True)
    records = []
    seen = {selected_mask}
    def try_step(direction, step, kind):
        proposal = np.clip(q + step * direction, -args.logit_bound, args.logit_bound)
        rho = density(proposal)
        cell, solid, positions, volume, report = reconstruct(rho, settings)
        mask = report['solid_mask_sha256']
        fraction = single_voxel_run_fraction(cell)
        record = {'step': step, 'direction': kind, 'solid_mask_sha256': mask,
                  'single_voxel_fraction': fraction}
        bottom = np.asarray(solid[0], dtype=bool)
        bottom_fraction = float(np.mean(bottom))
        record['bottom_coverage'] = bottom_fraction
        if reference_bottom is not None:
            if reference_bottom.shape != bottom.shape: raise ValueError('Reference bottom grid differs')
            record['common_bottom_coverage'] = float(np.mean(reference_bottom & bottom))
        if mask in seen:
            record['status'] = 'duplicate'
        elif (report['components'] != 1 or not report['faces_connected'] or
              report['populated_column_fraction'] != 1.):
            record['status'] = 'infeasible'
        elif args.max_single_voxel_fraction is not None and fraction > args.max_single_voxel_fraction:
            record['status'] = 'feature-limit'
        elif ((args.min_bottom_coverage is not None and bottom_fraction < args.min_bottom_coverage) or
              (args.min_common_bottom_coverage is not None and
               record['common_bottom_coverage'] < args.min_common_bottom_coverage)):
            record['status'] = 'bottom-coverage-limit'
        elif not np.isclose(report['binary_mass_kg'], base['binary_mass_kg'], rtol=1e-12, atol=1e-12):
            raise ValueError('Binary material mass changed')
        else:
            label = f'candidate-{len(records):03d}'
            design = args.output / (label + '-design.npz')
            geometry = args.output / (label + '-geometry.npz')
            np.savez_compressed(design, logits=proposal, rho_cell=rho, step=step)
            np.savez_compressed(geometry, cell_mask=cell, solid_mask=solid, positions=positions,
                                particle_volume=np.asarray(volume))
            record.update(status='proposed', design=design.name, design_sha256=sha256(design),
                          geometry=geometry.name, geometry_sha256=sha256(geometry),
                          binary_mass_kg=report['binary_mass_kg'])
        seen.add(mask)
        records.append(record)
    unit_gradient = gradient / norm
    for step in args.steps:
        try_step(-unit_gradient, step, 'gradient')
    if args.fallback_directions and not any(x['status'] == 'proposed' for x in records):
        rng = np.random.default_rng(args.fallback_seed)
        for index in range(args.fallback_directions):
            noise = rng.normal(size=q.shape)
            noise -= np.dot(noise, unit_gradient) * unit_gradient
            noise_norm = np.linalg.norm(noise)
            if not noise_norm > 0:
                raise FloatingPointError('Could not generate a distinct exploration direction')
            direction = -unit_gradient + noise / noise_norm
            direction /= np.linalg.norm(direction)
            for step in args.steps:
                try_step(direction, step, f'gradient-plus-orthogonal-{index}')
            if sum(x['status'] == 'proposed' for x in records) >= 4:
                break
    manifest = {'signature': meta['signature'], 'inputs_sha256': sha256(args.inputs),
                'gradient_sha256': sha256(args.gradient), 'baseline_solid_mask_sha256': selected_mask,
                'max_single_voxel_fraction': args.max_single_voxel_fraction,
                'feature_filter_sha256': sha256(root/'tools/multicell_feature_filter.py'),
                'fallback_seed': args.fallback_seed if args.fallback_directions else None,
                'steps': records, 'qualification': 'Proposals require complete binary MPM replay before acceptance.'}
    if args.min_bottom_coverage is not None:
        manifest['min_bottom_coverage'] = args.min_bottom_coverage
    if args.min_common_bottom_coverage is not None:
        manifest['min_common_bottom_coverage'] = args.min_common_bottom_coverage
        manifest['constraint_reference_geometry_sha256'] = sha256(args.constraint_reference_geometry)
    temp = args.output / 'manifest.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(args.output / 'manifest.json')
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
