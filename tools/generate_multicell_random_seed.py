"""Generate a connected shared-cell solid from smoothed random logits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filtered(field, passes):
    for _ in range(passes):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis=axis) + 2*field + np.roll(field, -1, axis=axis))*.25
        z = np.pad(field, ((1, 1), (0, 0), (0, 0)), mode='edge')
        field = (z[:-2] + 2*z[1:-1] + z[2:])*.25
    return field


def one_voxel_fraction(mask):
    z0 = np.pad(mask, ((1, 0), (0, 0), (0, 0)))[:-1]
    z1 = np.pad(mask, ((0, 1), (0, 0), (0, 0)))[1:]
    single = ((mask & ~z0 & ~z1) | (mask & ~np.roll(mask, 1, 1) & ~np.roll(mask, -1, 1)) |
              (mask & ~np.roll(mask, 1, 2) & ~np.roll(mask, -1, 2)))
    return float(single.sum()/mask.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('optimizer_inputs', type=Path)
    parser.add_argument('binary_inputs', type=Path)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma-zyx', type=float, nargs=3, default=[2., 1., 1.])
    parser.add_argument('--max-single-run-fraction', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if (args.trials < 1 or not np.isfinite(args.sigma_zyx).all() or min(args.sigma_zyx) <= 0 or
            not 0 <= args.max_single_run_fraction <= 1):
        raise ValueError('Invalid search settings')
    optimizer = json.loads(args.optimizer_inputs.read_text(encoding='utf-8'))
    binary = json.loads(args.binary_inputs.read_text(encoding='utf-8'))
    binding = optimizer['binding']
    if hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != optimizer['signature']:
        raise ValueError('Optimizer inputs signature differs')
    settings = binding['settings']; physical = binary['binding']['settings']
    fields = ('cell', 'cell_size', 'skin', 'skin_layers', 'volume_fraction', 'skin_young',
              'core_young', 'solid_density', 'pressure', 'pulse', 'blast_radius')
    if any(settings[key] != physical[key] for key in fields):
        raise ValueError('Optimizer and binary physics differ')
    root = Path(__file__).resolve().parents[1]
    for name in ('mpm_jax/shared_cell_design.py', 'mpm_jax/periodic_filter.py',
                 'tools/reconstruct_multicell_blast.py'):
        if sha256(root / name) != binding['source_sha256'][name]:
            raise ValueError('Frozen initialization source differs: ' + name)
    import jax
    import jaxlib
    if (jax.__version__, jaxlib.__version__, np.__version__) != (binding['jax'], binding['jaxlib'], binding['numpy']):
        raise ValueError('Runtime differs from optimizer binding')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from mpm_jax.periodic_filter import filter_cell_logits
    from mpm_jax.shared_cell_design import project_fixed_volume
    from tools.reconstruct_multicell_blast import reconstruct

    nx, ny, nz = map(int, settings['cell'])
    shape = (nz, ny, nx); nsolid = int(np.floor(np.prod(shape)*settings['volume_fraction'] + .5))
    if min(shape) < 1 or not 0 < nsolid < np.prod(shape) or int(settings['skin_layers']) != 0:
        raise ValueError('Expected a nonempty open cell without prescribed face sheets')
    best = None; valid = 0
    for seed in range(args.seed, args.seed + args.trials):
        raw = np.random.default_rng(seed).normal(size=shape)
        q = gaussian_filter(raw, sigma=args.sigma_zyx, mode=('nearest', 'wrap', 'wrap'))
        q = .1*q/max(float(q.std()), 1e-30)
        field = filtered(q, settings['filter_passes'])
        indices = np.lexsort((np.arange(field.size), -field.ravel()))[:nsolid]
        mask = np.zeros(field.size, dtype=bool); mask[indices] = True; mask = mask.reshape(shape)
        if not np.any(mask, axis=0).all() or not mask[0].any() or not mask[-1].any(): continue
        if label(np.tile(mask, (1, 3, 3)))[1] != 1: continue
        score = one_voxel_fraction(mask); valid += 1
        if score <= args.max_single_run_fraction and (best is None or score < best['score']):
            best = dict(seed=seed, score=score, logits=q, mask=mask)
    if best is None:
        raise ValueError('No connected candidate satisfied the requested feature threshold')
    rho = np.asarray(project_fixed_volume(filter_cell_logits(jnp.asarray(best['logits']), settings['filter_passes']),
                                          settings['volume_fraction'], beta=settings['beta']))
    cell, solid, positions, volume, report = reconstruct(rho, physical)
    if (not np.array_equal(cell, best['mask']) or report['components'] != 1 or not report['faces_connected'] or
            report['populated_column_fraction'] != 1.):
        raise ValueError('Selected geometry failed independent reconstruction')
    args.output.mkdir(parents=True)
    design = args.output / 'design.npz'
    geometry = args.output / 'geometry.npz'
    np.savez_compressed(design, logits=best['logits'].reshape(-1), rho_cell=rho)
    np.savez_compressed(geometry, cell_mask=cell, solid_mask=solid, positions=positions, particle_volume=np.asarray(volume))
    report.update(geometry_sha256=sha256(geometry), inputs_sha256=sha256(args.binary_inputs),
                  optimization_signature=binary['signature'], design_sha256=sha256(design),
                  optimizer_inputs_sha256=sha256(args.optimizer_inputs), initialization='smoothed random logits',
                  random_seed=best['seed'], smoothing_sigma_zyx=args.sigma_zyx,
                  trials=args.trials, geometrically_valid_trials=valid,
                  single_voxel_any_axis_fraction=best['score'],
                  qualification='Geometry resolution proxy only; complete dynamic and mesh validation are required.')
    geometry.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
