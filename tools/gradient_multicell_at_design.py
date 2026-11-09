"""Evaluate the shared-cell surrogate gradient at a selected design."""
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
    parser.add_argument('design', type=Path)
    parser.add_argument('--fixed-bottom-geometry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads(args.inputs.read_text(encoding='utf-8'))
    binding = meta['binding']
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if signature != meta['signature']:
        raise ValueError('Optimizer inputs signature does not match its physics binding')
    root = Path(__file__).resolve().parents[1]
    for name, expected in binding['source_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Frozen optimizer source differs: {name}')

    import jax
    import jaxlib
    if (jax.__version__, jaxlib.__version__, np.__version__) != (
            binding['jax'], binding['jaxlib'], binding['numpy']):
        raise ValueError('Optimizer runtime versions differ from the physics binding')
    from examples.multicell_blast_3d import build
    from mpm_jax.periodic_filter import filter_cell_logits
    from mpm_jax.shared_cell_design import project_fixed_volume
    from tools.reconstruct_multicell_blast import reconstruct

    settings = binding['settings']
    keys = ('cell', 'cell_size', 'skin', 'skin_layers', 'volume_fraction', 'beta',
            'filter_passes', 'skin_young', 'core_young', 'solid_density', 'pressure',
            'pulse', 'blast_radius', 'dt', 'cfl_limit', 'steps', 'checkpoint_block',
            'displacement_scale')
    model = {key: settings[key] for key in keys}
    for key in ('pressure_placement', 'density_floor', 'fixed_edge_width'):
        if key in settings:
            model[key] = settings[key]
    if args.fixed_bottom_geometry is not None:
        with np.load(args.fixed_bottom_geometry, allow_pickle=False) as data:
            reference = np.asarray(data['solid_mask'], dtype=bool)
        nx, ny, nz = settings['cell']
        if reference.shape != (nz + 2*settings['skin_layers'], 3*ny, 3*nx) or not np.any(reference[0]):
            raise ValueError('Fixed bottom reference geometry differs from the full-domain grid')
        model['fixed_bottom_mask'] = reference[0]
    objective, plan = build(argparse.Namespace(**model))
    import jax.numpy as jnp
    with np.load(args.design, allow_pickle=False) as data:
        q = np.asarray(data['logits'], dtype=np.float64).reshape(-1)
        saved_rho = np.asarray(data['rho_cell'], dtype=np.float64)
    nx, ny, nz = settings['cell']
    if q.shape != (nx * ny * nz,) or not np.isfinite(q).all():
        raise ValueError('Invalid design logits')
    rho = np.asarray(project_fixed_volume(filter_cell_logits(
        jnp.asarray(q).reshape(nz, ny, nx), settings['filter_passes']),
        settings['volume_fraction'], beta=settings['beta']))
    if not np.allclose(rho, saved_rho, rtol=0, atol=1e-12):
        raise ValueError('Saved density is inconsistent with the logits')
    report = reconstruct(rho, settings)[-1]
    if report['components'] != 1 or not report['faces_connected']:
        raise ValueError('Selected solid geometry is disconnected')
    print(json.dumps({'stage': 'evaluating-gradient', 'solid_mask_sha256': report['solid_mask_sha256']}), flush=True)
    value, grad = jax.jit(jax.value_and_grad(objective))(jnp.asarray(q))
    value, grad = float(value), np.asarray(grad)
    norm = float(np.linalg.norm(grad))
    if not np.isfinite(value) or not np.isfinite(grad).all() or not norm > 0:
        raise FloatingPointError('Invalid objective or gradient')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.stem + '.tmp.npz')
    output = dict(logits=q, rho_cell=rho, gradient=grad, objective=value,
                  inputs_sha256=sha256(args.inputs), design_sha256=sha256(args.design),
                  signature=signature, solid_mask_sha256=report['solid_mask_sha256'])
    if args.fixed_bottom_geometry is not None:
        output['fixed_bottom_geometry_sha256'] = sha256(args.fixed_bottom_geometry)
    np.savez_compressed(temp, **output)
    temp.replace(args.output)
    print(json.dumps({'stage': 'complete', 'objective': value, 'gradient_norm': norm,
                      'solid_mask_sha256': report['solid_mask_sha256'],
                      'full_domain_tiles': plan['tiles'], 'gradient_sha256': sha256(args.output)}), flush=True)


if __name__ == '__main__':
    main()
