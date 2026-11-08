"""Reconstruct a shared-cell blast design as solid voxels and voids."""
import argparse
import hashlib
import json
import math
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np


def mask_digest(mask):
    mask = np.asarray(mask, dtype=bool)
    return hashlib.sha256(np.asarray(mask.shape, dtype='<i8').tobytes() +
                          np.packbits(mask.reshape(-1)).tobytes()).hexdigest()


def reconstruct(rho, settings):
    nx, ny, nz = map(int, settings['cell'])
    rho = np.asarray(rho, dtype=np.float64)
    fraction = float(settings['volume_fraction'])
    layers = int(settings['skin_layers'])
    size = np.asarray(settings['cell_size'], dtype=np.float64)
    if rho.shape != (nz, ny, nx) or not np.all(np.isfinite(rho)):
        raise ValueError('Invalid cell density field')
    if min(nx, ny, nz) < 1 or layers < 0 or not 0 < fraction < 1 or np.any(size <= 0):
        raise ValueError('Invalid cell geometry or volume fraction')
    if np.any((rho < 0) | (rho > 1)) or not np.isclose(rho.mean(), fraction, atol=1e-9, rtol=0):
        raise ValueError('Density field does not match the prescribed fraction')
    h = size / (nx, ny, nz)
    if not np.isclose(layers * h[2], settings['skin'], atol=1e-12, rtol=0):
        raise ValueError('Face-sheet thickness does not match the grid')

    count = int(np.floor(rho.size * fraction + 0.5))
    if count == 0 or count == rho.size:
        raise ValueError('The material fraction cannot be represented on this grid')
    indices = np.arange(rho.size)
    selected = np.lexsort((indices, -rho.ravel()))[:count]
    cell = np.zeros(rho.size, dtype=bool)
    cell[selected] = True
    cell = cell.reshape(rho.shape)
    solid = np.concatenate((np.ones((layers, 3 * ny, 3 * nx), bool),
                            np.tile(cell, (1, 3, 3)),
                            np.ones((layers, 3 * ny, 3 * nx), bool)))

    # Connectivity is assessed through shared voxel faces in the finite plate.
    seen = np.zeros_like(solid)
    components = []
    for z, y, x in np.argwhere(solid):
        if seen[z, y, x]:
            continue
        seen[z, y, x] = True
        queue = deque([(int(z), int(y), int(x))])
        n = 0
        bottom = top = False
        while queue:
            iz, iy, ix = queue.popleft()
            n += 1
            bottom |= iz == 0
            top |= iz == solid.shape[0] - 1
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                zz, yy, xx = iz + dz, iy + dy, ix + dx
                if (0 <= zz < solid.shape[0] and 0 <= yy < solid.shape[1]
                        and 0 <= xx < solid.shape[2] and solid[zz, yy, xx]
                        and not seen[zz, yy, xx]):
                    seen[zz, yy, xx] = True
                    queue.append((zz, yy, xx))
        components.append((n, bottom, top))

    zyx = np.argwhere(solid)
    centers = (zyx[:, ::-1] + 0.5) * h
    offsets = np.asarray(list(product((-1 / (2 * math.sqrt(3)),
                                       1 / (2 * math.sqrt(3))), repeat=3))) * h
    positions = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    particle_volume = float(np.prod(h) / 8)
    density = float(settings['solid_density'])
    target_mass = density * (9 * np.prod(size) * fraction +
                             18 * size[0] * size[1] * float(settings['skin']))
    mass = len(positions) * particle_volume * density
    sheet_connected = sum(n for n, bottom, top in components if bottom or top)
    report = dict(cell_shape_xyz=[nx, ny, nz], repeats_xyz=[3, 3, 1],
                  solid_voxels_per_cell=count, total_voxels_per_cell=rho.size,
                  requested_core_fraction=fraction, binary_core_fraction=count / rho.size,
                  target_mass_kg=target_mass, binary_mass_kg=mass,
                  mass_error_kg=mass - target_mass, particles=len(positions),
                  voxel_size_m=h.tolist(), components=len(components),
                  detached_solid_voxels=int(solid.sum()) - sheet_connected,
                  faces_connected=any(bottom and top for _, bottom, top in components),
                  populated_column_fraction=float(np.mean(np.any(solid, axis=0))),
                  solid_mask_sha256=mask_digest(solid),
                  description='Binary geometric reconstruction; no dynamic validation has been performed.')
    return cell, solid, positions, particle_volume, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('output must have an .npz suffix')
    metadata_path = args.output.with_suffix('.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError('Refusing to overwrite an existing reconstruction')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    with np.load(args.design, allow_pickle=False) as data:
        cell, solid, positions, volume, report = reconstruct(data['rho_cell'], inputs['binding']['settings'])
    report.update(design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest(),
                  inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                  optimization_signature=inputs['signature'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, cell_mask=cell, solid_mask=solid,
                        positions=positions, particle_volume=np.asarray(volume))
    report['geometry_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
