"""Export completed binary-cell replay frames as a ParaView volume series."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpm_jax.compact_tl import surface_weights
from tools.compare_multicell_binary_blast import read_run
from tools.reconstruct_multicell_blast import mask_digest


def xml_array(name, values, kind='Float64', components=1):
    flat = np.asarray(values).reshape(-1)
    data = ' '.join(format(float(x), '.17g') if kind == 'Float64' else str(int(x)) for x in flat)
    return f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" format="ascii">{data}</DataArray>'


def export(source, geometry, output):
    meta, rows, energy_error = read_run(source, .02)
    if hashlib.sha256(geometry.read_bytes()).hexdigest() != meta['geometry_sha256']:
        raise ValueError('Geometry does not match the replay')
    with np.load(geometry, allow_pickle=False) as saved:
        mask = np.asarray(saved['solid_mask'], dtype=bool)
    if mask_digest(mask) != meta['solid_mask_sha256']:
        raise ValueError('Solid mask does not match the replay')
    with np.load(source/'reference.npz', allow_pickle=False) as saved:
        nodes, spacing = saved['nodes'], saved['spacing']
        shape = tuple(saved['shape'])
    voxels = np.argwhere(mask)[:, ::-1]
    corners = np.array(((0,0,0), (1,0,0), (1,1,0), (0,1,0),
                        (0,0,1), (1,0,1), (1,1,1), (0,1,1)))
    unique, inverse = np.unique((voxels[:, None, :]+corners[None, :, :]).reshape(-1, 3),
                                axis=0, return_inverse=True)
    points = unique * spacing
    cells = inverse.reshape(-1, 8)
    support = SimpleNamespace(nodes=nodes, spacing=spacing, shape=shape)
    indices, weights = surface_weights(support, points)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        frame = source/f"frame-{row['step']:07d}.npz"
        with np.load(frame, allow_pickle=False) as saved:
            nodal = saved['nodal_displacement']
            stress = saved['von_mises']
        if nodal.shape != nodes.shape or stress.shape != (len(voxels)*8,):
            raise ValueError(f'Invalid frame arrays: {frame}')
        u = np.sum(nodal[indices] * weights[:, :, None], axis=1)
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(stress)):
            raise ValueError(f'Non-finite frame: {frame}')
        cell_stress = np.mean(stress.reshape(-1, 8), axis=1)
        xml = ('<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
               '<UnstructuredGrid>'
               f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">'
               '<PointData>'+xml_array('displacement_m', u, components=3)+
               xml_array('displacement_magnitude_m', np.linalg.norm(u, axis=1))+'</PointData>'
               '<CellData>'+xml_array('von_mises_Pa', cell_stress)+'</CellData>'
               '<Points>'+xml_array('Points', points+u, components=3)+'</Points><Cells>'+
               xml_array('connectivity', cells, 'Int64')+
               xml_array('offsets', 8*np.arange(1, len(cells)+1), 'Int64')+
               xml_array('types', np.full(len(cells), 12), 'UInt8')+
               '</Cells></Piece></UnstructuredGrid></VTKFile>')
        name = f"frame-{row['step']:07d}.vtu"
        (output/name).write_text(xml, encoding='utf-8')
        records.append(dict(file=name, time_s=row['time_s'], step=row['step'],
                            source_sha256=hashlib.sha256(frame.read_bytes()).hexdigest()))
    datasets = ''.join(f'<DataSet timestep="{r["time_s"]:.17g}" part="0" file="{r["file"]}"/>' for r in records)
    (output/'series.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+datasets+'</Collection></VTKFile>', encoding='utf-8')
    settings=meta['settings']
    size=np.asarray(settings['cell_size'],dtype=float)
    top=float(size[2]*(1+2*settings['skin_layers']/settings['cell'][2]))
    receipt = dict(geometry_sha256=meta['geometry_sha256'], frames=records,
                   blast_center_projection_m=[float(1.5*size[0]),float(1.5*size[1]),top],
                   max_energy_residual=energy_error, physical_deformation_scale=1,
                   interpolation='Original quadratic reference-grid basis at voxel corners; cell stress is the mean of eight quadrature samples.')
    (output/'export.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = export(args.source, args.geometry, args.output)
    print(json.dumps(dict(frames=len(result['frames']), series=str(args.output/'series.pvd'))))


if __name__ == '__main__':
    main()
