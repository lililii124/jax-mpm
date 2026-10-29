"""Recover and export a full-MPM face from tensor-product Gauss snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def spline_matrix(points, nodes, spacing):
    distance = np.abs((np.asarray(points)[:, None] - nodes[None, :]) / spacing)
    return np.where(distance < .5, .75-distance**2,
                    np.where(distance < 1.5, .5*(1.5-distance)**2, 0.))


def apply_axis(matrix, values, axis):
    result = np.tensordot(matrix, values, axes=(1, axis))
    return np.moveaxis(result, 0, axis)


def recover_nodes(reference, displacement, spacing):
    spacing = np.asarray(spacing, dtype=float)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or reference.ndim != 2 or reference.shape[1] != 3
            or displacement.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(displacement).all()):
        raise ValueError('Expected finite 3D reference/displacement arrays and positive spacing')
    axes, inverse = zip(*(np.unique(reference[:, a], return_inverse=True) for a in range(3)))
    shape = tuple(len(a) for a in axes)
    if np.prod(shape) != len(reference) or any(n % 2 for n in shape):
        raise ValueError('Expected a complete two-point Gauss tensor grid')
    indices = np.ravel_multi_index(inverse, shape)
    if len(np.unique(indices)) != len(reference):
        raise ValueError('Duplicate reference points')
    sample = np.empty(shape+(3,), dtype=np.float64)
    sample.reshape(-1, 3)[indices] = displacement
    nodes, matrices, conditions = [], [], []
    result = sample
    for a, (points, h) in enumerate(zip(axes, spacing)):
        origin = points[0]-(.5-.5/np.sqrt(3))*h
        node = origin + np.arange(-1, len(points)//2+2)*h
        expected = origin + (np.arange(len(points)//2)[:, None] +
                             [.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)]).ravel()*h
        np.testing.assert_allclose(points, expected, rtol=0, atol=1e-12*h)
        matrix = spline_matrix(points, node, h)
        if np.linalg.matrix_rank(matrix) != len(node):
            raise ValueError('Snapshot does not uniquely determine nodal displacement')
        conditions.append(float(np.linalg.cond(matrix)))
        result = apply_axis(np.linalg.pinv(matrix, rcond=1e-13), result, a)
        nodes.append(node)
        matrices.append(matrix)
    restored = result
    for a, matrix in enumerate(matrices):
        restored = apply_axis(matrix, restored, a)
    error = float(np.max(np.abs(restored-sample)))
    tolerance = 1e-12*max(float(np.max(np.abs(sample))), 1e-6)
    if not np.isfinite(result).all() or error > tolerance:
        raise ValueError(f'Particle reconstruction error {error:g} exceeds {tolerance:g}')
    return axes, nodes, result, dict(particle_reconstruction_error_m=error,
                                    axis_condition_numbers=conditions)


def export_face(source, destination, spacing, face='bottom'):
    with np.load(source, allow_pickle=False) as saved:
        axes, nodes, u, receipt = recover_nodes(saved['reference'], saved['displacement'], spacing)
        time = float(saved['time'])
    z = nodes[2][1] if face == 'bottom' else nodes[2][-2]
    result = u
    for a, points in enumerate((axes[0], axes[1], np.array([z]))):
        result = apply_axis(spline_matrix(points, nodes[a], spacing[a]), result, a)
    displacement = result[:, :, 0].reshape(-1, 3)
    xx, yy = np.meshgrid(axes[0], axes[1], indexing='ij')
    reference = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
    position = reference+displacement
    ids = np.arange(xx.size).reshape(xx.shape)
    cells = np.stack((ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]), axis=-1).reshape(-1, 4)

    def array(name, values, kind='Float64', components=1):
        values = np.asarray(values)
        data = ' '.join(format(float(x), '.17g') if kind == 'Float64' else str(int(x))
                        for x in values.ravel())
        return f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" format="ascii">{data}</DataArray>'

    xml = ('<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
           '<UnstructuredGrid><FieldData>'+array('TimeValue', [time])+'</FieldData>'
           f'<Piece NumberOfPoints="{len(position)}" NumberOfCells="{len(cells)}">'
           '<PointData>'+array('displacement_m', displacement, components=3)+
           array('abs_displacement_z_m', np.abs(displacement[:, 2]))+'</PointData>'
           '<Points>'+array('Points', position, components=3)+'</Points><Cells>'+
           array('connectivity', cells, 'Int64')+array('offsets', 4*np.arange(1, len(cells)+1), 'Int64')+
           array('types', np.full(len(cells), 9), 'UInt8')+'</Cells></Piece></UnstructuredGrid></VTKFile>')
    Path(destination).write_text(xml, encoding='utf-8')
    return dict(receipt, time_s=time, face=face, physical_deformation_scale=1,
                peak_face_displacement_m=float(np.max(np.abs(displacement[:, 2]))),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                method='Unique tensor-product inversion of saved quadratic-MPM Gauss samples; evaluated at original face quadrature.')


def export_series(source, destination, spacing, face):
    summary = json.loads((source/'summary.json').read_text())
    if not summary['completed']:
        raise ValueError('Series export requires the completed physical loading history')
    files = sorted(source.glob('frame_*.npz'))
    if len(files) < 2 or files[0].stem != 'frame_0000000' or files[-1].stem != f"frame_{summary['steps']:07d}":
        raise ValueError('Series is missing its initial or final state')
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source_frame in files:
        target = destination/(source_frame.stem+'.vtu')
        record = export_face(source_frame, target, spacing, face)
        record['file'] = target.name
        records.append(record)
    times = np.asarray([r['time_s'] for r in records])
    if (abs(times[0]) > 1e-15 or np.any(np.diff(times) <= 0)
            or abs(times[-1]-summary['time_s']) > 1e-14):
        raise ValueError('Invalid saved physical times')
    datasets = ''.join(f'<DataSet timestep="{r["time_s"]:.17g}" part="0" file="{r["file"]}"/>' for r in records)
    (destination/'face.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+datasets+'</Collection></VTKFile>')
    receipt = dict(source_summary=summary, frames=records, renderer_input='face.pvd',
                   field='abs_displacement_z_m', physical_deformation_scale=1)
    (destination/'export.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return dict(frames=len(records), first_time_s=float(times[0]), last_time_s=float(times[-1]),
                series=str(destination/'face.pvd'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--spacing', type=float, nargs=3, required=True)
    parser.add_argument('--face', choices=('bottom', 'top'), default='bottom')
    args = parser.parse_args()
    if args.source.is_dir():
        receipt = export_series(args.source, args.destination, args.spacing, args.face)
    else:
        receipt = export_face(args.source, args.destination, args.spacing, args.face)
        args.destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
