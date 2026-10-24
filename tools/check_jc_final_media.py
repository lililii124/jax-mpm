"""Check every JC ParaView export, then verify actual PNG/GIF deliverables."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def vtk_audit(folder):
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    files = {}
    for index in range(61):
        raw = folder / 'frames_data' / f'frame_{index:05d}.npz'
        vtp = folder / 'vtk' / f'particles_{index:05d}.vtp'
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp))
        reader.Update()
        result = reader.GetOutput()
        with np.load(raw, allow_pickle=False) as frame:
            if result.GetNumberOfPoints() != len(frame['x']):
                raise ValueError('ParaView export lost particles')
            np.testing.assert_array_equal(vtk_to_numpy(result.GetPoints().GetData()), frame['x'].astype(np.float32))
            for name in ('speed', 'damage', 'temperature', 'failed'):
                actual = vtk_to_numpy(result.GetPointData().GetArray(name))
                expected = frame[name] if name == 'failed' else frame[name].astype(np.float32)
                np.testing.assert_array_equal(actual, expected)
        files[vtp.relative_to(folder).as_posix()] = digest(vtp)
    return {'schema': 'jc-final-vtk-audit-v1', 'passed': True, 'frames_checked': 61,
            'geometry': 'All saved coordinates cast to export float32, without displacement scaling', 'files': files}


def media_audit(folder):
    from PIL import Image
    render = json.loads((folder/'render.json').read_text(encoding='utf-8'))
    vtk = json.loads((folder/'vtk-full-audit.json').read_text(encoding='utf-8'))
    review = json.loads((folder/'full-result-review.json').read_text(encoding='utf-8'))
    if not (vtk['passed'] is True and vtk['frames_checked'] == 61 and render['renderer'] == 'ParaView'
            and render['frames_rendered'] == 61 and render['geometry_scale'] == 1.0
            and render['view_scope'] == 'complete scene' and render['field'] == 'damage'
            and render['color_range'] == [0, 1] and render['resolution'] == [1200, 1200]
            and render['animation'] == 'simulation.gif' and render['particles'] == 341584
            and render['failure_display']['all_particles_retained'] is True):
        raise ValueError('Unexpected rendering provenance')
    counts = render['failure_display']['displayed_counts']
    if len(counts) != 61:
        raise ValueError('Missing displayed particle counts')
    files = {}
    for index, count in enumerate(counts):
        row = review['separation_history'][index]
        if not (count['active_particles'] == row['active_particles']
                and count['debris_particles'] == row['failed_particles']
                and abs(count['time'] - row['time_s']) < 1e-15):
            raise ValueError('Displayed active/debris counts differ from raw history')
        path = folder/'frames'/f'frame_{index:05d}.png'
        with Image.open(path) as frame:
            frame.load()
            if frame.size != (1200, 1200):
                raise ValueError('Wrong PNG dimensions')
        files[path.relative_to(folder).as_posix()] = digest(path)
    with Image.open(folder/'simulation.gif') as gif:
        if gif.n_frames != 61 or gif.size != (1200, 1200):
            raise ValueError('GIF omitted rendered frames or has wrong dimensions')
        for index in range(61):
            gif.seek(index)
            gif.load()
            if gif.info.get('duration') != 60:
                raise ValueError('GIF timing differs from real render encoding')
    for name, expected in review['input_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Numerical raw input changed during rendering: ' + name)
    for name in ('simulation.gif', 'scene.pvsm', 'render.json', 'vtk_metadata.json', 'vtk-full-audit.json'):
        files[name] = digest(folder/name)
    return {'schema': 'jc-final-media-audit-v1', 'passed': True, 'real_paraview': True,
            'full_protocol_completed': review['full_protocol_completed'],
            'resolved_saved_history_fracture': review['resolved_saved_history_fracture'],
            'production_eligible': False, 'files': files,
            'visual_review_required': True,
            'caption': 'JC ballistic particle-model fracture' if review['resolved_saved_history_fracture']
                else 'Completed JC diagnostic; saved-history fracture gates not all satisfied',
            'review_source_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--vtk', action='store_true', help='Run with real pvpython')
    args = parser.parse_args()
    report = vtk_audit(args.folder) if args.vtk else media_audit(args.folder)
    name = 'vtk-full-audit.json' if args.vtk else 'media-audit.json'
    (args.folder/name).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'audit': name}))


if __name__ == '__main__':
    main()
