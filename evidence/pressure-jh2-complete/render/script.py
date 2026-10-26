"""Render actual JH-2 pressure-panel frames in ParaView; no physics is run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from paraview import servermanager
from paraview.simple import (ColorBy, Delete, GetActiveViewOrCreate, Hide,
    Render, SaveScreenshot, Show, TrivialProducer, _DisableFirstRenderCameraReset)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def polydata(points, displacement=None):
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
    poly = vtkPolyData()
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    poly.SetPoints(vtk_points)
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(len(points) + 1, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True))
    poly.SetVerts(cells)
    if displacement is not None:
        values = numpy_to_vtk(np.asarray(displacement, dtype=np.float32), deep=True)
        values.SetName('displacement_m')
        poly.GetPointData().AddArray(values)
    return poly


def configure_view(view, geometry):
    lower = geometry['lower_m']
    upper = geometry['upper_m']
    centre = [(lower[i] + upper[i]) * 0.5 for i in range(3)]
    _DisableFirstRenderCameraReset()
    view.ViewSize = [400, 400]
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    if hasattr(view, 'CenterAxesVisibility'):
        view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = centre
    view.CameraPosition = [centre[0] + 0.23, centre[1] + 0.24, centre[2] + 0.27]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.14


def style_gaussian(display, radius, color=None):
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = float(radius)
    display.ShaderPreset = 'Sphere'
    display.Emissive = 0
    display.Ambient = 0.3
    display.Diffuse = 0.7
    display.Specular = 0.35
    display.SpecularPower = 35
    display.Opacity = 1.0
    if color is not None:
        display.ColorArrayName = ['POINTS', '']
        display.DiffuseColor = color
        display.AmbientColor = color


def render_one(view, input_dir, output_dir, frame_index, reference, clamp_mask,
               maximum_displacement, gaussian_radius):
    path = input_dir / 'frames_data' / f'frame_{frame_index:05d}.npz'
    digest = sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        x = archive['x'].astype(np.float64, copy=False)
        time_s = float(np.asarray(archive['time']).reshape(-1)[0])
        speed = archive['speed']
        damage = archive['damage']
        temperature = archive['temperature']
    assert x.shape == reference.shape and np.isfinite(x).all(), (path.name, x.shape)
    displacement = np.linalg.norm(x - reference, axis=1)
    assert np.isfinite(displacement).all()
    fixed = x[clamp_mask]
    free = x[~clamp_mask]
    free_displacement = displacement[~clamp_mask]
    assert len(fixed) + len(free) == len(x) and len(free) > 0

    body = TrivialProducer(registrationName=f'JH-2 frame {frame_index} free')
    body.GetClientSideObject().SetOutput(polydata(free, free_displacement))
    body.UpdatePipeline()
    fetched = servermanager.Fetch(body)
    fetched_x = vtk_to_numpy(fetched.GetPoints().GetData())
    assert fetched_x.shape == free.shape and np.array_equal(fetched_x, free), frame_index
    body_display = Show(body, view)
    style_gaussian(body_display, gaussian_radius)
    ColorBy(body_display, ('POINTS', 'displacement_m'))
    body_display.SetScalarBarVisibility(view, False)
    from paraview.simple import GetColorTransferFunction
    lut = GetColorTransferFunction('displacement_m')
    peak = max(float(maximum_displacement), np.finfo(float).tiny)
    lut.ColorSpace = 'RGB'
    lut.RGBPoints = [0.0, 0.07, 0.20, 0.52,
                     0.25 * peak, 0.04, 0.65, 0.82,
                     0.60 * peak, 0.98, 0.78, 0.20,
                     peak, 0.74, 0.08, 0.10]
    lut.RescaleTransferFunction(0.0, peak)
    lut.AutomaticRescaleRangeMode = 'Never'
    body_display.LookupTable = lut

    clamp = TrivialProducer(registrationName=f'JH-2 frame {frame_index} clamp strips')
    clamp.GetClientSideObject().SetOutput(polydata(fixed))
    clamp.UpdatePipeline()
    fetched_clamp = servermanager.Fetch(clamp)
    fetched_fixed = vtk_to_numpy(fetched_clamp.GetPoints().GetData())
    assert fetched_fixed.shape == fixed.shape and np.array_equal(fetched_fixed, fixed), frame_index
    clamp_display = Show(clamp, view)
    style_gaussian(clamp_display, gaussian_radius, [0.58, 0.61, 0.64])

    Render(view)
    png = output_dir / 'png' / f'frame_{frame_index:05d}.png'
    SaveScreenshot(str(png), view, ImageResolution=[400, 400])
    record = {'index': frame_index, 'file': path.name, 'input_sha256': digest,
        'time_s': time_s, 'point_count': int(len(x)), 'clamp_point_count': int(len(fixed)),
        'free_point_count': int(len(free)), 'frame_displacement_peak_m': float(displacement.max()),
        'max_speed_m_per_s': float(np.max(speed)), 'max_damage': float(np.max(damage)),
        'temperature_range_K': [float(np.min(temperature)), float(np.max(temperature))],
        'fixed_strip_max_motion_m': float(np.linalg.norm(fixed-reference[clamp_mask], axis=1).max()),
        'colour_range_m': [0.0, peak], 'fetched_free_coordinates_exact': True,
        'fetched_clamp_coordinates_exact': True, 'png': str(png.name),
        'png_sha256': sha256(png)}
    Hide(body, view); Hide(clamp, view)
    Delete(body_display); Delete(clamp_display); Delete(body); Delete(clamp)
    return record


def build_gif(records, png_dir, gif_path):
    times = [r['time_s'] for r in records]
    delta = np.diff(times)
    median_delta = float(np.median(delta))
    playback_ms_per_median_interval = 90.0
    durations = [max(10, int(round((float(dt) / median_delta * playback_ms_per_median_interval) / 10) * 10))
                 for dt in np.r_[delta, median_delta]]
    grouped = []
    for row, duration in zip(records, durations):
        image = Image.open(png_dir / row['png']).convert('RGB')
        pixels = image.tobytes()
        if grouped and grouped[-1]['pixels'] == pixels:
            grouped[-1]['duration_ms'] += duration
            grouped[-1]['source_indices'].append(row['index'])
        else:
            grouped.append({'image': image, 'pixels': pixels, 'duration_ms': duration,
                            'source_indices': [row['index']]})
    assert len(grouped) >= 2, 'The actual frames rendered to fewer than two distinct images'
    gif_frames = [item['image'].convert('P', palette=Image.ADAPTIVE, colors=256)
                  for item in grouped]
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
        duration=[item['duration_ms'] for item in grouped], loop=0, optimize=False, disposal=2)
    with Image.open(gif_path) as check:
        decoded = [frame.convert('RGB').tobytes() for frame in ImageSequence.Iterator(check)]
    distinct = len(set(decoded))
    assert distinct >= 2, 'Encoded GIF has fewer than two distinct frames'
    return {'source_png_frames': len(records), 'consecutive_duplicate_groups': len(records) - len(grouped),
        'gif_frames_after_duplicate_merge': len(grouped), 'distinct_decoded_gif_frames': distinct,
        'playback_ms_per_median_simulation_interval': playback_ms_per_median_interval,
        'source_simulation_span_s': times[-1] - times[0],
        'encoded_playback_span_ms': int(sum(item['duration_ms'] for item in grouped)),
        'frame_durations_ms_after_merge': [item['duration_ms'] for item in grouped],
        'merged_source_indices': [item['source_indices'] for item in grouped],
        'gif_sha256': sha256(gif_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    input_dir, output_dir = args.input, args.output
    png_dir = output_dir / 'png'
    checks_dir = output_dir / 'gif-checks'
    png_dir.mkdir(parents=True, exist_ok=False)
    checks_dir.mkdir(parents=True, exist_ok=False)
    case = json.loads((input_dir / 'case.json').read_text())
    geometry = case['geometry']
    frames = int(case['frames'])
    assert case['model'] == 'jh2' and case['planned_steps'] == frames * int(case['substeps']) == 8460
    paths = [input_dir / 'frames_data' / f'frame_{i:05d}.npz' for i in range(frames + 1)]
    assert all(path.is_file() for path in paths), 'Incomplete full-frame archive'
    with np.load(paths[0], allow_pickle=False) as archive:
        reference = archive['x'].astype(np.float64, copy=True)
        time_zero = float(np.asarray(archive['time']).reshape(-1)[0])
    assert reference.shape == (case['particle_count'], 3)
    spacing = np.asarray(case['particle_spacing_m'], dtype=np.float64)
    gaussian_radius = 0.6 * float(spacing.min())
    lower, upper = geometry['lower_m'], geometry['upper_m']
    clamp_width = float(geometry['clamp_width_m'])
    clamp_mask = ((reference[:, 0] <= lower[0] + clamp_width) |
                  (reference[:, 0] >= upper[0] - clamp_width) |
                  (reference[:, 2] <= lower[2] + clamp_width) |
                  (reference[:, 2] >= upper[2] - clamp_width))

    # Scan all actual saved coordinates to set one fixed physical colour range.
    whole_course_peak = 0.0
    whole_course_time = time_zero
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            x = archive['x']
            peak = float(np.linalg.norm(x - reference, axis=1).max())
            t = float(np.asarray(archive['time']).reshape(-1)[0])
        if peak > whole_course_peak:
            whole_course_peak, whole_course_time = peak, t
        if i % 50 == 0:
            print(json.dumps({'global_peak_scan_frame': i, 'time_s': t,
                              'course_peak_so_far_m': whole_course_peak}), flush=True)

    selected = np.unique(np.rint(np.linspace(0, frames, 61)).astype(np.int32))
    assert len(selected) == 61 and selected[0] == 0 and selected[-1] == frames
    _DisableFirstRenderCameraReset()
    view = GetActiveViewOrCreate('RenderView')
    configure_view(view, geometry)
    records = []
    for pos, index in enumerate(selected):
        row = render_one(view, input_dir, output_dir, int(index), reference, clamp_mask,
                         whole_course_peak, gaussian_radius)
        records.append(row)
        print(json.dumps({'rendered_frame': pos, **row}), flush=True)
    final_png = png_dir / records[-1]['png']
    (output_dir / 'final.png').write_bytes(final_png.read_bytes())
    gif_path = output_dir / 'simulation.gif'
    gif_checks = build_gif(records, png_dir, gif_path)
    (checks_dir / 'gif-checks.json').write_text(json.dumps(gif_checks, indent=2) + '\n')
    checkpoint_manifest = json.loads((input_dir / 'checkpoint.json').read_text())
    checkpoint_file = input_dir / checkpoint_manifest['states'][0]
    render_record = {'scope': 'Actual saved JH-2 pressure-panel coordinates; no interpolation, no deformation scale, private render only.',
        'model': 'jh2', 'source_case_sha256': sha256(input_dir / 'case.json'),
        'source_checkpoint_sha256': sha256(checkpoint_file),
        'source_checkpoint_frame': frames, 'source_steps': int(case['planned_steps']),
        'source_time_s': [records[0]['time_s'], records[-1]['time_s']],
        'source_frame_count': len(paths), 'rendered_real_frame_indices': selected.tolist(),
        'rendered_frame_count': len(records), 'particle_count_per_frame': int(len(reference)),
        'retained_all_points': True, 'clamp_strip_definition': 'Original case four x/z strips, each width clamp_width_m, selected from reference coordinates; grey only.',
        'clamp_point_count': int(clamp_mask.sum()), 'free_point_count': int((~clamp_mask).sum()),
        'point_gaussian_radius_m': gaussian_radius, 'actual_spacing_m': spacing.tolist(),
        'whole_course_displacement_peak_m': whole_course_peak,
        'whole_course_peak_time_s': whole_course_time,
        'colour_range_m': [0.0, whole_course_peak], 'scale_factor': 1.0,
        'camera': {'position': list(view.CameraPosition), 'focal_point': list(view.CameraFocalPoint),
            'view_up': list(view.CameraViewUp), 'parallel_projection': True,
            'parallel_scale': float(view.CameraParallelScale), 'size_px': [400, 400]},
        'background': 'white', 'axes_and_text': False,
        'frame_records': records, 'gif_checks': gif_checks,
        'final_png_sha256': sha256(output_dir / 'final.png'),
        'simulation_gif_sha256': sha256(gif_path)}
    (output_dir / 'render.json').write_text(json.dumps(render_record, indent=2) + '\n')
    print(json.dumps({'render_complete': True, 'frame_count': len(records),
                      'global_displacement_peak_m': whole_course_peak,
                      'gif_sha256': render_record['simulation_gif_sha256']}), flush=True)


if __name__ == '__main__':
    main()
