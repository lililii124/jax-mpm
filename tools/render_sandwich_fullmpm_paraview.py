"""Render a completed full-MPM sandwich history with ParaView."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sandwich_fullmpm_face import export_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--spacing', type=float, nargs=3, required=True)
    parser.add_argument('--color-limit-mm', type=float, default=1.)
    parser.add_argument('--camera-side', action='store_true')
    parser.add_argument('--show-blast-center', action='store_true')
    args = parser.parse_args()
    if args.color_limit_mm <= 0:
        parser.error('color-limit-mm must be positive')
    from PIL import Image
    from paraview import simple as p
    args.output.mkdir(parents=True, exist_ok=False)
    export_series(args.source, args.output/'face', args.spacing, 'bottom')
    receipt = json.loads((args.output/'face/export.json').read_text())
    times = [row['time_s'] for row in receipt['frames']]
    p._DisableFirstRenderCameraReset()
    reader = p.PVDReader(FileName=str(args.output/'face/face.pvd'))
    view = p.CreateView('RenderView')
    view.ViewSize = [1200, 1200]
    view.Background = [1, 1, 1]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [.15, -.65, .05] if args.camera_side else [.6, -.55, .65]
    view.CameraFocalPoint = [.15, .15, 0]
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelScale = .17 if args.camera_side else .21
    display = p.Show(reader, view)
    display.Representation = 'Surface'
    display.Ambient = 1
    display.Diffuse = 0
    p.ColorBy(display, ('POINTS', 'abs_displacement_z_m'))
    lut = p.GetColorTransferFunction('abs_displacement_z_m')
    maximum = args.color_limit_mm/1000
    lut.ColorSpace = 'RGB'
    lut.RGBPoints = [0, .08, .18, .38, maximum*.25, .12, .47, .70,
                     maximum*.5, .25, .73, .65, maximum*.75, .96, .83, .30,
                     maximum, .80, .20, .15]
    lut.RescaleTransferFunction(0, maximum)
    lut.AutomaticRescaleRangeMode = 'Never'
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, False)
    if args.show_blast_center:
        summary = receipt['source_summary']
        if summary['case'] != 'central':
            raise ValueError('Blast center marker requires the central-pressure case')
        settings = summary['plan']['settings']
        center = [settings['length']/2, settings['width']/2, 0.]
        marker = p.Disk()
        marker.InnerRadius = .004
        marker.OuterRadius = .0055
        marker.CircumferentialResolution = 64
        location = p.Transform(Input=marker)
        location.Transform.Translate = [center[0], center[1], .04]
        marker_display = p.Show(location, view)
        marker_display.DiffuseColor = [.72, .08, .04]
        marker_display.AmbientColor = [.72, .08, .04]
        marker_display.Ambient = 1
        marker_display.Diffuse = 0
        projection = p.Line(Point1=center, Point2=[center[0], center[1], .04])
        projection_display = p.Show(projection, view)
        projection_display.DiffuseColor = [.72, .08, .04]
        projection_display.AmbientColor = [.72, .08, .04]
        projection_display.Ambient = 1
        projection_display.Diffuse = 0
        projection_display.LineWidth = 2
    frames = []
    for i, time in enumerate(times):
        view.ViewTime = time
        reader.UpdatePipeline(time)
        p.Render(view)
        path = args.output/f'frame-{i:04d}.png'
        p.SaveScreenshot(str(path), view, ImageResolution=[1200, 1200])
        frame = Image.open(path).convert('RGB').resize((400, 400), Image.Resampling.LANCZOS)
        frame.save(path)
        frames.append(frame)
    duration = [max(10, round((b-a)/(times[-1]-times[0])*5000/10)*10)
                for a, b in zip(times[:-1], times[1:])]
    duration.append(duration[-1])
    gif = args.output/'fullmpm-backface.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, disposal=2, optimize=False)
    receipt.update(renderer='ParaView', size=[400, 400], color_range_m=[0, maximum],
                   camera_side=args.camera_side, blast_center_marker=args.show_blast_center,
                   gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                   duration_ms=duration, source='Completed independent full MPM; no ROM state.')
    (args.output/'render.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(gif=str(gif), frames=len(frames), interval_s=[times[0], times[-1]])))


if __name__ == '__main__':
    main()
