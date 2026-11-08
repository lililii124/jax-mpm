"""Render a binary-cell volume series with ParaView."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stress-limit-mpa', type=float, required=True)
    parser.add_argument('--cutaway', action='store_true')
    args = parser.parse_args()
    if args.stress_limit_mpa <= 0:
        parser.error('Stress limit must be positive')
    from PIL import Image
    from paraview import simple as p
    receipt = json.loads((args.source/'export.json').read_text())
    records = receipt['frames']
    if len(records) < 2:
        raise ValueError('An animation requires at least two frames')
    args.output.mkdir(parents=True, exist_ok=False)
    p._DisableFirstRenderCameraReset()
    reader = p.PVDReader(FileName=str(args.source/'series.pvd'))
    reader.UpdatePipeline()
    target = reader
    if args.cutaway:
        target = p.Clip(Input=reader)
        target.ClipType = 'Plane'
        target.ClipType.Origin = [.15, .15, .012]
        target.ClipType.Normal = [0, 1, 0]
        target.UpdatePipeline()
    view = p.CreateView('RenderView')
    view.ViewSize = [1200, 1200]
    view.Background = [1, 1, 1]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    bounds = target.GetDataInformation().GetBounds()
    center = [(bounds[2*i]+bounds[2*i+1])*.5 for i in range(3)]
    view.CameraFocalPoint = center
    view.CameraPosition = [center[0]+.38, center[1]-.67, center[2]+.308]
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelScale = .23
    display = p.Show(target, view)
    display.Representation = 'Surface'
    display.Ambient = .3
    display.Diffuse = .7
    p.ColorBy(display, ('CELLS', 'von_mises_Pa'))
    maximum = args.stress_limit_mpa*1e6
    lut = p.GetColorTransferFunction('von_mises_Pa')
    lut.RGBPoints = [0, .08, .18, .38, maximum*.25, .12, .47, .70,
                     maximum*.5, .25, .73, .65, maximum*.75, .96, .83, .30,
                     maximum, .80, .20, .15]
    lut.RescaleTransferFunction(0, maximum)
    lut.AutomaticRescaleRangeMode = 'Never'
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, False)
    frames = []
    times = [row['time_s'] for row in records]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Frame times must be increasing')
    for index, time in enumerate(times):
        view.ViewTime = time
        target.UpdatePipeline(time)
        p.Render(view)
        path = args.output/f'frame-{index:04d}.png'
        p.SaveScreenshot(str(path), view, ImageResolution=[1200, 1200])
        frames.append(Image.open(path).convert('RGB').resize((400, 400), Image.Resampling.LANCZOS))
        path.unlink()
    durations = [max(10, round((b-a)/(times[-1]-times[0])*5000/10)*10)
                 for a, b in zip(times[:-1], times[1:])]
    durations.append(durations[-1])
    gif = args.output/'binary-blast.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    result = dict(renderer='ParaView', frames=len(frames), stress_limit_pa=maximum,
                  physical_deformation_scale=1, cutaway=args.cutaway,
                  duration_ms=durations, gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                  source_geometry_sha256=receipt['geometry_sha256'])
    (args.output/'render.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
