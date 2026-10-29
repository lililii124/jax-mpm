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
    view.CameraPosition = [.6, -.55, .65]
    view.CameraFocalPoint = [.15, .15, 0]
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelScale = .21
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
                   gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                   duration_ms=duration, source='Completed independent full MPM; no ROM state.')
    (args.output/'render.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(gif=str(gif), frames=len(frames), interval_s=[times[0], times[-1]])))


if __name__ == '__main__':
    main()
