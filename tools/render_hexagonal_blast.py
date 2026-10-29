"""Run with pvpython: native computed VTU/PVD, clean 400px stress animation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--end-step',type=int);p.add_argument('--color-max',type=float);p.add_argument('--assemble-gif',action='store_true');args=p.parse_args()
if args.assemble_gif:
    from PIL import Image
    paths=sorted(Path(args.input).glob('*.png'))
    if not paths:raise ValueError('No ParaView PNG frames')
    images=[Image.open(path).convert('RGB') for path in paths]
    if any(im.size!=(400,400) for im in images):raise ValueError('Expected 400x400 frames')
    sheet=Image.new('RGB',(400,400*len(images)))
    for i,im in enumerate(images):sheet.paste(im,(0,400*i))
    palette=sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(dest,save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=False,disposal=2)
    with Image.open(dest) as gif:
        count=gif.n_frames
        unique=set()
        for i in range(count):gif.seek(i);unique.add(hashlib.sha256(gif.convert('RGB').tobytes()).hexdigest())
    dest.with_suffix('.receipt.json').write_text(json.dumps(dict(frames=count,distinct_frames=len(unique),dimensions=[400,400],sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
    raise SystemExit(0)
from paraview.simple import XMLUnstructuredGridReader, CreateView, Show, ColorBy, GetColorTransferFunction, Render, SaveScreenshot
from paraview.simple import _DisableFirstRenderCameraReset
_DisableFirstRenderCameraReset()
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy

source=Path(args.input);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
files=sorted((source/'vtk').glob('*.vtu'))
if args.end_step is not None:files=[f for f in files if int(f.stem.split('_')[-1])<=args.end_step]
metadata=json.loads((source/'angular_metrics.json').read_text())
reader=XMLUnstructuredGridReader(FileName=[str(f) for f in files])
view=CreateView('RenderView');view.ViewSize=[400,400];view.UseColorPaletteForBackground=0;view.Background=[1.,1.,1.]
view.OrientationAxesVisibility=0;view.CenterAxesVisibility=0;view.CameraParallelProjection=1
view.CameraPosition=[0.,0.,1.];view.CameraFocalPoint=[0.,0.,0.];view.CameraViewUp=[0.,1.,0.];view.CameraParallelScale=.162
rep=Show(reader,view);rep.Representation='Surface';rep.Ambient=1.;rep.Diffuse=0.;rep.SetScalarBarVisibility(view,False)
ColorBy(rep,('POINTS','von_mises_Pa'));lut=GetColorTransferFunction('von_mises_Pa')
records=[];maximum=0.
for i,f in enumerate(files):
    reader.UpdatePipeline(float(i));data=servermanager.Fetch(reader)
    a=vtk_to_numpy(data.GetPointData().GetArray('von_mises_Pa'));maximum=max(maximum,float(a.max()))
    actual_sha256=hashlib.sha256(a.tobytes()).hexdigest()
    if metadata[i].get('surface_vm_sha256',actual_sha256)!=actual_sha256:raise ValueError('ParaView/source field hash mismatch')
    records.append(dict(above_display_max_count=int(np.count_nonzero(a>args.color_max)) if args.color_max is not None else 0,above_display_max_fraction=float(np.mean(a>args.color_max)) if args.color_max is not None else 0.,time_s=metadata[i]['time_s'],frame=f.name,points=int(data.GetNumberOfPoints()),min_Pa=float(a.min()),max_Pa=float(a.max()),sha256=hashlib.sha256(a.tobytes()).hexdigest()))
actual_maximum=maximum
if args.color_max is not None:maximum=args.color_max
lut.RGBPoints=[0.,.267,.005,.329,maximum*.25,.230,.322,.546,maximum*.5,.128,.567,.551,maximum*.75,.369,.789,.383,maximum,.993,.906,.144]
lut.ColorSpace='RGB';lut.ScalarRangeInitialized=1.;lut.RescaleTransferFunction(0.,maximum)
for i,f in enumerate(files):
    reader.UpdatePipeline(float(i));view.ViewTime=float(i);rep.SetScalarBarVisibility(view,False);Render(view)
    SaveScreenshot(str(out/f'{i:04d}.png'),view,ImageResolution=[400,400])
(out/'render_receipt.json').write_text(json.dumps(dict(renderer='ParaView pvpython',dimensions=[400,400],field='actual upper quadrature-layer Cauchy von Mises, Pa',actual_global_maximum_Pa=actual_maximum,color_range_Pa=[0.,maximum],background='white',axes=False,text=False,frames=records),indent=2))
