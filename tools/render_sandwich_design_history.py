"""Render discrete material designs with ParaView and a shared colour scale."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image
from paraview import simple as p,servermanager
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter


def render(source,output):
    source,output=Path(source),Path(output)
    metadata=json.loads((source/'export.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    p._DisableFirstRenderCameraReset()
    view=p.CreateView('RenderView');view.ViewSize=[1200,1200]
    view.Background=[1,1,1];view.UseColorPaletteForBackground=0
    view.OrientationAxesVisibility=0;view.CenterAxesVisibility=0;view.CameraParallelProjection=1
    view.CameraPosition=[.15,.15,.6];view.CameraFocalPoint=[.15,.15,.012]
    view.CameraViewUp=[0,1,0];view.CameraParallelScale=.162
    images=[];records=[]
    for frame in metadata['frames']:
        filename=source/frame['file']
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=frame['sha256']:
            raise ValueError('Exported field changed')
        native=vtkXMLRectilinearGridReader();native.SetFileName(str(filename));native.Update()
        converted=vtkAppendFilter();converted.AddInputData(native.GetOutput());converted.Update()
        reader=p.TrivialProducer();reader.GetClientSideObject().SetOutput(converted.GetOutput());reader.UpdatePipeline()
        dataset=servermanager.Fetch(reader)
        values=vtk_to_numpy(dataset.GetCellData().GetArray('core_fraction')).copy()
        if hashlib.sha256(values.tobytes()).hexdigest()!=frame['field_sha256']:
            raise ValueError('VTK material field differs from the original cell averages')
        display=p.Show(reader,view);display.Representation='Surface';display.Ambient=1;display.Diffuse=0
        p.ColorBy(display,('CELLS','core_fraction'));lut=p.GetColorTransferFunction('core_fraction')
        lut.ColorSpace='RGB';lut.RGBPoints=[.08,.10,.22,.47,.17,.12,.64,.73,.26,.98,.78,.24,.35,.76,.12,.12]
        lut.RescaleTransferFunction(.08,.35);lut.AutomaticRescaleRangeMode='Never';display.LookupTable=lut
        display.SetScalarBarVisibility(view,False)
        p.Render(view);path=output/f"frame-{frame['index']:03d}.png"
        p.SaveScreenshot(str(path),view,ImageResolution=[1200,1200])
        with Image.open(path) as image:reduced=image.convert('RGB').resize((400,400),Image.Resampling.LANCZOS)
        reduced.save(path);images.append(reduced)
        records.append(dict(index=frame['index'],file=path.name,source=frame['source'],
            material_field_bitwise_verified=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        p.Delete(display);p.Delete(reader)
    atlas=Image.new('RGB',(400,400*len(images)))
    for index,image in enumerate(images):atlas.paste(image,(0,400*index))
    palette=atlas.quantize(colors=255,method=Image.Quantize.MEDIANCUT)
    frames=[image.quantize(palette=palette,dither=Image.Dither.NONE) for image in images]
    path=output/'material-redistribution.gif';durations=[650]*len(frames);durations[-1]=2000
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=2)
    with Image.open(path) as gif:count=gif.n_frames
    report=dict(renderer='ParaView',render_states=len(frames),gif_frames=count,size=[400,400],color_range=[.08,.35],
        reader='VTK rectilinear grid converted to an unstructured grid without resampling; full cell array hash checked after ParaView transfer.',
        physical_scale=1,scope='Discrete retained design sequence; no interpolation, deformation or physical-time animation.',
        export_receipt_sha256=hashlib.sha256((source/'export.json').read_bytes()).hexdigest(),
        gif_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'render.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(render_states=len(frames),gif_frames=count,gif=str(path))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();render(args.source,args.output)
