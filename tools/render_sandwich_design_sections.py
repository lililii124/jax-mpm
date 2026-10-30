"""Render full-span and magnified central material sections with ParaView."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image
from paraview import simple as p, servermanager
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter


def render(source, output):
    source, output = Path(source), Path(output)
    metadata = json.loads((source/'export.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    p._DisableFirstRenderCameraReset()
    records, symmetry = [], []
    for plane, normal in [('xz', [0, 1, 0]), ('yz', [1, 0, 0])]:
        view = p.CreateView('RenderView')
        view.Background = [1, 1, 1]
        view.UseColorPaletteForBackground = 0
        view.OrientationAxesVisibility = 0
        view.CenterAxesVisibility = 0
        view.CameraParallelProjection = 1
        view.CameraFocalPoint = [.15, .15, .012]
        view.CameraPosition = [.15, -.6, .012] if plane == 'xz' else [.9, .15, .012]
        view.CameraViewUp = [0, 0, 1]
        images = []
        for frame in metadata['frames']:
            filename = source/frame['file']
            assert hashlib.sha256(filename.read_bytes()).hexdigest() == frame['sha256']
            native = vtkXMLRectilinearGridReader()
            native.SetFileName(str(filename)); native.Update()
            converted = vtkAppendFilter(); converted.AddInputData(native.GetOutput()); converted.Update()
            reader = p.TrivialProducer()
            reader.GetClientSideObject().SetOutput(converted.GetOutput()); reader.UpdatePipeline()
            fetched = servermanager.Fetch(reader)
            values = vtk_to_numpy(fetched.GetCellData().GetArray('core_fraction')).copy()
            assert hashlib.sha256(values.tobytes()).hexdigest() == frame['field_sha256']
            if plane == 'xz':
                shape = tuple(n-1 for n in native.GetOutput().GetDimensions())
                field = values.reshape(shape, order='F')
                norm = np.linalg.norm(field)
                symmetry.append(dict(index=frame['index'], controls_sha256=frame['source']['controls_sha256'],
                    mirror_x_relative_l2=float(np.linalg.norm(field-field[::-1,:,:])/norm),
                    mirror_y_relative_l2=float(np.linalg.norm(field-field[:,::-1,:])/norm),
                    layer_mean_fraction=field.mean(axis=(0,1)).tolist(),
                    maximum_thickness_fraction_range=float(np.max(np.ptp(field,axis=2)))))
            section = p.Slice(Input=reader)
            section.SliceType = 'Plane'; section.SliceType.Origin = [.15, .15, .012]
            section.SliceType.Normal = normal
            section.UpdatePipeline()
            if servermanager.Fetch(section).GetNumberOfCells() == 0:
                raise ValueError('The requested central section is empty')
            display = p.Show(section, view)
            display.Representation = 'Surface'; display.Ambient = 1; display.Diffuse = 0
            p.ColorBy(display, ('CELLS', 'core_fraction'))
            lut = p.GetColorTransferFunction('core_fraction')
            lut.ColorSpace = 'RGB'
            lut.RGBPoints = [.08,.10,.22,.47,.17,.12,.64,.73,.26,.98,.78,.24,.35,.76,.12,.12]
            lut.RescaleTransferFunction(.08, .35); lut.AutomaticRescaleRangeMode = 'Never'
            display.LookupTable = lut; display.SetScalarBarVisibility(view, False)
            skins = []
            for z in [.001, .023]:
                box = p.Box(); box.Center = [.15, .15, z]
                box.XLength = .3 if plane == 'xz' else .00001
                box.YLength = .00001 if plane == 'xz' else .3
                box.ZLength = .002
                skin = p.Show(box, view); skin.ColorArrayName = ['POINTS', '']
                skin.DiffuseColor = [.56, .59, .62]; skin.AmbientColor = [.56, .59, .62]
                skin.Ambient = 1; skin.Diffuse = 0
                skins.append((skin, box))
            composite = Image.new('RGB', (400, 400), 'white')
            for label, height, scale, y in [('full', 120, .0486, 0), ('detail', 280, .0322, 120)]:
                view.ViewSize = [1200, height*3]; view.CameraParallelScale = scale
                p.Render(view)
                temporary = output/f'{plane}-{frame["index"]:03d}-{label}.png'
                p.SaveScreenshot(str(temporary), view, ImageResolution=[1200, height*3])
                with Image.open(temporary) as image:
                    reduced = image.convert('RGB').resize((400, height), Image.Resampling.LANCZOS)
                composite.paste(reduced, (0, y)); temporary.unlink()
            composite.save(output/f'{plane}-{frame["index"]:03d}.png')
            images.append(composite)
            for actor, box in skins: p.Delete(actor); p.Delete(box)
            p.Delete(display); p.Delete(section); p.Delete(reader)
        atlas = Image.new('RGB', (400, 400*len(images)))
        for i, image in enumerate(images): atlas.paste(image, (0, 400*i))
        palette = atlas.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
        durations = [650]*len(frames); durations[-1] = 2000
        path = output/f'material-section-{plane}.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=False, disposal=2)
        with Image.open(path) as gif: assert gif.size == (400, 400) and gif.n_frames == len(images)
        records.append(dict(plane=plane, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            center_m=[.15,.15,.012], normal=normal, frames=len(images)))
        p.Delete(view)
    report = dict(renderer='ParaView', scope='Central material sections, not deformation; grey skins have their actual 2mm thickness.',
                  overview_width_m=.324, detail_width_m=.0322*2*400/280,
                  panels='Top: complete span. Bottom: central crop with isotropic magnification; no vertical exaggeration.',
                  export_sha256=hashlib.sha256((source/'export.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  records=records, material_reflection_differences=symmetry)
    (output/'sections-render.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); render(args.source, args.output)
